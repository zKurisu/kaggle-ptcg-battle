#!/usr/bin/env python3
"""Conservative PPO fine-tuning from a BC2 checkpoint against a policy pool.

This is intentionally a small, inspectable training loop for targeted matchup
work. It does not replace BC training: start from a strong BC/shadow checkpoint,
train against a curated weakness pool, and validate every saved checkpoint with
the existing random/RR/baseline-delta tools.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, sequence_nll
from ptcg_rl.encoder import FastEncoder
from ptcg_rl.model import NEG_INF, PolicyValueNet
from ptcg_rl.numpy_policy import NumpyPolicy
from tools.bc2_train import (
    _configure_cuda_memory_limit,
    _load_npz_init,
    _save_npz,
)
from tools.eval_baseline_delta import read_manifest_entries
from tools.eval_round_robin import Entry, clean_entry_name, parse_entry, policy_action, read_deck


METRIC_FIELDS = [
    "iter",
    "games",
    "decisions",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "avg_reward",
    "avg_decisions_per_game",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "bc_anchor_loss",
    "t_collect",
    "t_update",
    "checkpoint",
]


def has_cg_engine() -> bool:
    try:
        return importlib.util.find_spec("cg.game") is not None
    except ModuleNotFoundError:
        return False


def checkpoint_config(path: str) -> dict:
    with np.load(path) as z:
        ec = int(z["card_emb.weight"].shape[1])
        width = ec / 64.0
        state_in = int(z["state_fc1.weight"].shape[1])
        slot_feat_dim = state_in - 5 * ec
        legacy_feat_dim = state_in - 3 * ec
        slot_state = 8 <= slot_feat_dim <= 256
        state_feat_dim = slot_feat_dim if slot_state else legacy_feat_dim
        option_context = "context_emb.weight" in z.files
        opt_extra = 0
        if option_context:
            opt_extra = (
                z["context_emb.weight"].shape[1]
                + z["select_type_emb.weight"].shape[1]
                + z["area_emb.weight"].shape[1]
                + z["index_emb.weight"].shape[1]
                + z["inplay_area_emb.weight"].shape[1]
                + z["inplay_index_emb.weight"].shape[1]
            )
        opt_feat_dim = int(z["opt_fc.weight"].shape[1]) - (
            2 * ec
            + int(z["attack_emb.weight"].shape[1])
            + int(z["opt_type_emb.weight"].shape[1])
            + opt_extra
        )
    return {
        "width": float(width),
        "slot_state": bool(slot_state),
        "option_context": bool(option_context),
        "state_feat_dim": int(state_feat_dim),
        "opt_feat_dim": int(opt_feat_dim),
    }


def legal_random(sel: dict) -> list[int]:
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []


@torch.no_grad()
def sample_trainable_action(
    model: PolicyValueNet,
    decision,
    *,
    greedy: bool = False,
) -> tuple[list[int], bool, float, float]:
    dev = next(model.parameters()).device
    board = torch.from_numpy(decision.board_cards).unsqueeze(0).to(dev)
    hand = torch.from_numpy(decision.hand_cards).unsqueeze(0).to(dev)
    feats = torch.from_numpy(decision.state_feats).unsqueeze(0).to(dev)
    h = model.encode_state(board, hand, feats)
    value = float(model.value(h)[0])

    n_options = len(decision.opt_type)
    ot = torch.from_numpy(decision.opt_type).unsqueeze(0).to(dev)
    oc = torch.from_numpy(decision.opt_card).unsqueeze(0).to(dev)
    oc2 = torch.from_numpy(decision.opt_card2).unsqueeze(0).to(dev)
    oa = torch.from_numpy(decision.opt_attack).unsqueeze(0).to(dev)
    of = torch.from_numpy(decision.opt_feats).unsqueeze(0).to(dev)
    opts = model.encode_options(ot, oc, oc2, oa, of)

    picks: list[int] = []
    stopped = False
    logprob = 0.0
    picked_sum = torch.zeros(1, model._oe, device=dev)
    avail = torch.ones(1, n_options + 1, dtype=torch.bool, device=dev)
    max_count = int(decision.max_count)
    min_count = int(decision.min_count)

    while len(picks) < max_count:
        avail[0, n_options] = len(picks) >= min_count
        logits = model.option_logits(h, opts, picked_sum, avail)
        logp = F.log_softmax(logits, dim=-1)
        if greedy:
            idx = int(logp.argmax(dim=-1)[0])
        else:
            probs = torch.exp(logp)
            idx = int(torch.multinomial(probs, 1)[0, 0])
        logprob += float(logp[0, idx])
        if idx == n_options:
            stopped = True
            break
        picks.append(idx)
        picked_sum = picked_sum + opts[0, idx]
        avail[0, idx] = False

    return picks, stopped, logprob, value


def outcome_for_candidate(result: int, candidate_side: int) -> str:
    if result == candidate_side:
        return "win"
    if result in (0, 1):
        return "loss"
    return "draw"


def play_training_game(
    model: PolicyValueNet,
    candidate_deck: list[int],
    opponent: Entry,
    encoder: FastEncoder,
    args: argparse.Namespace,
    *,
    game_index: int,
    seed: int,
) -> tuple[list, dict]:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    swapped = bool(game_index % 2)
    candidate_side = 1 if swapped else 0
    first_deck, second_deck = (opponent.deck, candidate_deck) if swapped else (candidate_deck, opponent.deck)
    obs, sd = battle_start(first_deck, second_deck)
    decisions = []
    result = 2
    steps = 0

    if obs is None:
        return decisions, {
            "outcome": "draw",
            "opponent": opponent.name,
            "steps": 0,
            "candidate_side": candidate_side,
        }

    try:
        model.eval()
        for steps in range(args.max_turns):
            cur = obs.get("current") or {}
            res = int(cur.get("result", -1))
            if res != -1:
                result = res if res in (0, 1) else 2
                break
            sel = obs.get("select")
            if sel is None:
                result = 2
                break
            side = int(cur.get("yourIndex", 0))
            if side == candidate_side:
                try:
                    decision = encoder.encode(obs)
                    action, stopped, logprob, value = sample_trainable_action(
                        model, decision, greedy=args.greedy_rollout
                    )
                    decision.action = action
                    decision.logprob = logprob
                    decision.value = value
                    setattr(decision, "stopped", stopped)
                    decisions.append(decision)
                except Exception:
                    action = legal_random(sel)
            else:
                action = policy_action(
                    opponent,
                    obs,
                    use_mcts=args.opponent_mcts,
                    sims=args.opponent_mcts_sims,
                    time_budget=args.opponent_time_budget,
                )
            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()

    outcome = outcome_for_candidate(result, candidate_side)
    reward = args.win_reward if outcome == "win" else args.loss_reward if outcome == "loss" else args.draw_reward
    if decisions:
        decisions[-1].reward = float(reward)
    return decisions, {
        "outcome": outcome,
        "opponent": opponent.name,
        "steps": steps,
        "candidate_side": candidate_side,
    }


def compute_gae(samples: list, gamma: float, lam: float) -> None:
    gae = 0.0
    for i in range(len(samples) - 1, -1, -1):
        next_value = samples[i + 1].value if i + 1 < len(samples) else 0.0
        delta = samples[i].reward + gamma * next_value - samples[i].value
        gae = delta + gamma * lam * gae
        samples[i].adv = gae
        samples[i].ret = gae + samples[i].value


def load_opponent_specs(args: argparse.Namespace) -> tuple[list[str], dict[str, float]]:
    specs: list[str] = list(args.opponent)
    weights: dict[str, float] = {}
    for spec in specs:
        name = clean_entry_name(spec.split("=", 1)[0] if "=" in spec else spec)
        weights[name] = 1.0

    name_res = [re.compile(pat) for pat in args.manifest_name_regex]
    archetype_res = [re.compile(pat, re.I) for pat in args.manifest_archetype_regex]
    for manifest in args.opponent_manifest:
        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        if name_res or archetype_res:
            kept = []
            for row in rows:
                raw_name = row.get("shadow_name") or row.get("name") or row.get("team_name") or row.get("deck_sig") or ""
                archetype = row.get("archetype") or ""
                if name_res and not any(rx.search(raw_name) for rx in name_res):
                    continue
                if archetype_res and not any(rx.search(archetype) for rx in archetype_res):
                    continue
                kept.append(row)
            tmp = Path(args.tmp_manifest_dir) / f"rl_manifest_filter_{len(specs)}_{len(kept)}.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
                writer.writeheader()
                writer.writerows(kept)
            manifest_path = str(tmp)
        else:
            manifest_path = manifest
        for name, spec, weight in read_manifest_entries(
            manifest_path, limit=args.manifest_limit, random_from_deck=args.manifest_random
        ):
            specs.append(spec)
            weights[name] = float(weight)

    if not specs:
        raise ValueError("provide --opponent or --opponent-manifest with at least one valid opponent")
    return specs, weights


def sample_opponent(opponents: list[Entry], weights: dict[str, float], mode: str) -> Entry:
    if mode == "uniform":
        return random.choice(opponents)
    raw = np.asarray([max(float(weights.get(opp.name, 1.0)), 1e-6) for opp in opponents], dtype=np.float64)
    probs = raw / raw.sum()
    return opponents[int(np.random.choice(len(opponents), p=probs))]


def load_opponents(specs: list[str], default_deck: str, *, skip_bad_entries: bool) -> list[Entry]:
    entries: list[Entry] = []
    seen_names: dict[str, int] = {}
    seen_specs: set[tuple[str, str]] = set()
    for spec in specs:
        name, policy_path, deck_path = parse_entry(spec, default_deck)
        key = (policy_path, deck_path)
        if key in seen_specs:
            continue
        seen_specs.add(key)
        n = seen_names.get(name, 0) + 1
        seen_names[name] = n
        unique_name = name if n == 1 else f"{name}_{n}"
        try:
            deck = read_deck(deck_path)
            policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
        except Exception as exc:
            if skip_bad_entries:
                print(f"Skipping bad opponent {unique_name}: {exc}", flush=True)
                continue
            raise
        entries.append(Entry(unique_name, policy_path, deck_path, policy, deck))
    if not entries:
        raise ValueError("no valid opponents loaded")
    return entries


def setup_anchor_corpus(args: argparse.Namespace, state_feat_dim: int, opt_feat_dim: int):
    if args.bc_anchor_weight <= 0:
        return None, []
    if not args.bc_anchor_corpus:
        raise ValueError("--bc-anchor-weight requires --bc-anchor-corpus")
    score_bands = args.bc_anchor_score_bands or args.score_bands
    archetype = args.bc_anchor_archetype or args.archetype
    paths = discover_npz_paths(args.bc_anchor_corpus, archetype, score_bands)
    corpus = BCCorpus(
        paths,
        include_empty=args.bc_anchor_include_empty,
        state_feat_dim=state_feat_dim,
        opt_feat_dim=opt_feat_dim,
        deck_sigs=args.bc_anchor_deck_sig,
        team_names=args.bc_anchor_team_name,
        opponent_deck_sigs=args.bc_anchor_opponent_deck_sig,
        opponent_archetypes=args.bc_anchor_opponent_archetype,
        opponent_team_names=args.bc_anchor_opponent_team_name,
        winner_only=args.bc_anchor_winner_only,
        win_weight=args.bc_anchor_win_weight,
        loss_weight=args.bc_anchor_loss_weight,
        draw_weight=args.bc_anchor_draw_weight,
        load_progress_every=args.bc_anchor_load_progress_every,
    )
    indices = corpus.all_indices()
    if not indices:
        raise RuntimeError("BC anchor corpus has no usable samples after filters")
    return corpus, indices


def sample_anchor_batch(corpus: BCCorpus, indices: list, batch_size: int, device: torch.device):
    if batch_size <= 0:
        return None
    replace = len(indices) < batch_size
    picked = np.random.choice(len(indices), size=batch_size, replace=replace)
    return corpus.collate([indices[int(i)] for i in picked], device)


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    samples: list,
    args: argparse.Namespace,
    *,
    device: torch.device,
    anchor_corpus: BCCorpus | None,
    anchor_indices: list,
) -> dict:
    model.train()
    advs = np.asarray([s.adv for s in samples], dtype=np.float32)
    adv_mean = float(advs.mean())
    adv_std = float(advs.std() + 1e-8)
    order = np.arange(len(samples))
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "bc_anchor_loss": 0.0,
        "n": 0,
    }

    for _ in range(args.ppo_epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), args.minibatch):
            mb_idx = order[start : start + args.minibatch]
            if len(mb_idx) < 2:
                continue
            mb = [samples[int(i)] for i in mb_idx]
            old_logprob = torch.as_tensor([s.logprob for s in mb], dtype=torch.float32, device=device)
            adv = torch.as_tensor([s.adv for s in mb], dtype=torch.float32, device=device)
            ret = torch.as_tensor([s.ret for s in mb], dtype=torch.float32, device=device)
            adv = (adv - adv_mean) / adv_std

            new_logprob, entropy, value = model.evaluate_actions(mb)
            ratio = torch.exp(new_logprob - old_logprob)
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            value_loss = (value - ret).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()

            anchor_loss = torch.tensor(0.0, device=device)
            if anchor_corpus is not None and args.bc_anchor_weight > 0:
                batch = sample_anchor_batch(anchor_corpus, anchor_indices, args.bc_anchor_batch_size, device)
                anchor_loss = sequence_nll(
                    model,
                    batch,
                    first_action_weight=args.bc_anchor_first_action_weight,
                    value_weight=args.bc_anchor_value_weight,
                    set_loss_weight=args.bc_anchor_set_loss_weight,
                    set_loss_min_count=args.bc_anchor_set_loss_min_count,
                    set_loss_negative_weight=args.bc_anchor_set_loss_negative_weight,
                )
                loss = loss + args.bc_anchor_weight * anchor_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - (new_logprob - old_logprob)).mean()
                clip_frac = (ratio - 1.0).abs().gt(args.clip_eps).float().mean()
            stats["policy_loss"] += float(policy_loss.detach().cpu())
            stats["value_loss"] += float(value_loss.detach().cpu())
            stats["entropy"] += float(entropy.mean().detach().cpu())
            stats["approx_kl"] += float(approx_kl.detach().cpu())
            stats["clip_frac"] += float(clip_frac.detach().cpu())
            stats["bc_anchor_loss"] += float(anchor_loss.detach().cpu())
            stats["n"] += 1

    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}


def append_metrics(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-init", required=True, help="BC2 .npz checkpoint to fine-tune")
    p.add_argument("--deck", required=True, help="candidate deck CSV")
    p.add_argument("--save", required=True, help="final/best output .npz path")
    p.add_argument("--checkpoint-dir", default="", help="optional directory for per-iteration checkpoints")
    p.add_argument("--metrics-csv", default="", help="append per-iteration metrics here")
    p.add_argument("--archetype", default="", help="label used only for logs and BC anchor defaults")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])

    p.add_argument("--opponent", action="append", default=[], help="NAME=POLICY:DECK, or NAME=random:DECK")
    p.add_argument("--opponent-manifest", action="append", default=[], help="shadow manifest CSV")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--manifest-random", action="store_true", help="use manifest decks as random opponents")
    p.add_argument("--manifest-name-regex", action="append", default=[])
    p.add_argument("--manifest-archetype-regex", action="append", default=[])
    p.add_argument("--tmp-manifest-dir", default="/tmp")
    p.add_argument("--skip-bad-entries", action="store_true")
    p.add_argument("--opponent-weight-mode", choices=["uniform", "manifest"], default="uniform")
    p.add_argument("--opponent-mcts", action="store_true")
    p.add_argument("--opponent-mcts-sims", type=int, default=48)
    p.add_argument("--opponent-time-budget", type=float, default=4.0)

    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip-eps", type=float, default=0.1)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--win-reward", type=float, default=1.0)
    p.add_argument("--loss-reward", type=float, default=-1.0)
    p.add_argument("--draw-reward", type=float, default=0.0)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--greedy-rollout", action="store_true", help="debug only; PPO should normally sample")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuda-memory-gb", type=float, default=0.0)
    p.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    p.add_argument("--width", type=float, default=0.0, help="override width; default inferred from --policy-init")
    p.add_argument("--state-feat-dim", type=int, default=0, help="override state features; default inferred")
    p.add_argument("--opt-feat-dim", type=int, default=0, help="override option features; default inferred")
    p.add_argument("--legacy-state-pool", action="store_true")
    p.add_argument("--no-option-context", action="store_true")

    p.add_argument("--bc-anchor-weight", type=float, default=0.0)
    p.add_argument("--bc-anchor-corpus", default="")
    p.add_argument("--bc-anchor-archetype", default="")
    p.add_argument("--bc-anchor-score-bands", nargs="+", default=[])
    p.add_argument("--bc-anchor-deck-sig", action="append", default=[])
    p.add_argument("--bc-anchor-team-name", action="append", default=[])
    p.add_argument("--bc-anchor-opponent-deck-sig", action="append", default=[])
    p.add_argument("--bc-anchor-opponent-archetype", action="append", default=[])
    p.add_argument("--bc-anchor-opponent-team-name", action="append", default=[])
    p.add_argument("--bc-anchor-winner-only", action="store_true")
    p.add_argument("--bc-anchor-include-empty", action="store_true")
    p.add_argument("--bc-anchor-win-weight", type=float, default=1.0)
    p.add_argument("--bc-anchor-loss-weight", type=float, default=1.0)
    p.add_argument("--bc-anchor-draw-weight", type=float, default=1.0)
    p.add_argument("--bc-anchor-batch-size", type=int, default=512)
    p.add_argument("--bc-anchor-first-action-weight", type=float, default=1.0)
    p.add_argument("--bc-anchor-value-weight", type=float, default=0.0)
    p.add_argument("--bc-anchor-set-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-anchor-set-loss-min-count", type=int, default=2)
    p.add_argument("--bc-anchor-set-loss-negative-weight", type=float, default=0.25)
    p.add_argument("--bc-anchor-load-progress-every", type=int, default=0)

    p.add_argument("--dry-run", action="store_true", help="load model/opponent/anchor and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not has_cg_engine():
        raise SystemExit(
            "cg.game not found. Run this in the Kaggle/remote repo with the cg engine available."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = checkpoint_config(args.policy_init)
    width = args.width or cfg["width"]
    state_feat_dim = args.state_feat_dim or cfg["state_feat_dim"]
    opt_feat_dim = args.opt_feat_dim or cfg["opt_feat_dim"]
    slot_state = not args.legacy_state_pool and bool(cfg["slot_state"])
    option_context = (not args.no_option_context) and bool(cfg["option_context"])

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    memory_msg = _configure_cuda_memory_limit(
        device, gb=args.cuda_memory_gb, fraction=args.cuda_memory_fraction
    )
    model = PolicyValueNet(
        width=width,
        slot_state=slot_state,
        option_context=option_context,
        state_feat_dim=state_feat_dim,
        opt_feat_dim=opt_feat_dim,
    ).to(device)
    loaded, skipped = _load_npz_init(model, args.policy_init, device, partial=False)
    if skipped:
        raise RuntimeError(f"strict init unexpectedly skipped tensors: {skipped[:8]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    candidate_deck = read_deck(args.deck)
    specs, weights = load_opponent_specs(args)
    opponents = load_opponents(
        specs,
        default_deck=args.deck,
        skip_bad_entries=args.skip_bad_entries,
    )
    anchor_corpus, anchor_indices = setup_anchor_corpus(args, state_feat_dim, opt_feat_dim)

    print(
        f"RL fine-tune: init={args.policy_init} save={args.save} device={device} "
        f"{memory_msg + ' ' if memory_msg else ''}"
        f"width={width:g} slot_state={slot_state} option_context={option_context} "
        f"state_feat_dim={state_feat_dim} opt_feat_dim={opt_feat_dim} "
        f"opponents={len(opponents)} opponent_weight_mode={args.opponent_weight_mode} "
        f"bc_anchor={bool(anchor_corpus)} anchor_weight={args.bc_anchor_weight}",
        flush=True,
    )
    for opp in opponents[:20]:
        w = weights.get(opp.name, 1.0)
        kind = "random" if opp.policy is None else opp.policy_path
        print(f"  opponent {opp.name} w={w:.3f} policy={kind} deck={opp.deck_path}", flush=True)
    if len(opponents) > 20:
        print(f"  ... {len(opponents) - 20} more opponents", flush=True)
    if anchor_corpus is not None:
        print(f"BC anchor stats: {anchor_corpus.stats} samples={len(anchor_indices)}", flush=True)
    if args.dry_run:
        return

    encoder = FastEncoder()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_wr = -1.0
    global_game = 0
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        samples = []
        episodes = []
        games_meta = []
        for local_game in range(args.games_per_iter):
            opponent = sample_opponent(opponents, weights, args.opponent_weight_mode)
            decisions, meta = play_training_game(
                model,
                candidate_deck,
                opponent,
                encoder,
                args,
                game_index=global_game,
                seed=args.seed + global_game * 1009,
            )
            global_game += 1
            if decisions:
                episodes.append(decisions)
                samples.extend(decisions)
            games_meta.append(meta)
            done = local_game + 1
            if args.progress_every and (
                done == 1 or done % args.progress_every == 0 or done == args.games_per_iter
            ):
                wins = sum(1 for row in games_meta if row["outcome"] == "win")
                losses = sum(1 for row in games_meta if row["outcome"] == "loss")
                draws = sum(1 for row in games_meta if row["outcome"] == "draw")
                print(
                    f"  iter {iteration} rollout {done}/{args.games_per_iter} "
                    f"W/L/D={wins}/{losses}/{draws} wr={wins/done:.3f} decisions={len(samples)}",
                    flush=True,
                )

        t_collect = time.time() - t0
        wins = sum(1 for row in games_meta if row["outcome"] == "win")
        losses = sum(1 for row in games_meta if row["outcome"] == "loss")
        draws = sum(1 for row in games_meta if row["outcome"] == "draw")
        win_rate = wins / max(args.games_per_iter, 1)
        avg_reward = (
            wins * args.win_reward + losses * args.loss_reward + draws * args.draw_reward
        ) / max(args.games_per_iter, 1)

        if len(samples) < 2:
            print(f"iter {iteration}: only {len(samples)} decisions collected; skipping update", flush=True)
            continue

        for episode in episodes:
            compute_gae(episode, args.gamma, args.gae_lambda)
        stats = ppo_update(
            model,
            optimizer,
            samples,
            args,
            device=device,
            anchor_corpus=anchor_corpus,
            anchor_indices=anchor_indices,
        )
        t_update = time.time() - t0 - t_collect

        checkpoint = ""
        should_save = args.save_every > 0 and (iteration % args.save_every == 0 or iteration == args.iterations)
        if should_save and args.checkpoint_dir:
            checkpoint = str(Path(args.checkpoint_dir) / f"rl_iter{iteration:04d}_wr{win_rate:.3f}.npz")
            _save_npz(model, checkpoint)
        if win_rate >= best_wr:
            best_wr = win_rate
            _save_npz(model, args.save)

        row = {
            "iter": iteration,
            "games": args.games_per_iter,
            "decisions": len(samples),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_decisions_per_game": len(samples) / max(args.games_per_iter, 1),
            **stats,
            "t_collect": t_collect,
            "t_update": t_update,
            "checkpoint": checkpoint or args.save,
        }
        append_metrics(args.metrics_csv, row)
        print(
            f"iter {iteration:04d} games={args.games_per_iter} W/L/D={wins}/{losses}/{draws} "
            f"wr={win_rate:.3f} decisions={len(samples)} "
            f"pol={stats['policy_loss']:.4f} val={stats['value_loss']:.4f} "
            f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
            f"clip={stats['clip_frac']:.3f} bc={stats['bc_anchor_loss']:.4f} "
            f"collect={t_collect:.1f}s update={t_update:.1f}s save={checkpoint or args.save}",
            flush=True,
        )

    _save_npz(model, args.save)
    print(f"Final saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
