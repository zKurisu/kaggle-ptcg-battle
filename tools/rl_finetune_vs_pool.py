#!/usr/bin/env python3
"""PPO training against a policy pool.

The active 2026-08-09 direction is not BC fine-tuning. BC checkpoints are locked:
use them as baselines, opponents, submission references, or architecture
templates. For new structural-weakness work, prefer `--init-mode random` and
train from scratch against a curriculum/league.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if os.environ.get("PTCG_DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, sequence_nll
from ptcg_rl.encoder import FastEncoder
from ptcg_rl.history_features import (
    action_event_from_encoded,
    board_snapshot_from_encoded,
    pack_action_history,
    pack_board_history,
    pack_log_history_from_obs,
)
from ptcg_rl.model import (
    build_policy_model,
    checkpoint_arch,
    checkpoint_board_history_dims,
    checkpoint_feature_dims,
    checkpoint_hierarchical_plan,
    checkpoint_history_k,
    checkpoint_log_history_k,
    checkpoint_opp_history_k,
    checkpoint_plan_dim,
    checkpoint_width,
)
from ptcg_rl.numpy_policy import NumpyPolicy
from tools.bc2_train import (
    _configure_cuda_memory_limit,
    _load_npz_init,
    _save_npz,
)
from tools.eval_baseline_delta import read_manifest_entries
from tools.eval_round_robin import Entry, clean_entry_name, parse_entry, policy_action, read_deck


_ACTOR_POLICY: NumpyPolicy | None = None
_ACTOR_CANDIDATE_DECK: list[int] = []
_ACTOR_OPPONENTS: list[Entry] = []
_ACTOR_ARGS: SimpleNamespace | None = None


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
    "value_clip_frac",
    "ref_kl_loss",
    "ref_logprob_delta",
    "bc_anchor_loss",
    "avg_reward_weight",
    "rollout_temperature_eff",
    "rollout_top_k_eff",
    "entropy_coef_eff",
    "ref_kl_coef_eff",
    "bc_anchor_weight_eff",
    "opponent_top_name",
    "opponent_top_weight",
    "league_snapshots",
    "early_stop",
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
        state_feat_dim, opt_feat_dim, option_context, slot_state = checkpoint_feature_dims(z)
        board_history_k, board_history_feat_dim = checkpoint_board_history_dims(z)
        state_layers = 0
        while f"state_layers.{state_layers}.q.weight" in z.files:
            state_layers += 1
        return {
            "arch": checkpoint_arch(z.files),
            "width": float(checkpoint_width(z)),
            "slot_state": bool(slot_state),
            "option_context": bool(option_context),
            "state_feat_dim": int(state_feat_dim),
            "opt_feat_dim": int(opt_feat_dim),
            "plan_dim": int(checkpoint_plan_dim(z)),
            "hierarchical_plan": bool(checkpoint_hierarchical_plan(z)),
            "history_k": int(checkpoint_history_k(z)),
            "opp_history_k": int(checkpoint_opp_history_k(z)),
            "log_history_k": int(checkpoint_log_history_k(z)),
            "board_history_k": int(board_history_k),
            "board_history_feat_dim": int(board_history_feat_dim),
            "state_layers": max(1, int(state_layers)),
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


def _history_numpy_from_policy(policy: NumpyPolicy, obs: dict) -> dict[str, np.ndarray] | None:
    try:
        hist = policy._history_arrays(obs)  # noqa: SLF001 - rollout must match submission-time history.
    except Exception:
        return None
    if not hist:
        return None
    return {k: np.asarray(v).copy() for k, v in hist.items()}


def _history_numpy_from_trackers(
    action_history: list[dict],
    board_history: list[dict],
    obs: dict,
    args: argparse.Namespace,
) -> dict[str, np.ndarray] | None:
    out: dict[str, np.ndarray] = {}
    if int(getattr(args, "history_k", 0)) > 0:
        out.update({k: np.asarray(v) for k, v in pack_action_history(action_history, args.history_k).items()})
    if int(getattr(args, "opp_history_k", 0)) > 0:
        opp = pack_action_history([], args.opp_history_k)
        out.update({f"opp_{k}": np.asarray(v) for k, v in opp.items()})
    if int(getattr(args, "log_history_k", 0)) > 0:
        logs = pack_log_history_from_obs(obs, args.log_history_k)
        out.update({f"log_{k}": np.asarray(v) for k, v in logs.items()})
    if int(getattr(args, "board_history_k", 0)) > 0:
        boards = pack_board_history(
            board_history,
            args.board_history_k,
            int(getattr(args, "board_history_feat_dim", 0) or 0),
        )
        out["board_cards"] = np.asarray(boards["cards"])
        out["board_feats"] = np.asarray(boards["feats"])
        out["board_mask"] = np.asarray(boards["mask"])
    return out or None


def _history_torch(
    history: dict[str, np.ndarray] | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not history:
        return None
    return {k: torch.as_tensor(np.asarray(v), device=device).unsqueeze(0) for k, v in history.items()}


def _legalize_action(action: list[int], sel: dict) -> tuple[list[int], bool]:
    opts = sel.get("option", [])
    n = len(opts)
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if n == 0 or mx <= 0:
        return [], True
    picks = [int(p) for p in action if 0 <= int(p) < n]
    picks = list(dict.fromkeys(picks))[:mx]
    stopped = len(picks) < mx and len(picks) >= mn
    if mn <= len(picks) <= mx:
        return picks, stopped
    fallback = legal_random(sel)
    return fallback, len(fallback) < mx and len(fallback) >= mn


def _state_potential(decision) -> float:
    """Small general potential for dense RL shaping from current-player features."""
    f = np.asarray(decision.state_feats, dtype=np.float32)
    def at(i: int) -> float:
        return float(f[i]) if i < len(f) else 0.0

    prize_term = 2.0 * (at(7) - at(6))
    setup_term = 0.35 * at(60) + 0.20 * at(76) + 0.12 * at(56)
    pressure_term = 0.35 * at(63) + 0.20 * at(62)
    safety_term = -0.25 * at(25) - 0.10 * at(61)
    tempo_term = 0.08 * at(55) - 0.05 * at(5)
    return float(np.clip(prize_term + setup_term + pressure_term + safety_term + tempo_term, -4.0, 4.0))


def assign_episode_rewards(decisions: list, outcome: str, args: argparse.Namespace) -> float:
    terminal = (
        float(args.win_reward)
        if outcome == "win"
        else float(args.loss_reward)
        if outcome == "loss"
        else float(args.draw_reward)
    )
    for d in decisions:
        d.reward = 0.0
    shaping = float(getattr(args, "shaping_weight", 0.0) or 0.0)
    if shaping > 0 and len(decisions) > 1:
        potentials = [_state_potential(d) for d in decisions]
        for i, d in enumerate(decisions[:-1]):
            d.reward += shaping * (potentials[i + 1] - potentials[i])
    turn_penalty = float(getattr(args, "turn_penalty", 0.0) or 0.0)
    if turn_penalty:
        for d in decisions:
            d.reward -= turn_penalty
    if decisions:
        decisions[-1].reward += terminal
    return terminal


@torch.no_grad()
def refresh_old_policy_stats(model: nn.Module, samples: list, minibatch: int) -> None:
    """Fill logprob/value for sampled actions under the pre-update policy."""
    model.eval()
    for start in range(0, len(samples), max(1, int(minibatch))):
        mb = samples[start : start + max(1, int(minibatch))]
        logprob, _, value = model.evaluate_actions(mb)
        for d, lp, val in zip(mb, logprob.detach().cpu().numpy(), value.detach().cpu().numpy()):
            d.logprob = float(lp)
            d.value = float(val)


@torch.no_grad()
def refresh_reference_policy_stats(model: nn.Module, samples: list, minibatch: int) -> None:
    """Fill reference-policy log-probs for sampled actions.

    This is an action-level trust region against the initial policy. It is not a
    full distribution KL, but it is cheap, architecture-agnostic, and catches the
    destructive drift observed in the first RL wave.
    """
    model.eval()
    for start in range(0, len(samples), max(1, int(minibatch))):
        mb = samples[start : start + max(1, int(minibatch))]
        logprob, _, _ = model.evaluate_actions(mb)
        for d, lp in zip(mb, logprob.detach().cpu().numpy()):
            d.ref_logprob = float(lp)


def attach_episode_metadata(decisions: list, *, opponent: str, outcome: str) -> None:
    for d in decisions:
        d.opponent = opponent
        d.outcome = outcome


def apply_batch_reward_weighting(
    episodes: list[list],
    games_meta: list[dict],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Reweight collected rewards before GAE.

    `opponent_inverse_winrate` emphasizes matchups where the current policy is
    struggling in this batch, reducing the tendency to improve already-easy
    opponents while losing the target weakness.
    """
    mode = str(getattr(args, "reward_weight_mode", "none") or "none")
    if mode == "none" or not episodes:
        for episode in episodes:
            for d in episode:
                d.reward_weight = 1.0
        return {"avg_reward_weight": 1.0}

    counts: dict[str, int] = {}
    wins: dict[str, int] = {}
    for meta in games_meta:
        opp = str(meta.get("opponent", ""))
        counts[opp] = counts.get(opp, 0) + 1
        if meta.get("outcome") == "win":
            wins[opp] = wins.get(opp, 0) + 1

    coef = float(getattr(args, "reward_weight_coef", 1.0) or 0.0)
    lo = float(getattr(args, "reward_weight_min", 0.25) or 0.25)
    hi = float(getattr(args, "reward_weight_max", 2.5) or 2.5)
    weights: list[float] = []
    for episode, meta in zip(episodes, games_meta):
        opp = str(meta.get("opponent", ""))
        wr = wins.get(opp, 0) / max(counts.get(opp, 1), 1)
        if mode == "opponent_inverse_winrate":
            weight = 1.0 + coef * (0.5 - wr)
        elif mode == "opponent_lossrate":
            weight = 1.0 + coef * (1.0 - wr)
        else:
            weight = 1.0
        weight = float(np.clip(weight, lo, hi))
        weights.append(weight)
        for d in episode:
            d.reward *= weight
            d.reward_weight = weight
    return {"avg_reward_weight": float(np.mean(weights)) if weights else 1.0}


def _linear_schedule(start: float, final: float | None, schedule_iters: int, iteration: int) -> float:
    if final is None or int(schedule_iters) <= 0:
        return float(start)
    progress = min(1.0, max(0.0, (int(iteration) - 1) / max(1, int(schedule_iters) - 1)))
    return float(start + progress * (float(final) - float(start)))


def _linear_schedule_int(start: int, final: int | None, schedule_iters: int, iteration: int) -> int:
    if final is None or int(schedule_iters) <= 0:
        return int(start)
    value = _linear_schedule(float(start), float(final), schedule_iters, iteration)
    return int(round(value))


def apply_iteration_schedule(args: argparse.Namespace, iteration: int) -> dict[str, float | int]:
    """Mutate runtime knobs for the current iteration and return loggable values."""
    schedule_iters = int(getattr(args, "schedule_iters", 0) or 0)
    args.rollout_temperature = _linear_schedule(
        args._base_rollout_temperature,
        args.rollout_temperature_final,
        schedule_iters,
        iteration,
    )
    args.rollout_top_k = _linear_schedule_int(
        args._base_rollout_top_k,
        args.rollout_top_k_final,
        schedule_iters,
        iteration,
    )
    args.entropy_coef = _linear_schedule(
        args._base_entropy_coef,
        args.entropy_final_coef,
        schedule_iters,
        iteration,
    )
    args.ref_kl_coef = _linear_schedule(
        args._base_ref_kl_coef,
        args.ref_kl_final_coef,
        schedule_iters,
        iteration,
    )
    args.bc_anchor_weight = _linear_schedule(
        args._base_bc_anchor_weight,
        args.bc_anchor_final_weight,
        schedule_iters,
        iteration,
    )
    return {
        "rollout_temperature_eff": float(args.rollout_temperature),
        "rollout_top_k_eff": int(args.rollout_top_k),
        "entropy_coef_eff": float(args.entropy_coef),
        "ref_kl_coef_eff": float(args.ref_kl_coef),
        "bc_anchor_weight_eff": float(args.bc_anchor_weight),
    }


@torch.no_grad()
def sample_trainable_action(
    model: nn.Module,
    decision,
    *,
    greedy: bool = False,
    history: dict[str, torch.Tensor] | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> tuple[list[int], bool, float, float]:
    dev = next(model.parameters()).device
    board = torch.from_numpy(decision.board_cards).unsqueeze(0).to(dev)
    hand = torch.from_numpy(decision.hand_cards).unsqueeze(0).to(dev)
    feats = torch.from_numpy(decision.state_feats).unsqueeze(0).to(dev)
    h = model.encode_state(board, hand, feats, history)
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
        if float(temperature) != 1.0:
            logits = logits / max(float(temperature), 1e-6)
        if not greedy and int(top_k) > 0:
            legal = torch.nonzero(avail[0], as_tuple=True)[0]
            if legal.numel() > int(top_k):
                keep = legal[torch.topk(logits[0, legal], int(top_k)).indices]
                top_mask = torch.zeros_like(avail)
                top_mask[0, keep] = True
                logits = logits.masked_fill(~top_mask, -1e9)
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
    model: nn.Module,
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
        if opponent.policy is not None and hasattr(opponent.policy, "reset_history"):
            opponent.policy.reset_history()
        if getattr(opponent, "planner", None) is not None:
            opponent.planner.reset(opponent.deck)
        action_history: list[dict] = []
        board_history: list[dict] = []
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
                    hist_np = _history_numpy_from_trackers(action_history, board_history, obs, args)
                    action, stopped, logprob, value = sample_trainable_action(
                        model,
                        decision,
                        greedy=args.greedy_rollout,
                        history=_history_torch(hist_np, next(model.parameters()).device),
                        temperature=args.rollout_temperature,
                        top_k=args.rollout_top_k,
                    )
                    action, stopped = _legalize_action(action, sel)
                    decision.action = action
                    decision.logprob = logprob
                    decision.value = value
                    decision.history = hist_np
                    setattr(decision, "stopped", stopped)
                    decisions.append(decision)
                    event = action_event_from_encoded(decision, action)
                    if event is not None:
                        action_history.append(event)
                        if len(action_history) > max(args.history_k, 1) * 4:
                            del action_history[:-max(args.history_k, 1) * 4]
                    if args.board_history_k > 0:
                        board_history.append(
                            board_snapshot_from_encoded(decision, args.board_history_feat_dim)
                        )
                        if len(board_history) > max(args.board_history_k, 1) * 4:
                            del board_history[:-max(args.board_history_k, 1) * 4]
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
    assign_episode_rewards(decisions, outcome, args)
    attach_episode_metadata(decisions, opponent=opponent.name, outcome=outcome)
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


def _entry_payload(entry: Entry) -> tuple[str, str, str]:
    return entry.name, entry.policy_path, entry.deck_path


def _entry_from_payload(payload: tuple[str, str, str]) -> Entry:
    name, policy_path, deck_path = payload
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
    return Entry(name, policy_path, deck_path, policy, deck)


def _init_rollout_worker(
    policy_path: str,
    candidate_deck_path: str,
    opponent_payloads: list[tuple[str, str, str]],
    args_payload: dict,
) -> None:
    global _ACTOR_POLICY, _ACTOR_CANDIDATE_DECK, _ACTOR_OPPONENTS, _ACTOR_ARGS
    _ACTOR_POLICY = NumpyPolicy.load(policy_path)
    _ACTOR_CANDIDATE_DECK = read_deck(candidate_deck_path)
    _ACTOR_OPPONENTS = [_entry_from_payload(p) for p in opponent_payloads]
    _ACTOR_ARGS = SimpleNamespace(**args_payload)


def _actor_play_training_game(task: tuple[int, int, int]) -> tuple[list, dict]:
    from cg.game import battle_finish, battle_select, battle_start

    if _ACTOR_POLICY is None or _ACTOR_ARGS is None:
        raise RuntimeError("rollout worker was not initialized")

    game_index, seed, opponent_index = task
    args = _ACTOR_ARGS
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    policy = _ACTOR_POLICY
    opponent = _ACTOR_OPPONENTS[int(opponent_index) % len(_ACTOR_OPPONENTS)]
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    if opponent.policy is not None and hasattr(opponent.policy, "reset_history"):
        opponent.policy.reset_history()

    swapped = bool(game_index % 2)
    candidate_side = 1 if swapped else 0
    first_deck, second_deck = (
        (opponent.deck, _ACTOR_CANDIDATE_DECK)
        if swapped
        else (_ACTOR_CANDIDATE_DECK, opponent.deck)
    )
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
        for steps in range(int(args.max_turns)):
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
                    decision = policy.encoder.encode(obs)
                    decision.history = _history_numpy_from_policy(policy, obs)
                    action = policy.select(
                        obs,
                        greedy=bool(args.greedy_rollout),
                        temperature=float(args.rollout_temperature),
                        top_k=int(args.rollout_top_k),
                        update_history=False,
                    )
                    action, stopped = _legalize_action(action, sel)
                    decision.action = action
                    setattr(decision, "stopped", stopped)
                    decisions.append(decision)
                    try:
                        policy._remember_decision(decision, action)  # noqa: SLF001
                    except Exception:
                        pass
                except Exception:
                    action = legal_random(sel)
            else:
                action = policy_action(
                    opponent,
                    obs,
                    use_mcts=bool(args.opponent_mcts),
                    sims=int(args.opponent_mcts_sims),
                    time_budget=float(args.opponent_time_budget),
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
    assign_episode_rewards(decisions, outcome, args)
    attach_episode_metadata(decisions, opponent=opponent.name, outcome=outcome)
    return decisions, {
        "outcome": outcome,
        "opponent": opponent.name,
        "steps": steps,
        "candidate_side": candidate_side,
    }


def sample_opponent_index(opponents: list[Entry], weights: dict[str, float], mode: str) -> int:
    if mode == "uniform":
        return int(np.random.randint(len(opponents)))
    raw = np.asarray([max(float(weights.get(opp.name, 1.0)), 1e-6) for opp in opponents], dtype=np.float64)
    probs = raw / raw.sum()
    return int(np.random.choice(len(opponents), p=probs))


def summarize_opponents(
    opponents: list[Entry],
    weights: dict[str, float],
    games_meta: list[dict],
) -> tuple[list[dict], dict[str, str | float]]:
    counts: dict[str, int] = {}
    wins: dict[str, int] = {}
    draws: dict[str, int] = {}
    for meta in games_meta:
        name = str(meta.get("opponent", ""))
        counts[name] = counts.get(name, 0) + 1
        if meta.get("outcome") == "win":
            wins[name] = wins.get(name, 0) + 1
        elif meta.get("outcome") == "draw":
            draws[name] = draws.get(name, 0) + 1

    rows = []
    for opp in opponents:
        n = counts.get(opp.name, 0)
        w = wins.get(opp.name, 0)
        d = draws.get(opp.name, 0)
        rows.append({
            "opponent": opp.name,
            "games": n,
            "wins": w,
            "losses": max(0, n - w - d),
            "draws": d,
            "win_rate": w / max(n, 1),
            "weight": float(weights.get(opp.name, 1.0)),
            "policy": opp.policy_path,
            "deck": opp.deck_path,
        })

    top = max(rows, key=lambda r: float(r["weight"])) if rows else {"opponent": "", "weight": 0.0}
    return rows, {
        "opponent_top_name": str(top["opponent"]),
        "opponent_top_weight": float(top["weight"]),
    }


def append_opponent_stats(path: str, iteration: int, rows: list[dict]) -> None:
    if not path:
        return
    fields = [
        "iter",
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "weight",
        "policy",
        "deck",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["iter"] = iteration
            formatted["win_rate"] = f"{float(formatted['win_rate']):.6f}"
            formatted["weight"] = f"{float(formatted['weight']):.6f}"
            writer.writerow(formatted)


def update_adaptive_opponent_weights(
    opponents: list[Entry],
    weights: dict[str, float],
    base_weights: dict[str, float],
    games_meta: list[dict],
    args: argparse.Namespace,
) -> None:
    mode = str(getattr(args, "opponent_weight_mode", "uniform") or "uniform")
    if not mode.startswith("adaptive"):
        return
    counts: dict[str, int] = {}
    wins: dict[str, int] = {}
    for meta in games_meta:
        opp = str(meta.get("opponent", ""))
        counts[opp] = counts.get(opp, 0) + 1
        if meta.get("outcome") == "win":
            wins[opp] = wins.get(opp, 0) + 1

    min_games = int(getattr(args, "opponent_adaptive_min_games", 4) or 0)
    coef = float(getattr(args, "opponent_adaptive_coef", 2.0) or 0.0)
    ema = float(getattr(args, "opponent_adaptive_ema", 0.35) or 0.0)
    lo = float(getattr(args, "opponent_weight_min", 0.2) or 0.0)
    hi = float(getattr(args, "opponent_weight_max", 6.0) or 0.0)
    for opp in opponents:
        n = counts.get(opp.name, 0)
        if n < min_games:
            continue
        wr = wins.get(opp.name, 0) / max(n, 1)
        base = float(base_weights.get(opp.name, 1.0))
        if mode == "adaptive_inverse_winrate":
            target = base * (1.0 + coef * max(0.0, 0.5 - wr))
        else:
            target = base * (1.0 + coef * (1.0 - wr))
        if hi > 0:
            target = min(target, hi)
        if lo > 0:
            target = max(target, lo)
        old = float(weights.get(opp.name, base))
        weights[opp.name] = (1.0 - ema) * old + ema * target


def maybe_add_league_snapshot(
    model: nn.Module,
    opponents: list[Entry],
    weights: dict[str, float],
    base_weights: dict[str, float],
    snapshots: list[Entry],
    args: argparse.Namespace,
    *,
    iteration: int,
    win_rate: float,
    force: bool = False,
) -> None:
    every = int(getattr(args, "league_snapshot_every", 0) or 0)
    if not force and every <= 0:
        return
    if not force and iteration < int(getattr(args, "league_start_iter", 1) or 1):
        return
    if not force and iteration % every != 0 and iteration != 1:
        return
    min_wr = float(getattr(args, "league_min_batch_win_rate", 0.0) or 0.0)
    if not force and win_rate < min_wr:
        return
    snap_dir_arg = getattr(args, "league_snapshot_dir", "") or getattr(args, "checkpoint_dir", "")
    if not snap_dir_arg:
        return
    snap_dir = Path(snap_dir_arg)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"league_iter{iteration:04d}_wr{win_rate:.3f}.npz"
    _save_npz(model, str(snap_path))
    name = clean_entry_name(f"league_iter{iteration:04d}_wr{win_rate:.3f}")
    deck = read_deck(args.deck)
    entry = Entry(name, str(snap_path), args.deck, NumpyPolicy.load(str(snap_path)), deck)
    opponents.append(entry)
    snapshots.append(entry)
    weight = float(getattr(args, "league_snapshot_weight", 1.0) or 1.0)
    weights[entry.name] = weight
    base_weights[entry.name] = weight
    max_snaps = int(getattr(args, "league_max_snapshots", 0) or 0)
    while max_snaps > 0 and len(snapshots) > max_snaps:
        old = snapshots.pop(0)
        opponents[:] = [opp for opp in opponents if opp.name != old.name]
        weights.pop(old.name, None)
        base_weights.pop(old.name, None)


def collect_parallel_rollouts(
    model: nn.Module,
    opponents: list[Entry],
    weights: dict[str, float],
    args: argparse.Namespace,
    *,
    iteration: int,
    global_game: int,
) -> tuple[list[list], list[dict], int, float]:
    actor_dir = Path(args.actor_tmp_dir)
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_policy = actor_dir / f"rl_actor_policy_{os.getpid()}_{iteration:04d}.npz"
    _save_npz(model, str(actor_policy))
    tasks = []
    for local_game in range(args.games_per_iter):
        game_index = global_game + local_game
        tasks.append((
            game_index,
            args.seed + game_index * 1009,
            sample_opponent_index(opponents, weights, args.opponent_weight_mode),
        ))
    args_payload = {
        "max_turns": args.max_turns,
        "greedy_rollout": args.greedy_rollout,
        "rollout_temperature": args.rollout_temperature,
        "rollout_top_k": args.rollout_top_k,
        "opponent_mcts": args.opponent_mcts,
        "opponent_mcts_sims": args.opponent_mcts_sims,
        "opponent_time_budget": args.opponent_time_budget,
        "win_reward": args.win_reward,
        "loss_reward": args.loss_reward,
        "draw_reward": args.draw_reward,
        "shaping_weight": args.shaping_weight,
        "turn_penalty": args.turn_penalty,
    }
    opponent_payloads = [_entry_payload(o) for o in opponents]
    episodes: list[list] = []
    games_meta: list[dict] = []
    t0 = time.time()
    workers = max(1, min(int(args.rollout_workers), int(args.games_per_iter)))
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_rollout_worker,
            initargs=(str(actor_policy), args.deck, opponent_payloads, args_payload),
        ) as ex:
            futs = [ex.submit(_actor_play_training_game, task) for task in tasks]
            for done, fut in enumerate(as_completed(futs), 1):
                decisions, meta = fut.result()
                if decisions:
                    episodes.append(decisions)
                games_meta.append(meta)
                if args.progress_every and (
                    done == 1 or done % args.progress_every == 0 or done == args.games_per_iter
                ):
                    wins = sum(1 for row in games_meta if row["outcome"] == "win")
                    losses = sum(1 for row in games_meta if row["outcome"] == "loss")
                    draws = sum(1 for row in games_meta if row["outcome"] == "draw")
                    n_decisions = sum(len(ep) for ep in episodes)
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-9)
                    print(
                        f"  iter {iteration} parallel rollout {done}/{args.games_per_iter} "
                        f"W/L/D={wins}/{losses}/{draws} wr={wins/done:.3f} "
                        f"decisions={n_decisions} {rate:.2f} games/s",
                        flush=True,
                    )
    finally:
        if not args.keep_actor_policy:
            try:
                actor_policy.unlink()
            except FileNotFoundError:
                pass
    return episodes, games_meta, global_game + args.games_per_iter, time.time() - t0


def setup_anchor_corpus(args: argparse.Namespace, state_feat_dim: int, opt_feat_dim: int):
    max_weight = max(
        float(getattr(args, "bc_anchor_weight", 0.0) or 0.0),
        float(getattr(args, "bc_anchor_final_weight", 0.0) or 0.0),
    )
    if max_weight <= 0:
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
        history_k=max(0, int(getattr(args, "history_k", 0))),
        opp_history_k=max(0, int(getattr(args, "opp_history_k", 0))),
        log_history_k=max(0, int(getattr(args, "log_history_k", 0))),
        board_history_k=max(0, int(getattr(args, "board_history_k", 0))),
        board_history_feat_dim=max(0, int(getattr(args, "board_history_feat_dim", 0))),
        split_by_game=(
            int(getattr(args, "history_k", 0)) > 0
            or int(getattr(args, "opp_history_k", 0)) > 0
            or int(getattr(args, "log_history_k", 0)) > 0
            or int(getattr(args, "board_history_k", 0)) > 0
        ),
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


def normalized_advantage_map(samples: list, args: argparse.Namespace) -> tuple[dict[int, float], dict[str, float]]:
    raw = np.asarray([float(s.adv) for s in samples], dtype=np.float32)
    mode = str(getattr(args, "advantage_normalization", "global") or "global")
    out = np.zeros_like(raw)
    if mode == "none":
        out = raw
    elif mode == "opponent":
        groups: dict[str, list[int]] = {}
        for i, sample in enumerate(samples):
            groups.setdefault(str(getattr(sample, "opponent", "")), []).append(i)
        for idxs in groups.values():
            vals = raw[idxs]
            out[idxs] = (vals - vals.mean()) / (vals.std() + 1e-8)
    else:
        out = (raw - raw.mean()) / (raw.std() + 1e-8)
    clip = float(getattr(args, "advantage_clip", 0.0) or 0.0)
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return (
        {id(sample): float(out[i]) for i, sample in enumerate(samples)},
        {
            "adv_mean": float(raw.mean()) if raw.size else 0.0,
            "adv_std": float(raw.std()) if raw.size else 0.0,
        },
    )


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list,
    args: argparse.Namespace,
    *,
    device: torch.device,
    anchor_corpus: BCCorpus | None,
    anchor_indices: list,
) -> dict:
    model.train()
    adv_map, _adv_stats = normalized_advantage_map(samples, args)
    order = np.arange(len(samples))
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "value_clip_frac": 0.0,
        "ref_kl_loss": 0.0,
        "ref_logprob_delta": 0.0,
        "bc_anchor_loss": 0.0,
        "early_stop": 0.0,
        "n": 0,
    }
    early_stop = False

    for _ in range(args.ppo_epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), args.minibatch):
            mb_idx = order[start : start + args.minibatch]
            if len(mb_idx) < 2:
                continue
            mb = [samples[int(i)] for i in mb_idx]
            old_logprob = torch.as_tensor([s.logprob for s in mb], dtype=torch.float32, device=device)
            old_value = torch.as_tensor([s.value for s in mb], dtype=torch.float32, device=device)
            adv = torch.as_tensor([adv_map[id(s)] for s in mb], dtype=torch.float32, device=device)
            ret = torch.as_tensor([s.ret for s in mb], dtype=torch.float32, device=device)

            new_logprob, entropy, value = model.evaluate_actions(mb)
            ratio = torch.exp(new_logprob - old_logprob)
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            value_clip_frac = torch.tensor(0.0, device=device)
            if float(getattr(args, "value_clip_eps", 0.0) or 0.0) > 0:
                value_clipped = old_value + torch.clamp(
                    value - old_value,
                    -float(args.value_clip_eps),
                    float(args.value_clip_eps),
                )
                value_loss_unclipped = (value - ret).pow(2)
                value_loss_clipped = (value_clipped - ret).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                value_clip_frac = (value - old_value).abs().gt(float(args.value_clip_eps)).float().mean()
            else:
                value_loss = (value - ret).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()

            ref_kl_loss = torch.tensor(0.0, device=device)
            ref_logprob_delta = torch.tensor(0.0, device=device)
            if float(getattr(args, "ref_kl_coef", 0.0) or 0.0) > 0:
                if not all(hasattr(s, "ref_logprob") for s in mb):
                    raise RuntimeError("--ref-kl-coef requires reference logprobs")
                ref_logprob = torch.as_tensor([s.ref_logprob for s in mb], dtype=torch.float32, device=device)
                delta_ref = new_logprob - ref_logprob
                ref_kl_loss = delta_ref.pow(2).mean()
                ref_logprob_delta = delta_ref.abs().mean()
                loss = loss + float(args.ref_kl_coef) * ref_kl_loss

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
            stats["value_clip_frac"] += float(value_clip_frac.detach().cpu())
            stats["ref_kl_loss"] += float(ref_kl_loss.detach().cpu())
            stats["ref_logprob_delta"] += float(ref_logprob_delta.detach().cpu())
            stats["bc_anchor_loss"] += float(anchor_loss.detach().cpu())
            stats["n"] += 1
            if float(getattr(args, "target_kl", 0.0) or 0.0) > 0 and abs(float(approx_kl)) > float(args.target_kl):
                early_stop = True
                break
        if early_stop:
            break

    n = max(stats.pop("n"), 1)
    out = {k: v / n for k, v in stats.items()}
    out["early_stop"] = 1.0 if early_stop else 0.0
    return out


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
    p.add_argument("--policy-init", required=True,
                   help="BC2 .npz checkpoint used as an architecture template, or explicit load init")
    p.add_argument("--init-mode", choices=["random", "load"], default="random",
                   help="'random' uses --policy-init only as a template; 'load' is deprecated BC fine-tuning")
    p.add_argument("--deck", required=True, help="candidate deck CSV")
    p.add_argument("--save", required=True, help="final/best output .npz path")
    p.add_argument("--save-policy", choices=["final", "best", "both"], default="final",
                   help="what to write to --save; use best/both for RL sweeps to avoid final drift")
    p.add_argument("--save-final", default="",
                   help="optional final checkpoint path when --save-policy is best or both")
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
    p.add_argument("--opponent-weight-mode",
                   choices=["uniform", "manifest", "adaptive_lossrate", "adaptive_inverse_winrate"],
                   default="uniform")
    p.add_argument("--opponent-adaptive-coef", type=float, default=2.0,
                   help="adaptive opponent sampler multiplier; larger values focus harder opponents")
    p.add_argument("--opponent-adaptive-ema", type=float, default=0.35,
                   help="EMA applied when updating adaptive opponent weights from batch results")
    p.add_argument("--opponent-adaptive-min-games", type=int, default=4,
                   help="minimum games against an opponent before adapting its weight")
    p.add_argument("--opponent-weight-min", type=float, default=0.2)
    p.add_argument("--opponent-weight-max", type=float, default=6.0)
    p.add_argument("--opponent-mcts", action="store_true")
    p.add_argument("--opponent-mcts-sims", type=int, default=48)
    p.add_argument("--opponent-time-budget", type=float, default=4.0)

    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--rollout-workers", type=int, default=1,
                   help="parallel CPU actor workers for game collection; 1 keeps the old in-process path")
    p.add_argument("--rollout-temperature", type=float, default=1.15,
                   help="sampling temperature used by NumPy actor workers")
    p.add_argument("--rollout-temperature-final", type=float, default=None,
                   help="linearly anneal rollout temperature to this value over --schedule-iters")
    p.add_argument("--rollout-top-k", type=int, default=8,
                   help="when sampling, restrict actor choices to top K legal options; 0 disables")
    p.add_argument("--rollout-top-k-final", type=int, default=None,
                   help="linearly anneal rollout top-k to this value over --schedule-iters")
    p.add_argument("--actor-tmp-dir", default="/tmp/ptcg_rl_actor_policies")
    p.add_argument("--keep-actor-policy", action="store_true",
                   help="keep per-iteration actor .npz files for debugging")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip-eps", type=float, default=0.1)
    p.add_argument("--target-kl", type=float, default=0.0,
                   help="stop the current PPO update early when approximate KL exceeds this value")
    p.add_argument("--value-clip-eps", type=float, default=0.0,
                   help="PPO2-style value clipping; 0 disables")
    p.add_argument("--advantage-normalization", choices=["global", "opponent", "none"], default="global",
                   help="normalize GAE globally or within each opponent group")
    p.add_argument("--advantage-clip", type=float, default=0.0,
                   help="clip normalized advantages to +/- this value; 0 disables")
    p.add_argument("--ref-policy", default="",
                   help="reference checkpoint for action-level KL; default is --policy-init")
    p.add_argument("--ref-kl-coef", type=float, default=0.0,
                   help="penalty coefficient for squared logprob drift from the reference policy")
    p.add_argument("--ref-kl-final-coef", type=float, default=None,
                   help="linearly anneal reference KL coefficient to this value over --schedule-iters")
    p.add_argument("--reward-weight-mode", choices=["none", "opponent_inverse_winrate", "opponent_lossrate"],
                   default="none",
                   help="batch-level episode reward reweighting before GAE")
    p.add_argument("--reward-weight-coef", type=float, default=1.0)
    p.add_argument("--reward-weight-min", type=float, default=0.25)
    p.add_argument("--reward-weight-max", type=float, default=2.5)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--entropy-final-coef", type=float, default=None,
                   help="linearly anneal entropy coefficient to this value over --schedule-iters")
    p.add_argument("--schedule-iters", type=int, default=0,
                   help="number of iterations for temperature/top-k/entropy/KL/anchor schedules")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--win-reward", type=float, default=1.0)
    p.add_argument("--loss-reward", type=float, default=-1.0)
    p.add_argument("--draw-reward", type=float, default=0.0)
    p.add_argument("--shaping-weight", type=float, default=0.0,
                   help="dense potential-difference reward multiplier; 0 keeps terminal-only PPO")
    p.add_argument("--turn-penalty", type=float, default=0.0,
                   help="small per-decision penalty to discourage long non-progress loops")
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--greedy-rollout", action="store_true", help="debug only; PPO should normally sample")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuda-memory-gb", type=float, default=0.0)
    p.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    p.add_argument("--width", type=float, default=0.0, help="override width; default inferred from --policy-init")
    p.add_argument("--arch", choices=["pointer", "cross_attn"], default="",
                   help="override model arch; default inferred from --policy-init")
    p.add_argument("--state-layers", type=int, default=0,
                   help="override cross-attn state layers; default inferred from --policy-init")
    p.add_argument("--state-feat-dim", type=int, default=0, help="override state features; default inferred")
    p.add_argument("--opt-feat-dim", type=int, default=0, help="override option features; default inferred")
    p.add_argument("--legacy-state-pool", action="store_true")
    p.add_argument("--no-option-context", action="store_true")
    p.add_argument("--plan-dim", type=int, default=-1,
                   help="override auxiliary/hierarchical plan dim; -1 infers from checkpoint")
    p.add_argument("--hierarchical-plan", action="store_true",
                   help="force hierarchical plan conditioning on even if checkpoint does not contain it")
    p.add_argument("--history-k", type=int, default=-1,
                   help="override own action history length; -1 infers from checkpoint")
    p.add_argument("--opp-history-k", type=int, default=-1,
                   help="override opponent action history length; -1 infers from checkpoint")
    p.add_argument("--log-history-k", type=int, default=-1,
                   help="override public log history length; -1 infers from checkpoint")
    p.add_argument("--board-history-k", type=int, default=-1,
                   help="override board snapshot history length; -1 infers from checkpoint")
    p.add_argument("--board-history-feat-dim", type=int, default=0,
                   help="override board history feature width; 0 infers from checkpoint")

    p.add_argument("--bc-anchor-weight", type=float, default=0.0)
    p.add_argument("--bc-anchor-final-weight", type=float, default=None,
                   help="linearly anneal BC anchor weight to this value over --schedule-iters")
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

    p.add_argument("--opponent-stats-csv", default="",
                   help="append per-opponent rollout stats and adaptive weights here")
    p.add_argument("--league-bootstrap-current", action="store_true",
                   help="add the initial policy as a league opponent before the first rollout")
    p.add_argument("--league-snapshot-dir", default="",
                   help="directory for league opponent snapshots; default is --checkpoint-dir")
    p.add_argument("--league-snapshot-every", type=int, default=0,
                   help="add the current policy to the opponent pool every N iterations")
    p.add_argument("--league-start-iter", type=int, default=1)
    p.add_argument("--league-max-snapshots", type=int, default=6)
    p.add_argument("--league-snapshot-weight", type=float, default=1.0)
    p.add_argument("--league-min-batch-win-rate", type=float, default=0.0,
                   help="skip adding league snapshots if the latest training batch is below this win rate")

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
    args._base_rollout_temperature = float(args.rollout_temperature)
    args._base_rollout_top_k = int(args.rollout_top_k)
    args._base_entropy_coef = float(args.entropy_coef)
    args._base_ref_kl_coef = float(args.ref_kl_coef)
    args._base_bc_anchor_weight = float(args.bc_anchor_weight)

    cfg = checkpoint_config(args.policy_init)
    arch = args.arch or cfg["arch"]
    width = args.width or cfg["width"]
    state_feat_dim = args.state_feat_dim or cfg["state_feat_dim"]
    opt_feat_dim = args.opt_feat_dim or cfg["opt_feat_dim"]
    slot_state = not args.legacy_state_pool and bool(cfg["slot_state"])
    option_context = (not args.no_option_context) and bool(cfg["option_context"])
    plan_dim = cfg["plan_dim"] if args.plan_dim < 0 else args.plan_dim
    hierarchical_plan = bool(args.hierarchical_plan or cfg["hierarchical_plan"])
    history_k = cfg["history_k"] if args.history_k < 0 else args.history_k
    opp_history_k = cfg["opp_history_k"] if args.opp_history_k < 0 else args.opp_history_k
    log_history_k = cfg["log_history_k"] if args.log_history_k < 0 else args.log_history_k
    board_history_k = cfg["board_history_k"] if args.board_history_k < 0 else args.board_history_k
    board_history_feat_dim = args.board_history_feat_dim or cfg["board_history_feat_dim"]
    state_layers = args.state_layers or cfg["state_layers"]
    args.history_k = int(history_k)
    args.opp_history_k = int(opp_history_k)
    args.log_history_k = int(log_history_k)
    args.board_history_k = int(board_history_k)
    args.board_history_feat_dim = int(board_history_feat_dim)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    memory_msg = _configure_cuda_memory_limit(
        device, gb=args.cuda_memory_gb, fraction=args.cuda_memory_fraction
    )
    model = build_policy_model(
        arch=arch,
        width=width,
        slot_state=slot_state,
        option_context=option_context,
        state_feat_dim=state_feat_dim,
        opt_feat_dim=opt_feat_dim,
        plan_dim=plan_dim,
        hierarchical_plan=hierarchical_plan,
        history_k=history_k,
        opp_history_k=opp_history_k,
        log_history_k=log_history_k,
        board_history_k=board_history_k,
        board_history_feat_dim=board_history_feat_dim,
        state_layers=state_layers,
    ).to(device)
    if args.init_mode == "load":
        loaded, skipped = _load_npz_init(model, args.policy_init, device, partial=False)
        if skipped:
            raise RuntimeError(f"strict init unexpectedly skipped tensors: {skipped[:8]}")
        init_msg = f"loaded_tensors={loaded} deprecated_finetune_mode=true"
    else:
        init_msg = "random_init=true template_only=true"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ref_model = None
    if max(
        float(args.ref_kl_coef or 0.0),
        float(args.ref_kl_final_coef or 0.0),
    ) > 0:
        ref_path = args.ref_policy or args.policy_init
        ref_model = build_policy_model(
            arch=arch,
            width=width,
            slot_state=slot_state,
            option_context=option_context,
            state_feat_dim=state_feat_dim,
            opt_feat_dim=opt_feat_dim,
            plan_dim=plan_dim,
            hierarchical_plan=hierarchical_plan,
            history_k=history_k,
            opp_history_k=opp_history_k,
            log_history_k=log_history_k,
            board_history_k=board_history_k,
            board_history_feat_dim=board_history_feat_dim,
            state_layers=state_layers,
        ).to(device)
        loaded, skipped = _load_npz_init(ref_model, ref_path, device, partial=False)
        if skipped:
            raise RuntimeError(f"strict ref init unexpectedly skipped tensors: {skipped[:8]}")
        ref_model.eval()
        for p_ref in ref_model.parameters():
            p_ref.requires_grad_(False)

    candidate_deck = read_deck(args.deck)
    specs, weights = load_opponent_specs(args)
    opponents = load_opponents(
        specs,
        default_deck=args.deck,
        skip_bad_entries=args.skip_bad_entries,
    )
    base_weights = dict(weights)
    league_snapshots: list[Entry] = []
    anchor_corpus, anchor_indices = setup_anchor_corpus(args, state_feat_dim, opt_feat_dim)

    print(
        f"RL train: init={args.policy_init} save={args.save} device={device} "
        f"{memory_msg + ' ' if memory_msg else ''}"
        f"init_mode={args.init_mode} {init_msg} "
        f"arch={arch} width={width:g} state_layers={state_layers} "
        f"slot_state={slot_state} option_context={option_context} "
        f"state_feat_dim={state_feat_dim} opt_feat_dim={opt_feat_dim} "
        f"plan_dim={plan_dim} hierarchical_plan={hierarchical_plan} "
        f"history_k={history_k} opp_history_k={opp_history_k} "
        f"log_history_k={log_history_k} board_history_k={board_history_k} "
        f"opponents={len(opponents)} opponent_weight_mode={args.opponent_weight_mode} "
        f"rollout_workers={args.rollout_workers} rollout_temperature={args.rollout_temperature} "
        f"rollout_temperature_final={args.rollout_temperature_final} "
        f"rollout_top_k={args.rollout_top_k} rollout_top_k_final={args.rollout_top_k_final} "
        f"schedule_iters={args.schedule_iters} shaping_weight={args.shaping_weight} "
        f"bc_anchor={bool(anchor_corpus)} anchor_weight={args.bc_anchor_weight} "
        f"anchor_final_weight={args.bc_anchor_final_weight} "
        f"ref_kl_coef={args.ref_kl_coef} ref_kl_final_coef={args.ref_kl_final_coef} "
        f"target_kl={args.target_kl} "
        f"adv_norm={args.advantage_normalization} reward_weight={args.reward_weight_mode} "
        f"league_every={args.league_snapshot_every} league_max={args.league_max_snapshots} "
        f"save_policy={args.save_policy}",
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
    if (
        args.init_mode == "load"
        or max(float(args.bc_anchor_weight or 0.0), float(args.bc_anchor_final_weight or 0.0)) > 0
        or max(float(args.ref_kl_coef or 0.0), float(args.ref_kl_final_coef or 0.0)) > 0
    ):
        print(
            "WARNING: BC is considered locked for the current project direction. "
            "init-mode=load, BC anchor, or ref-KL means this run is a deprecated "
            "fine-tuning/control run, not the main path.",
            flush=True,
        )
    if args.dry_run:
        return

    encoder = FastEncoder()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.league_bootstrap_current:
        maybe_add_league_snapshot(
            model,
            opponents,
            weights,
            base_weights,
            league_snapshots,
            args,
            iteration=0,
            win_rate=0.0,
            force=True,
        )
        print(f"League bootstrap added; opponents={len(opponents)}", flush=True)

    best_wr = -1.0
    global_game = 0
    for iteration in range(1, args.iterations + 1):
        schedule_stats = apply_iteration_schedule(args, iteration)
        t0 = time.time()
        if int(args.rollout_workers) > 1:
            episodes, games_meta, global_game, t_collect = collect_parallel_rollouts(
                model,
                opponents,
                weights,
                args,
                iteration=iteration,
                global_game=global_game,
            )
        else:
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
                games_meta.append(meta)
                done = local_game + 1
                if args.progress_every and (
                    done == 1 or done % args.progress_every == 0 or done == args.games_per_iter
                ):
                    wins = sum(1 for row in games_meta if row["outcome"] == "win")
                    losses = sum(1 for row in games_meta if row["outcome"] == "loss")
                    draws = sum(1 for row in games_meta if row["outcome"] == "draw")
                    n_decisions = sum(len(ep) for ep in episodes)
                    print(
                        f"  iter {iteration} rollout {done}/{args.games_per_iter} "
                        f"W/L/D={wins}/{losses}/{draws} wr={wins/done:.3f} decisions={n_decisions}",
                        flush=True,
                    )
            t_collect = time.time() - t0
        samples = [d for episode in episodes for d in episode]
        wins = sum(1 for row in games_meta if row["outcome"] == "win")
        losses = sum(1 for row in games_meta if row["outcome"] == "loss")
        draws = sum(1 for row in games_meta if row["outcome"] == "draw")
        win_rate = wins / max(args.games_per_iter, 1)
        avg_reward = (
            wins * args.win_reward + losses * args.loss_reward + draws * args.draw_reward
        ) / max(args.games_per_iter, 1)
        _opp_rows, _opp_weight_stats = summarize_opponents(opponents, weights, games_meta)
        update_adaptive_opponent_weights(opponents, weights, base_weights, games_meta, args)
        opponent_rows, opponent_weight_stats = summarize_opponents(opponents, weights, games_meta)
        append_opponent_stats(args.opponent_stats_csv, iteration, opponent_rows)

        if len(samples) < 2:
            print(f"iter {iteration}: only {len(samples)} decisions collected; skipping update", flush=True)
            continue

        reward_stats = apply_batch_reward_weighting(episodes, games_meta, args)
        refresh_old_policy_stats(model, samples, args.minibatch)
        if ref_model is not None:
            refresh_reference_policy_stats(ref_model, samples, args.minibatch)
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
        maybe_add_league_snapshot(
            model,
            opponents,
            weights,
            base_weights,
            league_snapshots,
            args,
            iteration=iteration,
            win_rate=win_rate,
        )

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
            **reward_stats,
            **schedule_stats,
            **opponent_weight_stats,
            "league_snapshots": len(league_snapshots),
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
            f"clip={stats['clip_frac']:.3f} vclip={stats['value_clip_frac']:.3f} "
            f"ref={stats['ref_kl_loss']:.4f}/{stats['ref_logprob_delta']:.3f} "
            f"bc={stats['bc_anchor_loss']:.4f} rw={reward_stats['avg_reward_weight']:.3f} "
            f"temp={schedule_stats['rollout_temperature_eff']:.2f} "
            f"topk={schedule_stats['rollout_top_k_eff']} "
            f"eref={schedule_stats['ref_kl_coef_eff']:.4g} "
            f"ebc={schedule_stats['bc_anchor_weight_eff']:.4g} "
            f"opp_top={opponent_weight_stats['opponent_top_name']}:{opponent_weight_stats['opponent_top_weight']:.2f} "
            f"league={len(league_snapshots)} "
            f"early={int(stats['early_stop'])} "
            f"collect={t_collect:.1f}s update={t_update:.1f}s save={checkpoint or args.save}",
            flush=True,
        )

    if args.save_policy == "final":
        _save_npz(model, args.save)
        print(f"Final saved -> {args.save}", flush=True)
    else:
        if args.save_final:
            _save_npz(model, args.save_final)
            print(f"Final saved -> {args.save_final}", flush=True)
        elif args.save_policy == "both":
            final_path = str(Path(args.save).with_suffix("")) + ".final.npz"
            _save_npz(model, final_path)
            print(f"Final saved -> {final_path}", flush=True)
        print(f"Best rollout saved -> {args.save} best_wr={best_wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
