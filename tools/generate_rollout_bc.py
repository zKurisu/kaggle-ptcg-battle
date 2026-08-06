#!/usr/bin/env python3
"""Generate BCCorpus-compatible trajectories from local rollout search.

This is meant for weak-matchup work where replay BC has too few successful
examples. It plays local games with a candidate policy against one or more
opponent policies, explores with stochastic/rule/MCTS actor modes, and writes
the candidate-side decisions from successful games as standard BC .npz files.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.deck_registry import deck_signature
from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM, FastEncoder
from ptcg_rl.numpy_policy import NumpyPolicy
from ptcg_rl.rule_overlay import RULE_MODES, apply_rule_overlay
from tools.bc_extract_v2 import FEATURE_VERSION, classify
from tools.eval_round_robin import (
    Entry,
    clean_entry_name,
    parse_entry,
    policy_action,
    read_deck,
    read_manifest_entry_specs,
)


@dataclass(frozen=True)
class ActorSpec:
    raw: str
    base: str
    temperature: float
    top_k: int
    rules: str
    weight: float


@dataclass(frozen=True)
class EntrySpec:
    name: str
    policy_path: str
    deck_path: str


def _has_cg_engine() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("cg.game") is not None
    except ModuleNotFoundError:
        return False


def _parse_actor(spec: str) -> ActorSpec:
    if "=" in spec:
        mode, weight_s = spec.split("=", 1)
        weight = float(weight_s)
    else:
        mode, weight = spec, 1.0
    mode = mode.strip()
    if not mode:
        raise ValueError("empty actor mode")
    if weight <= 0:
        raise ValueError(f"actor weight must be positive: {spec!r}")

    rules = ""
    base = mode
    if "+rules:" in mode:
        base, rules = mode.split("+rules:", 1)
    elif mode.startswith("rules:"):
        base, rules = "greedy", mode.split(":", 1)[1]
    base = base.strip() or "greedy"
    rules = rules.strip()
    if rules and rules not in RULE_MODES:
        raise ValueError(f"unknown rule mode {rules!r}; expected one of: {', '.join(RULE_MODES)}")

    temperature = 1.0
    if base.startswith("sample@"):
        temperature = float(base.split("@", 1)[1])
        base = "sample"
    top_k = 0
    if base.startswith("topk@"):
        top_k = int(base.split("@", 1)[1])
        base = "topk"
    if base not in {"greedy", "sample", "topk", "random", "mcts"}:
        raise ValueError(f"unknown actor base {base!r}; use greedy, sample[@T], topk@K, random, mcts")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive: {spec!r}")
    if base == "topk" and top_k <= 0:
        raise ValueError(f"topk actor requires a positive K: {spec!r}")
    return ActorSpec(raw=mode, base=base, temperature=temperature, top_k=top_k, rules=rules, weight=weight)


def _choose_actor(actors: list[ActorSpec], rng: random.Random) -> ActorSpec:
    total = sum(a.weight for a in actors)
    x = rng.random() * total
    acc = 0.0
    for actor in actors:
        acc += actor.weight
        if x <= acc:
            return actor
    return actors[-1]


def _legal_random(sel: dict[str, Any], rng: random.Random) -> list[int]:
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = rng.randint(lo, hi)
    return rng.sample(range(len(opts)), k) if k > 0 else []


def _valid_action(action: list[int], sel: dict[str, Any]) -> bool:
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if len(action) < mn or len(action) > mx:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(x, int) and 0 <= x < len(opts) for x in action)


def _sanitize_action(action: list[int], sel: dict[str, Any], rng: random.Random) -> list[int]:
    n = len(sel.get("option", []))
    action = [int(x) for x in action if isinstance(x, (int, np.integer)) and 0 <= int(x) < n]
    action = list(dict.fromkeys(action))
    if _valid_action(action, sel):
        return action
    return _legal_random(sel, rng)


def _load_entry(spec: EntrySpec) -> Entry:
    deck = read_deck(spec.deck_path)
    policy = None if spec.policy_path == "random" else NumpyPolicy.load(spec.policy_path)
    return Entry(spec.name, spec.policy_path, spec.deck_path, policy, deck)


def _entry_spec_from_cli(spec: str, default_deck: str, registry: str = "") -> EntrySpec:
    name, policy_path, deck_path = parse_entry(spec, default_deck, registry)
    return EntrySpec(name, policy_path, deck_path)


def _select_candidate_action(
    entry: Entry,
    obs: dict[str, Any],
    actor: ActorSpec,
    rng: random.Random,
    *,
    epsilon_random: float,
    mcts_sims: int,
    time_budget: float,
) -> tuple[list[int], str]:
    sel = obs.get("select") or {}
    if not sel.get("option"):
        return [], actor.raw
    if rng.random() < epsilon_random:
        return _legal_random(sel, rng), f"{actor.raw}|epsilon_random"

    actor_label = actor.raw
    try:
        if entry.policy is None or actor.base == "random":
            action = _legal_random(sel, rng)
        elif actor.base == "mcts":
            action = entry.policy.select_mcts(obs, entry.deck, sims=mcts_sims, time_budget=time_budget)
        elif actor.base == "topk":
            action = entry.policy.select(obs, greedy=False, temperature=actor.temperature, top_k=actor.top_k)
        elif actor.base == "sample":
            action = entry.policy.select(obs, greedy=False, temperature=actor.temperature)
        else:
            action = entry.policy.select(obs, greedy=True)
        if actor.rules:
            decision = apply_rule_overlay(obs, action, entry.deck, mode=actor.rules)
            action = decision.action
            if decision.reason:
                actor_label = f"{actor.raw}|{decision.reason}"
    except Exception:
        action = _legal_random(sel, rng)
        actor_label = f"{actor.raw}|fallback_random"
    return _sanitize_action(action, sel, rng), actor_label


def _encode_decision(
    encoder: FastEncoder,
    obs: dict[str, Any],
    action: list[int],
    actor_mode: str,
) -> dict[str, Any] | None:
    sel = obs.get("select") or {}
    if not sel.get("option") or not _valid_action(action, sel):
        return None
    try:
        ed = encoder.encode(obs)
    except Exception:
        return None
    return {
        "board": ed.board_cards.astype(np.int16),
        "hand": ed.hand_cards.astype(np.int16),
        "feats": ed.state_feats.astype(np.float16),
        "ot": ed.opt_type.astype(np.int16),
        "oc": ed.opt_card.astype(np.int16),
        "oc2": ed.opt_card2.astype(np.int16),
        "oa": ed.opt_attack.astype(np.int16),
        "of": ed.opt_feats.astype(np.float16),
        "action": np.asarray(action, dtype=np.int16),
        "min_c": int(ed.min_count),
        "max_c": int(ed.max_count),
        "actor_mode": actor_mode,
    }


def _final_status(result: int, candidate_side: int, timed_out: bool = False) -> tuple[bool, bool, str, float]:
    if timed_out:
        return False, True, "timeout", 0.0
    if result == candidate_side:
        return True, False, "win", 1.0
    if result in (0, 1):
        return False, False, "loss", -1.0
    return False, True, "draw", 0.0


def _should_keep(keep_outcomes: str, won: bool, draw: bool) -> bool:
    if keep_outcomes == "all":
        return True
    if keep_outcomes == "nonloss":
        return won or draw
    return won


def _play_rollout_game(
    candidate: Entry,
    opponents: list[Entry],
    actors: list[ActorSpec],
    *,
    encoder: FastEncoder,
    game_i: int,
    seed: int,
    max_turns: int,
    keep_outcomes: str,
    actor_scope: str,
    epsilon_random: float,
    mcts_sims: int,
    time_budget: float,
    opponent_mcts: bool,
    opponent_mcts_sims: int,
    opponent_time_budget: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from cg.game import battle_finish, battle_select, battle_start

    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    opponent = opponents[game_i % len(opponents)]
    candidate_first = bool(game_i % 2 == 0)
    first, second = (candidate, opponent) if candidate_first else (opponent, candidate)
    candidate_side = 0 if candidate_first else 1
    game_actor = _choose_actor(actors, rng) if actor_scope == "game" else None
    decisions: list[dict[str, Any]] = []
    stats = {
        "games": 1,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "errors": 0,
        "timeouts": 0,
        "decisions_seen": 0,
        "decisions_written": 0,
        "steps": 0,
        "opponent": opponent.name,
        "actor": game_actor.raw if game_actor else "decision_mix",
    }

    obs, _sd = battle_start(first.deck, second.deck)
    if obs is None:
        stats["errors"] = 1
        stats["draws"] = 1
        return [], stats

    try:
        for step in range(max_turns):
            stats["steps"] = step + 1
            cur = obs.get("current") or {}
            result = int(cur.get("result", -1))
            if result != -1:
                won, draw, status, reward = _final_status(result, candidate_side)
                break
            if obs.get("select") is None:
                stats["errors"] = 1
                won, draw, status, reward = False, True, "bad_select", 0.0
                break
            side = int(cur.get("yourIndex", 0) or 0)
            if side == candidate_side:
                actor = game_actor or _choose_actor(actors, rng)
                action, actor_label = _select_candidate_action(
                    candidate,
                    obs,
                    actor,
                    rng,
                    epsilon_random=epsilon_random,
                    mcts_sims=mcts_sims,
                    time_budget=time_budget,
                )
                stats["decisions_seen"] += 1
                row = _encode_decision(encoder, obs, action, actor_label)
                if row is not None:
                    decisions.append(row)
            else:
                action = policy_action(
                    opponent,
                    obs,
                    use_mcts=opponent_mcts,
                    sims=opponent_mcts_sims,
                    time_budget=opponent_time_budget,
                )
            obs = battle_select(action)
            if obs is None:
                stats["errors"] = 1
                won, draw, status, reward = False, True, "engine_none", 0.0
                break
        else:
            stats["timeouts"] = 1
            won, draw, status, reward = _final_status(-1, candidate_side, timed_out=True)
    finally:
        battle_finish()

    stats["wins"] = int(won)
    stats["losses"] = int((not won) and (not draw))
    stats["draws"] = int(draw)
    if not _should_keep(keep_outcomes, won, draw):
        return [], stats

    candidate_sig = deck_signature(candidate.deck)
    opponent_sig = deck_signature(opponent.deck)
    opponent_arch = classify(opponent.deck)
    episode_id = f"rollout_{candidate.name}_{opponent.name}_{game_i:08d}"
    for row in decisions:
        row.update(
            {
                "deck_sig": candidate_sig,
                "team_name": f"rollout:{candidate.name}",
                "score": 0.0,
                "opponent_deck_sig": opponent_sig,
                "opponent_archetype": opponent_arch,
                "opponent_team_name": f"rollout:{opponent.name}",
                "opponent_score": 0.0,
                "opponent_score_band": "generated",
                "episode_id": episode_id,
                "player_index": candidate_side,
                "reward": reward,
                "won": int(won),
                "draw": int(draw),
                "final_status": status,
                "game_steps": int(stats["steps"]),
                "opponent_name": opponent.name,
            }
        )
    stats["decisions_written"] = len(decisions)
    return decisions, stats


def _write_npz(path: str, rows: list[dict[str, Any]], feature_version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        board=np.asarray([r["board"] for r in rows], dtype=object),
        hand=np.asarray([r["hand"] for r in rows], dtype=object),
        feats=np.asarray([r["feats"] for r in rows], dtype=object),
        ot=np.asarray([r["ot"] for r in rows], dtype=object),
        oc=np.asarray([r["oc"] for r in rows], dtype=object),
        oc2=np.asarray([r["oc2"] for r in rows], dtype=object),
        oa=np.asarray([r["oa"] for r in rows], dtype=object),
        of_arr=np.asarray([r["of"] for r in rows], dtype=object),
        action=np.asarray([r["action"] for r in rows], dtype=object),
        min_c=np.asarray([r["min_c"] for r in rows], dtype=np.int16),
        max_c=np.asarray([r["max_c"] for r in rows], dtype=np.int16),
        deck_sig=np.asarray([r["deck_sig"] for r in rows], dtype=object),
        team_name=np.asarray([r["team_name"] for r in rows], dtype=object),
        score=np.asarray([r["score"] for r in rows], dtype=np.float32),
        opponent_deck_sig=np.asarray([r["opponent_deck_sig"] for r in rows], dtype=object),
        opponent_archetype=np.asarray([r["opponent_archetype"] for r in rows], dtype=object),
        opponent_team_name=np.asarray([r["opponent_team_name"] for r in rows], dtype=object),
        opponent_score=np.asarray([r["opponent_score"] for r in rows], dtype=np.float32),
        opponent_score_band=np.asarray([r["opponent_score_band"] for r in rows], dtype=object),
        episode_id=np.asarray([r["episode_id"] for r in rows], dtype=object),
        player_index=np.asarray([r["player_index"] for r in rows], dtype=np.int8),
        reward=np.asarray([r["reward"] for r in rows], dtype=np.float32),
        won=np.asarray([r["won"] for r in rows], dtype=np.int8),
        draw=np.asarray([r["draw"] for r in rows], dtype=np.int8),
        final_status=np.asarray([r["final_status"] for r in rows], dtype=object),
        game_steps=np.asarray([r["game_steps"] for r in rows], dtype=np.int16),
        actor_mode=np.asarray([r["actor_mode"] for r in rows], dtype=object),
        opponent_name=np.asarray([r["opponent_name"] for r in rows], dtype=object),
        feature_version=np.asarray(feature_version, dtype=object),
        state_feat_dim=np.asarray(STATE_FEAT_DIM, dtype=np.int16),
        opt_feat_dim=np.asarray(OPT_FEAT_DIM, dtype=np.int16),
    )


def _worker_run(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = _load_entry(payload["candidate"])
    opponents = [_load_entry(s) for s in payload["opponents"]]
    encoder = FastEncoder()
    actors = payload["actors"]
    out_root = payload["out_root"]
    out_band = payload["out_band"]
    arch = classify(candidate.deck).replace(" ", "_")
    rows: list[dict[str, Any]] = []
    written_paths: list[str] = []
    rows_total = 0
    totals = Counter()
    opponent_counts = Counter()
    actor_counts = Counter()
    actor_wins = Counter()
    t0 = time.time()
    arch_dir = os.path.join(out_root, arch, out_band.replace(" ", "_"))
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", payload["tag"]).strip("_") or "rollout"
    part_i = 0

    def flush_rows(*, final: bool = False) -> None:
        nonlocal rows, part_i
        if not rows:
            return
        if payload["flush_every_games"] > 0:
            filename = f"{safe_tag}_w{payload['worker_id']:03d}_p{part_i:03d}.npz"
        else:
            filename = f"{safe_tag}_w{payload['worker_id']:03d}.npz"
        path = os.path.join(arch_dir, filename)
        _write_npz(path, rows, f"{FEATURE_VERSION}_rollout_search")
        written_paths.append(path)
        rows = []
        part_i += 1

    for local_i, game_i in enumerate(payload["game_indices"], 1):
        game_rows, stats = _play_rollout_game(
            candidate,
            opponents,
            actors,
            encoder=encoder,
            game_i=game_i,
            seed=payload["seed"] + game_i,
            max_turns=payload["max_turns"],
            keep_outcomes=payload["keep_outcomes"],
            actor_scope=payload["actor_scope"],
            epsilon_random=payload["epsilon_random"],
            mcts_sims=payload["mcts_sims"],
            time_budget=payload["time_budget"],
            opponent_mcts=payload["opponent_mcts"],
            opponent_mcts_sims=payload["opponent_mcts_sims"],
            opponent_time_budget=payload["opponent_time_budget"],
        )
        rows.extend(game_rows)
        rows_total += len(game_rows)
        for key in ("games", "wins", "losses", "draws", "errors", "timeouts", "decisions_seen", "decisions_written", "steps"):
            totals[key] += int(stats.get(key, 0))
        opponent_counts[str(stats.get("opponent", ""))] += 1
        actor = str(stats.get("actor", ""))
        actor_counts[actor] += 1
        if int(stats.get("wins", 0)):
            actor_wins[actor] += 1
        if payload["flush_every_games"] > 0 and local_i % int(payload["flush_every_games"]) == 0:
            flush_rows()
        if payload["worker_progress_every"] and (
            local_i == 1
            or local_i % int(payload["worker_progress_every"]) == 0
            or local_i == len(payload["game_indices"])
        ):
            elapsed = time.time() - t0
            rate = int(totals["games"]) / max(elapsed, 1e-9)
            print(
                f"  {payload['tag']} w{payload['worker_id']:03d}: "
                f"{local_i}/{len(payload['game_indices'])} games "
                f"wins={totals['wins']} wr={totals['wins']/max(totals['games'],1):.3f} "
                f"rows={rows_total} {rate:.2f} games/s",
                flush=True,
            )

    flush_rows(final=True)

    totals["rows"] = rows_total
    result = dict(totals)
    result.update(
        {
            "worker_id": payload["worker_id"],
            "path": ";".join(written_paths),
            "elapsed": time.time() - t0,
            "opponent_counts": dict(opponent_counts),
            "actor_counts": dict(actor_counts),
            "actor_wins": dict(actor_wins),
        }
    )
    return result


def _write_summary(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "worker_id",
        "games",
        "wins",
        "losses",
        "draws",
        "errors",
        "timeouts",
        "decisions_seen",
        "decisions_written",
        "rows",
        "elapsed",
        "path",
        "actor_counts",
        "actor_wins",
        "opponent_counts",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            for k in ("actor_counts", "actor_wins", "opponent_counts"):
                out[k] = repr(out[k])
            w.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--opponent", action="append", default=[], help="NAME=POLICY:DECK; repeatable")
    parser.add_argument("--opponent-manifest", action="append", default=[],
                        help="CSV manifest with eval entries; repeatable")
    parser.add_argument("--opponent-manifest-limit", type=int, default=0)
    parser.add_argument("--deck", default="deck.csv", help="default deck for entries without :DECK")
    parser.add_argument("--registry", default="")
    parser.add_argument("--actor", action="append", default=None,
                        help="actor mode=weight. Modes: greedy, sample[@T], random, mcts, +rules:MODE")
    parser.add_argument("--actor-scope", choices=["game", "decision"], default="decision")
    parser.add_argument("--epsilon-random", type=float, default=0.02)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--keep-outcomes", choices=["win", "nonloss", "all"], default="win")
    parser.add_argument("--mcts-sims", type=int, default=48)
    parser.add_argument("--time-budget", type=float, default=4.0)
    parser.add_argument("--opponent-mcts", action="store_true")
    parser.add_argument("--opponent-mcts-sims", type=int, default=48)
    parser.add_argument("--opponent-time-budget", type=float, default=4.0)
    parser.add_argument("--out-root", default="data/generated_rollout_bc")
    parser.add_argument("--out-band", default="generated")
    parser.add_argument("--tag", default="")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="print worker completions every N finished workers; 0 disables")
    parser.add_argument("--worker-progress-every", type=int, default=0,
                        help="print progress inside each worker every N games; 0 disables")
    parser.add_argument("--flush-every-games", type=int, default=0,
                        help="write partial NPZ chunks every N games per worker; 0 writes once at worker end")
    args = parser.parse_args()

    if not _has_cg_engine():
        parser.error("cg.game not found; run on the remote/Kaggle environment with the battle engine available")
    if args.games <= 0:
        parser.error("--games must be positive")
    if not (0.0 <= args.epsilon_random <= 1.0):
        parser.error("--epsilon-random must be in [0, 1]")

    candidate = _entry_spec_from_cli(args.candidate, args.deck, args.registry)
    opponent_specs = list(args.opponent)
    for manifest in args.opponent_manifest:
        opponent_specs.extend(
            read_manifest_entry_specs(manifest, limit=args.opponent_manifest_limit, random_from_deck=False)
        )
    if not opponent_specs:
        parser.error("provide at least one --opponent or --opponent-manifest")
    opponents = [_entry_spec_from_cli(s, args.deck, args.registry) for s in opponent_specs]
    actor_specs = args.actor or ["greedy=1", "sample@1.15=1", "sample@1.6=1", "random=0.1"]
    actors = [_parse_actor(s) for s in actor_specs]
    workers = max(1, min(int(args.workers), int(args.games)))
    tag = args.tag or f"{candidate.name}_rollout"

    print(f"Rollout BC generation: candidate={candidate.name} games={args.games} workers={workers}", flush=True)
    print(f"  candidate policy={candidate.policy_path} deck={candidate.deck_path}", flush=True)
    print(f"  opponents={len(opponents)} keep={args.keep_outcomes} actor_scope={args.actor_scope}", flush=True)
    for opp in opponents:
        print(f"    opponent {opp.name}: {opp.policy_path} | {opp.deck_path}", flush=True)
    print("  actors=" + ", ".join(f"{a.raw}:{a.weight:g}" for a in actors), flush=True)

    game_indices = list(range(args.games))
    chunks = [game_indices[i::workers] for i in range(workers)]
    payloads = [
        {
            "worker_id": wi,
            "candidate": candidate,
            "opponents": opponents,
            "actors": actors,
            "game_indices": chunk,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "keep_outcomes": args.keep_outcomes,
            "actor_scope": args.actor_scope,
            "epsilon_random": args.epsilon_random,
            "mcts_sims": args.mcts_sims,
            "time_budget": args.time_budget,
            "opponent_mcts": args.opponent_mcts,
            "opponent_mcts_sims": args.opponent_mcts_sims,
            "opponent_time_budget": args.opponent_time_budget,
            "out_root": args.out_root,
            "out_band": args.out_band,
            "tag": tag,
            "worker_progress_every": args.worker_progress_every,
            "flush_every_games": args.flush_every_games,
        }
        for wi, chunk in enumerate(chunks)
        if chunk
    ]

    t0 = time.time()
    results: list[dict[str, Any]] = []
    if workers == 1:
        results.append(_worker_run(payloads[0]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_run, p) for p in payloads]
            for done, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                results.append(row)
                if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == len(futs)):
                    games_done = sum(int(r.get("games", 0)) for r in results)
                    wins = sum(int(r.get("wins", 0)) for r in results)
                    rows = sum(int(r.get("rows", 0)) for r in results)
                    elapsed = time.time() - t0
                    rate = games_done / max(elapsed, 1e-9)
                    print(
                        f"  workers {done}/{len(futs)} games={games_done}/{args.games} "
                        f"wins={wins} wr={wins/max(games_done,1):.3f} rows={rows} "
                        f"{rate:.2f} games/s",
                        flush=True,
                    )

    totals = Counter()
    actor_counts = Counter()
    actor_wins = Counter()
    opponent_counts = Counter()
    for row in results:
        for key in ("games", "wins", "losses", "draws", "errors", "timeouts", "decisions_seen", "decisions_written", "rows"):
            totals[key] += int(row.get(key, 0))
        actor_counts.update(row.get("actor_counts", {}))
        actor_wins.update(row.get("actor_wins", {}))
        opponent_counts.update(row.get("opponent_counts", {}))

    if args.summary_csv:
        _write_summary(args.summary_csv, sorted(results, key=lambda r: int(r.get("worker_id", 0))))
        print(f"Wrote summary {args.summary_csv}", flush=True)
    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s: games={totals['games']} wins={totals['wins']} "
        f"wr={totals['wins']/max(totals['games'],1):.3f} rows={totals['rows']} "
        f"decisions_seen={totals['decisions_seen']} errors={totals['errors']} timeouts={totals['timeouts']}",
        flush=True,
    )
    if actor_counts:
        print("Actor games/wins:", flush=True)
        for actor, count in actor_counts.most_common():
            wins = actor_wins.get(actor, 0)
            print(f"  {actor}: {wins}/{count} wr={wins/max(count,1):.3f}", flush=True)
    if opponent_counts:
        print("Opponent games:", flush=True)
        for opp, count in opponent_counts.most_common():
            print(f"  {opp}: {count}", flush=True)


if __name__ == "__main__":
    main()
