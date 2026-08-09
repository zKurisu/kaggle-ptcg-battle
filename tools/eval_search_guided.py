#!/usr/bin/env python3
"""Evaluate an online search-guided candidate policy against one opponent.

This is a direct validation harness for search_action_teacher.py. At each
candidate-side decision, it builds the same root-action teacher problem and
executes the best searched action in the real battle. It is intentionally slow
and meant for small game counts before turning motifs into rules or labels.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
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

import tools.search_action_teacher as teacher
from tools.eval_round_robin import Entry, parse_entry, policy_action, read_deck
from tools.trace_matchup_decisions import card_name, type_name
from ptcg_rl.numpy_policy import NumpyPolicy


GAME_FIELDS = [
    "game",
    "seed",
    "candidate_side",
    "result",
    "outcome",
    "steps",
    "candidate_decisions",
    "search_decisions",
    "fallback_decisions",
]

DECISION_FIELDS = [
    "game",
    "step",
    "turn",
    "context",
    "option_count",
    "baseline_action",
    "baseline_desc",
    "best_action",
    "best_desc",
    "baseline_score",
    "best_score",
    "delta_score",
    "evaluated_actions",
]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def has_cg_engine() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("cg.game") is not None
    except ModuleNotFoundError:
        return False


def load_entry(spec: str) -> Entry:
    name, policy_path, deck_path = parse_entry(spec, default_deck="deck.csv")
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
    return Entry(name, policy_path, deck_path, policy, deck)


def reset_entry(entry: Entry) -> None:
    if entry.policy is not None and hasattr(entry.policy, "reset_history"):
        entry.policy.reset_history()


def describe_action(obs: dict[str, Any], action: list[int]) -> str:
    try:
        encoded = teacher.FastEncoder().encode(obs)
    except Exception:
        encoded = None
    parts = []
    for idx in action:
        opts = (obs.get("select") or {}).get("option", [])
        typ = safe_int(opts[idx].get("type")) if 0 <= idx < len(opts) else 0
        cid = 0
        cid2 = 0
        if encoded is not None and 0 <= idx < len(encoded.opt_type):
            typ = int(encoded.opt_type[idx])
            cid = int(encoded.opt_card[idx])
            cid2 = int(encoded.opt_card2[idx])
        label = f"{idx}:{type_name(typ)}"
        if cid:
            label += f":{card_name(cid)}"
        if cid2 and cid2 != cid:
            label += f"->{card_name(cid2)}"
        parts.append(label)
    return " | ".join(parts) if parts else "STOP"


def ranking(policy, obs: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if policy is None:
        return []
    try:
        return policy.first_step_ranking(obs)[:limit]
    except Exception:
        return []


def baseline_action(entry: Entry, obs: dict[str, Any]) -> list[int]:
    if entry.policy is None:
        return []
    try:
        action = entry.policy.select(obs, greedy=True, update_history=False)
    except Exception:
        return []
    sel = obs.get("select") or {}
    return action if teacher.valid_action(action, sel) else []


def search_guided_action(
    candidate: Entry,
    opponent: Entry,
    obs: dict[str, Any],
    candidate_side: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    base = baseline_action(candidate, obs)
    rec = {
        "state_id": f"online_g{args._game:05d}_s{args._step:04d}",
        "candidate": candidate.name,
        "opponent": opponent.name,
        "candidate_side": candidate_side,
        "turn": safe_int((obs.get("current") or {}).get("turn")),
        "context": safe_int((obs.get("select") or {}).get("context")),
        "option_count": len((obs.get("select") or {}).get("option") or []),
        "baseline_action": base,
        "policy_ranking": ranking(candidate.policy, obs, args.root_top_options),
        "obs": obs,
    }
    teacher._CANDIDATE = candidate
    teacher._OPPONENT = opponent
    teacher._ARGS = argparse.Namespace(
        rollouts_per_action=args.rollouts_per_action,
        rollout_horizon=args.rollout_horizon,
        root_top_options=args.root_top_options,
        combo_top_options=args.combo_top_options,
        max_combo_size=args.max_combo_size,
        random_actions=args.random_actions,
        max_actions=args.max_actions,
        candidate_rollout_temperature=args.candidate_rollout_temperature,
        opponent_rollout_temperature=args.opponent_rollout_temperature,
    )
    try:
        _, best = teacher.process_record((0, rec, seed))
        best_action = teacher.parse_action(best.get("best_action", ""))
        if teacher.valid_action(best_action, obs.get("select") or {}):
            return best_action, {
                "baseline_action": teacher.action_key(base),
                "baseline_desc": describe_action(obs, base),
                "best_action": teacher.action_key(best_action),
                "best_desc": describe_action(obs, best_action),
                "baseline_score": best.get("baseline_score", ""),
                "best_score": best.get("best_score", ""),
                "delta_score": best.get("delta_score_vs_baseline", ""),
                "evaluated_actions": best.get("evaluated_actions", 0),
            }
    except Exception as exc:
        return base, {
            "baseline_action": teacher.action_key(base),
            "baseline_desc": describe_action(obs, base),
            "best_action": teacher.action_key(base),
            "best_desc": f"fallback:{type(exc).__name__}",
            "baseline_score": "",
            "best_score": "",
            "delta_score": "",
            "evaluated_actions": 0,
        }
    return base, {
        "baseline_action": teacher.action_key(base),
        "baseline_desc": describe_action(obs, base),
        "best_action": teacher.action_key(base),
        "best_desc": "fallback:no_valid_search_action",
        "baseline_score": "",
        "best_score": "",
        "delta_score": "",
        "evaluated_actions": 0,
    }


def play_game(candidate: Entry, opponent: Entry, game: int, seed: int, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    swapped = bool(game % 2)
    first, second = (opponent, candidate) if swapped else (candidate, opponent)
    candidate_side = 1 if swapped else 0
    for entry in (first, second):
        reset_entry(entry)
    obs, _ = battle_start(first.deck, second.deck)
    decisions: list[dict] = []
    result = 2
    candidate_decisions = 0
    search_decisions = 0
    fallback_decisions = 0
    steps = 0
    if obs is None:
        return {
            "game": game,
            "seed": seed,
            "candidate_side": candidate_side,
            "result": result,
            "outcome": "draw",
            "steps": 0,
            "candidate_decisions": 0,
            "search_decisions": 0,
            "fallback_decisions": 0,
        }, []

    try:
        for steps in range(args.max_turns):
            cur = obs.get("current") or {}
            res = safe_int(cur.get("result"), -1)
            if res != -1:
                result = res if res in (0, 1) else 2
                break
            sel = obs.get("select")
            if sel is None:
                result = 2
                break
            side = safe_int(cur.get("yourIndex"))
            entry = first if side == 0 else second
            if side == candidate_side:
                candidate_decisions += 1
                turn = safe_int(cur.get("turn"))
                context = safe_int(sel.get("context"))
                option_count = len(sel.get("option") or [])
                context_ok = not args.search_context or str(context) in args.search_context
                search_ok = (
                    turn >= args.search_min_turn
                    and (not args.search_max_turn or turn <= args.search_max_turn)
                    and context_ok
                    and option_count >= args.search_min_option_count
                )
                if search_ok:
                    args._game = game
                    args._step = steps
                    action, meta = search_guided_action(candidate, opponent, obs, candidate_side, args, seed + steps * 1297)
                    search_decisions += int(int(meta.get("evaluated_actions", 0) or 0) > 0)
                    fallback_decisions += int(not meta.get("evaluated_actions"))
                    meta.update({
                        "game": game,
                        "step": steps,
                        "turn": turn,
                        "context": context,
                        "option_count": option_count,
                    })
                    decisions.append(meta)
                    teacher.remember_external_action(candidate.policy, obs, action)
                else:
                    action = policy_action(entry, obs, use_mcts=False, sims=0, time_budget=0.0)
            else:
                action = policy_action(entry, obs, use_mcts=False, sims=0, time_budget=0.0)
            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()

    if result == 2:
        outcome = "draw"
    elif result == candidate_side:
        outcome = "win"
    else:
        outcome = "loss"
    return {
        "game": game,
        "seed": seed,
        "candidate_side": candidate_side,
        "result": result,
        "outcome": outcome,
        "steps": steps,
        "candidate_decisions": candidate_decisions,
        "search_decisions": search_decisions,
        "fallback_decisions": fallback_decisions,
    }, decisions


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=700)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--rollout-horizon", type=int, default=100)
    parser.add_argument("--root-top-options", type=int, default=6)
    parser.add_argument("--combo-top-options", type=int, default=4)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--random-actions", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--candidate-rollout-temperature", type=float, default=0.0)
    parser.add_argument("--opponent-rollout-temperature", type=float, default=0.0)
    parser.add_argument("--search-min-turn", type=int, default=0)
    parser.add_argument("--search-max-turn", type=int, default=0, help="0 disables upper bound")
    parser.add_argument("--search-context", action="append", default=[])
    parser.add_argument("--search-min-option-count", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out-games-csv", required=True)
    parser.add_argument("--out-decisions-csv", required=True)
    args = parser.parse_args()

    if not has_cg_engine():
        parser.error("cg.game not found. Run this in the remote/Kaggle engine environment.")

    candidate = load_entry(args.candidate)
    opponent = load_entry(args.opponent)
    game_rows: list[dict] = []
    decision_rows: list[dict] = []
    t0 = time.time()
    for game in range(args.games):
        grow, drows = play_game(candidate, opponent, game, args.seed + game, args)
        game_rows.append(grow)
        decision_rows.extend(drows)
        done = game + 1
        if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
            wins = sum(1 for r in game_rows if r["outcome"] == "win")
            losses = sum(1 for r in game_rows if r["outcome"] == "loss")
            draws = sum(1 for r in game_rows if r["outcome"] == "draw")
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"  {done}/{args.games} W/L/D={wins}/{losses}/{draws} "
                f"wr={wins/done:.3f} {rate:.2f} games/s",
                flush=True,
            )

    write_csv(Path(args.out_games_csv), GAME_FIELDS, game_rows)
    write_csv(Path(args.out_decisions_csv), DECISION_FIELDS, decision_rows)
    wins = sum(1 for r in game_rows if r["outcome"] == "win")
    losses = sum(1 for r in game_rows if r["outcome"] == "loss")
    draws = sum(1 for r in game_rows if r["outcome"] == "draw")
    print(
        f"Wrote {args.out_games_csv} and {args.out_decisions_csv}; "
        f"final W/L/D={wins}/{losses}/{draws} wr={wins/max(args.games,1):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
