#!/usr/bin/env python3
"""Trace rule-overlay triggers in a concrete matchup."""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.numpy_policy import NumpyPolicy
from ptcg_rl.rule_overlay import RULE_MODES, apply_rule_overlay
from tools.eval_round_robin import legal_random, read_deck


def _sanitize(action: list[int], sel: dict) -> list[int]:
    opts = sel.get("option") or []
    mn = int(sel.get("minCount", 0) or 0)
    mc = int(sel.get("maxCount", 0) or 0)
    n = len(opts)
    picks = [p for p in action if 0 <= p < n]
    picks = list(dict.fromkeys(picks))
    if mn <= len(picks) <= mc:
        return picks[:mc]
    return legal_random(sel)


def _policy_action(policy: NumpyPolicy | None, obs: dict, deck: list[int],
                   rules: str = "") -> tuple[list[int], str]:
    sel = obs.get("select") or {}
    if policy is None:
        return legal_random(sel), ""
    try:
        raw = policy.select(obs, greedy=True)
    except Exception:
        return legal_random(sel), ""
    reason = ""
    action = raw
    if rules:
        try:
            decision = apply_rule_overlay(obs, raw, deck, mode=rules)
            action = decision.action
            reason = decision.reason if decision.action != raw else ""
        except Exception:
            action = raw
    return _sanitize(action, sel), reason


def _play_game(
    *,
    candidate_policy: NumpyPolicy,
    candidate_deck: list[int],
    opponent_policy: NumpyPolicy | None,
    opponent_deck: list[int],
    rules: str,
    swapped: bool,
    seed: int,
    max_turns: int,
) -> dict:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    candidate_policy.reset_history()
    if opponent_policy is not None:
        opponent_policy.reset_history()

    first_deck, second_deck = (
        (opponent_deck, candidate_deck) if swapped else (candidate_deck, opponent_deck)
    )
    candidate_side = 1 if swapped else 0
    obs, sd = battle_start(first_deck, second_deck)
    if obs is None:
        return {"result": 2, "win": 0, "draw": 1, "turns": 0, "reasons": Counter({"battle_start_error": 1})}

    reasons: Counter[str] = Counter()
    result = 2
    try:
        for turn in range(max_turns):
            cur = obs.get("current") or {}
            result = int(cur.get("result", -1))
            if result != -1:
                break
            if obs.get("select") is None:
                result = 2
                break
            side = int(cur.get("yourIndex", 0) or 0)
            if side == candidate_side:
                action, reason = _policy_action(candidate_policy, obs, candidate_deck, rules)
                if reason:
                    reasons[reason] += 1
            else:
                action, _ = _policy_action(opponent_policy, obs, opponent_deck, "")
            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()

    return {
        "result": result,
        "win": 1 if result == candidate_side else 0,
        "draw": 1 if result not in (0, 1) else 0,
        "turns": turn + 1 if "turn" in locals() else 0,
        "reasons": reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-policy", default="random")
    p.add_argument("--opponent-deck", required=True)
    p.add_argument("--rules", choices=RULE_MODES, required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alternate-first", action="store_true")
    p.add_argument("--out-csv", default="logs/rule_overlay_stats.csv")
    args = p.parse_args()

    candidate_policy = NumpyPolicy.load(args.policy)
    candidate_deck = read_deck(args.deck)
    opponent_policy = None if args.opponent_policy == "random" else NumpyPolicy.load(args.opponent_policy)
    opponent_deck = read_deck(args.opponent_deck)

    rows: list[dict] = []
    total_reasons: Counter[str] = Counter()
    wins = draws = 0
    for g in range(args.games):
        swapped = bool(args.alternate_first and g % 2)
        row = _play_game(
            candidate_policy=candidate_policy,
            candidate_deck=candidate_deck,
            opponent_policy=opponent_policy,
            opponent_deck=opponent_deck,
            rules=args.rules,
            swapped=swapped,
            seed=args.seed + g,
            max_turns=args.max_turns,
        )
        reasons = row.pop("reasons")
        total_reasons.update(reasons)
        wins += int(row["win"])
        draws += int(row["draw"])
        rows.append({
            "game": g,
            "swapped": int(swapped),
            "result": row["result"],
            "win": row["win"],
            "draw": row["draw"],
            "turns": row["turns"],
            "triggers": sum(reasons.values()),
            "reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "swapped", "result", "win", "draw", "turns", "triggers", "reasons"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out}")
    print(f"wins={wins}/{args.games} wr={wins / max(args.games, 1):.3f} draws={draws}")
    print("rule reasons:")
    for reason, count in total_reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
