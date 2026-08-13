#!/usr/bin/env python3
"""Parallel rule-overlay trigger diagnostics for a concrete matchup."""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.numpy_policy import NumpyPolicy
from ptcg_rl.resource_planner import apply_rule_decision, make_rule_planner
from ptcg_rl.rule_overlay import RULE_MODES
from tools.eval_round_robin import legal_random, read_deck


@dataclass
class DiagEntry:
    policy_path: str
    deck_path: str
    policy: NumpyPolicy | None
    deck: list[int]
    rules: str = ""
    planner: object | None = None


_CANDIDATE: DiagEntry | None = None
_OPPONENT: DiagEntry | None = None
_MAX_TURNS = 700


def _load_entry(policy_path: str, deck_path: str, rules: str = "") -> DiagEntry:
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
    return DiagEntry(policy_path, deck_path, policy, deck, rules, make_rule_planner(rules, deck))


def _sanitize(action: list[int], sel: dict) -> list[int]:
    opts = sel.get("option") or []
    mn = int(sel.get("minCount", 0) or 0)
    mc = int(sel.get("maxCount", 0) or 0)
    n = len(opts)
    picks = [int(p) for p in action if 0 <= int(p) < n]
    picks = list(dict.fromkeys(picks))
    if mn <= len(picks) <= mc:
        return picks[:mc]
    return legal_random(sel)


def _policy_action(entry: DiagEntry, obs: dict) -> tuple[list[int], str]:
    sel = obs.get("select") or {}
    if entry.policy is None:
        return legal_random(sel), ""
    try:
        raw = entry.policy.select(obs, greedy=True, update_history=False)
    except Exception:
        action = legal_random(sel)
        try:
            entry.policy.remember_decision(obs, action)
        except Exception:
            pass
        return action, "policy_error"
    action = raw
    reason = ""
    if entry.rules:
        try:
            decision = apply_rule_decision(obs, raw, entry.deck, mode=entry.rules, planner=entry.planner)
            action = decision.action
            if decision.action != raw:
                reason = decision.reason
        except Exception as exc:
            reason = f"rule_error:{type(exc).__name__}"
            action = raw
    final = _sanitize(action, sel)
    try:
        entry.policy.remember_decision(obs, final)
    except Exception:
        pass
    return final, reason


def _reset_entry(entry: DiagEntry) -> None:
    if entry.policy is not None and hasattr(entry.policy, "reset_history"):
        entry.policy.reset_history()
    if entry.planner is not None:
        entry.planner.reset(entry.deck)


def _init_worker(candidate_policy: str, candidate_deck: str, opponent_policy: str,
                 opponent_deck: str, rules: str, max_turns: int) -> None:
    global _CANDIDATE, _OPPONENT, _MAX_TURNS
    _CANDIDATE = _load_entry(candidate_policy, candidate_deck, rules)
    _OPPONENT = _load_entry(opponent_policy, opponent_deck, "")
    _MAX_TURNS = int(max_turns)


def _play_one(payload: tuple[int, int, bool]) -> dict:
    from cg.game import battle_finish, battle_select, battle_start

    game_i, seed, alternate_first = payload
    random.seed(seed)
    assert _CANDIDATE is not None and _OPPONENT is not None
    _reset_entry(_CANDIDATE)
    _reset_entry(_OPPONENT)

    swapped = bool(alternate_first and game_i % 2)
    first, second = (_OPPONENT, _CANDIDATE) if swapped else (_CANDIDATE, _OPPONENT)
    candidate_side = 1 if swapped else 0
    reasons: Counter[str] = Counter()
    result = 2
    turns = 0

    obs, _sd = battle_start(first.deck, second.deck)
    if obs is None:
        return {"game": game_i, "swapped": int(swapped), "result": 2, "win": 0, "draw": 1, "turns": 0, "reasons": reasons}

    try:
        for turns in range(1, _MAX_TURNS + 1):
            cur = obs.get("current") or {}
            raw_result = int(cur.get("result", -1))
            if raw_result != -1:
                result = raw_result
                break
            if obs.get("select") is None:
                result = 2
                break
            side = int(cur.get("yourIndex", 0) or 0)
            entry = first if side == 0 else second
            action, reason = _policy_action(entry, obs)
            if side == candidate_side and reason:
                reasons[reason] += 1
            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()

    win = int(result == candidate_side)
    draw = int(result not in (0, 1))
    return {
        "game": game_i,
        "swapped": int(swapped),
        "result": result,
        "win": win,
        "draw": draw,
        "turns": turns,
        "reasons": reasons,
    }


def _write_rows(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "swapped", "result", "win", "draw", "turns", "triggers", "reasons"])
        w.writeheader()
        for row in rows:
            reasons = row.pop("reasons")
            w.writerow({
                **row,
                "triggers": sum(reasons.values()),
                "reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
            })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-policy", default="random")
    p.add_argument("--opponent-deck", required=True)
    p.add_argument("--rules", choices=RULE_MODES, required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alternate-first", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--out-csv", default="logs/rule_overlay_stats_fast.csv")
    args = p.parse_args()

    workers = max(1, min(int(args.workers), int(args.games)))
    tasks = [(g, args.seed + g, args.alternate_first) for g in range(args.games)]
    rows: list[dict] = []
    total_reasons: Counter[str] = Counter()
    wins = draws = 0
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.policy, args.deck, args.opponent_policy, args.opponent_deck, args.rules, args.max_turns),
    ) as ex:
        futs = [ex.submit(_play_one, task) for task in tasks]
        for done, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            reasons = row["reasons"]
            total_reasons.update(reasons)
            wins += int(row["win"])
            draws += int(row["draw"])
            rows.append(row)
            if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                print(
                    f"  {done}/{args.games} wins={wins} wr={wins / done:.3f} "
                    f"draws={draws} triggers={sum(total_reasons.values())} "
                    f"{rate:.2f} games/s eta={(args.games - done) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )
    rows.sort(key=lambda r: int(r["game"]))
    _write_rows(args.out_csv, rows)
    print(f"Wrote {args.out_csv}", flush=True)
    print(f"wins={wins}/{args.games} wr={wins / max(args.games, 1):.3f} draws={draws}", flush=True)
    print("rule reasons:", flush=True)
    for reason, count in total_reasons.most_common():
        print(f"  {reason}: {count}", flush=True)


if __name__ == "__main__":
    main()
