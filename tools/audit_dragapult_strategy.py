#!/usr/bin/env python3
"""Summarize Dragapult trajectory targets by team/opponent/score bucket."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = [
    "outcome_win",
    "setup_success",
    "tempo_success",
    "strategy_success",
    "dreepy_board_by_2",
    "drakloak_board_by_4",
    "dragapult_board_by_6",
    "dusknoir_line_by_6",
    "dragapult_attack_by_6",
    "damage_counter_used",
    "drakloak_before_dragapult_evolve",
    "dragapult_strategy_success",
]


def fnum(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_bucket(score: float) -> str:
    if score >= 1200:
        return "1200+"
    if score >= 1100:
        return "1100-1199"
    if score >= 1000:
        return "1000-1099"
    if score >= 900:
        return "900-999"
    if score >= 800:
        return "800-899"
    return "<800"


def make_key(row: dict[str, str], group_by: list[str]) -> str:
    parts = []
    for col in group_by:
        if col == "score_bucket":
            parts.append(score_bucket(fnum(row.get("score", ""))))
        elif col == "opponent_score_bucket":
            parts.append(score_bucket(fnum(row.get("opponent_score", ""))))
        else:
            parts.append(str(row.get(col, "")).strip() or "unknown")
    return " / ".join(parts)


def empty() -> dict[str, float]:
    d = {
        "games": 0.0,
        "score_sum": 0.0,
        "opponent_score_sum": 0.0,
        "decisions_sum": 0.0,
    }
    for metric in METRICS:
        d[f"{metric}_sum"] = 0.0
    return d


def summarize(rows: list[dict[str, str]], group_by: list[str], min_games: int) -> list[dict[str, str]]:
    buckets: dict[str, dict[str, float]] = defaultdict(empty)
    for row in rows:
        key = make_key(row, group_by)
        acc = buckets[key]
        acc["games"] += 1.0
        acc["score_sum"] += fnum(row.get("score", ""))
        acc["opponent_score_sum"] += fnum(row.get("opponent_score", ""))
        acc["decisions_sum"] += fnum(row.get("decisions", ""))
        for metric in METRICS:
            acc[f"{metric}_sum"] += fnum(row.get(metric, ""))

    out = []
    for key, acc in buckets.items():
        games = int(acc["games"])
        if games < min_games:
            continue
        row = {
            "group_by": ",".join(group_by),
            "key": key,
            "games": games,
            "avg_score": f"{acc['score_sum'] / max(games, 1):.1f}",
            "avg_opponent_score": f"{acc['opponent_score_sum'] / max(games, 1):.1f}",
            "avg_decisions": f"{acc['decisions_sum'] / max(games, 1):.1f}",
        }
        for metric in METRICS:
            row[metric] = f"{acc[f'{metric}_sum'] / max(games, 1):.4f}"
        out.append(row)
    out.sort(key=lambda r: (-float(r["dragapult_strategy_success"]), -float(r["outcome_win"]), -int(r["games"]), r["key"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-csv", required=True)
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help=(
            "comma-separated columns. Repeatable. Special columns: score_bucket, "
            "opponent_score_bucket. Defaults to several useful Dragapult views."
        ),
    )
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    with open(args.trajectory_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty trajectory CSV")

    group_specs = args.group_by or [
        "team_name",
        "opponent_archetype",
        "score_bucket",
        "team_name,opponent_archetype",
        "team_name,opponent_archetype,score_bucket",
    ]

    all_rows = []
    for spec in group_specs:
        cols = [x.strip() for x in spec.split(",") if x.strip()]
        if not cols:
            continue
        all_rows.extend(summarize(rows, cols, args.min_games))

    fields = [
        "group_by",
        "key",
        "games",
        "avg_score",
        "avg_opponent_score",
        "avg_decisions",
        *METRICS,
    ]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {out} rows={len(all_rows)} source_games={len(rows)}")
    print("Top Dragapult strategy-success groups:")
    for row in all_rows[: args.top]:
        print(
            f"  {row['group_by']}={row['key']} games={row['games']} "
            f"wr={row['outcome_win']} strat={row['dragapult_strategy_success']} "
            f"dreepy2={row['dreepy_board_by_2']} drak4={row['drakloak_board_by_4']} "
            f"drag6={row['dragapult_board_by_6']} dmg={row['damage_counter_used']}"
        )


if __name__ == "__main__":
    main()
