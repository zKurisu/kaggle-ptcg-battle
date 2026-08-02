#!/usr/bin/env python3
"""Summarize an eval_round_robin.py CSV into ranking metrics."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "name",
    "games",
    "avg_all",
    "avg_no_random",
    "random_wr",
    "min_no_random",
    "worst_opponent",
    "worst_wr",
    "losses_no_random",
]


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], random_name: str) -> list[dict]:
    by: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for row in rows:
        name = row["row"]
        opp = row["column"]
        wr = float(row["row_win_rate"])
        games = int(row["games"])
        by[name].append((opp, wr, games))

    out = []
    for name, vals in by.items():
        all_rates = [wr for _, wr, _ in vals]
        nr = [(opp, wr, games) for opp, wr, games in vals if opp != random_name]
        random_wr = next((wr for opp, wr, _ in vals if opp == random_name), None)
        if nr:
            worst_opp, worst_wr, _ = min(nr, key=lambda x: x[1])
            avg_no_random = sum(wr for _, wr, _ in nr) / len(nr)
            losses_no_random = sum(1 for _, wr, _ in nr if wr < 0.5)
        else:
            worst_opp, worst_wr, avg_no_random, losses_no_random = "", 0.0, 0.0, 0
        out.append(
            {
                "name": name,
                "games": sum(g for _, _, g in vals),
                "avg_all": sum(all_rates) / max(len(all_rates), 1),
                "avg_no_random": avg_no_random,
                "random_wr": random_wr if random_wr is not None else "",
                "min_no_random": worst_wr,
                "worst_opponent": worst_opp,
                "worst_wr": worst_wr,
                "losses_no_random": losses_no_random,
            }
        )
    return sorted(out, key=lambda r: (float(r["avg_no_random"]), float(r["min_no_random"])), reverse=True)


def write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(value) -> str:
    if value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--random-name", default="random")
    p.add_argument("--out", default="", help="optional ranking CSV output")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    ranked = summarize(read_rows(args.csv_path), args.random_name)
    print("Round-robin ranking")
    print("name        avg_no_random  min_no_random  random_wr  losses  worst")
    for row in ranked[: args.top]:
        print(
            f"{row['name']:<12} {fmt(row['avg_no_random']):>13} "
            f"{fmt(row['min_no_random']):>14} {fmt(row['random_wr']):>10} "
            f"{row['losses_no_random']:>6}  {row['worst_opponent']}={fmt(row['worst_wr'])}",
            flush=True,
        )
    if args.out:
        write_csv(args.out, ranked)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
