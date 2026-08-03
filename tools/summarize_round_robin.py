#!/usr/bin/env python3
"""Summarize an eval_round_robin.py CSV into ranking metrics."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "name",
    "games",
    "avg_all",
    "avg_no_random",
    "random_wr",
    "weighted_avg_no_random",
    "weighted_games",
    "min_no_random",
    "worst_opponent",
    "worst_wr",
    "losses_no_random",
]


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_name(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text


def load_manifest_weights(path: str) -> tuple[dict[str, float], dict[str, str]]:
    if not path:
        return {}, {}
    weights: dict[str, float] = {}
    labels: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sig = (row.get("deck_sig") or "").strip()
            if not sig:
                continue
            try:
                weight = float(row.get("weight") or 0.0)
            except ValueError:
                weight = 0.0
            arch = row.get("archetype", "")
            team = row.get("team_name", "")
            for key in {sig, sig[:8]}:
                weights[key] = max(weight, 1e-9)
                labels[key] = f"{arch}:{team}" if team else arch
    return weights, labels


def infer_opponent_weight(name: str, weights: dict[str, float]) -> float:
    if not weights:
        return 1.0
    safe = _safe_name(name)
    for key, weight in weights.items():
        if key and key in safe:
            return weight
    return 1.0


def summarize(rows: list[dict], random_name: str, manifest_weights: dict[str, float] | None = None) -> list[dict]:
    manifest_weights = manifest_weights or {}
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
            wsum = sum(infer_opponent_weight(opp, manifest_weights) for opp, _, _ in nr)
            weighted_avg = sum(wr * infer_opponent_weight(opp, manifest_weights) for opp, wr, _ in nr) / max(wsum, 1e-9)
            losses_no_random = sum(1 for _, wr, _ in nr if wr < 0.5)
        else:
            worst_opp, worst_wr, avg_no_random, weighted_avg, wsum, losses_no_random = "", 0.0, 0.0, 0.0, 0.0, 0
        out.append(
            {
                "name": name,
                "games": sum(g for _, _, g in vals),
                "avg_all": sum(all_rates) / max(len(all_rates), 1),
                "avg_no_random": avg_no_random,
                "random_wr": random_wr if random_wr is not None else "",
                "weighted_avg_no_random": weighted_avg,
                "weighted_games": wsum,
                "min_no_random": worst_wr,
                "worst_opponent": worst_opp,
                "worst_wr": worst_wr,
                "losses_no_random": losses_no_random,
            }
        )
    sort_key = "weighted_avg_no_random" if manifest_weights else "avg_no_random"
    return sorted(out, key=lambda r: (float(r[sort_key]), float(r["min_no_random"])), reverse=True)


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
    p.add_argument("--manifest", default="",
                   help="optional ladder pool manifest; use deck weights for weighted_avg_no_random")
    p.add_argument("--out", default="", help="optional ranking CSV output")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    weights, _ = load_manifest_weights(args.manifest)
    ranked = summarize(read_rows(args.csv_path), args.random_name, weights)
    print("Round-robin ranking")
    if weights:
        print("name        weighted_avg  avg_no_random  min_no_random  random_wr  losses  worst")
    else:
        print("name        avg_no_random  min_no_random  random_wr  losses  worst")
    for row in ranked[: args.top]:
        if weights:
            print(
                f"{row['name']:<12} {fmt(row['weighted_avg_no_random']):>12} "
                f"{fmt(row['avg_no_random']):>13} {fmt(row['min_no_random']):>14} "
                f"{fmt(row['random_wr']):>10} {row['losses_no_random']:>6}  "
                f"{row['worst_opponent']}={fmt(row['worst_wr'])}",
                flush=True,
            )
        else:
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
