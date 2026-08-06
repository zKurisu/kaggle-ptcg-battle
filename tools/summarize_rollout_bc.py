#!/usr/bin/env python3
"""Summarize rollout-search logs and generated BC corpora."""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}


def _parse_counter_cell(value: str) -> Counter:
    if not value:
        return Counter()
    try:
        raw = ast.literal_eval(value)
    except Exception:
        return Counter()
    if isinstance(raw, dict):
        return Counter({str(k): int(v) for k, v in raw.items()})
    return Counter()


def summarize_worker_csv(path: str) -> dict:
    totals = Counter()
    actor_counts = Counter()
    actor_wins = Counter()
    opponent_counts = Counter()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "worker_id" not in (reader.fieldnames or []):
            raise ValueError(f"not a rollout worker summary CSV: {path}")
        for row in reader:
            for key in (
                "games",
                "wins",
                "losses",
                "draws",
                "errors",
                "timeouts",
                "decisions_seen",
                "decisions_written",
                "rows",
            ):
                totals[key] += int(float(row.get(key) or 0))
            totals["elapsed"] += float(row.get("elapsed") or 0.0)
            actor_counts.update(_parse_counter_cell(row.get("actor_counts", "")))
            actor_wins.update(_parse_counter_cell(row.get("actor_wins", "")))
            opponent_counts.update(_parse_counter_cell(row.get("opponent_counts", "")))
    games = max(totals["games"], 1)
    wins = totals["wins"]
    return {
        "name": Path(path).stem,
        "games": totals["games"],
        "wins": wins,
        "win_rate": wins / games,
        "losses": totals["losses"],
        "draws": totals["draws"],
        "errors": totals["errors"],
        "timeouts": totals["timeouts"],
        "decisions_seen": totals["decisions_seen"],
        "rows": totals["rows"],
        "rows_per_win": totals["rows"] / max(wins, 1),
        "elapsed_sum": totals["elapsed"],
        "actor_counts": dict(actor_counts),
        "actor_wins": dict(actor_wins),
        "opponent_counts": dict(opponent_counts),
    }


def _tag_from_npz(path: str) -> str:
    stem = Path(path).stem
    if "_w" in stem and stem.rsplit("_w", 1)[1].isdigit():
        return stem.rsplit("_w", 1)[0]
    return stem


def summarize_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        n = len(z["board"])
        actor = Counter(str(x) for x in z["actor_mode"]) if "actor_mode" in z.files else Counter()
        opp = Counter(str(x) for x in z["opponent_name"]) if "opponent_name" in z.files else Counter()
        status = Counter(str(x) for x in z["final_status"]) if "final_status" in z.files else Counter()
        first_type = Counter()
        contexts = Counter()
        episodes = set()
        if "episode_id" in z.files:
            episodes = {str(x) for x in z["episode_id"]}
        for i in range(n):
            action = np.asarray(z["action"][i], dtype=np.int64)
            ot = np.asarray(z["ot"][i], dtype=np.int64)
            if len(action) > 0 and 0 <= int(action[0]) < len(ot):
                first_type[TYPE_NAMES.get(int(ot[int(action[0])]), str(int(ot[int(action[0])])))] += 1
            feats = np.asarray(z["feats"][i], dtype=np.float32)
            if len(feats) > 17:
                contexts[int(round(float(feats[17]) * 64.0))] += 1
    return {
        "tag": _tag_from_npz(path),
        "path": path,
        "rows": n,
        "episodes": len(episodes),
        "actor": actor,
        "opponent": opp,
        "status": status,
        "first_type": first_type,
        "context": contexts,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "games",
        "wins",
        "win_rate",
        "losses",
        "draws",
        "errors",
        "timeouts",
        "decisions_seen",
        "rows",
        "rows_per_win",
        "elapsed_sum",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-glob", action="append", default=[],
                        help="glob for per-worker summary CSVs from generate_rollout_bc.py")
    parser.add_argument("--corpus", action="append", default=[],
                        help="generated corpus root to scan recursively")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    summary_paths: list[str] = []
    for pat in args.summary_glob:
        summary_paths.extend(sorted(glob.glob(pat)))
    summary_rows = []
    for path in summary_paths:
        try:
            summary_rows.append(summarize_worker_csv(path))
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
    summary_rows.sort(key=lambda r: (float(r["win_rate"]), int(r["rows"])), reverse=True)
    if summary_rows:
        print("Rollout summaries:")
        for row in summary_rows:
            print(
                f"  {row['name']}: wins={row['wins']}/{row['games']} "
                f"wr={row['win_rate']:.3f} rows={row['rows']} "
                f"errors={row['errors']} timeouts={row['timeouts']}"
            )
        write_csv(args.out_csv, summary_rows)
        if args.out_csv:
            print(f"Wrote {args.out_csv}")

    npz_paths: list[str] = []
    for root in args.corpus:
        npz_paths.extend(sorted(glob.glob(os.path.join(root, "*", "*", "*.npz"))))
    if not npz_paths:
        return

    by_tag: dict[str, dict] = defaultdict(lambda: {
        "rows": 0,
        "episodes": 0,
        "actor": Counter(),
        "opponent": Counter(),
        "status": Counter(),
        "first_type": Counter(),
        "context": Counter(),
    })
    for path in npz_paths:
        row = summarize_npz(path)
        agg = by_tag[row["tag"]]
        agg["rows"] += int(row["rows"])
        agg["episodes"] += int(row["episodes"])
        for key in ("actor", "opponent", "status", "first_type", "context"):
            agg[key].update(row[key])

    print("\nGenerated corpus:")
    for tag, row in sorted(by_tag.items(), key=lambda kv: int(kv[1]["rows"]), reverse=True):
        print(f"  {tag}: rows={row['rows']} episodes={row['episodes']} status={dict(row['status'])}")
        for label, key in (("actor", "actor"), ("opponent", "opponent"), ("first_type", "first_type")):
            top_items = row[key].most_common(args.top)
            if top_items:
                text = ", ".join(f"{k}:{v}" for k, v in top_items)
                print(f"    {label}: {text}")


if __name__ == "__main__":
    main()
