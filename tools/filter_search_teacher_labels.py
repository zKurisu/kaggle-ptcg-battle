#!/usr/bin/env python3
"""Filter root-search teacher labels into high-confidence rule/training seeds.

The online search-guided agent was too greedy: good one-step root actions did
not compose into whole-game wins. This tool keeps the useful part of that work:
clear, repeated local corrections that can be reviewed as rule motifs or used
as scratch distillation labels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


OUT_FIELDS = [
    "source",
    "state_id",
    "candidate",
    "opponent",
    "outcome",
    "turn",
    "context",
    "option_count",
    "baseline_action",
    "baseline_desc",
    "baseline_type",
    "baseline_card",
    "best_action",
    "best_desc",
    "best_type",
    "best_card",
    "baseline_win_rate",
    "best_win_rate",
    "delta_vs_baseline",
    "baseline_score",
    "best_score",
    "delta_score_vs_baseline",
    "evaluated_actions",
    "rollouts_per_action",
    "motif_key",
    "motif_count",
    "trust_reason",
]


def read_rows(patterns: list[str]) -> list[dict[str, Any]]:
    paths: list[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        paths.extend(matched or [pat])
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source"] = path
                rows.append(row)
    return rows


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except Exception:
        return default


def parse_desc(desc: str) -> tuple[str, str]:
    desc = str(desc or "").strip()
    if not desc or desc == "STOP":
        return "STOP", ""
    first = desc.split("|", 1)[0].strip()
    parts = first.split(":", 2)
    if len(parts) >= 2:
        typ = parts[1].strip()
        card = parts[2].split("->", 1)[0].strip() if len(parts) >= 3 else ""
        return typ, card
    return "", ""


def motif_key(row: dict[str, Any]) -> str:
    best_type, best_card = parse_desc(row.get("best_desc", ""))
    return "|".join([
        str(row.get("candidate", "")),
        str(row.get("opponent", "")),
        best_type,
        best_card,
    ])


def keep_row(row: dict[str, Any], args: argparse.Namespace, motif_counts: Counter[str]) -> tuple[bool, str]:
    best_type, _ = parse_desc(row.get("best_desc", ""))
    if best_type.upper() in args.exclude_action_type:
        return False, f"excluded_type:{best_type}"
    if fnum(row, "delta_score_vs_baseline") < args.min_delta_score:
        return False, "low_delta_score"
    if fnum(row, "best_score") < args.min_best_score:
        return False, "low_best_score"
    if fnum(row, "best_win_rate") < args.min_best_win_rate:
        return False, "low_best_win_rate"
    if fnum(row, "delta_vs_baseline") < args.min_winrate_gain:
        return False, "low_winrate_gain"
    if int(motif_counts[motif_key(row)]) < args.min_motif_count:
        return False, "rare_motif"
    return True, (
        f"delta_score>={args.min_delta_score:g};"
        f"best_score>={args.min_best_score:g};"
        f"motif_count>={args.min_motif_count}"
    )


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-csv", action="append", required=True,
                        help="search_action_teacher *_teacher_best.csv path/glob; repeatable")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--min-delta-score", type=float, default=0.20)
    parser.add_argument("--min-best-score", type=float, default=0.05)
    parser.add_argument("--min-best-win-rate", type=float, default=0.0)
    parser.add_argument("--min-winrate-gain", type=float, default=0.0)
    parser.add_argument("--min-motif-count", type=int, default=1)
    parser.add_argument("--exclude-action-type", action="append",
                        default=["STOP", "END", "YES", "NO", "NUMBER", "CARD"],
                        help="best action type to exclude; repeatable")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    args.exclude_action_type = {str(x).upper() for x in args.exclude_action_type}

    rows = read_rows(args.best_csv)
    motif_counts = Counter(motif_key(r) for r in rows)
    kept: list[dict[str, Any]] = []
    rejects = Counter()
    for row in rows:
        ok, reason = keep_row(row, args, motif_counts)
        if not ok:
            rejects[reason] += 1
            continue
        best_type, best_card = parse_desc(row.get("best_desc", ""))
        base_type, base_card = parse_desc(row.get("baseline_desc", ""))
        key = motif_key(row)
        out = {
            "source": row.get("_source", ""),
            "baseline_type": base_type,
            "baseline_card": base_card,
            "best_type": best_type,
            "best_card": best_card,
            "motif_key": key,
            "motif_count": motif_counts[key],
            "trust_reason": reason,
        }
        out.update(row)
        kept.append(out)

    kept.sort(key=lambda r: (
        float(r.get("delta_score_vs_baseline") or 0.0),
        int(r.get("motif_count") or 0),
    ), reverse=True)
    write_csv(args.out_csv, kept)
    write_jsonl(args.out_jsonl, kept)

    print(f"Read {len(rows)} labels; kept {len(kept)} -> {args.out_csv}")
    if args.out_jsonl:
        print(f"Wrote {args.out_jsonl}")
    if rejects:
        print("Rejects:")
        for key, n in rejects.most_common():
            print(f"  {key}: {n}")
    print("Top motifs:")
    for key, n in Counter(r["motif_key"] for r in kept).most_common(args.top):
        print(f"  {n} {key}")


if __name__ == "__main__":
    main()
