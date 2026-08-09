#!/usr/bin/env python3
"""Summarize search_action_teacher.py outputs."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = [
    "tag",
    "states",
    "improved",
    "improved_rate",
    "strong_delta_ge",
    "strong",
    "strong_rate",
    "mean_delta_score",
    "median_delta_score",
    "mean_baseline_score",
    "mean_best_score",
    "mean_baseline_wr",
    "mean_best_wr",
]

MOTIF_FIELDS = [
    "tag",
    "kind",
    "motif",
    "n",
    "rate",
    "mean_delta_score",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except Exception:
        return default


def tag_for_path(path: Path) -> str:
    name = path.name
    name = re.sub(r"_teacher_best\.csv$", "", name)
    name = re.sub(r"^wave\d+_", "", name)
    return name


def action_type(desc: str) -> str:
    if not desc or desc == "STOP":
        return "STOP"
    first = desc.split("|", 1)[0].strip()
    parts = first.split(":")
    return parts[1].strip() if len(parts) >= 2 else first


def action_card(desc: str) -> str:
    if not desc or desc == "STOP":
        return "STOP"
    first = desc.split("|", 1)[0].strip()
    parts = first.split(":")
    return parts[2].strip() if len(parts) >= 3 else action_type(desc)


def summarize_file(path: Path, strong_delta: float, motif_limit: int) -> tuple[dict, list[dict]]:
    rows = read_rows(path)
    tag = tag_for_path(path)
    deltas = np.asarray([f(r, "delta_score_vs_baseline") for r in rows], dtype=np.float32)
    baseline_scores = np.asarray([f(r, "baseline_score") for r in rows], dtype=np.float32)
    best_scores = np.asarray([f(r, "best_score") for r in rows], dtype=np.float32)
    baseline_wrs = np.asarray([f(r, "baseline_win_rate") for r in rows], dtype=np.float32)
    best_wrs = np.asarray([f(r, "best_win_rate") for r in rows], dtype=np.float32)
    improved = int((deltas > 0).sum()) if len(deltas) else 0
    strong = int((deltas >= strong_delta).sum()) if len(deltas) else 0
    summary = {
        "tag": tag,
        "states": len(rows),
        "improved": improved,
        "improved_rate": improved / max(len(rows), 1),
        "strong_delta_ge": strong_delta,
        "strong": strong,
        "strong_rate": strong / max(len(rows), 1),
        "mean_delta_score": float(deltas.mean()) if len(deltas) else 0.0,
        "median_delta_score": float(np.median(deltas)) if len(deltas) else 0.0,
        "mean_baseline_score": float(baseline_scores.mean()) if len(baseline_scores) else 0.0,
        "mean_best_score": float(best_scores.mean()) if len(best_scores) else 0.0,
        "mean_baseline_wr": float(baseline_wrs.mean()) if len(baseline_wrs) else 0.0,
        "mean_best_wr": float(best_wrs.mean()) if len(best_wrs) else 0.0,
    }

    motif_rows: list[dict] = []
    strong_rows = [r for r in rows if f(r, "delta_score_vs_baseline") >= strong_delta]
    tables: dict[str, Counter] = {
        "best_desc": Counter(r.get("best_desc", "") for r in strong_rows),
        "best_type": Counter(action_type(r.get("best_desc", "")) for r in strong_rows),
        "best_card": Counter(action_card(r.get("best_desc", "")) for r in strong_rows),
        "baseline_type": Counter(action_type(r.get("baseline_desc", "")) for r in strong_rows),
        "baseline_card": Counter(action_card(r.get("baseline_desc", "")) for r in strong_rows),
    }
    delta_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in strong_rows:
        delta = f(r, "delta_score_vs_baseline")
        values = {
            "best_desc": r.get("best_desc", ""),
            "best_type": action_type(r.get("best_desc", "")),
            "best_card": action_card(r.get("best_desc", "")),
            "baseline_type": action_type(r.get("baseline_desc", "")),
            "baseline_card": action_card(r.get("baseline_desc", "")),
        }
        for kind, value in values.items():
            delta_by[(kind, value)].append(delta)
    for kind, counts in tables.items():
        total = max(sum(counts.values()), 1)
        for motif, n in counts.most_common(motif_limit):
            motif_rows.append({
                "tag": tag,
                "kind": kind,
                "motif": motif,
                "n": n,
                "rate": n / total,
                "mean_delta_score": float(np.mean(delta_by[(kind, motif)])) if delta_by[(kind, motif)] else 0.0,
            })
    return summary, motif_rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--strong-delta", type=float, default=0.20)
    parser.add_argument("--motif-limit", type=int, default=8)
    parser.add_argument("--out-summary-csv", default="")
    parser.add_argument("--out-motifs-csv", default="")
    args = parser.parse_args()

    summaries = []
    motifs = []
    for raw in args.paths:
        summary, motif_rows = summarize_file(Path(raw), args.strong_delta, args.motif_limit)
        summaries.append(summary)
        motifs.extend(motif_rows)

    if args.out_summary_csv:
        write_csv(Path(args.out_summary_csv), SUMMARY_FIELDS, summaries)
    if args.out_motifs_csv:
        write_csv(Path(args.out_motifs_csv), MOTIF_FIELDS, motifs)

    for row in summaries:
        print(
            f"{row['tag']}: states={row['states']} improved={row['improved']} "
            f"({row['improved_rate']:.1%}) strong={row['strong']} "
            f"mean_delta_score={row['mean_delta_score']:.3f} "
            f"median={row['median_delta_score']:.3f} "
            f"wr={row['mean_baseline_wr']:.3f}->{row['mean_best_wr']:.3f}",
            flush=True,
        )
    print("\nTop motifs on strong-delta states:", flush=True)
    for row in motifs:
        if row["kind"] not in {"best_type", "best_card", "baseline_type"}:
            continue
        print(
            f"{row['tag']} {row['kind']} {row['motif']}: "
            f"n={row['n']} rate={row['rate']:.1%} delta={row['mean_delta_score']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
