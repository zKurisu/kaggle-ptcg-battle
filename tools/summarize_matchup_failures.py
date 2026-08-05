#!/usr/bin/env python3
"""Rank failure signals from one or more trace_matchup_decisions summaries."""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path


METRICS = [
    "miss_attack_rate",
    "miss_ability_rate",
    "miss_attach_rate",
    "miss_evolve_rate",
    "miss_play_rate",
    "miss_retreat_rate",
    "early_end_rate",
    "short_optional_multi_rate",
]

OUT_FIELDS = [
    "source",
    "context",
    "loss_decisions",
    "win_decisions",
    "loss_games",
    "win_games",
    "priority",
    "top_signal",
    "loss_rate",
    "win_rate",
]
for metric in METRICS:
    OUT_FIELDS.extend([f"loss_{metric}", f"win_{metric}", f"delta_{metric}"])


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def inum(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def source_name(path: str, strip_suffix: bool) -> str:
    base = os.path.basename(path)
    if strip_suffix and base.endswith(".summary.csv"):
        base = base[: -len(".summary.csv")]
    return base


def compare_summary(path: str, args: argparse.Namespace) -> list[dict]:
    rows = read_rows(path)
    by_key: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("table") != "outcome_context":
            continue
        key = str(row.get("key") or "")
        if ":" not in key:
            continue
        outcome, context = key.split(":", 1)
        if outcome not in ("win", "loss"):
            continue
        by_key.setdefault(context, {})[outcome] = row

    out = []
    src = source_name(path, args.strip_suffix)
    for context, pair in by_key.items():
        if "loss" not in pair or "win" not in pair:
            continue
        loss = pair["loss"]
        win = pair["win"]
        loss_decisions = inum(loss, "decisions")
        win_decisions = inum(win, "decisions")
        if loss_decisions < args.min_loss_decisions:
            continue
        row = {
            "source": src,
            "context": context,
            "loss_decisions": loss_decisions,
            "win_decisions": win_decisions,
            "loss_games": inum(loss, "losses"),
            "win_games": inum(win, "wins"),
        }
        top_signal = ""
        top_delta = -999.0
        priority = 0.0
        for metric in METRICS:
            loss_v = fnum(loss, metric)
            win_v = fnum(win, metric)
            delta = loss_v - win_v
            row[f"loss_{metric}"] = loss_v
            row[f"win_{metric}"] = win_v
            row[f"delta_{metric}"] = delta
            if delta > top_delta:
                top_delta = delta
                top_signal = metric
            if delta > 0:
                priority += delta
        row["priority"] = priority * (loss_decisions ** 0.5)
        row["top_signal"] = top_signal
        row["loss_rate"] = loss_decisions / max(inum(loss, "losses"), 1)
        row["win_rate"] = win_decisions / max(inum(win, "wins"), 1)
        out.append(row)
    out.sort(key=lambda r: (-float(r["priority"]), r["source"], r["context"]))
    return out


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (-float(r["priority"]), r["source"], r["context"]))


def write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict], limit: int) -> None:
    for row in rows[:limit]:
        print(
            f"{row['priority']:7.3f} {row['source']} {row['context']} "
            f"loss_decisions={row['loss_decisions']} top={row['top_signal']} "
            f"delta={float(row.get('delta_' + row['top_signal'], 0.0)):+.3f}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", nargs="+", help="trace .summary.csv files or glob patterns")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--min-loss-decisions", type=int, default=10)
    p.add_argument("--min-priority", type=float, default=0.0)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--strip-suffix", action="store_true", default=True)
    args = p.parse_args()

    paths: list[str] = []
    for spec in args.summary:
        matches = sorted(glob.glob(spec))
        paths.extend(matches or [spec])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no summary files found")

    rows: list[dict] = []
    for path in paths:
        rows.extend(compare_summary(path, args))
    rows = sort_rows([row for row in rows if float(row["priority"]) >= args.min_priority])
    write_csv(args.out_csv, rows)
    print(f"Wrote {args.out_csv} rows={len(rows)} from summaries={len(paths)}", flush=True)
    print_top(rows, args.top)


if __name__ == "__main__":
    main()
