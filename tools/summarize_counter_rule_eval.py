#!/usr/bin/env python3
"""Summarize generated counter_plan RR runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_wr(path: str, prefix: str, opp_prefix: str = "opp_") -> float | None:
    if not path or not Path(path).exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = str(row.get("row", ""))
            c = str(row.get("column", ""))
            if r.startswith(prefix) and c.startswith(opp_prefix):
                return float(row["row_win_rate"])
            if c.startswith(prefix) and r.startswith(opp_prefix):
                return 1.0 - float(row["row_win_rate"])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-csv", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    rows = []
    with open(args.plan_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "planned":
                row["base_wr"] = ""
                row["rule_wr"] = ""
                row["delta"] = ""
                rows.append(row)
                continue
            csv_path = row.get("csv_path", "")
            base_wr = read_wr(csv_path, "base_")
            rule_wr = read_wr(csv_path, "rule_")
            row["base_wr"] = "" if base_wr is None else f"{base_wr:.6f}"
            row["rule_wr"] = "" if rule_wr is None else f"{rule_wr:.6f}"
            row["delta"] = "" if base_wr is None or rule_wr is None else f"{rule_wr - base_wr:.6f}"
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    for name in ("base_wr", "rule_wr", "delta"):
        if name not in fieldnames:
            fieldnames.append(name)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    good = bad = valid = 0
    deltas = []
    for row in rows:
        if row.get("delta") in ("", None):
            continue
        delta = float(row["delta"])
        valid += 1
        deltas.append(delta)
        if delta > 0:
            good += 1
        elif delta < 0:
            bad += 1
    avg = sum(deltas) / len(deltas) if deltas else 0.0
    print(f"wrote {out} valid={valid} improved={good} worsened={bad} avg_delta={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
