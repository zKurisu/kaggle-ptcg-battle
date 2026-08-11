#!/usr/bin/env python3
"""Summarize eval_baseline_delta.py output by opponent archetype."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def fnum(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def load_manifest(path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("name") or row.get("shadow_name") or row.get("deck_sig") or ""
            if not name and row.get("eval_entry"):
                name = row["eval_entry"].split("=", 1)[0]
            if name:
                out[name] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", required=True, help="eval_baseline_delta.py output CSV")
    parser.add_argument("--opponent-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args()

    manifest = load_manifest(args.opponent_manifest)
    acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    meta: dict[str, dict[str, str]] = {}
    archetypes = set()
    with open(args.delta, newline="") as f:
        for row in csv.DictReader(f):
            opp = row["opponent"]
            opp_arch = manifest.get(opp, {}).get("archetype", opp)
            archetypes.add(opp_arch)
            games = fnum(row["games"])
            cand = row["candidate"]
            meta.setdefault(cand, {"name": cand, "kind": "candidate"})
            acc[(cand, opp_arch)][0] += fnum(row["candidate_wins"])
            acc[(cand, opp_arch)][1] += games
            if args.include_baseline:
                base = row["baseline"]
                meta.setdefault(base, {"name": base, "kind": "baseline"})
                acc[(base, opp_arch)][0] += fnum(row["baseline_wins"])
                acc[(base, opp_arch)][1] += games

    arches = sorted(archetypes)
    names = list(meta)
    fieldnames = ["name", "kind", "macro_avg", "weighted_avg", "worst_archetype", "worst_wr"] + arches
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in names:
            rates = []
            row = {"name": name, "kind": meta[name]["kind"]}
            for arch in arches:
                wins, games = acc.get((name, arch), [0.0, 0.0])
                if games > 0:
                    rate = wins / games
                    rates.append((arch, rate, games))
                    row[arch] = f"{rate:.{args.digits}f}"
                else:
                    row[arch] = ""
            if rates:
                row["macro_avg"] = f"{sum(r for _, r, _ in rates) / len(rates):.{args.digits}f}"
                row["weighted_avg"] = f"{sum(r * g for _, r, g in rates) / sum(g for _, _, g in rates):.{args.digits}f}"
                worst = min(rates, key=lambda x: x[1])
                row["worst_archetype"] = worst[0]
                row["worst_wr"] = f"{worst[1]:.{args.digits}f}"
            writer.writerow(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
