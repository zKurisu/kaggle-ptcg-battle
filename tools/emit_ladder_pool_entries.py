#!/usr/bin/env python3
"""Emit eval_round_robin.py --entry args from a ladder pool manifest."""
from __future__ import annotations

import argparse
import csv
import shlex


def safe_entry_name(row: dict, idx: int) -> str:
    arch = row.get("archetype", "opp").lower().replace(" ", "_").replace("+", "")
    sig = row.get("deck_sig", "")[:8]
    return f"opp{idx:02d}_{arch[:18]}_{sig}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--policy", default="random", help="policy path or random for emitted opponents")
    p.add_argument("--one-per-archetype", action="store_true")
    p.add_argument("--exclude-archetype", action="append", default=[])
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    excluded = set(args.exclude_archetype)
    emitted = []
    seen_arch = set()
    for row in rows:
        arch = row.get("archetype", "")
        if arch in excluded:
            continue
        if args.one_per_archetype and arch in seen_arch:
            continue
        seen_arch.add(arch)
        emitted.append(row)
        if args.top and len(emitted) >= args.top:
            break
    parts = []
    for i, row in enumerate(emitted, 1):
        name = safe_entry_name(row, i)
        spec = f"{name}={args.policy}:{row['deck_path']}"
        parts.extend(["--entry", spec])
    print(" ".join(shlex.quote(p) for p in parts))


if __name__ == "__main__":
    main()
