#!/usr/bin/env python3
"""Select top rows per archetype from a policy/deck manifest."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-arch", type=int, default=1)
    p.add_argument("--archetype", action="append", default=[])
    p.add_argument("--sort-key", action="append", default=[],
                   help="optional numeric sort key, descending; repeatable")
    args = p.parse_args()

    wanted = {x.lower() for x in args.archetype}
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "archetype" not in fields:
        raise ValueError(f"{args.manifest} has no archetype column")

    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arch = str(row.get("archetype", "")).strip()
        if not arch:
            continue
        if wanted and arch.lower() not in wanted:
            continue
        by_arch[arch].append(row)

    sort_keys = [x for x in args.sort_key if x]
    out_rows: list[dict[str, str]] = []
    for arch in sorted(by_arch):
        arch_rows = by_arch[arch]
        if sort_keys:
            arch_rows = sorted(
                arch_rows,
                key=lambda r: tuple(as_float(r.get(k, "")) for k in sort_keys),
                reverse=True,
            )
        out_rows.extend(arch_rows[: max(1, args.max_per_arch)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(
        f"wrote {out} rows={len(out_rows)} archetypes={len(by_arch)} "
        f"max_per_arch={args.max_per_arch}",
        flush=True,
    )
    for row in out_rows:
        print(
            f"{row.get('archetype', '')}: {row.get('name') or row.get('team_name') or row.get('deck_sig')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
