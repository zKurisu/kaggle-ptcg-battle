#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PAIR_RE = re.compile(r"^\s*(?P<a>.+?)\s+vs\s+(?P<b>.+?):\s+(?P<done>\d+)/(?P<total>\d+)\s+(?P<w>\d+)-(?P<l>\d+)-(?P<d>\d+)\s+wr=(?P<wr>[0-9.]+)")


def archetype(name: str) -> str:
    text = str(name).strip()
    if "_" not in text:
        return text
    parts = text.split("_")
    if len(parts) >= 2 and len(parts[-1]) in (8, 12):
        parts = parts[:-1]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def parse_log(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = PAIR_RE.match(line)
        if not m:
            continue
        a = m.group("a").strip()
        b = m.group("b").strip()
        latest[(a, b)] = {
            "a": a,
            "b": b,
            "done": int(m.group("done")),
            "total": int(m.group("total")),
            "wins": int(m.group("w")),
            "losses": int(m.group("l")),
            "draws": int(m.group("d")),
            "wr": float(m.group("wr")),
        }
    return latest


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    except OSError as exc:
        print(f"warning: could not write {path}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--out-prefix", default="")
    p.add_argument("--min-done", type=int, default=1)
    args = p.parse_args()

    pairs = parse_log(Path(args.log))
    rows = [r for r in pairs.values() if int(r["done"]) >= int(args.min_done)]
    rows.sort(key=lambda r: (str(r["a"]), str(r["b"])))

    by_row: dict[str, list[float]] = defaultdict(list)
    worst: dict[str, tuple[str, float]] = {}
    arch_rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        a = str(r["a"])
        b = str(r["b"])
        wr = float(r["wr"])
        by_row[a].append(wr)
        if a not in worst or wr < worst[a][1]:
            worst[a] = (b, wr)
        aa, bb = archetype(a), archetype(b)
        arch_rows[(aa, bb)].append(wr)
        arch_rows[(bb, aa)].append(1.0 - wr)

    summary = []
    for name, vals in sorted(by_row.items()):
        summary.append({
            "name": name,
            "completed_pairs": len(vals),
            "mean_wr": round(sum(vals) / max(len(vals), 1), 4),
            "worst_opponent": worst.get(name, ("", 0.0))[0],
            "worst_wr": round(worst.get(name, ("", 0.0))[1], 4),
        })

    matrix_rows = []
    for (aa, bb), vals in sorted(arch_rows.items()):
        matrix_rows.append({
            "row_arch": aa,
            "col_arch": bb,
            "pairs": len(vals),
            "mean_wr": round(sum(vals) / max(len(vals), 1), 4),
        })

    print(f"parsed_pairs={len(rows)} from={args.log}")
    for row in sorted(summary, key=lambda r: float(r["mean_wr"])):
        print(
            f"{row['name']}: n={row['completed_pairs']} mean={row['mean_wr']} "
            f"worst={row['worst_opponent']}:{row['worst_wr']}"
        )

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        write_csv(prefix.with_suffix(".pairs.csv"), rows, ["a", "b", "done", "total", "wins", "losses", "draws", "wr"])
        write_csv(prefix.with_suffix(".summary.csv"), summary, ["name", "completed_pairs", "mean_wr", "worst_opponent", "worst_wr"])
        write_csv(prefix.with_suffix(".arch_matrix.csv"), matrix_rows, ["row_arch", "col_arch", "pairs", "mean_wr"])


if __name__ == "__main__":
    main()
