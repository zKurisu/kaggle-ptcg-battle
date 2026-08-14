#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def clean(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--score-bands", nargs="+", default=["900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--top-per-archetype", type=int, default=2)
    p.add_argument("--min-rows", type=int, default=5000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    arch_sig_rows: dict[tuple[str, str], int] = Counter()
    arch_sig_score: dict[tuple[str, str], float] = defaultdict(float)
    arch_sig_team: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for arch_dir in sorted(Path(args.corpus).iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name.replace("_", " ")
        for band in args.score_bands:
            for path in glob.glob(str(arch_dir / band.replace(" ", "_") / "*.npz")):
                with np.load(path, allow_pickle=True) as z:
                    for sig, score, team in zip(z["deck_sig"], z["score"], z["team_name"]):
                        key = (arch, str(sig))
                        arch_sig_rows[key] += 1
                        arch_sig_score[key] = max(float(arch_sig_score[key]), float(score))
                        if str(team):
                            arch_sig_team[key][str(team)] += 1

    rows: list[dict[str, str | int | float]] = []
    by_arch: dict[str, list[tuple[str, int, float, str]]] = defaultdict(list)
    for (arch, sig), n in arch_sig_rows.items():
        if n < args.min_rows:
            continue
        team = arch_sig_team[(arch, sig)].most_common(1)[0][0] if arch_sig_team[(arch, sig)] else ""
        by_arch[arch].append((sig, n, arch_sig_score[(arch, sig)], team))
    for arch, vals in sorted(by_arch.items()):
        vals.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for rank, (sig, n, score, team) in enumerate(vals[: args.top_per_archetype], 1):
            name = f"v14seq_{clean(arch)}_{sig[:8]}_{rank}"
            rows.append({
                "name": name,
                "archetype": arch,
                "deck_sig": sig,
                "team_name": "",
                "score_bands": " ".join(args.score_bands),
                "rows": n,
                "max_score": f"{score:.1f}",
                "top_team": team,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "archetype", "deck_sig", "team_name", "score_bands", "rows", "max_score", "top_team"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")
    for row in rows[:30]:
        print(f"{row['name']} arch={row['archetype']} sig={row['deck_sig']} rows={row['rows']} score={row['max_score']} team={row['top_team']}")


if __name__ == "__main__":
    main()
