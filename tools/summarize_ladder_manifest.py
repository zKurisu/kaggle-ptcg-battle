#!/usr/bin/env python3
"""Summarize a ladder pool manifest by score band and archetype."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _float(text: str) -> float:
    try:
        return float(text or 0.0)
    except ValueError:
        return 0.0


def _int(text: str) -> int:
    try:
        return int(text or 0)
    except ValueError:
        return 0


def _add(dst: dict, row: dict) -> None:
    dst["decks"] += 1
    dst["games"] += _int(row.get("games", ""))
    dst["wins"] += _int(row.get("wins", ""))
    dst["losses"] += _int(row.get("losses", ""))
    dst["weight"] += _float(row.get("weight", ""))


def _empty() -> dict:
    return {"decks": 0, "games": 0, "wins": 0, "losses": 0, "weight": 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", help="pool_manifest.csv from build_ladder_pool.py")
    p.add_argument("--out", default="", help="band x archetype CSV")
    p.add_argument("--top", type=int, default=8, help="rows printed per score band; 0 prints all")
    args = p.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out) if args.out else manifest.with_name("band_archetype_stats.csv")
    band_arch = defaultdict(_empty)
    band_totals = defaultdict(_empty)
    arch_totals = defaultdict(_empty)

    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            band = row.get("score_band") or "unknown"
            arch = row.get("archetype") or "Other"
            _add(band_arch[(band, arch)], row)
            _add(band_totals[band], row)
            _add(arch_totals[arch], row)

    bands = sorted(band_totals, key=lambda b: band_totals[b]["weight"], reverse=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "score_band",
                "archetype",
                "decks",
                "games",
                "wins",
                "losses",
                "win_rate",
                "weight",
                "band_game_share",
                "band_weight_share",
            ]
        )
        for band in bands:
            rows = [(arch, stats) for (b, arch), stats in band_arch.items() if b == band]
            rows.sort(key=lambda item: item[1]["weight"], reverse=True)
            total = band_totals[band]
            for arch, stats in rows:
                denom = max(stats["wins"] + stats["losses"], 1)
                w.writerow(
                    [
                        band,
                        arch,
                        stats["decks"],
                        stats["games"],
                        stats["wins"],
                        stats["losses"],
                        f"{stats['wins'] / denom:.4f}",
                        f"{stats['weight']:.4f}",
                        f"{stats['games'] / max(total['games'], 1):.4f}",
                        f"{stats['weight'] / max(total['weight'], 1e-9):.4f}",
                    ]
                )

    print(f"Wrote {out}")
    print("\nBand totals:")
    for band in bands:
        stats = band_totals[band]
        print(
            f"{band:9s} decks={stats['decks']:3d} "
            f"games={stats['games']:6d} weight={stats['weight']:.1f}",
            flush=True,
        )

    print("\nOverall archetypes:")
    for arch, stats in sorted(arch_totals.items(), key=lambda item: item[1]["weight"], reverse=True):
        print(
            f"{arch:22s} decks={stats['decks']:3d} "
            f"games={stats['games']:6d} weight={stats['weight']:.1f}",
            flush=True,
        )

    print("\nTop archetypes by band:")
    for band in bands:
        print(f"== {band}")
        rows = [(arch, stats) for (b, arch), stats in band_arch.items() if b == band]
        rows.sort(key=lambda item: item[1]["weight"], reverse=True)
        if args.top:
            rows = rows[: args.top]
        total = band_totals[band]
        for arch, stats in rows:
            print(
                f"  {arch:22s} decks={stats['decks']:3d} "
                f"games={stats['games']:6d} weight={stats['weight']:.1f} "
                f"band_w={stats['weight'] / max(total['weight'], 1e-9):.1%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
