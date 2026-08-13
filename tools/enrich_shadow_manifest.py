#!/usr/bin/env python3
"""Add ladder-pool score context to a shadow manifest.

build_shadow_pool.py records the source team trajectory score from the BC
corpus. For climb-aware validation we also want the current ladder-pool score
for the same deck signature, because strong submissions may climb from low
bands with a deck whose current best public score is much higher.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


POOL_FIELDS = [
    "pool_deck_score",
    "pool_score_label",
    "pool_score_tier",
    "pool_team_name",
    "pool_games",
    "pool_wins",
    "pool_losses",
    "pool_weight",
    "pool_deck_path",
    "climb_score",
    "climb_score_label",
    "climb_score_tier",
]


def fnum(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except Exception:
        return default


def inum(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except Exception:
        return default


def score_label(score: float) -> str:
    if score <= 0:
        return "sunk"
    return f"s{int(round(score)):04d}"


def score_tier(score: float) -> str:
    if score >= 1200:
        return "1200+"
    if score >= 1100:
        return "1100-1199"
    if score >= 1000:
        return "1000-1099"
    if score >= 900:
        return "900-999"
    if score >= 800:
        return "800-899"
    if score >= 700:
        return "700-799"
    if score >= 600:
        return "600-699"
    return "<600"


def load_pool(path: str) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sig = str(row.get("deck_sig") or "").strip()
            if not sig:
                continue
            current = best.get(sig)
            key = (
                fnum(row.get("score")),
                fnum(row.get("weight")),
                inum(row.get("games")),
            )
            cur_key = (
                fnum(current.get("score") if current else None),
                fnum(current.get("weight") if current else None),
                inum(current.get("games") if current else None),
            )
            if current is None or key > cur_key:
                best[sig] = row
    return best


def enrich_row(row: dict[str, str], pool_by_sig: dict[str, dict[str, str]]) -> dict[str, str]:
    out = dict(row)
    sig = str(row.get("deck_sig") or "").strip()
    pool = pool_by_sig.get(sig, {})
    pool_score = fnum(pool.get("score"))
    source_score = fnum(row.get("source_score"), fnum(row.get("max_score")))
    climb_score = max(source_score, pool_score)

    out.update({
        "pool_deck_score": f"{pool_score:.1f}" if pool_score else "",
        "pool_score_label": score_label(pool_score) if pool_score else "",
        "pool_score_tier": score_tier(pool_score) if pool_score else "",
        "pool_team_name": pool.get("team_name", ""),
        "pool_games": pool.get("games", ""),
        "pool_wins": pool.get("wins", ""),
        "pool_losses": pool.get("losses", ""),
        "pool_weight": pool.get("weight", ""),
        "pool_deck_path": pool.get("deck_path", ""),
        "climb_score": f"{climb_score:.1f}" if climb_score else "",
        "climb_score_label": score_label(climb_score) if climb_score else "",
        "climb_score_tier": score_tier(climb_score) if climb_score else "",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="shadow manifest CSV")
    parser.add_argument("--pool-manifest", required=True, help="ladder pool_manifest.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pool_by_sig = load_pool(args.pool_manifest)
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows = [enrich_row(row, pool_by_sig) for row in reader]
        base_fields = list(reader.fieldnames or [])
    fieldnames = base_fields + [field for field in POOL_FIELDS if field not in base_fields]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(1 for row in rows if row.get("pool_deck_score"))
    print(f"Wrote {out}: rows={len(rows)} pool_matched={matched}/{len(rows)}")
    for row in rows[:12]:
        print(
            f"  {row.get('shadow_name','')} source={row.get('source_score','')} "
            f"pool={row.get('pool_deck_score','')} climb={row.get('climb_score','')} "
            f"{row.get('climb_score_tier','')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
