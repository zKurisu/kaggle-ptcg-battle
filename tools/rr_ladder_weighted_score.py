#!/usr/bin/env python3
"""Score RR candidates against a score-band archetype distribution."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_CLIMB_MULTIPLIER = {
    "600-699": 8.0,
    "700-799": 6.0,
    "800-899": 4.0,
    "900-999": 2.5,
    "1000-1099": 1.5,
    "1100-1199": 1.0,
    "1200+": 0.8,
}


def fnum(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def load_matrix(path: str) -> tuple[list[str], list[dict[str, str]]]:
    rows = list(csv.DictReader(open(path, newline="")))
    meta = {"name", "kind", "archetype", "deck_sig", "random_wr", "macro_avg", "weighted_avg", "worst_archetype", "worst_wr"}
    arches = [c for c in (rows[0].keys() if rows else []) if c not in meta]
    return arches, rows


def load_distribution(path: str, *, value_col: str, band_multiplier: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            band = row["score_band"]
            arch = row["archetype"]
            value = fnum(row.get(value_col))
            if value <= 0:
                continue
            out["all"][arch] += value
            out[band][arch] += value
            out["climb"][arch] += value * band_multiplier.get(band, 1.0)
    return out


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items() if v > 0}


def score_row(row: dict[str, str], weights: dict[str, float], *, unknown_policy: str) -> tuple[float, float, str, float]:
    total = 0.0
    covered = 0.0
    worst_arch = ""
    worst_wr = 2.0
    for arch, w in normalize(weights).items():
        value = row.get(arch, "")
        if value == "":
            if unknown_policy == "neutral":
                wr = 0.5
                total += w * wr
                continue
            if unknown_policy == "zero":
                wr = 0.0
                total += w * wr
                continue
            continue
        wr = fnum(value)
        total += w * wr
        covered += w
        if wr < worst_wr:
            worst_wr = wr
            worst_arch = arch
    if unknown_policy == "renormalize" and covered > 0:
        total /= covered
    return total, covered, worst_arch, (worst_wr if worst_wr <= 1.0 else 0.0)


def parse_multiplier(spec: str) -> dict[str, float]:
    if not spec:
        return DEFAULT_CLIMB_MULTIPLIER
    out = dict(DEFAULT_CLIMB_MULTIPLIER)
    for part in spec.split(","):
        if not part:
            continue
        band, value = part.split("=", 1)
        out[band.strip()] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="rr_archetype_matrix.py output")
    parser.add_argument("--distribution", required=True, help="band_archetype_*_0810.csv")
    parser.add_argument("--value-col", default="player_entries", help="distribution count column")
    parser.add_argument("--out", required=True)
    parser.add_argument("--band-multiplier", default="", help="comma list, e.g. 600-699=8,700-799=6")
    parser.add_argument(
        "--unknown-policy",
        choices=["renormalize", "neutral", "zero"],
        default="renormalize",
        help="How to handle distribution archetypes absent from the RR matrix.",
    )
    args = parser.parse_args()

    band_multiplier = parse_multiplier(args.band_multiplier)
    _, matrix_rows = load_matrix(args.matrix)
    dist = load_distribution(args.distribution, value_col=args.value_col, band_multiplier=band_multiplier)
    bands = [b for b in ["600-699", "700-799", "800-899", "900-999", "1000-1099", "1100-1199", "1200+"] if b in dist]
    scenarios = ["all", "climb"] + bands

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "archetype",
        "deck_sig",
        "random_wr",
        "matrix_macro",
        "matrix_worst_arch",
        "matrix_worst_wr",
        "all_weighted",
        "climb_weighted",
        "coverage_all",
        "coverage_climb",
        "climb_worst_arch",
        "climb_worst_wr",
    ] + [f"band_{b}" for b in bands]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_rows:
            all_score, all_cov, _, _ = score_row(row, dist["all"], unknown_policy=args.unknown_policy)
            climb_score, climb_cov, climb_worst_arch, climb_worst_wr = score_row(row, dist["climb"], unknown_policy=args.unknown_policy)
            out = {
                "name": row.get("name", ""),
                "archetype": row.get("archetype", ""),
                "deck_sig": row.get("deck_sig", ""),
                "random_wr": row.get("random_wr", ""),
                "matrix_macro": row.get("macro_avg", ""),
                "matrix_worst_arch": row.get("worst_archetype", ""),
                "matrix_worst_wr": row.get("worst_wr", ""),
                "all_weighted": f"{all_score:.4f}",
                "climb_weighted": f"{climb_score:.4f}",
                "coverage_all": f"{all_cov:.4f}",
                "coverage_climb": f"{climb_cov:.4f}",
                "climb_worst_arch": climb_worst_arch,
                "climb_worst_wr": f"{climb_worst_wr:.4f}",
            }
            for band in bands:
                score, cov, _, _ = score_row(row, dist[band], unknown_policy=args.unknown_policy)
                out[f"band_{band}"] = f"{score:.4f}" if cov > 0 else ""
            writer.writerow(out)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
