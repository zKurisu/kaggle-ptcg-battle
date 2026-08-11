#!/usr/bin/env python3
"""Summarize round-robin results as deck-vs-archetype win-rate matrices."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def fnum(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def load_manifest(path: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("name") or row.get("shadow_name") or row.get("deck_sig") or ""
            if not name and row.get("eval_entry"):
                name = row["eval_entry"].split("=", 1)[0]
            if not name:
                continue
            rows[name] = row
    return rows


def add_cell(acc: dict[tuple[str, str], list[float]], row_name: str, opp_arch: str, wins: float, games: float) -> None:
    if games <= 0:
        return
    cur = acc[(row_name, opp_arch)]
    cur[0] += wins
    cur[1] += games


def read_rr(rr_csv: str, manifest: dict[str, dict[str, str]], *, exclude_same_archetype: bool) -> dict[tuple[str, str], list[float]]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    with open(rr_csv, newline="") as f:
        for row in csv.DictReader(f):
            a = row["row"]
            b = row["column"]
            if a not in manifest or b not in manifest:
                continue
            a_arch = manifest[a].get("archetype", "")
            b_arch = manifest[b].get("archetype", "")
            if exclude_same_archetype and a_arch == b_arch:
                continue
            games = fnum(row.get("games"))
            a_wins = fnum(row.get("row_wins"))
            b_wins = fnum(row.get("column_wins"))
            add_cell(acc, a, b_arch, a_wins, games)
            add_cell(acc, b, a_arch, b_wins, games)
    return acc


def load_random(path: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "")
            if name:
                out[name] = row.get("win_rate", "")
    return out


def fmt_rate(wins_games: list[float] | None, digits: int) -> str:
    if not wins_games or wins_games[1] <= 0:
        return ""
    return f"{wins_games[0] / wins_games[1]:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True, help="eval_round_robin.py CSV output")
    parser.add_argument("--manifest", required=True, help="manifest CSV with name/archetype/deck_sig")
    parser.add_argument("--random", default="", help="optional eval_manifest_random.py CSV")
    parser.add_argument("--out", required=True, help="deck x opponent archetype matrix CSV")
    parser.add_argument("--counts-out", default="", help="optional game-count matrix CSV")
    parser.add_argument("--include-same-archetype", action="store_true")
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    random_wr = load_random(args.random)
    acc = read_rr(args.rr, manifest, exclude_same_archetype=not args.include_same_archetype)

    arches = sorted({row.get("archetype", "") for row in manifest.values() if row.get("archetype", "")})
    names = [name for name in manifest if any((name, arch) in acc for arch in arches)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "name",
        "archetype",
        "deck_sig",
        "random_wr",
        "macro_avg",
        "weighted_avg",
        "worst_archetype",
        "worst_wr",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + arches)
        writer.writeheader()
        for name in names:
            row_meta = manifest[name]
            values = [(arch, acc[(name, arch)]) for arch in arches if (name, arch) in acc]
            rates = [(arch, wg[0] / wg[1], wg[1]) for arch, wg in values if wg[1] > 0]
            if rates:
                macro = sum(rate for _, rate, _ in rates) / len(rates)
                weighted = sum(rate * games for _, rate, games in rates) / sum(games for _, _, games in rates)
                worst_arch, worst_rate, _ = min(rates, key=lambda x: x[1])
            else:
                macro = weighted = 0.0
                worst_arch, worst_rate = "", 0.0
            out_row = {
                "name": name,
                "archetype": row_meta.get("archetype", ""),
                "deck_sig": row_meta.get("deck_sig", ""),
                "random_wr": random_wr.get(name, row_meta.get("random_win_rate", "")),
                "macro_avg": f"{macro:.{args.digits}f}" if rates else "",
                "weighted_avg": f"{weighted:.{args.digits}f}" if rates else "",
                "worst_archetype": worst_arch,
                "worst_wr": f"{worst_rate:.{args.digits}f}" if rates else "",
            }
            for arch in arches:
                out_row[arch] = fmt_rate(acc.get((name, arch)), args.digits)
            writer.writerow(out_row)

    if args.counts_out:
        with open(args.counts_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "archetype", "deck_sig"] + arches)
            writer.writeheader()
            for name in names:
                row_meta = manifest[name]
                out_row = {
                    "name": name,
                    "archetype": row_meta.get("archetype", ""),
                    "deck_sig": row_meta.get("deck_sig", ""),
                }
                for arch in arches:
                    wg = acc.get((name, arch))
                    out_row[arch] = int(wg[1]) if wg else ""
                writer.writerow(out_row)

    print(f"Wrote {args.out}")
    if args.counts_out:
        print(f"Wrote {args.counts_out}")


if __name__ == "__main__":
    main()
