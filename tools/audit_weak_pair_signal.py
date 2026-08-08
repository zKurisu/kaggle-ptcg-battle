#!/usr/bin/env python3
"""Measure how much weak-matchup success signal is diluted in a BC corpus."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def clean_arch(name: str) -> str:
    return str(name or "").strip().replace(" ", "_")


def norm_arch(name: str) -> str:
    return str(name or "").strip().replace("_", " ").lower()


def date_from_path(path: str) -> str:
    m = re.search(r"episodes-(\d{4}-\d{2}-\d{2})", path)
    return m.group(1) if m else ""


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if row.get(name):
            return str(row[name]).strip()
    return ""


def read_weak_pairs(path: str, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cand = first_present(row, ("cand_arch", "archetype", "candidate_archetype", "row_archetype", "row"))
            opp = first_present(row, ("opp_arch", "opponent_archetype", "target_archetype", "column_archetype", "column", "opponent"))
            if not cand or not opp:
                continue
            key = (cand, opp)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if limit and len(pairs) >= limit:
                break
    if not pairs:
        raise RuntimeError(f"no weak pairs found in {path}")
    return pairs


def read_clean_teacher_games(path: str) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not path:
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("clean_win", "0")).strip() not in ("1", "1.0", "true", "True"):
                continue
            arch = first_present(row, ("archetype", "candidate_archetype"))
            opp = first_present(row, ("opponent_archetype", "target_archetype"))
            game_key = first_present(row, ("game_key",))
            if not game_key and row.get("episode_id") and row.get("player_index"):
                game_key = f"{row['episode_id']}:{row['player_index']}"
            if arch and opp and game_key:
                out[(norm_arch(arch), norm_arch(opp))].add(game_key)
    return out


def discover_paths(corpus: str, archetype: str, score_bands: list[str], date_from: str, date_to: str) -> list[str]:
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(os.path.join(corpus, clean_arch(archetype), band.replace(" ", "_"), "*.npz"))))
    if date_from or date_to:
        filtered = []
        for path in paths:
            d = date_from_path(path)
            if date_from and d and d < date_from:
                continue
            if date_to and d and d > date_to:
                continue
            filtered.append(path)
        paths = filtered
    return paths


def as_str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def game_keys(data: dict[str, np.ndarray], mask: np.ndarray) -> list[str]:
    if "episode_id" not in data or "player_index" not in data or not mask.any():
        return []
    ep = as_str(data["episode_id"])[mask]
    pi = np.asarray(data["player_index"])[mask].astype(str)
    return [f"{e}:{p}" for e, p in zip(ep, pi)]


def count_pair(
    paths: list[str],
    pair_opponents: set[str],
    clean_games_by_pair: dict[tuple[str, str], set[str]],
    arch: str,
) -> dict[tuple[str, str], dict[str, float]]:
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for opp in pair_opponents:
        stats[(arch, opp)] = {
            "files": 0.0,
            "all_decisions": 0.0,
            "all_games": 0.0,
            "weak_decisions": 0.0,
            "weak_games": 0.0,
            "weak_win_decisions": 0.0,
            "weak_win_games": 0.0,
            "clean_teacher_decisions": 0.0,
            "clean_teacher_games": 0.0,
        }
    all_games: set[str] = set()
    weak_games: dict[str, set[str]] = {opp: set() for opp in pair_opponents}
    weak_win_games: dict[str, set[str]] = {opp: set() for opp in pair_opponents}
    clean_seen: dict[str, set[str]] = {opp: set() for opp in pair_opponents}

    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        n = len(data["board"])
        keys_all = game_keys(data, np.ones(n, dtype=bool))
        all_games.update(keys_all)
        opp_arr = as_str(data.get("opponent_archetype", np.asarray([""] * n)))
        won = np.asarray(data.get("won", np.zeros(n, dtype=np.int8)), dtype=np.int8)
        for opp in pair_opponents:
            row = stats[(arch, opp)]
            row["files"] += 1.0
            row["all_decisions"] += float(n)
            mask = np.char.lower(opp_arr) == norm_arch(opp)
            row["weak_decisions"] += float(mask.sum())
            wk = game_keys(data, mask)
            weak_games[opp].update(wk)
            win_mask = mask & (won == 1)
            row["weak_win_decisions"] += float(win_mask.sum())
            weak_win_games[opp].update(game_keys(data, win_mask))
            clean_games = clean_games_by_pair.get((norm_arch(arch), norm_arch(opp)), set())
            if clean_games and keys_all:
                clean_mask = np.fromiter((k in clean_games for k in keys_all), dtype=bool, count=n)
                row["clean_teacher_decisions"] += float(clean_mask.sum())
                clean_seen[opp].update(k for k, keep in zip(keys_all, clean_mask) if keep)

    for opp in pair_opponents:
        row = stats[(arch, opp)]
        row["all_games"] = float(len(all_games))
        row["weak_games"] = float(len(weak_games[opp]))
        row["weak_win_games"] = float(len(weak_win_games[opp]))
        row["clean_teacher_games"] = float(len(clean_seen[opp]))
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--weak-pairs-csv", required=True)
    p.add_argument("--teacher-games-csv", default="")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    pairs = read_weak_pairs(args.weak_pairs_csv, args.limit)
    clean_games = read_clean_teacher_games(args.teacher_games_csv)
    by_arch: dict[str, set[str]] = defaultdict(set)
    for arch, opp in pairs:
        by_arch[arch].add(opp)

    rows: list[dict[str, object]] = []
    for arch, opponents in by_arch.items():
        paths = discover_paths(args.corpus, arch, args.score_bands, args.date_from, args.date_to)
        print(f"{arch}: files={len(paths)} opponents={len(opponents)}", flush=True)
        if not paths:
            for opp in sorted(opponents):
                rows.append({
                    "archetype": arch,
                    "opponent_archetype": opp,
                    "files": 0,
                    "missing_corpus": 1,
                })
            continue
        stats = count_pair(paths, opponents, clean_games, arch)
        for (pair_arch, opp), row in stats.items():
            all_dec = max(float(row.get("all_decisions", 0.0)), 1.0)
            weak_dec = float(row.get("weak_decisions", 0.0))
            weak_win_dec = float(row.get("weak_win_decisions", 0.0))
            clean_dec = float(row.get("clean_teacher_decisions", 0.0))
            row.update({
                "archetype": pair_arch,
                "opponent_archetype": opp,
                "missing_corpus": 0,
                "weak_decision_share": weak_dec / all_dec,
                "weak_win_decision_share": weak_win_dec / all_dec,
                "clean_teacher_decision_share": clean_dec / all_dec,
                "clean_within_weak_share": clean_dec / max(weak_dec, 1.0),
            })
            rows.append(row)

    fieldnames = [
        "archetype",
        "opponent_archetype",
        "missing_corpus",
        "files",
        "all_decisions",
        "all_games",
        "weak_decisions",
        "weak_games",
        "weak_win_decisions",
        "weak_win_games",
        "clean_teacher_decisions",
        "clean_teacher_games",
        "weak_decision_share",
        "weak_win_decision_share",
        "clean_teacher_decision_share",
        "clean_within_weak_share",
    ]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
