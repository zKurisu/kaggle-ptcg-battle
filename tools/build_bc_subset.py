#!/usr/bin/env python3
"""Build a BCCorpus-compatible subset from extracted BC npz files."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def clean_arch(name: str) -> str:
    return name.replace(" ", "_")


def as_str_array(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def discover_paths(corpus: str, archetype: str, score_bands: list[str]) -> list[str]:
    arch = clean_arch(archetype)
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(os.path.join(corpus, arch, band.replace(" ", "_"), "*.npz"))))
    return paths


def row_mask(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    n = len(data["board"])
    mask = np.ones(n, dtype=bool)
    if args.deck_sig:
        if "deck_sig" not in data:
            raise ValueError("deck_sig filter requested but corpus has no deck_sig")
        mask &= np.isin(as_str_array(data["deck_sig"]), args.deck_sig)
    if args.team_name:
        if "team_name" not in data:
            raise ValueError("team_name filter requested but corpus has no team_name")
        wanted = {x.lower() for x in args.team_name}
        mask &= np.isin(np.char.lower(as_str_array(data["team_name"])), list(wanted))
    if args.opponent_deck_sig:
        if "opponent_deck_sig" not in data:
            raise ValueError("opponent_deck_sig filter requested but corpus has no opponent_deck_sig")
        mask &= np.isin(as_str_array(data["opponent_deck_sig"]), args.opponent_deck_sig)
    if args.opponent_archetype:
        if "opponent_archetype" not in data:
            raise ValueError("opponent_archetype filter requested but corpus has no opponent_archetype")
        wanted = {x.lower() for x in args.opponent_archetype}
        mask &= np.isin(np.char.lower(as_str_array(data["opponent_archetype"])), list(wanted))
    if args.outcome != "all":
        if "won" not in data:
            raise ValueError("outcome filter requested but corpus has no won metadata")
        won = np.asarray(data["won"], dtype=np.int8)
        draw = np.asarray(data["draw"], dtype=np.int8) if "draw" in data else np.zeros(n, dtype=np.int8)
        if args.outcome == "win":
            mask &= won == 1
        elif args.outcome == "loss":
            mask &= (won != 1) & (draw != 1)
        elif args.outcome == "draw":
            mask &= (won != 1) & (draw == 1)
    return mask


def sample_by_game(data: dict[str, np.ndarray], idx: np.ndarray, max_games: int, seed: int) -> np.ndarray:
    if max_games <= 0 or len(idx) == 0:
        return idx
    if "episode_id" not in data or "player_index" not in data:
        rng = np.random.default_rng(seed)
        if len(idx) <= max_games:
            return idx
        return np.sort(rng.choice(idx, size=max_games, replace=False))
    episodes = as_str_array(data["episode_id"])
    players = np.asarray(data["player_index"], dtype=np.int16)
    game_to_rows: dict[tuple[str, int], list[int]] = {}
    for i in idx:
        game_to_rows.setdefault((episodes[i], int(players[i])), []).append(int(i))
    keys = list(game_to_rows)
    if len(keys) <= max_games:
        return idx
    rng = np.random.default_rng(seed)
    keep_keys = set(keys[i] for i in rng.choice(len(keys), size=max_games, replace=False))
    rows: list[int] = []
    for key in keys:
        if key in keep_keys:
            rows.extend(game_to_rows[key])
    return np.asarray(sorted(rows), dtype=np.int64)


def subset_data(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    n = len(data["board"])
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[idx]
        else:
            out[key] = arr
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--outcome", choices=["all", "win", "loss", "draw"], default="all")
    p.add_argument("--out", required=True, help="output corpus root")
    p.add_argument("--out-band", required=True, help="synthetic score band/folder name")
    p.add_argument("--name", default="subset", help="output npz stem")
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    paths = discover_paths(args.corpus, args.archetype, args.score_bands)
    if not paths:
        raise FileNotFoundError("no source npz files found")
    out_dir = Path(args.out) / clean_arch(args.archetype) / args.out_band.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{args.name}.npz"
    summary_csv = out_dir / f"{args.name}_summary.csv"

    arrays_by_key: dict[str, list[np.ndarray]] = {}
    scalar_by_key: dict[str, np.ndarray] = {}
    rows = []
    total_raw = 0
    total_kept = 0
    total_games: set[tuple[str, int]] = set()
    for path_i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        raw = len(data["board"])
        total_raw += raw
        mask = row_mask(data, args)
        idx = np.flatnonzero(mask)
        idx = sample_by_game(data, idx, args.max_games, args.seed + path_i)
        kept = len(idx)
        total_kept += kept
        if kept:
            part = subset_data(data, idx)
            for key, arr in part.items():
                if np.asarray(arr).shape[:1] == (kept,):
                    arrays_by_key.setdefault(key, []).append(np.asarray(arr))
                elif key not in scalar_by_key:
                    scalar_by_key[key] = np.asarray(arr)
            if "episode_id" in data and "player_index" in data:
                episodes = as_str_array(data["episode_id"])
                players = np.asarray(data["player_index"], dtype=np.int16)
                for i in idx:
                    total_games.add((episodes[i], int(players[i])))
        rows.append({"path": path, "raw": raw, "kept": kept})
        print(f"{path_i}/{len(paths)} {path} raw={raw} kept={kept}", flush=True)

    if total_kept <= 0:
        raise RuntimeError("no rows kept")

    output: dict[str, np.ndarray] = {}
    for key, parts in arrays_by_key.items():
        output[key] = np.concatenate(parts, axis=0)
    for key, arr in scalar_by_key.items():
        output.setdefault(key, arr)
    output["subset_source_corpus"] = np.asarray(args.corpus)
    output["subset_source_archetype"] = np.asarray(args.archetype)
    output["subset_source_score_bands"] = np.asarray(args.score_bands, dtype=object)
    output["subset_outcome"] = np.asarray(args.outcome)

    np.savez_compressed(out_npz, **output)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "raw", "kept"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"path": "TOTAL", "raw": total_raw, "kept": total_kept})

    print(
        f"wrote {out_npz} rows={total_kept} games={len(total_games)} raw={total_raw} "
        f"summary={summary_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
