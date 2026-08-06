#!/usr/bin/env python3
"""Audit matchup success slices by deck, team, and opponent metadata."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


COUNT_FIELDS = (
    "games",
    "wins",
    "losses",
    "draws",
    "decisions",
    "win_decisions",
    "loss_decisions",
    "draw_decisions",
)

DEFAULT_GROUP_BY = (
    "archetype",
    "deck_sig",
    "team_name",
    "opponent_archetype",
    "opponent_deck_sig",
    "opponent_team_name",
)


def clean_arch(name: str) -> str:
    return str(name).replace(" ", "_")


def display_arch(path_name: str) -> str:
    return path_name.replace("_", " ")


def as_str_array(arr: np.ndarray, n: int, default: str = "") -> np.ndarray:
    if arr is None:
        return np.full(n, default, dtype=object)
    return np.asarray(arr).astype(str)


def normalized_set(values: list[str]) -> set[str]:
    return {str(x).strip().lower() for x in values if str(x).strip()}


def discover_paths(corpus: str, archetypes: list[str], score_bands: list[str]) -> list[tuple[str, str]]:
    root = Path(corpus)
    if archetypes:
        arch_dirs = [(arch, root / clean_arch(arch)) for arch in archetypes]
    else:
        arch_dirs = [(display_arch(p.name), p) for p in sorted(root.iterdir()) if p.is_dir()]
    out: list[tuple[str, str]] = []
    for arch, arch_dir in arch_dirs:
        if not arch_dir.exists():
            continue
        if score_bands:
            band_dirs = [arch_dir / b.replace(" ", "_") for b in score_bands]
        else:
            band_dirs = [p for p in sorted(arch_dir.iterdir()) if p.is_dir()]
        for band_dir in band_dirs:
            out.extend((arch, p) for p in sorted(glob.glob(str(band_dir / "*.npz"))))
    return out


def blank_counts() -> Counter:
    return Counter({k: 0 for k in COUNT_FIELDS})


def outcome_at(won: np.ndarray, draw: np.ndarray, i: int) -> str:
    if int(won[i]) == 1:
        return "win"
    if int(draw[i]) == 1:
        return "draw"
    return "loss"


def add_outcome(counts: Counter, outcome: str, amount: int = 1, *, decisions: bool = False) -> None:
    if decisions:
        counts["decisions"] += amount
        if outcome == "win":
            counts["win_decisions"] += amount
        elif outcome == "draw":
            counts["draw_decisions"] += amount
        else:
            counts["loss_decisions"] += amount
        return
    counts["games"] += amount
    if outcome == "win":
        counts["wins"] += amount
    elif outcome == "draw":
        counts["draws"] += amount
    else:
        counts["losses"] += amount


def label_status(data: dict[str, np.ndarray], i: int, include_empty: bool) -> str:
    action = np.asarray(data["action"][i], dtype=np.int64)
    n_opt = len(data["ot"][i])
    mn = int(data["min_c"][i])
    mx = int(data["max_c"][i])
    if len(action) == 0:
        return "keep" if include_empty and mn == 0 else "empty"
    if len(action) < mn or len(action) > mx:
        return "bad"
    if len(set(action.tolist())) != len(action):
        return "bad"
    if not ((action >= 0) & (action < n_opt)).all():
        return "bad"
    return "keep"


def read_key(data: dict[str, np.ndarray], arch: str, key: str, i: int) -> str:
    if key == "archetype":
        return arch
    if key in data:
        return str(data[key][i])
    return ""


def passes_filters(meta: dict[str, str], args: argparse.Namespace) -> bool:
    if args.deck_sig and meta.get("deck_sig") not in set(args.deck_sig):
        return False
    if args.team_name and meta.get("team_name", "").lower() not in normalized_set(args.team_name):
        return False
    if args.opponent_deck_sig and meta.get("opponent_deck_sig") not in set(args.opponent_deck_sig):
        return False
    if args.opponent_team_name and meta.get("opponent_team_name", "").lower() not in normalized_set(args.opponent_team_name):
        return False
    if args.opponent_archetype and meta.get("opponent_archetype", "").lower() not in normalized_set(args.opponent_archetype):
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", action="append", default=[], help="exact archetype; repeatable. Default scans all")
    p.add_argument("--score-bands", nargs="*", default=[], help="default scans all score bands")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--opponent-team-name", action="append", default=[])
    p.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_BY), choices=list(DEFAULT_GROUP_BY))
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--min-games", type=int, default=0)
    p.add_argument("--min-wins", type=int, default=0)
    p.add_argument("--min-decisions", type=int, default=0)
    p.add_argument("--top", type=int, default=0, help="limit written rows after sorting; 0 means all")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    paths = discover_paths(args.corpus, args.archetype, args.score_bands)
    if not paths:
        raise FileNotFoundError("no corpus .npz files found")

    counts: dict[tuple[str, ...], Counter] = {}
    key_meta: dict[tuple[str, ...], dict[str, str]] = {}
    seen_games: set[tuple[str, int, tuple[str, ...]]] = set()

    for path_i, (arch, path) in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        n = len(data["board"])
        if "won" not in data:
            raise ValueError(f"{path} lacks won metadata")
        won = np.asarray(data["won"], dtype=np.int8)
        draw = np.asarray(data["draw"], dtype=np.int8) if "draw" in data else np.zeros(n, dtype=np.int8)
        episode_id = as_str_array(data.get("episode_id"), n)
        player_index = np.asarray(data["player_index"], dtype=np.int16) if "player_index" in data else np.zeros(n, dtype=np.int16)

        for i in range(n):
            meta = {key: read_key(data, arch, key, i) for key in DEFAULT_GROUP_BY}
            if not passes_filters(meta, args):
                continue
            if label_status(data, i, args.include_empty) != "keep":
                continue
            key = tuple(meta[k] for k in args.group_by)
            row = counts.setdefault(key, blank_counts())
            key_meta.setdefault(key, {k: meta[k] for k in DEFAULT_GROUP_BY})
            outcome = outcome_at(won, draw, i)
            add_outcome(row, outcome, decisions=True)
            game_key = (str(episode_id[i]), int(player_index[i]), key)
            if game_key not in seen_games:
                seen_games.add(game_key)
                add_outcome(row, outcome, decisions=False)

        if args.progress_every and (path_i == 1 or path_i % args.progress_every == 0 or path_i == len(paths)):
            print(f"scanned {path_i}/{len(paths)} groups={len(counts)}", flush=True)

    rows = []
    for key, row in counts.items():
        games = int(row["games"])
        wins = int(row["wins"])
        decisions = int(row["decisions"])
        if games < args.min_games or wins < args.min_wins or decisions < args.min_decisions:
            continue
        meta = key_meta[key]
        out = {k: meta.get(k, "") for k in DEFAULT_GROUP_BY}
        out.update({
            "games": games,
            "wins": wins,
            "losses": int(row["losses"]),
            "draws": int(row["draws"]),
            "game_wr": f"{(wins / games) if games else 0.0:.6f}",
            "decisions": decisions,
            "win_decisions": int(row["win_decisions"]),
            "loss_decisions": int(row["loss_decisions"]),
            "draw_decisions": int(row["draw_decisions"]),
            "decision_win_share": f"{(int(row['win_decisions']) / decisions) if decisions else 0.0:.6f}",
        })
        rows.append(out)

    rows.sort(
        key=lambda r: (
            float(r["game_wr"]),
            int(r["wins"]),
            int(r["games"]),
            int(r["win_decisions"]),
        ),
        reverse=True,
    )
    if args.top:
        rows = rows[: args.top]

    fields = [
        *DEFAULT_GROUP_BY,
        *COUNT_FIELDS,
        "game_wr",
        "decision_win_share",
    ]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
