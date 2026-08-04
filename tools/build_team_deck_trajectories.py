#!/usr/bin/env python3
"""Summarize team + deck-signature trajectories from extracted BC corpora."""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2.data import discover_npz_paths


FIELDS = [
    "trajectory_score",
    "archetype",
    "team_name",
    "deck_sig",
    "bands",
    "dates",
    "files",
    "episodes",
    "decisions",
    "wins",
    "losses",
    "draws",
    "decision_win_rate",
    "avg_score",
    "max_score",
    "first_date",
    "last_date",
    "opponent_filters",
]


def _date_from_name(name: str) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    return m.group(1) if m else ""


def _score_row(row: dict) -> float:
    decisions = int(row["decisions"])
    episodes = int(row["episodes"] or 0)
    dates = len(str(row["dates"]).split()) if row["dates"] else 0
    wr = float(row["decision_win_rate"])
    max_score = float(row["max_score"])
    # Prefer high final strength, cross-day persistence, winner density, and
    # enough labels. This is a ranking heuristic for teacher selection, not Elo.
    return (
        np.log1p(decisions)
        * (1.0 + 0.20 * max(dates - 1, 0))
        * (0.60 + wr)
        * (0.50 + min(max_score, 1300.0) / 1300.0)
        * (1.0 + min(episodes, 500) / 1000.0)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded_v9")
    p.add_argument("--archetype", action="append", default=[],
                   help="repeatable; omit to scan known top archetypes")
    p.add_argument("--score-bands", nargs="+",
                   default=["1200+", "1100-1199", "1000-1099", "900-999", "800-899"])
    p.add_argument("--min-decisions", type=int, default=1000)
    p.add_argument("--min-episodes", type=int, default=10)
    p.add_argument("--opponent-deck-sig", action="append", default=[],
                   help="filter decisions to games against one or more opponent deck signatures")
    p.add_argument("--opponent-archetype", action="append", default=[],
                   help="filter decisions to games against one or more opponent archetypes")
    p.add_argument("--opponent-team-name", action="append", default=[],
                   help="filter decisions to games against one or more exact opponent team names")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--progress-every-files", type=int, default=10)
    p.add_argument("--out", default="logs/team_deck_trajectories.csv")
    args = p.parse_args()

    archetypes = args.archetype or [
        "Marnie Grimmsnarl",
        "Teal Mask Ogerpon",
        "Mega Lopunny",
        "Mega Lucario",
        "Alakazam",
        "Dragapult",
        "Festival Lead",
        "Crustle Wall",
        "Cynthia Garchomp",
        "Team Rocket Mewtwo",
    ]

    rows = defaultdict(lambda: {
        "bands": Counter(),
        "dates": Counter(),
        "files": Counter(),
        "episodes": set(),
        "decisions": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score_sum": 0.0,
        "score_n": 0,
        "max_score": 0.0,
    })
    opponent_deck_sigs = {str(x) for x in args.opponent_deck_sig}
    opponent_archetypes = {str(x).lower() for x in args.opponent_archetype}
    opponent_team_names = {str(x).lower() for x in args.opponent_team_name}
    total_files = total_rows = 0
    all_paths: list[tuple[str, str]] = []
    for arch in archetypes:
        paths = discover_npz_paths(args.corpus, arch, args.score_bands)
        all_paths.extend((arch, path) for path in paths)
    t0 = time.time()
    for file_i, (arch, path) in enumerate(all_paths, 1):
        total_files += 1
        date = _date_from_name(Path(path).name)
        band = Path(path).parent.name.replace("_", " ")
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        if "team_name" not in data or "deck_sig" not in data:
            raise ValueError(f"{path} lacks team_name/deck_sig metadata; re-extract corpus")
        if opponent_deck_sigs and "opponent_deck_sig" not in data:
            raise ValueError(f"{path} lacks opponent_deck_sig metadata; re-extract corpus")
        if opponent_archetypes and "opponent_archetype" not in data:
            raise ValueError(f"{path} lacks opponent_archetype metadata; re-extract corpus")
        if opponent_team_names and "opponent_team_name" not in data:
            raise ValueError(f"{path} lacks opponent_team_name metadata; re-extract corpus")
        n = len(data["board"])
        total_rows += n
        for i in range(n):
            if opponent_deck_sigs and str(data["opponent_deck_sig"][i]) not in opponent_deck_sigs:
                continue
            if opponent_archetypes and str(data["opponent_archetype"][i]).lower() not in opponent_archetypes:
                continue
            if opponent_team_names and str(data["opponent_team_name"][i]).lower() not in opponent_team_names:
                continue
            team = str(data["team_name"][i])
            sig = str(data["deck_sig"][i])
            if not team or not sig:
                continue
            key = (arch, team, sig)
            row = rows[key]
            row["bands"][band] += 1
            if date:
                row["dates"][date] += 1
            row["files"][Path(path).name] += 1
            if "episode_id" in data:
                row["episodes"].add(str(data["episode_id"][i]))
            row["decisions"] += 1
            if "won" in data:
                row["wins"] += int(data["won"][i])
            if "draw" in data:
                row["draws"] += int(data["draw"][i])
            if "won" in data and "draw" in data:
                row["losses"] += int(int(data["won"][i]) == 0 and int(data["draw"][i]) == 0)
            if "score" in data:
                score = float(data["score"][i])
                row["score_sum"] += score
                row["score_n"] += 1
                row["max_score"] = max(float(row["max_score"]), score)
        if args.progress_every_files and (
            file_i == 1 or file_i % args.progress_every_files == 0 or file_i == len(all_paths)
        ):
            rate = total_rows / max(time.time() - t0, 1e-9)
            print(
                f"  files {file_i}/{len(all_paths)} rows={total_rows} "
                f"trajectories={len(rows)} {rate:.0f} rows/s",
                flush=True,
            )

    out_rows = []
    for (arch, team, sig), r in rows.items():
        episodes = len(r["episodes"])
        decisions = int(r["decisions"])
        if decisions < args.min_decisions or episodes < args.min_episodes:
            continue
        dates = sorted(r["dates"])
        row = {
            "trajectory_score": 0.0,
            "archetype": arch,
            "team_name": team,
            "deck_sig": sig,
            "bands": " ".join(f"{k}:{v}" for k, v in r["bands"].most_common()),
            "dates": " ".join(dates),
            "files": len(r["files"]),
            "episodes": episodes,
            "decisions": decisions,
            "wins": int(r["wins"]),
            "losses": int(r["losses"]),
            "draws": int(r["draws"]),
            "decision_win_rate": int(r["wins"]) / max(decisions, 1),
            "avg_score": float(r["score_sum"]) / max(int(r["score_n"]), 1),
            "max_score": float(r["max_score"]),
            "first_date": dates[0] if dates else "",
            "last_date": dates[-1] if dates else "",
            "opponent_filters": " ".join(
                [*(f"deck_sig={x}" for x in args.opponent_deck_sig),
                 *(f"archetype={x}" for x in args.opponent_archetype),
                 *(f"team={x}" for x in args.opponent_team_name)]
            ),
        }
        row["trajectory_score"] = _score_row(row)
        out_rows.append(row)
    out_rows.sort(key=lambda x: float(x["trajectory_score"]), reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in out_rows:
            w.writerow({
                **row,
                "trajectory_score": f"{row['trajectory_score']:.4f}",
                "decision_win_rate": f"{row['decision_win_rate']:.4f}",
                "avg_score": f"{row['avg_score']:.1f}",
                "max_score": f"{row['max_score']:.1f}",
            })

    print(f"Corpus: {args.corpus}")
    print(f"Scanned: files={total_files} rows={total_rows}")
    print(f"Wrote {out}: {len(out_rows)} trajectories")
    print("\nTop trajectories:")
    for row in out_rows[: args.top]:
        print(
            f"  {row['trajectory_score']:7.2f} {row['archetype']:<20} "
            f"sig={row['deck_sig']} dec={row['decisions']:7d} eps={row['episodes']:5d} "
            f"wr={row['decision_win_rate']:.2f} max={row['max_score']:.1f} "
            f"dates={row['dates']} team={row['team_name'][:32]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
