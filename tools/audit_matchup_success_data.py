#!/usr/bin/env python3
"""Audit available winning demonstrations for weak matchups in a BC corpus."""
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


def clean_arch(name: str) -> str:
    return name.replace(" ", "_")


def display_arch(path_name: str) -> str:
    return path_name.replace("_", " ")


def as_str_array(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def read_weak_plan(path: str) -> list[dict]:
    if not path:
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"cand_arch", "opp_arch"}
    if not rows:
        return rows
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"weak plan missing columns: {sorted(missing)}")
    return rows


def discover_paths(corpus: str, archetypes: set[str], score_bands: list[str]) -> list[tuple[str, str]]:
    root = Path(corpus)
    if archetypes:
        arch_dirs = [(arch, root / clean_arch(arch)) for arch in sorted(archetypes)]
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


def add_outcome(counts: Counter, outcome: str, amount: int = 1, *, decisions: bool = False) -> None:
    if decisions:
        counts["decisions"] += amount
        if outcome == "win":
            counts["win_decisions"] += amount
        elif outcome == "loss":
            counts["loss_decisions"] += amount
        elif outcome == "draw":
            counts["draw_decisions"] += amount
    else:
        counts["games"] += amount
        if outcome == "win":
            counts["wins"] += amount
        elif outcome == "loss":
            counts["losses"] += amount
        elif outcome == "draw":
            counts["draws"] += amount


def outcome_at(won: np.ndarray, draw: np.ndarray, i: int) -> str:
    if int(won[i]) == 1:
        return "win"
    if int(draw[i]) == 1:
        return "draw"
    return "loss"


def scan_corpus(paths: list[tuple[str, str]], progress_every: int) -> dict[tuple[str, str, str], Counter]:
    counts: dict[tuple[str, str, str], Counter] = {}
    seen_games: set[tuple[str, int, str, str, str]] = set()
    for path_i, (arch, path) in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            if "deck_sig" not in z or "opponent_archetype" not in z:
                continue
            deck_sigs = as_str_array(z["deck_sig"])
            opp_arches = as_str_array(z["opponent_archetype"])
            won = np.asarray(z["won"], dtype=np.int8)
            draw = np.asarray(z["draw"], dtype=np.int8) if "draw" in z else np.zeros(len(deck_sigs), dtype=np.int8)
            episode_id = as_str_array(z["episode_id"]) if "episode_id" in z else np.arange(len(deck_sigs)).astype(str)
            player_index = np.asarray(z["player_index"], dtype=np.int16) if "player_index" in z else np.zeros(len(deck_sigs), dtype=np.int16)

            for i in range(len(deck_sigs)):
                key = (arch, str(deck_sigs[i]), str(opp_arches[i]))
                row_counts = counts.setdefault(key, blank_counts())
                outcome = outcome_at(won, draw, i)
                add_outcome(row_counts, outcome, decisions=True)
                game_key = (str(episode_id[i]), int(player_index[i]), key[0], key[1], key[2])
                if game_key not in seen_games:
                    seen_games.add(game_key)
                    add_outcome(row_counts, outcome)

        if progress_every and path_i % progress_every == 0:
            print(f"scanned {path_i}/{len(paths)} files groups={len(counts)}", flush=True)
    return counts


def row_from_counts(arch: str, deck_sig: str, opp_arch: str, counts: Counter) -> dict:
    games = int(counts["games"])
    decisions = int(counts["decisions"])
    wins = int(counts["wins"])
    win_decisions = int(counts["win_decisions"])
    return {
        "archetype": arch,
        "deck_sig": deck_sig,
        "opponent_archetype": opp_arch,
        "games": games,
        "wins": wins,
        "losses": int(counts["losses"]),
        "draws": int(counts["draws"]),
        "game_wr": f"{(wins / games) if games else 0.0:.6f}",
        "decisions": decisions,
        "win_decisions": win_decisions,
        "loss_decisions": int(counts["loss_decisions"]),
        "draw_decisions": int(counts["draw_decisions"]),
        "decision_win_share": f"{(win_decisions / decisions) if decisions else 0.0:.6f}",
    }


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def combine_counts(counts: dict[tuple[str, str, str], Counter], arch: str, deck_sigs: list[str], opp_arch: str) -> Counter:
    out = blank_counts()
    for sig in deck_sigs:
        out.update(counts.get((arch, sig, opp_arch), Counter()))
    return out


def combine_arch_counts(counts: dict[tuple[str, str, str], Counter], arch: str, opp_arch: str) -> Counter:
    out = blank_counts()
    for (row_arch, _sig, row_opp), row_counts in counts.items():
        if row_arch == arch and row_opp == opp_arch:
            out.update(row_counts)
    return out


def recommendation(counts: Counter, arch_counts: Counter, args: argparse.Namespace) -> str:
    wins = int(counts["wins"])
    win_decisions = int(counts["win_decisions"])
    arch_wins = int(arch_counts["wins"])
    arch_win_decisions = int(arch_counts["win_decisions"])
    if wins >= args.min_success_games and win_decisions >= args.min_success_decisions:
        return "same_sig_success_bc_ok"
    if wins > 0 and win_decisions > 0:
        if arch_wins >= args.min_success_games and arch_win_decisions >= args.min_success_decisions:
            return "same_sig_sparse_use_cross_sig_or_generate"
        return "same_sig_sparse_generate_more"
    if arch_wins >= args.min_success_games and arch_win_decisions >= args.min_success_decisions:
        return "cross_sig_teacher_needed"
    return "generate_success_needed"


def build_plan_rows(plan_rows: list[dict], counts: dict[tuple[str, str, str], Counter], args: argparse.Namespace) -> list[dict]:
    out: list[dict] = []
    for row in plan_rows[: args.limit or None]:
        arch = row["cand_arch"]
        opp_arch = row["opp_arch"]
        top_sigs = [x for x in str(row.get("top_deck_sigs", "")).split(";") if x]
        if args.top_sigs > 0:
            top_sigs = top_sigs[: args.top_sigs]
        if not top_sigs:
            top_sigs = sorted(sig for a, sig, o in counts if a == arch and o == opp_arch)
        arch_counts = combine_arch_counts(counts, arch, opp_arch)
        for scope, deck_sig, scoped_counts in [
            ("all_candidate_sigs", "*", arch_counts),
            ("top_sigs_combined", ";".join(top_sigs), combine_counts(counts, arch, top_sigs, opp_arch)),
        ]:
            base = row_from_counts(arch, deck_sig, opp_arch, scoped_counts)
            base.update({
                "scope": scope,
                "rr_mean_wr": row.get("rr_mean_wr", ""),
                "rr_pairs": row.get("rr_pairs", ""),
                "recommendation": recommendation(scoped_counts, arch_counts, args),
            })
            out.append(base)
        for sig in top_sigs:
            scoped_counts = counts.get((arch, sig, opp_arch), blank_counts())
            base = row_from_counts(arch, sig, opp_arch, scoped_counts)
            base.update({
                "scope": "deck_sig",
                "rr_mean_wr": row.get("rr_mean_wr", ""),
                "rr_pairs": row.get("rr_pairs", ""),
                "recommendation": recommendation(scoped_counts, arch_counts, args),
            })
            out.append(base)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--weak-plan", default="", help="CSV with cand_arch, opp_arch, top_deck_sigs")
    p.add_argument("--score-bands", nargs="*", default=[], help="default: all bands")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-plan-csv", default="")
    p.add_argument("--limit", type=int, default=0, help="limit weak-plan rows")
    p.add_argument("--top-sigs", type=int, default=4)
    p.add_argument("--min-success-games", type=int, default=30)
    p.add_argument("--min-success-decisions", type=int, default=2000)
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    plan_rows = read_weak_plan(args.weak_plan)
    archetypes = {r["cand_arch"] for r in plan_rows[: args.limit or None]} if plan_rows else set()
    paths = discover_paths(args.corpus, archetypes, args.score_bands)
    if not paths:
        raise FileNotFoundError("no BC corpus .npz files found")
    counts = scan_corpus(paths, args.progress_every)

    rows = [
        row_from_counts(arch, sig, opp, row_counts)
        for (arch, sig, opp), row_counts in sorted(
            counts.items(),
            key=lambda kv: (kv[0][0], kv[0][2], -int(kv[1]["decisions"]), kv[0][1]),
        )
    ]
    fields = [
        "archetype",
        "deck_sig",
        "opponent_archetype",
        *COUNT_FIELDS,
        "game_wr",
        "decision_win_share",
    ]
    write_csv(args.out_csv, rows, fields)
    print(f"wrote {args.out_csv} rows={len(rows)}")

    if args.out_plan_csv:
        if not plan_rows:
            raise ValueError("--out-plan-csv requires --weak-plan")
        plan = build_plan_rows(plan_rows, counts, args)
        plan_fields = [
            "archetype",
            "deck_sig",
            "opponent_archetype",
            "scope",
            "rr_mean_wr",
            "rr_pairs",
            *COUNT_FIELDS,
            "game_wr",
            "decision_win_share",
            "recommendation",
        ]
        write_csv(args.out_plan_csv, plan, plan_fields)
        print(f"wrote {args.out_plan_csv} rows={len(plan)}")


if __name__ == "__main__":
    main()
