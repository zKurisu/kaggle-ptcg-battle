#!/usr/bin/env python3
"""Audit whether weak-matchup wins look reproducible or lucky.

Input comes from build_trajectory_targets.py for both sides of a matchup.  This
pairs the two players in each episode and groups candidate wins by
deck_sig/team_name, separating wins where the opponent appears to have bricked
from wins where both sides had a plausible setup/tempo.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "archetype",
    "opponent_archetype",
    "deck_sig",
    "team_name",
    "games",
    "wins",
    "losses",
    "game_wr",
    "paired_games",
    "paired_wins",
    "strategy_wins",
    "clean_wins",
    "opponent_normal_wins",
    "opponent_brick_wins",
    "opponent_no_attack_wins",
    "opponent_no_primary_wins",
    "opponent_early_end_wins",
    "candidate_setup_wins",
    "candidate_tempo_wins",
    "candidate_no_early_end_wins",
    "clean_win_share",
    "brick_win_share",
    "avg_first_attack_turn_win",
    "avg_opp_first_attack_turn_win",
    "avg_primary_board_turn_win",
    "avg_opp_primary_board_turn_win",
    "avg_pressing_rate_win",
    "avg_opp_pressing_rate_win",
    "quality_score",
    "recommendation",
]


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def inum(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(round(float(row.get(key, default))))
    except Exception:
        return default


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def player_key(row: dict) -> tuple[str, str]:
    return str(row.get("episode_id", "")), str(row.get("player_index", ""))


def other_player(player: str) -> str:
    try:
        return str(1 - int(player))
    except Exception:
        return "1" if player == "0" else "0"


def build_opponent_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {player_key(row): row for row in rows}


def opponent_bricked(row: dict) -> bool:
    if not row:
        return False
    no_attack = inum(row, "attack_by_8") == 0
    no_primary = inum(row, "primary_board_by_6") == 0
    early_end = inum(row, "no_early_end") == 0
    low_pressing = fnum(row, "pressing_main_rate") < 0.25
    return no_attack or (no_primary and low_pressing) or (early_end and low_pressing)


def opponent_normal(row: dict) -> bool:
    if not row:
        return False
    normal_tempo = inum(row, "attack_by_8") == 1 or inum(row, "tempo_success") == 1
    normal_setup = inum(row, "primary_board_by_6") == 1 or inum(row, "setup_success") == 1
    no_bad_end = inum(row, "no_early_end") == 1
    return no_bad_end and (normal_tempo or normal_setup) and fnum(row, "pressing_main_rate") >= 0.25


def candidate_strategy(row: dict) -> bool:
    return (
        inum(row, "outcome_win") == 1
        and inum(row, "no_early_end") == 1
        and (inum(row, "setup_success") == 1 or inum(row, "tempo_success") == 1)
    )


def clean_win(row: dict, opp: dict) -> bool:
    return candidate_strategy(row) and opponent_normal(opp) and not opponent_bricked(opp)


def add_sum(stats: dict, key: str, value: float) -> None:
    stats[key] += value


def rank_quality(stats: dict) -> float:
    wins = max(int(stats["wins"]), 1)
    clean = int(stats["clean_wins"])
    strategy = int(stats["strategy_wins"])
    normal = int(stats["opponent_normal_wins"])
    brick = int(stats["opponent_brick_wins"])
    games = max(int(stats["games"]), 1)
    wr = int(stats["wins"]) / games
    clean_share = clean / wins
    strategy_share = strategy / wins
    normal_share = normal / wins
    brick_penalty = brick / wins
    support = math.sqrt(min(clean / 20.0, 1.0)) * math.sqrt(min(wins / 40.0, 1.0))
    return (
        math.log1p(clean)
        * (0.30 + 0.70 * clean_share)
        * (0.35 + 0.35 * strategy_share + 0.30 * normal_share)
        * (0.80 + 0.20 * wr)
        * (0.35 + 0.65 * support)
        * (1.0 - 0.55 * brick_penalty)
    )


def recommendation(stats: dict, args: argparse.Namespace) -> str:
    wins = int(stats["wins"])
    clean = int(stats["clean_wins"])
    brick = int(stats["opponent_brick_wins"])
    paired_wins = max(int(stats["paired_wins"]), 1)
    if clean >= args.min_clean_wins and clean / max(wins, 1) >= args.min_clean_share:
        return "clean_teacher"
    if clean > 0 and brick / paired_wins <= args.max_brick_share:
        return "usable_mixed_teacher"
    if wins > 0 and brick / paired_wins > args.max_brick_share:
        return "mostly_opponent_brick"
    if wins > 0:
        return "unpaired_or_sparse_win"
    return "no_success"


def finalize_row(stats: dict, args: argparse.Namespace) -> dict:
    games = int(stats["games"])
    wins = int(stats["wins"])
    losses = int(stats["losses"])
    paired_wins = max(int(stats["paired_wins"]), 1)
    clean = int(stats["clean_wins"])
    brick = int(stats["opponent_brick_wins"])
    out = {k: stats.get(k, "") for k in ("archetype", "opponent_archetype", "deck_sig", "team_name")}
    for key in FIELDS:
        if key in out:
            continue
        if key.startswith("avg_"):
            denom = max(paired_wins, 1)
            out[key] = f"{(stats.get(key + '_sum', 0.0) / denom):.4f}"
        elif key == "game_wr":
            out[key] = f"{(wins / max(games, 1)):.6f}"
        elif key == "clean_win_share":
            out[key] = f"{(clean / max(wins, 1)):.6f}"
        elif key == "brick_win_share":
            out[key] = f"{(brick / paired_wins):.6f}"
        elif key == "quality_score":
            out[key] = f"{rank_quality(stats):.6f}"
        elif key == "recommendation":
            out[key] = recommendation(stats, args)
        else:
            out[key] = int(stats.get(key, 0))
    out["losses"] = losses
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--opponent-csv", required=True)
    p.add_argument("--min-games", type=int, default=1)
    p.add_argument("--min-wins", type=int, default=1)
    p.add_argument("--min-clean-wins", type=int, default=5)
    p.add_argument("--min-clean-share", type=float, default=0.20)
    p.add_argument("--max-brick-share", type=float, default=0.55)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    cand_rows = read_rows(args.candidate_csv)
    opp_index = build_opponent_index(read_rows(args.opponent_csv))
    groups: dict[tuple[str, str, str, str], dict] = defaultdict(lambda: defaultdict(float))

    for row in cand_rows:
        key = (
            str(row.get("archetype", "")),
            str(row.get("opponent_archetype", "")),
            str(row.get("deck_sig", "")),
            str(row.get("team_name", "")),
        )
        stats = groups[key]
        stats["archetype"], stats["opponent_archetype"], stats["deck_sig"], stats["team_name"] = key
        stats["games"] += 1
        if inum(row, "outcome_win") == 1:
            stats["wins"] += 1
        elif inum(row, "outcome_loss") == 1:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        if inum(row, "outcome_win") != 1:
            continue
        episode, player = player_key(row)
        opp = opp_index.get((episode, other_player(player)))
        if opp:
            stats["paired_wins"] += 1
        if candidate_strategy(row):
            stats["strategy_wins"] += 1
        if inum(row, "setup_success") == 1:
            stats["candidate_setup_wins"] += 1
        if inum(row, "tempo_success") == 1:
            stats["candidate_tempo_wins"] += 1
        if inum(row, "no_early_end") == 1:
            stats["candidate_no_early_end_wins"] += 1
        if opp and opponent_normal(opp):
            stats["opponent_normal_wins"] += 1
        if opp and opponent_bricked(opp):
            stats["opponent_brick_wins"] += 1
        if opp and inum(opp, "attack_by_8") == 0:
            stats["opponent_no_attack_wins"] += 1
        if opp and inum(opp, "primary_board_by_6") == 0:
            stats["opponent_no_primary_wins"] += 1
        if opp and inum(opp, "no_early_end") == 0:
            stats["opponent_early_end_wins"] += 1
        if opp and clean_win(row, opp):
            stats["clean_wins"] += 1
        if opp:
            add_sum(stats, "avg_first_attack_turn_win_sum", min(fnum(row, "first_attack_turn", 999.0), 30.0))
            add_sum(stats, "avg_opp_first_attack_turn_win_sum", min(fnum(opp, "first_attack_turn", 999.0), 30.0))
            add_sum(stats, "avg_primary_board_turn_win_sum", min(fnum(row, "first_primary_board_turn", 999.0), 30.0))
            add_sum(stats, "avg_opp_primary_board_turn_win_sum", min(fnum(opp, "first_primary_board_turn", 999.0), 30.0))
            add_sum(stats, "avg_pressing_rate_win_sum", fnum(row, "pressing_main_rate"))
            add_sum(stats, "avg_opp_pressing_rate_win_sum", fnum(opp, "pressing_main_rate"))

    rows = [
        finalize_row(stats, args)
        for stats in groups.values()
        if int(stats["games"]) >= args.min_games and int(stats["wins"]) >= args.min_wins
    ]
    rows.sort(
        key=lambda r: (
            float(r["quality_score"]),
            int(r["clean_wins"]),
            float(r["game_wr"]),
            int(r["wins"]),
            int(r["games"]),
        ),
        reverse=True,
    )
    if args.top:
        rows = rows[: args.top]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")
    for row in rows[: min(20, len(rows))]:
        print(
            f"{row['archetype']} vs {row['opponent_archetype']} "
            f"sig={row['deck_sig'][:12]} team={row['team_name']} "
            f"wr={row['game_wr']} clean={row['clean_wins']}/{row['wins']} "
            f"brick_share={row['brick_win_share']} q={row['quality_score']} "
            f"rec={row['recommendation']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
