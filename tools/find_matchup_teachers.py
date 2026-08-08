#!/usr/bin/env python3
"""Find same-archetype teacher cohorts for weak matchups.

The goal is different from picking a submission candidate. For a weak pair like
Teal Mask Ogerpon vs Crustle Wall, this scans every Ogerpon deck signature and
team in the corpus, ranks the cohorts that actually beat Crustle, and reports
whether the archetype has enough successful games to learn from.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
import time
from collections import Counter, defaultdict
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

PAIR_FIELDS = (
    "archetype",
    "opponent_archetype",
    *COUNT_FIELDS,
    "game_wr",
    "decision_win_share",
    "avg_score",
    "avg_opponent_score",
    "recommendation",
)

TEACHER_FIELDS = (
    "archetype",
    "opponent_archetype",
    "rank",
    "deck_sig",
    "team_name",
    *COUNT_FIELDS,
    "game_wr",
    "decision_win_share",
    "avg_score",
    "avg_opponent_score",
    "pair_games",
    "pair_wins",
    "pair_game_wr",
    "pair_win_decisions",
    "share_pair_wins",
    "quality_score",
    "recommendation",
)


def clean_arch(name: str) -> str:
    return str(name).replace(" ", "_")


def display_arch(path_name: str) -> str:
    return path_name.replace("_", " ")


def normalized(values: list[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def as_str_array(arr: np.ndarray | None, n: int, default: str = "") -> np.ndarray:
    if arr is None:
        return np.full(n, default, dtype=object)
    return np.asarray(arr).astype(str)


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_weak_pair(raw: str) -> tuple[str, str]:
    if "=>" in raw:
        left, right = raw.split("=>", 1)
    elif ":" in raw:
        left, right = raw.split(":", 1)
    else:
        raise ValueError(f"weak pair must use 'A=>B': {raw}")
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"empty weak pair side: {raw}")
    return left, right


def discover_paths(corpus: str, archetypes: set[str], score_bands: list[str]) -> list[tuple[str, str]]:
    root = Path(corpus)
    if not root.exists():
        raise FileNotFoundError(corpus)
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


def count_to_row(counts: Counter) -> dict[str, int | float | str]:
    games = int(counts["games"])
    wins = int(counts["wins"])
    decisions = int(counts["decisions"])
    win_decisions = int(counts["win_decisions"])
    return {
        "games": games,
        "wins": wins,
        "losses": int(counts["losses"]),
        "draws": int(counts["draws"]),
        "decisions": decisions,
        "win_decisions": win_decisions,
        "loss_decisions": int(counts["loss_decisions"]),
        "draw_decisions": int(counts["draw_decisions"]),
        "game_wr": f"{(wins / games) if games else 0.0:.6f}",
        "decision_win_share": f"{(win_decisions / decisions) if decisions else 0.0:.6f}",
    }


def recommendation(
    counts: Counter,
    *,
    min_games: int,
    min_wins: int,
    min_win_decisions: int,
    min_wr: float,
) -> str:
    games = int(counts["games"])
    wins = int(counts["wins"])
    win_decisions = int(counts["win_decisions"])
    wr = wins / max(games, 1)
    if games >= min_games and wins >= min_wins and win_decisions >= min_win_decisions and wr >= min_wr:
        return "enough_success_teacher"
    if wins >= min_wins and win_decisions >= min_win_decisions:
        return "enough_wins_but_low_rate"
    if wins > 0 and win_decisions > 0:
        return "sparse_success_teacher"
    return "generate_success_needed"


def quality_score(
    counts: Counter,
    *,
    avg_score: float,
    support_games: int,
    support_wins: int,
    support_win_decisions: int,
) -> float:
    games = int(counts["games"])
    wins = int(counts["wins"])
    win_decisions = int(counts["win_decisions"])
    wr = wins / max(games, 1)
    game_term = math.sqrt(min(games / max(support_games, 1), 1.0))
    win_term = math.sqrt(min(wins / max(support_wins, 1), 1.0))
    decision_term = math.sqrt(min(win_decisions / max(support_win_decisions, 1), 1.0))
    score_term = 0.85 + min(max(avg_score, 0.0), 1500.0) / 10000.0
    return wr * (0.20 + 0.80 * game_term) * (0.25 + 0.75 * win_term) * (
        0.35 + 0.65 * decision_term
    ) * score_term * math.log1p(wins)


def selected_pair(
    arch: str,
    opp_arch: str,
    pairs: set[tuple[str, str]],
    selected_arches: set[str],
    selected_opps: set[str],
) -> bool:
    if pairs:
        return (arch, opp_arch) in pairs
    if selected_arches and arch.lower() not in selected_arches:
        return False
    if selected_opps and opp_arch.lower() not in selected_opps:
        return False
    return True


def scan(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    pairs = {parse_weak_pair(x) for x in args.weak_pair}
    pair_arches = {a for a, _ in pairs}
    selected_arches = normalized(args.archetype)
    selected_opps = normalized(args.opponent_archetype)
    path_arches = pair_arches | set(args.archetype)
    paths = discover_paths(args.corpus, path_arches, args.score_bands)
    if not paths:
        raise FileNotFoundError("no corpus .npz files found")

    pair_counts: dict[tuple[str, str], Counter] = defaultdict(blank_counts)
    teacher_counts: dict[tuple[str, str, str, str], Counter] = defaultdict(blank_counts)
    pair_scores: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    teacher_scores: dict[tuple[str, str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    seen_pair_games: set[tuple[str, int, str, str]] = set()
    seen_teacher_games: set[tuple[str, int, str, str, str, str]] = set()

    raw = kept = 0
    t0 = time.time()
    for path_i, (arch, path) in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        required = {"deck_sig", "team_name", "opponent_archetype", "won", "action", "ot", "min_c", "max_c"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path} missing metadata: {sorted(missing)}")
        n = len(data["board"])
        deck_sigs = as_str_array(data.get("deck_sig"), n)
        team_names = as_str_array(data.get("team_name"), n)
        opp_arches = as_str_array(data.get("opponent_archetype"), n)
        scores = np.asarray(data["score"], dtype=np.float32) if "score" in data else np.zeros(n, dtype=np.float32)
        opp_scores = (
            np.asarray(data["opponent_score"], dtype=np.float32)
            if "opponent_score" in data
            else np.zeros(n, dtype=np.float32)
        )
        won = np.asarray(data["won"], dtype=np.int8)
        draw = np.asarray(data["draw"], dtype=np.int8) if "draw" in data else np.zeros(n, dtype=np.int8)
        episode_id = as_str_array(data.get("episode_id"), n)
        player_index = (
            np.asarray(data["player_index"], dtype=np.int16) if "player_index" in data else np.zeros(n, dtype=np.int16)
        )

        for i in range(n):
            raw += 1
            opp_arch = str(opp_arches[i])
            if not selected_pair(arch, opp_arch, pairs, selected_arches, selected_opps):
                continue
            if label_status(data, i, args.include_empty) != "keep":
                continue
            kept += 1
            pair_key = (arch, opp_arch)
            teacher_key = (arch, opp_arch, str(deck_sigs[i]), str(team_names[i]))
            outcome = outcome_at(won, draw, i)
            add_outcome(pair_counts[pair_key], outcome, decisions=True)
            add_outcome(teacher_counts[teacher_key], outcome, decisions=True)

            game_base = (str(episode_id[i]), int(player_index[i]))
            pair_game = (*game_base, arch, opp_arch)
            if pair_game not in seen_pair_games:
                seen_pair_games.add(pair_game)
                add_outcome(pair_counts[pair_key], outcome)
                pair_scores[pair_key][0] += as_float(scores[i])
                pair_scores[pair_key][1] += as_float(opp_scores[i])
                pair_scores[pair_key][2] += 1.0
                pair_scores[pair_key][3] += 1.0 if outcome == "win" else 0.0

            teacher_game = (*game_base, arch, opp_arch, str(deck_sigs[i]), str(team_names[i]))
            if teacher_game not in seen_teacher_games:
                seen_teacher_games.add(teacher_game)
                add_outcome(teacher_counts[teacher_key], outcome)
                teacher_scores[teacher_key][0] += as_float(scores[i])
                teacher_scores[teacher_key][1] += as_float(opp_scores[i])
                teacher_scores[teacher_key][2] += 1.0
                teacher_scores[teacher_key][3] += 1.0 if outcome == "win" else 0.0

        if args.progress_every and (path_i == 1 or path_i % args.progress_every == 0 or path_i == len(paths)):
            rate = raw / max(time.time() - t0, 1e-9)
            print(
                f"scanned {path_i}/{len(paths)} raw={raw} kept={kept} "
                f"pairs={len(pair_counts)} teachers={len(teacher_counts)} rate={rate:.0f}/s",
                flush=True,
            )
    return pair_counts, teacher_counts, pair_scores, teacher_scores


def score_avgs(score_stats: dict, key: tuple) -> tuple[float, float]:
    score_sum, opp_sum, n_games, _win_games = score_stats.get(key, [0.0, 0.0, 0.0, 0.0])
    denom = max(float(n_games), 1.0)
    return score_sum / denom, opp_sum / denom


def build_pair_rows(args: argparse.Namespace, pair_counts: dict, pair_scores: dict) -> list[dict]:
    rows: list[dict] = []
    for pair_key, counts in pair_counts.items():
        arch, opp_arch = pair_key
        avg_score, avg_opp_score = score_avgs(pair_scores, pair_key)
        row = {"archetype": arch, "opponent_archetype": opp_arch}
        row.update(count_to_row(counts))
        row["avg_score"] = f"{avg_score:.3f}"
        row["avg_opponent_score"] = f"{avg_opp_score:.3f}"
        row["recommendation"] = recommendation(
            counts,
            min_games=args.min_pair_games,
            min_wins=args.min_pair_wins,
            min_win_decisions=args.min_pair_win_decisions,
            min_wr=args.min_pair_wr,
        )
        rows.append(row)
    rows.sort(key=lambda r: (float(r["game_wr"]), -int(r["games"]), r["archetype"], r["opponent_archetype"]))
    return rows


def build_teacher_rows(
    args: argparse.Namespace,
    pair_counts: dict,
    teacher_counts: dict,
    pair_scores: dict,
    teacher_scores: dict,
) -> list[dict]:
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for teacher_key, counts in teacher_counts.items():
        arch, opp_arch, deck_sig, team_name = teacher_key
        games = int(counts["games"])
        wins = int(counts["wins"])
        decisions = int(counts["decisions"])
        if games < args.min_games or wins < args.min_wins or decisions < args.min_decisions:
            continue
        pair_key = (arch, opp_arch)
        pair = pair_counts[pair_key]
        avg_score, avg_opp_score = score_avgs(teacher_scores, teacher_key)
        pair_avg_score, _pair_avg_opp = score_avgs(pair_scores, pair_key)
        q = quality_score(
            counts,
            avg_score=avg_score or pair_avg_score,
            support_games=args.support_games,
            support_wins=args.support_wins,
            support_win_decisions=args.support_win_decisions,
        )
        row = {
            "archetype": arch,
            "opponent_archetype": opp_arch,
            "deck_sig": deck_sig,
            "team_name": team_name,
        }
        row.update(count_to_row(counts))
        row["avg_score"] = f"{avg_score:.3f}"
        row["avg_opponent_score"] = f"{avg_opp_score:.3f}"
        row["pair_games"] = int(pair["games"])
        row["pair_wins"] = int(pair["wins"])
        row["pair_game_wr"] = f"{(int(pair['wins']) / max(int(pair['games']), 1)):.6f}"
        row["pair_win_decisions"] = int(pair["win_decisions"])
        row["share_pair_wins"] = f"{(wins / max(int(pair['wins']), 1)):.6f}"
        row["quality_score"] = f"{q:.6f}"
        row["recommendation"] = recommendation(
            counts,
            min_games=args.min_teacher_games,
            min_wins=args.min_teacher_wins,
            min_win_decisions=args.min_teacher_win_decisions,
            min_wr=args.min_teacher_wr,
        )
        by_pair[pair_key].append(row)

    rows: list[dict] = []
    for pair_key in sorted(by_pair):
        group = by_pair[pair_key]
        group.sort(
            key=lambda r: (
                float(r["quality_score"]),
                float(r["game_wr"]),
                int(r["wins"]),
                int(r["win_decisions"]),
                int(r["games"]),
            ),
            reverse=True,
        )
        if args.top_per_pair:
            group = group[: args.top_per_pair]
        for rank, row in enumerate(group, 1):
            row["rank"] = rank
            rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--score-bands", nargs="*", default=[])
    p.add_argument("--weak-pair", action="append", default=[], help="repeatable, e.g. 'Marnie Grimmsnarl=>Teal Mask Ogerpon'")
    p.add_argument("--archetype", action="append", default=[], help="scan these candidate archetypes; repeatable")
    p.add_argument("--opponent-archetype", action="append", default=[], help="scan these opponent archetypes; repeatable")
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--min-games", type=int, default=1, help="minimum games for a teacher row to be written")
    p.add_argument("--min-wins", type=int, default=0, help="minimum wins for a teacher row to be written")
    p.add_argument("--min-decisions", type=int, default=1, help="minimum decisions for a teacher row to be written")
    p.add_argument("--top-per-pair", type=int, default=20, help="0 writes all teacher rows")
    p.add_argument("--min-pair-games", type=int, default=80)
    p.add_argument("--min-pair-wins", type=int, default=20)
    p.add_argument("--min-pair-win-decisions", type=int, default=1200)
    p.add_argument("--min-pair-wr", type=float, default=0.20)
    p.add_argument("--min-teacher-games", type=int, default=20)
    p.add_argument("--min-teacher-wins", type=int, default=8)
    p.add_argument("--min-teacher-win-decisions", type=int, default=500)
    p.add_argument("--min-teacher-wr", type=float, default=0.25)
    p.add_argument("--support-games", type=int, default=80)
    p.add_argument("--support-wins", type=int, default=20)
    p.add_argument("--support-win-decisions", type=int, default=1200)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-pair-csv", required=True)
    args = p.parse_args()

    pair_counts, teacher_counts, pair_scores, teacher_scores = scan(args)
    pair_rows = build_pair_rows(args, pair_counts, pair_scores)
    teacher_rows = build_teacher_rows(args, pair_counts, teacher_counts, pair_scores, teacher_scores)
    write_csv(args.out_pair_csv, pair_rows, PAIR_FIELDS)
    write_csv(args.out_csv, teacher_rows, TEACHER_FIELDS)

    for row in teacher_rows[: min(20, len(teacher_rows))]:
        print(
            f"{row['archetype']} vs {row['opponent_archetype']} "
            f"#{row['rank']} sig={row['deck_sig'][:12]} team={row['team_name']} "
            f"wr={row['game_wr']} wins={row['wins']}/{row['games']} q={row['quality_score']} "
            f"rec={row['recommendation']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
