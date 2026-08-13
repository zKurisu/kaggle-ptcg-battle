#!/usr/bin/env python3
"""Build lightweight per-game trajectory targets for hierarchical BC training."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_plans import PLANS

MAIN = 0
PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14
PRESSING_TYPES = {PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK}
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
DUSKULL = 131
DUSCLOPS = 132
DUSKNOIR = 133
MUNKIDORI = 112
DRAGAPULT_LINE = {DREEPY, DRAKLOAK, DRAGAPULT_EX}
DRAGAPULT_COUNTER_ENGINE = {DUSKULL, DUSCLOPS, DUSKNOIR, MUNKIDORI}
DAMAGE_COUNTER_CONTEXTS = {13, 14, 16}

FIELDS = [
    "game_key",
    "episode_id",
    "player_index",
    "archetype",
    "deck_sig",
    "team_name",
    "opponent_archetype",
    "opponent_deck_sig",
    "opponent_team_name",
    "outcome",
    "score",
    "opponent_score",
    "decisions",
    "max_turn",
    "main_decisions",
    "attack_count",
    "attach_count",
    "evolve_count",
    "ability_count",
    "play_count",
    "retreat_count",
    "end_count",
    "early_end_count",
    "first_attack_turn",
    "attack_by_4",
    "attack_by_6",
    "attack_by_8",
    "active_primary_turns",
    "bench_primary_turns",
    "first_primary_active_turn",
    "first_primary_board_turn",
    "active_secondary_turns",
    "bench_secondary_turns",
    "first_secondary_active_turn",
    "first_secondary_board_turn",
    "first_setup_board_turn",
    "first_engine_board_turn",
    "primary_active_by_2",
    "primary_active_by_4",
    "primary_active_by_6",
    "primary_board_by_2",
    "primary_board_by_4",
    "primary_board_by_6",
    "secondary_active_by_2",
    "secondary_active_by_4",
    "secondary_active_by_6",
    "secondary_board_by_2",
    "secondary_board_by_4",
    "secondary_board_by_6",
    "setup_board_by_2",
    "setup_board_by_4",
    "setup_board_by_6",
    "engine_board_by_2",
    "engine_board_by_4",
    "engine_board_by_6",
    "outcome_win",
    "outcome_loss",
    "outcome_draw",
    "no_early_end",
    "pressing_main_rate",
    "primary_board_turn_norm",
    "primary_active_turn_norm",
    "attack_turn_norm",
    "setup_success",
    "tempo_success",
    "strategy_success",
    "strategy_weight",
    "first_dreepy_board_turn",
    "first_drakloak_board_turn",
    "first_dragapult_board_turn",
    "first_dragapult_evolve_turn",
    "first_dusknoir_line_board_turn",
    "first_dragapult_attack_turn",
    "drakloak_ability_count",
    "dragapult_evolve_count",
    "damage_counter_count",
    "dreepy_board_by_2",
    "drakloak_board_by_4",
    "dragapult_board_by_6",
    "dusknoir_line_by_6",
    "dragapult_attack_by_6",
    "damage_counter_used",
    "drakloak_before_dragapult_evolve",
    "dragapult_strategy_success",
]


def clean_arch(name: str) -> str:
    return name.replace(" ", "_")


def filter_paths_by_date(paths: list[str], *, date_from: str = "", date_to: str = "") -> list[str]:
    date_from = str(date_from or "").strip()
    date_to = str(date_to or "").strip()
    if not date_from and not date_to:
        return paths
    out: list[str] = []
    for path in paths:
        m = DATE_RE.search(os.path.basename(path))
        day = m.group(1) if m else ""
        if date_from and day and day < date_from:
            continue
        if date_to and day and day > date_to:
            continue
        out.append(path)
    return out


def discover_paths(
    corpus: str,
    archetype: str,
    score_bands: list[str],
    *,
    date_from: str = "",
    date_to: str = "",
) -> list[str]:
    root = Path(corpus) / clean_arch(archetype)
    if not root.exists():
        return []
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(str(root / band.replace(" ", "_") / "*.npz"))))
    if not score_bands:
        paths = sorted(glob.glob(str(root / "*" / "*.npz")))
    return filter_paths_by_date(paths, date_from=date_from, date_to=date_to)


def fnum(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def inum(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def lower_set(values: list[str]) -> set[str]:
    return {str(x).lower() for x in values if str(x)}


def outcome_at(data: dict[str, np.ndarray], i: int) -> str:
    if "won" in data and int(data["won"][i]) == 1:
        return "win"
    if "draw" in data and int(data["draw"][i]) == 1:
        return "draw"
    return "loss"


def turn_context(data: dict[str, np.ndarray], i: int) -> tuple[int, int]:
    feats = np.asarray(data["feats"][i], dtype=np.float32)
    turn = inum(fnum(feats[0]) * 30.0) if len(feats) > 0 else 0
    ctx = inum(fnum(feats[17]) * 64.0) if len(feats) > 17 else -1
    return turn, ctx


def first_type(data: dict[str, np.ndarray], i: int) -> int:
    action = np.asarray(data["action"][i], dtype=np.int64)
    if len(action) == 0:
        return -1
    first = int(action[0])
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    if not 0 <= first < len(ot):
        return -1
    return int(ot[first])


def first_action(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int, int]:
    action = np.asarray(data["action"][i], dtype=np.int64)
    if len(action) == 0:
        return -1, 0, 0, -1
    first = int(action[0])
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    if not 0 <= first < len(ot):
        return -1, 0, 0, -1
    oc = np.asarray(data["oc"][i], dtype=np.int64) if "oc" in data else np.zeros(0, dtype=np.int64)
    oc2 = np.asarray(data["oc2"][i], dtype=np.int64) if "oc2" in data else np.zeros(0, dtype=np.int64)
    feats = np.asarray(data["feats"][i], dtype=np.float32)
    ctx = inum(fnum(feats[17]) * 64.0) if len(feats) > 17 else -1
    return (
        int(ot[first]),
        int(oc[first]) if first < len(oc) else 0,
        int(oc2[first]) if first < len(oc2) else 0,
        ctx,
    )


def option_types(data: dict[str, np.ndarray], i: int) -> set[int]:
    try:
        return {int(x) for x in np.asarray(data["ot"][i], dtype=np.int64).tolist()}
    except Exception:
        return set()


def row_matches(
    data: dict[str, np.ndarray],
    i: int,
    *,
    deck_sigs: set[str],
    team_names: set[str],
    opponent_archetypes: set[str],
    opponent_deck_sigs: set[str],
    opponent_team_names: set[str],
) -> bool:
    if deck_sigs and str(data["deck_sig"][i]) not in deck_sigs:
        return False
    if team_names and str(data["team_name"][i]).lower() not in team_names:
        return False
    if opponent_archetypes and str(data["opponent_archetype"][i]).lower() not in opponent_archetypes:
        return False
    if opponent_deck_sigs and str(data["opponent_deck_sig"][i]) not in opponent_deck_sigs:
        return False
    if opponent_team_names and str(data["opponent_team_name"][i]).lower() not in opponent_team_names:
        return False
    return True


def new_row(args: argparse.Namespace, data: dict[str, np.ndarray], i: int) -> dict:
    episode_id = str(data["episode_id"][i])
    player_index = str(data["player_index"][i])
    return {
        "game_key": f"{episode_id}:{player_index}",
        "episode_id": episode_id,
        "player_index": player_index,
        "archetype": args.archetype,
        "deck_sig": str(data["deck_sig"][i]) if "deck_sig" in data else "",
        "team_name": str(data["team_name"][i]) if "team_name" in data else "",
        "opponent_archetype": str(data["opponent_archetype"][i]) if "opponent_archetype" in data else "",
        "opponent_deck_sig": str(data["opponent_deck_sig"][i]) if "opponent_deck_sig" in data else "",
        "opponent_team_name": str(data["opponent_team_name"][i]) if "opponent_team_name" in data else "",
        "outcome": outcome_at(data, i),
        "score": fnum(data["score"][i]) if "score" in data else 0.0,
        "opponent_score": fnum(data["opponent_score"][i]) if "opponent_score" in data else 0.0,
        "decisions": 0,
        "max_turn": 0,
        "main_decisions": 0,
        "attack_count": 0,
        "attach_count": 0,
        "evolve_count": 0,
        "ability_count": 0,
        "play_count": 0,
        "retreat_count": 0,
        "end_count": 0,
        "early_end_count": 0,
        "first_attack_turn": 999,
        "active_primary_turns": 0,
        "bench_primary_turns": 0,
        "first_primary_active_turn": 999,
        "first_primary_board_turn": 999,
        "active_secondary_turns": 0,
        "bench_secondary_turns": 0,
        "first_secondary_active_turn": 999,
        "first_secondary_board_turn": 999,
        "first_setup_board_turn": 999,
        "first_engine_board_turn": 999,
        "first_dreepy_board_turn": 999,
        "first_drakloak_board_turn": 999,
        "first_dragapult_board_turn": 999,
        "first_dragapult_evolve_turn": 999,
        "first_dusknoir_line_board_turn": 999,
        "first_dragapult_attack_turn": 999,
        "drakloak_ability_count": 0,
        "dragapult_evolve_count": 0,
        "damage_counter_count": 0,
        "_saw_drakloak_ability_before_dragapult_evolve": 0,
    }


def finalize(row: dict) -> dict:
    row["attack_by_4"] = int(row["first_attack_turn"] <= 4)
    row["attack_by_6"] = int(row["first_attack_turn"] <= 6)
    row["attack_by_8"] = int(row["first_attack_turn"] <= 8)
    row["primary_active_by_2"] = int(row["first_primary_active_turn"] <= 2)
    row["primary_active_by_4"] = int(row["first_primary_active_turn"] <= 4)
    row["primary_active_by_6"] = int(row["first_primary_active_turn"] <= 6)
    row["primary_board_by_2"] = int(row["first_primary_board_turn"] <= 2)
    row["primary_board_by_4"] = int(row["first_primary_board_turn"] <= 4)
    row["primary_board_by_6"] = int(row["first_primary_board_turn"] <= 6)
    row["secondary_active_by_2"] = int(row["first_secondary_active_turn"] <= 2)
    row["secondary_active_by_4"] = int(row["first_secondary_active_turn"] <= 4)
    row["secondary_active_by_6"] = int(row["first_secondary_active_turn"] <= 6)
    row["secondary_board_by_2"] = int(row["first_secondary_board_turn"] <= 2)
    row["secondary_board_by_4"] = int(row["first_secondary_board_turn"] <= 4)
    row["secondary_board_by_6"] = int(row["first_secondary_board_turn"] <= 6)
    row["setup_board_by_2"] = int(row["first_setup_board_turn"] <= 2)
    row["setup_board_by_4"] = int(row["first_setup_board_turn"] <= 4)
    row["setup_board_by_6"] = int(row["first_setup_board_turn"] <= 6)
    row["engine_board_by_2"] = int(row["first_engine_board_turn"] <= 2)
    row["engine_board_by_4"] = int(row["first_engine_board_turn"] <= 4)
    row["engine_board_by_6"] = int(row["first_engine_board_turn"] <= 6)
    row["outcome_win"] = int(row.get("outcome") == "win")
    row["outcome_loss"] = int(row.get("outcome") == "loss")
    row["outcome_draw"] = int(row.get("outcome") == "draw")
    row["no_early_end"] = int(int(row.get("early_end_count", 0)) == 0)
    main = max(int(row.get("main_decisions", 0)), 1)
    pressing = (
        int(row.get("attack_count", 0))
        + int(row.get("attach_count", 0))
        + int(row.get("evolve_count", 0))
        + int(row.get("ability_count", 0))
        + int(row.get("play_count", 0))
        + int(row.get("retreat_count", 0))
    )
    row["pressing_main_rate"] = min(1.0, pressing / float(main))
    # Bounded continuous timing signals for the plan head. 1.0 means early,
    # 0.0 means never/very late. These are deliberately coarse and game-level.
    row["primary_board_turn_norm"] = max(0.0, 1.0 - min(int(row["first_primary_board_turn"]), 12) / 12.0)
    row["primary_active_turn_norm"] = max(0.0, 1.0 - min(int(row["first_primary_active_turn"]), 12) / 12.0)
    row["attack_turn_norm"] = max(0.0, 1.0 - min(int(row["first_attack_turn"]), 12) / 12.0)
    row["setup_success"] = int(
        (int(row["primary_board_by_4"]) == 1 or int(row["setup_board_by_4"]) == 1)
        and int(row["no_early_end"]) == 1
        and float(row["pressing_main_rate"]) >= 0.35
    )
    row["tempo_success"] = int(
        int(row["attack_by_6"]) == 1
        and int(row["no_early_end"]) == 1
        and float(row["pressing_main_rate"]) >= 0.40
    )
    row["strategy_success"] = int(
        int(row["outcome_win"]) == 1
        and (int(row["setup_success"]) == 1 or int(row["tempo_success"]) == 1)
    )
    row["dreepy_board_by_2"] = int(int(row.get("first_dreepy_board_turn", 999)) <= 2)
    row["drakloak_board_by_4"] = int(int(row.get("first_drakloak_board_turn", 999)) <= 4)
    row["dragapult_board_by_6"] = int(int(row.get("first_dragapult_board_turn", 999)) <= 6)
    row["dusknoir_line_by_6"] = int(int(row.get("first_dusknoir_line_board_turn", 999)) <= 6)
    row["dragapult_attack_by_6"] = int(int(row.get("first_dragapult_attack_turn", 999)) <= 6)
    row["damage_counter_used"] = int(int(row.get("damage_counter_count", 0)) > 0)
    row["drakloak_before_dragapult_evolve"] = int(
        int(row.get("_saw_drakloak_ability_before_dragapult_evolve", 0)) > 0
    )
    row["dragapult_strategy_success"] = int(
        int(row["outcome_win"]) == 1
        and int(row["no_early_end"]) == 1
        and int(row["dreepy_board_by_2"]) == 1
        and (
            int(row["drakloak_board_by_4"]) == 1
            or int(row["dragapult_board_by_6"]) == 1
            or int(row["dragapult_attack_by_6"]) == 1
        )
        and (
            int(row["damage_counter_used"]) == 1
            or int(row["dragapult_attack_by_6"]) == 1
            or int(row["dusknoir_line_by_6"]) == 1
        )
    )
    # A numeric column for whole-game sample weighting. It intentionally keeps
    # losing trajectories present, but gives strongest mass to wins that also
    # exhibit coherent setup/tempo signals.
    weight = 1.0
    if int(row["outcome_win"]) == 1:
        weight *= 1.8
    elif int(row["outcome_draw"]) == 1:
        weight *= 0.9
    else:
        weight *= 0.45
    if int(row["setup_success"]) == 1:
        weight *= 1.25
    if int(row["tempo_success"]) == 1:
        weight *= 1.20
    if int(row.get("dragapult_strategy_success", 0)) == 1:
        weight *= 1.50
    if int(row["no_early_end"]) == 0:
        weight *= 0.75
    row["strategy_weight"] = min(weight, 4.0)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="*", default=[])
    p.add_argument("--date-from", default="",
                   help="keep only corpus npz files whose filename date is >= YYYY-MM-DD")
    p.add_argument("--date-to", default="",
                   help="keep only corpus npz files whose filename date is <= YYYY-MM-DD")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--opponent-team-name", action="append", default=[])
    p.add_argument("--progress-every", type=int, default=2)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    paths = discover_paths(
        args.corpus,
        args.archetype,
        args.score_bands,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    if not paths:
        raise FileNotFoundError("no corpus .npz files found")
    deck_sigs = {str(x) for x in args.deck_sig if str(x)}
    team_names = lower_set(args.team_name)
    opponent_archetypes = lower_set(args.opponent_archetype)
    opponent_deck_sigs = {str(x) for x in args.opponent_deck_sig if str(x)}
    opponent_team_names = lower_set(args.opponent_team_name)
    plan = PLANS.get(args.archetype)
    primary = set(int(x) for x in (plan.primary_attackers if plan else []))
    secondary = set(int(x) for x in (plan.secondary_attackers if plan else []))
    setup_basics = set(int(x) for x in (plan.setup_basics if plan else []))
    engine_cards = set(int(x) for x in (plan.engine_cards if plan else []))

    rows: dict[str, dict] = {}
    raw = kept = 0
    t0 = time.time()
    for path_i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        required = {"episode_id", "player_index", "deck_sig", "team_name", "opponent_archetype", "opponent_deck_sig"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path} missing metadata: {sorted(missing)}")
        n = len(data["board"])
        for i in range(n):
            raw += 1
            if not row_matches(
                data,
                i,
                deck_sigs=deck_sigs,
                team_names=team_names,
                opponent_archetypes=opponent_archetypes,
                opponent_deck_sigs=opponent_deck_sigs,
                opponent_team_names=opponent_team_names,
            ):
                continue
            kept += 1
            key = f"{data['episode_id'][i]}:{data['player_index'][i]}"
            row = rows.get(key)
            if row is None:
                row = new_row(args, data, i)
                rows[key] = row
            turn, ctx = turn_context(data, i)
            typ, first_card, first_card2, action_ctx = first_action(data, i)
            row["decisions"] += 1
            row["max_turn"] = max(int(row["max_turn"]), turn)
            if ctx == MAIN:
                row["main_decisions"] += 1
                if typ == END and option_types(data, i) & PRESSING_TYPES:
                    row["early_end_count"] += 1
            if typ == ATTACK:
                row["attack_count"] += 1
                row["first_attack_turn"] = min(int(row["first_attack_turn"]), turn)
            elif typ == ATTACH:
                row["attach_count"] += 1
            elif typ == EVOLVE:
                row["evolve_count"] += 1
            elif typ == ABILITY:
                row["ability_count"] += 1
            elif typ == PLAY:
                row["play_count"] += 1
            elif typ == RETREAT:
                row["retreat_count"] += 1
            elif typ == END:
                row["end_count"] += 1

            board = np.asarray(data["board"][i], dtype=np.int64)
            active = int(board[0]) if len(board) > 0 else 0
            bench = [int(x) for x in board[1:6].tolist() if int(x) > 0]
            board_set = {active, *bench}
            board_set.discard(0)
            if primary:
                if active in primary:
                    row["active_primary_turns"] += 1
                    row["first_primary_active_turn"] = min(int(row["first_primary_active_turn"]), turn)
                if any(x in primary for x in bench):
                    row["bench_primary_turns"] += 1
                if active in primary or any(x in primary for x in bench):
                    row["first_primary_board_turn"] = min(int(row["first_primary_board_turn"]), turn)
            if secondary:
                if active in secondary:
                    row["active_secondary_turns"] += 1
                    row["first_secondary_active_turn"] = min(int(row["first_secondary_active_turn"]), turn)
                if any(x in secondary for x in bench):
                    row["bench_secondary_turns"] += 1
                if board_set & secondary:
                    row["first_secondary_board_turn"] = min(int(row["first_secondary_board_turn"]), turn)
            if setup_basics and board_set & setup_basics:
                row["first_setup_board_turn"] = min(int(row["first_setup_board_turn"]), turn)
            if engine_cards and board_set & engine_cards:
                row["first_engine_board_turn"] = min(int(row["first_engine_board_turn"]), turn)
            if args.archetype == "Dragapult":
                if DREEPY in board_set:
                    row["first_dreepy_board_turn"] = min(int(row["first_dreepy_board_turn"]), turn)
                if DRAKLOAK in board_set:
                    row["first_drakloak_board_turn"] = min(int(row["first_drakloak_board_turn"]), turn)
                if DRAGAPULT_EX in board_set:
                    row["first_dragapult_board_turn"] = min(int(row["first_dragapult_board_turn"]), turn)
                if board_set & DRAGAPULT_COUNTER_ENGINE:
                    row["first_dusknoir_line_board_turn"] = min(
                        int(row["first_dusknoir_line_board_turn"]), turn
                    )
                if typ == ABILITY and first_card == DRAKLOAK:
                    row["drakloak_ability_count"] += 1
                    if int(row.get("first_dragapult_evolve_turn", 999)) == 999:
                        row["_saw_drakloak_ability_before_dragapult_evolve"] = 1
                if typ == EVOLVE and (first_card == DRAGAPULT_EX or first_card2 == DRAGAPULT_EX):
                    row["dragapult_evolve_count"] += 1
                    row["first_dragapult_evolve_turn"] = min(
                        int(row["first_dragapult_evolve_turn"]), turn
                    )
                if typ == ATTACK and active == DRAGAPULT_EX:
                    row["first_dragapult_attack_turn"] = min(
                        int(row["first_dragapult_attack_turn"]), turn
                    )
                if ctx in DAMAGE_COUNTER_CONTEXTS or action_ctx in DAMAGE_COUNTER_CONTEXTS:
                    row["damage_counter_count"] += 1
        if args.progress_every and (path_i == 1 or path_i % args.progress_every == 0 or path_i == len(paths)):
            rate = raw / max(time.time() - t0, 1e-9)
            print(
                f"scanned {path_i}/{len(paths)} raw={raw} kept_decisions={kept} "
                f"games={len(rows)} rate={rate:.0f}/s",
                flush=True,
            )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_rows = [finalize(row) for row in rows.values()]
    final_rows.sort(key=lambda r: (str(r["episode_id"]), int(r["player_index"])))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in final_rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    wins = sum(r["outcome"] == "win" for r in final_rows)
    losses = sum(r["outcome"] == "loss" for r in final_rows)
    draws = sum(r["outcome"] == "draw" for r in final_rows)
    print(
        f"Wrote {out} games={len(final_rows)} wins={wins} losses={losses} draws={draws} "
        f"wr={wins / max(len(final_rows), 1):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
