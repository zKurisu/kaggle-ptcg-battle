#!/usr/bin/env python3
"""Mine trajectory-level strategy signals from extracted BC corpora.

The existing BC diagnostics are mostly single-decision reports. This tool groups
decisions back into games, compares winning and losing trajectories, and emits
strategy seeds that can drive trace, rule-probe, or rollout-teacher jobs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_plans import CARD_NAMES, PLANS


TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}

CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE",
    2: "SETUP_BENCH",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    13: "DAMAGE_COUNTER",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    35: "ATTACK",
    37: "EVOLVE",
    43: "ACTIVATE",
}

MAIN = 0
TO_ACTIVE = 4
ATTACH_FROM = 21
ATTACH_TO = 22
PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14
PRESSING_TYPES = {PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK}

GAME_FIELDS = [
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
    "primary_active_by_2",
    "primary_active_by_4",
    "primary_active_by_6",
    "primary_board_by_2",
    "primary_board_by_4",
    "primary_board_by_6",
    "first_token_sequence",
]

METRIC_FIELDS = [
    "metric",
    "win_mean",
    "loss_mean",
    "delta_win_minus_loss",
    "win_n",
    "loss_n",
    "priority",
]

EVENT_FIELDS = [
    "kind",
    "key",
    "win_games",
    "loss_games",
    "win_rate",
    "loss_rate",
    "delta_win_minus_loss",
    "win_count_per_game",
    "loss_count_per_game",
    "delta_count_per_game",
    "priority",
    "example_win_game",
    "example_loss_game",
]

SEED_FIELDS = [
    "seed_id",
    "archetype",
    "deck_sigs",
    "opponent_archetypes",
    "opponent_deck_sigs",
    "direction",
    "kind",
    "key",
    "priority",
    "win_rate",
    "loss_rate",
    "delta_win_minus_loss",
    "recommendation",
    "next_action",
]


def clean_arch(name: str) -> str:
    return name.replace(" ", "_")


def slug(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value[:max_len] or "x"


def card_name(card_id: int) -> str:
    if card_id <= 0:
        return ""
    return CARD_NAMES.get(card_id, str(card_id))


def card_label(card_id: int) -> str:
    name = card_name(card_id)
    return f"{card_id}:{name}" if name else str(card_id)


def type_name(opt_type: int) -> str:
    return TYPE_NAMES.get(opt_type, str(opt_type))


def context_name(context: int) -> str:
    return CONTEXT_NAMES.get(context, str(context))


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


def outcome_at(data: dict[str, np.ndarray], i: int) -> str:
    if int(data.get("won", [0])[i]) == 1:
        return "win"
    if "draw" in data and int(data["draw"][i]) == 1:
        return "draw"
    return "loss"


def discover_paths(corpus: str, archetype: str, score_bands: list[str]) -> list[str]:
    root = Path(corpus) / clean_arch(archetype)
    if not root.exists():
        return []
    if score_bands:
        paths: list[str] = []
        for band in score_bands:
            paths.extend(sorted(glob.glob(str(root / band.replace(" ", "_") / "*.npz"))))
        return paths
    return sorted(glob.glob(str(root / "*" / "*.npz")))


def parse_track_cards(args: argparse.Namespace) -> dict[int, str]:
    cards: dict[int, str] = {}
    plan = PLANS.get(args.archetype)
    if plan:
        for cid in (
            *plan.signature_ids,
            *plan.primary_attackers,
            *plan.secondary_attackers,
            *plan.setup_basics,
            *plan.evolution_chain,
            *plan.engine_cards,
            *plan.energy_accel,
            *plan.draw_search,
        ):
            if cid:
                cards[int(cid)] = card_name(int(cid)) or str(cid)
    for spec in args.track_card:
        if ":" in spec:
            cid, label = spec.split(":", 1)
        else:
            cid, label = spec, ""
        cid_i = int(cid)
        cards[cid_i] = label or card_name(cid_i) or str(cid_i)
    return cards


def parse_card_specs(specs: list[str]) -> dict[int, str]:
    cards: dict[int, str] = {}
    for spec in specs:
        if ":" in spec:
            cid, label = spec.split(":", 1)
        else:
            cid, label = spec, ""
        cid_i = int(cid)
        cards[cid_i] = label or card_name(cid_i) or str(cid_i)
    return cards


def parse_track_opponent_cards(args: argparse.Namespace) -> dict[int, str]:
    cards = parse_card_specs(args.track_opponent_card)
    opp_arches = [x for x in args.opponent_archetype if x]
    if len(opp_arches) == 1:
        plan = PLANS.get(opp_arches[0])
        if plan:
            for cid in (
                *plan.signature_ids,
                *plan.primary_attackers,
                *plan.secondary_attackers,
                *plan.setup_basics,
                *plan.evolution_chain,
                *plan.engine_cards,
                *plan.energy_accel,
                *plan.draw_search,
            ):
                if cid:
                    cards.setdefault(int(cid), card_name(int(cid)) or str(cid))
    return cards


def first_action(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int, int]:
    action = np.asarray(data["action"][i], dtype=np.int64)
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    oc = np.asarray(data["oc"][i], dtype=np.int64)
    oc2 = np.asarray(data["oc2"][i], dtype=np.int64) if "oc2" in data else np.zeros_like(oc)
    if len(action) == 0:
        return -1, -1, -1, -1
    first = int(action[0])
    if not 0 <= first < len(ot):
        return first, -1, -1, -1
    return first, int(ot[first]), int(oc[first]) if first < len(oc) else 0, int(oc2[first]) if first < len(oc2) else 0


def option_type_set(data: dict[str, np.ndarray], i: int) -> set[int]:
    try:
        return {int(x) for x in np.asarray(data["ot"][i], dtype=np.int64).tolist()}
    except Exception:
        return set()


def turn_context(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int]:
    feats = np.asarray(data["feats"][i], dtype=np.float32)
    turn = inum(fnum(feats[0]) * 30.0) if len(feats) > 0 else 0
    tac = inum(fnum(feats[1]) * 50.0) if len(feats) > 1 else 0
    ctx = inum(fnum(feats[17]) * 64.0) if len(feats) > 17 else -1
    return turn, tac, ctx


def board_cards(data: dict[str, np.ndarray], i: int) -> tuple[list[int], list[int]]:
    board = np.asarray(data["board"][i], dtype=np.int64)
    mine = [int(x) for x in board[:6].tolist()]
    opp = [int(x) for x in board[6:12].tolist()]
    return mine, opp


def row_matches(
    data: dict[str, np.ndarray],
    i: int,
    args: argparse.Namespace,
    deck_sigs: set[str],
    team_names: set[str],
    opp_arches: set[str],
    opp_sigs: set[str],
    opp_teams: set[str],
) -> bool:
    if deck_sigs and str(data["deck_sig"][i]) not in deck_sigs:
        return False
    if team_names and str(data["team_name"][i]).lower() not in team_names:
        return False
    if opp_arches and str(data["opponent_archetype"][i]).lower() not in opp_arches:
        return False
    if opp_sigs and str(data["opponent_deck_sig"][i]) not in opp_sigs:
        return False
    if opp_teams and str(data["opponent_team_name"][i]).lower() not in opp_teams:
        return False
    if args.winner_only and outcome_at(data, i) != "win":
        return False
    return True


def token_for_row(data: dict[str, np.ndarray], i: int, *, with_turn: bool) -> str:
    turn, _tac, ctx = turn_context(data, i)
    _first, typ, card, card2 = first_action(data, i)
    c_name = card_name(card)
    c2_name = card_name(card2)
    card_part = c_name or str(card) if card > 0 else ""
    if card2 > 0 and c2_name:
        card_part = f"{card_part}>{c2_name}" if card_part else c2_name
    parts = []
    if with_turn:
        parts.append(f"t{min(turn, 12)}")
    parts.extend([context_name(ctx), type_name(typ)])
    if card_part:
        parts.append(card_part)
    return ":".join(parts)


def update_first(mapping: dict[str, int], key: str, turn: int) -> None:
    if key not in mapping or turn < mapping[key]:
        mapping[key] = turn


def summarize_game(
    game_key: tuple,
    indices: list[tuple[dict[str, np.ndarray], int]],
    args: argparse.Namespace,
    track_cards: dict[int, str],
    track_opp_cards: dict[int, str],
) -> tuple[dict, Counter, Counter]:
    first_data, first_i = indices[0]
    outcome = outcome_at(first_data, first_i)
    episode_id, player_index = game_key[0], game_key[1]
    meta = {
        "game_key": f"{episode_id}:{player_index}",
        "episode_id": episode_id,
        "player_index": player_index,
        "archetype": args.archetype,
        "deck_sig": str(first_data["deck_sig"][first_i]) if "deck_sig" in first_data else "",
        "team_name": str(first_data["team_name"][first_i]) if "team_name" in first_data else "",
        "opponent_archetype": str(first_data["opponent_archetype"][first_i]) if "opponent_archetype" in first_data else "",
        "opponent_deck_sig": str(first_data["opponent_deck_sig"][first_i]) if "opponent_deck_sig" in first_data else "",
        "opponent_team_name": str(first_data["opponent_team_name"][first_i]) if "opponent_team_name" in first_data else "",
        "outcome": outcome,
        "score": fnum(first_data["score"][first_i]) if "score" in first_data else 0.0,
        "opponent_score": fnum(first_data["opponent_score"][first_i]) if "opponent_score" in first_data else 0.0,
    }

    counters = Counter()
    events = Counter()
    seq: list[str] = []
    max_turn = 0
    first_turns: dict[str, int] = {}
    primary = set()
    plan = PLANS.get(args.archetype)
    if plan:
        primary = set(int(x) for x in plan.primary_attackers)
    track_ids = set(track_cards)
    track_opp_ids = set(track_opp_cards)

    for data, i in indices:
        turn, _tac, ctx = turn_context(data, i)
        max_turn = max(max_turn, turn)
        _first, typ, card, _card2 = first_action(data, i)
        my_board, opp_board = board_cards(data, i)
        active = my_board[0] if my_board else 0
        bench = [x for x in my_board[1:] if x]
        on_board = [x for x in my_board if x]
        opp_active = opp_board[0] if opp_board else 0
        opp_bench = [x for x in opp_board[1:] if x]
        opp_on_board = [x for x in opp_board if x]

        if ctx == MAIN:
            counters["main_decisions"] += 1
            counters[f"type_{typ}"] += 1
            if typ == END and option_type_set(data, i) & PRESSING_TYPES:
                counters["early_end_count"] += 1
            events[f"main_type={type_name(typ)}"] += 1
            if card > 0:
                events[f"main_card={type_name(typ)}:{card_label(card)}"] += 1
        if ctx == TO_ACTIVE and card > 0:
            events[f"to_active={card_label(card)}"] += 1
        if ctx == ATTACH_FROM and card > 0:
            events[f"attach_from={card_label(card)}"] += 1
        if ctx == ATTACH_TO and card > 0:
            events[f"attach_to={card_label(card)}"] += 1
        for cid in track_opp_ids:
            label = card_label(cid)
            if opp_active == cid:
                events[f"opp_active={label}"] += 1
            if cid in opp_bench:
                events[f"opp_bench={label}"] += 1
            if cid in opp_on_board:
                events[f"opp_board={label}"] += 1

        if typ == ATTACK:
            counters["attack_count"] += 1
            update_first(first_turns, "first_attack_turn", turn)
        elif typ == ATTACH:
            counters["attach_count"] += 1
        elif typ == EVOLVE:
            counters["evolve_count"] += 1
        elif typ == ABILITY:
            counters["ability_count"] += 1
        elif typ == PLAY:
            counters["play_count"] += 1
        elif typ == RETREAT:
            counters["retreat_count"] += 1
        elif typ == END:
            counters["end_count"] += 1

        for cid in track_ids:
            label = slug(track_cards[cid], 30)
            if active == cid:
                counters[f"active_turns_{label}"] += 1
                update_first(first_turns, f"first_active_turn_{label}", turn)
            if cid in bench:
                counters[f"bench_turns_{label}"] += 1
                update_first(first_turns, f"first_bench_turn_{label}", turn)
            if cid in on_board:
                counters[f"board_turns_{label}"] += 1
                update_first(first_turns, f"first_board_turn_{label}", turn)
        for cid in track_opp_ids:
            label = slug(track_opp_cards[cid], 30)
            if opp_active == cid:
                counters[f"opp_active_turns_{label}"] += 1
                update_first(first_turns, f"first_opp_active_turn_{label}", turn)
            if cid in opp_bench:
                counters[f"opp_bench_turns_{label}"] += 1
                update_first(first_turns, f"first_opp_bench_turn_{label}", turn)
            if cid in opp_on_board:
                counters[f"opp_board_turns_{label}"] += 1
                update_first(first_turns, f"first_opp_board_turn_{label}", turn)
        if primary:
            if active in primary:
                counters["active_primary_turns"] += 1
                update_first(first_turns, "first_primary_active_turn", turn)
            if any(x in primary for x in bench):
                counters["bench_primary_turns"] += 1
            if any(x in primary for x in on_board):
                update_first(first_turns, "first_primary_board_turn", turn)

        if len(seq) < args.max_tokens and ctx in {MAIN, TO_ACTIVE, ATTACH_FROM, ATTACH_TO, 3, 7, 8, 43}:
            seq.append(token_for_row(data, i, with_turn=args.ngram_with_turn))

    row = {
        **meta,
        "decisions": len(indices),
        "max_turn": max_turn,
        "main_decisions": counters["main_decisions"],
        "attack_count": counters["attack_count"],
        "attach_count": counters["attach_count"],
        "evolve_count": counters["evolve_count"],
        "ability_count": counters["ability_count"],
        "play_count": counters["play_count"],
        "retreat_count": counters["retreat_count"],
        "end_count": counters["end_count"],
        "early_end_count": counters["early_end_count"],
        "first_attack_turn": first_turns.get("first_attack_turn", 999),
        "attack_by_4": int(first_turns.get("first_attack_turn", 999) <= 4),
        "attack_by_6": int(first_turns.get("first_attack_turn", 999) <= 6),
        "attack_by_8": int(first_turns.get("first_attack_turn", 999) <= 8),
        "active_primary_turns": counters["active_primary_turns"],
        "bench_primary_turns": counters["bench_primary_turns"],
        "first_primary_active_turn": first_turns.get("first_primary_active_turn", 999),
        "first_primary_board_turn": first_turns.get("first_primary_board_turn", 999),
        "primary_active_by_2": int(first_turns.get("first_primary_active_turn", 999) <= 2),
        "primary_active_by_4": int(first_turns.get("first_primary_active_turn", 999) <= 4),
        "primary_active_by_6": int(first_turns.get("first_primary_active_turn", 999) <= 6),
        "primary_board_by_2": int(first_turns.get("first_primary_board_turn", 999) <= 2),
        "primary_board_by_4": int(first_turns.get("first_primary_board_turn", 999) <= 4),
        "primary_board_by_6": int(first_turns.get("first_primary_board_turn", 999) <= 6),
        "first_token_sequence": " > ".join(seq[: args.sequence_preview]),
    }
    for cid, label in track_cards.items():
        key = slug(label, 30)
        row[f"active_turns_{key}"] = counters[f"active_turns_{key}"]
        row[f"bench_turns_{key}"] = counters[f"bench_turns_{key}"]
        row[f"board_turns_{key}"] = counters[f"board_turns_{key}"]
        row[f"first_active_turn_{key}"] = first_turns.get(f"first_active_turn_{key}", 999)
        row[f"first_bench_turn_{key}"] = first_turns.get(f"first_bench_turn_{key}", 999)
        row[f"first_board_turn_{key}"] = first_turns.get(f"first_board_turn_{key}", 999)
        row[f"board_by_4_{key}"] = int(first_turns.get(f"first_board_turn_{key}", 999) <= 4)
        row[f"active_by_4_{key}"] = int(first_turns.get(f"first_active_turn_{key}", 999) <= 4)
    for cid, label in track_opp_cards.items():
        key = slug(label, 30)
        row[f"opp_active_turns_{key}"] = counters[f"opp_active_turns_{key}"]
        row[f"opp_bench_turns_{key}"] = counters[f"opp_bench_turns_{key}"]
        row[f"opp_board_turns_{key}"] = counters[f"opp_board_turns_{key}"]
        row[f"first_opp_active_turn_{key}"] = first_turns.get(f"first_opp_active_turn_{key}", 999)
        row[f"first_opp_bench_turn_{key}"] = first_turns.get(f"first_opp_bench_turn_{key}", 999)
        row[f"first_opp_board_turn_{key}"] = first_turns.get(f"first_opp_board_turn_{key}", 999)
        row[f"opp_board_by_4_{key}"] = int(first_turns.get(f"first_opp_board_turn_{key}", 999) <= 4)
        row[f"opp_active_by_4_{key}"] = int(first_turns.get(f"first_opp_active_turn_{key}", 999) <= 4)

    event_counts = Counter(events)
    ngrams = Counter()
    for n in args.ngram:
        if n <= 0:
            continue
        for j in range(0, max(0, len(seq) - n + 1)):
            ngrams[f"ngram{n}=" + " > ".join(seq[j:j + n])] += 1
    return row, event_counts, ngrams


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def metric_gap_rows(game_rows: list[dict], min_games: int) -> list[dict]:
    wins = [r for r in game_rows if r["outcome"] == "win"]
    losses = [r for r in game_rows if r["outcome"] == "loss"]
    if len(wins) < min_games or len(losses) < min_games:
        return []
    skip = {
        "game_key", "episode_id", "player_index", "archetype", "deck_sig", "team_name",
        "opponent_archetype", "opponent_deck_sig", "opponent_team_name", "outcome",
        "first_token_sequence",
    }
    metrics = []
    for key in game_rows[0]:
        if key in skip:
            continue
        try:
            win_vals = [float(r[key]) for r in wins]
            loss_vals = [float(r[key]) for r in losses]
        except Exception:
            continue
        wm = mean(win_vals)
        lm = mean(loss_vals)
        delta = wm - lm
        pooled = math.sqrt(max(1e-9, mean([(x - wm) ** 2 for x in win_vals] + [(x - lm) ** 2 for x in loss_vals])))
        priority = abs(delta) / pooled
        metrics.append({
            "metric": key,
            "win_mean": wm,
            "loss_mean": lm,
            "delta_win_minus_loss": delta,
            "win_n": len(wins),
            "loss_n": len(losses),
            "priority": priority,
        })
    metrics.sort(key=lambda r: float(r["priority"]), reverse=True)
    return metrics


def event_gap_rows(
    kind: str,
    counters_by_game: dict[str, Counter],
    game_rows: list[dict],
    min_games: int,
    min_rate_gap: float,
) -> list[dict]:
    wins = [r for r in game_rows if r["outcome"] == "win"]
    losses = [r for r in game_rows if r["outcome"] == "loss"]
    if len(wins) < min_games or len(losses) < min_games:
        return []
    win_keys = [r["game_key"] for r in wins]
    loss_keys = [r["game_key"] for r in losses]
    keys = set()
    for c in counters_by_game.values():
        keys.update(c)
    rows = []
    for key in keys:
        win_present = [g for g in win_keys if counters_by_game[g].get(key, 0) > 0]
        loss_present = [g for g in loss_keys if counters_by_game[g].get(key, 0) > 0]
        win_rate = len(win_present) / max(len(win_keys), 1)
        loss_rate = len(loss_present) / max(len(loss_keys), 1)
        win_cpg = sum(counters_by_game[g].get(key, 0) for g in win_keys) / max(len(win_keys), 1)
        loss_cpg = sum(counters_by_game[g].get(key, 0) for g in loss_keys) / max(len(loss_keys), 1)
        rate_gap = win_rate - loss_rate
        cpg_gap = win_cpg - loss_cpg
        if abs(rate_gap) < min_rate_gap and abs(cpg_gap) < min_rate_gap:
            continue
        priority = abs(rate_gap) + 0.25 * abs(cpg_gap)
        rows.append({
            "kind": kind,
            "key": key,
            "win_games": len(win_present),
            "loss_games": len(loss_present),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "delta_win_minus_loss": rate_gap,
            "win_count_per_game": win_cpg,
            "loss_count_per_game": loss_cpg,
            "delta_count_per_game": cpg_gap,
            "priority": priority,
            "example_win_game": win_present[0] if win_present else "",
            "example_loss_game": loss_present[0] if loss_present else "",
        })
    rows.sort(key=lambda r: float(r["priority"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_row(row: dict, fields: list[str]) -> dict:
    out = {}
    for k in fields:
        v = row.get(k, "")
        if isinstance(v, float):
            out[k] = f"{v:.6f}"
        else:
            out[k] = v
    return out


def recommendation(row: dict) -> tuple[str, str]:
    key = row["key"]
    delta = float(row["delta_win_minus_loss"])
    if delta > 0:
        direction = "win_overrepresented"
        if key.startswith("opp_"):
            rec = "opponent_state_seen_in_wins__inspect_choke_timing"
        elif key.startswith("ngram"):
            rec = "distill_as_teacher_trajectory"
        else:
            rec = "promote_as_strategy_seed"
        next_action = "filter winning games containing this signal, then inspect surrounding sequence before distillation"
    else:
        direction = "loss_overrepresented"
        if key.startswith("opp_board=") or key.startswith("opp_active=") or key.startswith("opp_bench="):
            rec = "opponent_choke_point_candidate"
        elif "to_active=" in key:
            rec = "probe_active_selection_rule_or_teacher"
        elif "main_card=" in key or "main_type=" in key:
            rec = "avoid_blind_action_bias__inspect_surrounding_ngram"
        else:
            rec = "negative_strategy_signal"
        next_action = "find earlier window to prevent or punish this opponent board state"
    return direction, f"{rec}", next_action


def build_strategy_seeds(
    args: argparse.Namespace,
    event_rows: list[dict],
    ngram_rows: list[dict],
) -> list[dict]:
    seeds = []
    selected = []
    selected.extend(event_rows[: args.seed_top])
    selected.extend(ngram_rows[: args.seed_top])
    selected.sort(key=lambda r: float(r["priority"]), reverse=True)
    for i, row in enumerate(selected[: args.seed_top * 2], 1):
        direction, rec, next_action = recommendation(row)
        seeds.append({
            "seed_id": f"{slug(args.archetype, 24)}_{i:03d}_{slug(direction, 24)}",
            "archetype": args.archetype,
            "deck_sigs": ";".join(args.deck_sig),
            "opponent_archetypes": ";".join(args.opponent_archetype),
            "opponent_deck_sigs": ";".join(args.opponent_deck_sig),
            "direction": direction,
            "kind": row["kind"],
            "key": row["key"],
            "priority": row["priority"],
            "win_rate": row["win_rate"],
            "loss_rate": row["loss_rate"],
            "delta_win_minus_loss": row["delta_win_minus_loss"],
            "recommendation": rec,
            "next_action": next_action,
        })
    return seeds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="*", default=[])
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--opponent-team-name", action="append", default=[])
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--track-card", action="append", default=[], help="CARD_ID or CARD_ID:label; repeats")
    p.add_argument("--track-opponent-card", action="append", default=[], help="opponent CARD_ID or CARD_ID:label; repeats")
    p.add_argument("--ngram", type=int, action="append", default=[3])
    p.add_argument("--ngram-with-turn", action="store_true")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--sequence-preview", type=int, default=32)
    p.add_argument("--min-games", type=int, default=10)
    p.add_argument("--min-rate-gap", type=float, default=0.12)
    p.add_argument("--seed-top", type=int, default=20)
    p.add_argument("--limit-games", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    deck_sigs = {str(x) for x in args.deck_sig}
    team_names = {str(x).lower() for x in args.team_name}
    opp_arches = {str(x).lower() for x in args.opponent_archetype}
    opp_sigs = {str(x) for x in args.opponent_deck_sig}
    opp_teams = {str(x).lower() for x in args.opponent_team_name}

    paths = discover_paths(args.corpus, args.archetype, args.score_bands)
    if not paths:
        raise FileNotFoundError("no corpus .npz files found")

    groups: dict[tuple, list[tuple[dict[str, np.ndarray], int]]] = defaultdict(list)
    raw = kept = 0
    for path_i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        required = {"deck_sig", "team_name", "opponent_archetype", "opponent_deck_sig", "episode_id", "player_index", "won"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path} missing metadata: {sorted(missing)}")
        n = len(data["board"])
        for i in range(n):
            raw += 1
            if not row_matches(data, i, args, deck_sigs, team_names, opp_arches, opp_sigs, opp_teams):
                continue
            key = (
                str(data["episode_id"][i]),
                int(data["player_index"][i]),
                str(data["deck_sig"][i]),
                str(data["opponent_archetype"][i]),
                str(data["opponent_deck_sig"][i]),
            )
            groups[key].append((data, i))
            kept += 1
        if args.progress_every and (path_i == 1 or path_i % args.progress_every == 0 or path_i == len(paths)):
            print(f"scanned {path_i}/{len(paths)} files raw={raw} kept_decisions={kept} games={len(groups)}", flush=True)

    track_cards = parse_track_cards(args)
    track_opp_cards = parse_track_opponent_cards(args)
    game_rows = []
    event_by_game: dict[str, Counter] = {}
    ngram_by_game: dict[str, Counter] = {}
    for game_i, (key, indices) in enumerate(groups.items(), 1):
        if args.limit_games and game_i > args.limit_games:
            break
        row, events, ngrams = summarize_game(key, indices, args, track_cards, track_opp_cards)
        game_rows.append(row)
        event_by_game[row["game_key"]] = events
        ngram_by_game[row["game_key"]] = ngrams

    wins = sum(r["outcome"] == "win" for r in game_rows)
    losses = sum(r["outcome"] == "loss" for r in game_rows)
    draws = sum(r["outcome"] == "draw" for r in game_rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dynamic_game_fields = list(GAME_FIELDS)
    for cid, label in track_cards.items():
        key = slug(label, 30)
        for col in (
            f"active_turns_{key}", f"bench_turns_{key}", f"board_turns_{key}",
            f"first_active_turn_{key}", f"first_bench_turn_{key}", f"first_board_turn_{key}",
            f"board_by_4_{key}", f"active_by_4_{key}",
        ):
            if col not in dynamic_game_fields:
                dynamic_game_fields.append(col)
    for cid, label in track_opp_cards.items():
        key = slug(label, 30)
        for col in (
            f"opp_active_turns_{key}", f"opp_bench_turns_{key}", f"opp_board_turns_{key}",
            f"first_opp_active_turn_{key}", f"first_opp_bench_turn_{key}", f"first_opp_board_turn_{key}",
            f"opp_board_by_4_{key}", f"opp_active_by_4_{key}",
        ):
            if col not in dynamic_game_fields:
                dynamic_game_fields.append(col)
    write_csv(out / "game_trajectories.csv", game_rows, dynamic_game_fields)

    metric_rows = metric_gap_rows(game_rows, args.min_games)
    event_rows = event_gap_rows("event", event_by_game, game_rows, args.min_games, args.min_rate_gap)
    ngram_rows = event_gap_rows("ngram", ngram_by_game, game_rows, args.min_games, args.min_rate_gap)
    seed_rows = build_strategy_seeds(args, event_rows, ngram_rows)
    write_csv(out / "metric_gaps.csv", [fmt_row(r, METRIC_FIELDS) for r in metric_rows], METRIC_FIELDS)
    write_csv(out / "event_gaps.csv", [fmt_row(r, EVENT_FIELDS) for r in event_rows], EVENT_FIELDS)
    write_csv(out / "ngram_gaps.csv", [fmt_row(r, EVENT_FIELDS) for r in ngram_rows], EVENT_FIELDS)
    write_csv(out / "strategy_seeds.csv", [fmt_row(r, SEED_FIELDS) for r in seed_rows], SEED_FIELDS)

    print(f"Corpus: {args.corpus}")
    print(f"Archetype: {args.archetype}")
    print(f"Filters: deck_sig={args.deck_sig} opponent_archetype={args.opponent_archetype} opponent_deck_sig={args.opponent_deck_sig}")
    print(f"Decisions kept: {kept}/{raw}")
    print(f"Games: {len(game_rows)} wins={wins} losses={losses} draws={draws} wr={wins / max(len(game_rows), 1):.3f}")
    print(f"Wrote: {out}")
    print("\nTop metric gaps:")
    for row in metric_rows[: args.top]:
        print(
            f"  {row['metric']:<38} win={row['win_mean']:.3f} loss={row['loss_mean']:.3f} "
            f"delta={row['delta_win_minus_loss']:+.3f} pr={row['priority']:.3f}"
        )
    print("\nTop event gaps:")
    for row in event_rows[: args.top]:
        print(
            f"  {row['key'][:90]:<90} win_rate={row['win_rate']:.3f} "
            f"loss_rate={row['loss_rate']:.3f} delta={row['delta_win_minus_loss']:+.3f} "
            f"cpg_delta={row['delta_count_per_game']:+.3f}"
        )
    print("\nTop ngram gaps:")
    for row in ngram_rows[: args.top]:
        print(
            f"  {row['key'][:120]:<120} win_rate={row['win_rate']:.3f} "
            f"loss_rate={row['loss_rate']:.3f} delta={row['delta_win_minus_loss']:+.3f}"
        )


if __name__ == "__main__":
    main()
