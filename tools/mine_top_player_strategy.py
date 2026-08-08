#!/usr/bin/env python3
"""Mine top-player strategy gaps from BC corpora.

This tool compares a target cohort such as one strong Kaggle team/deck
signature against a control cohort in the same matchup. It is designed to
produce rule/teacher hypotheses, not train labels directly.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import time
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
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    30: "DISCARD_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    37: "EVOLVE",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
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
TACTICAL_TYPES = {PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END}
PRESSING_TYPES = {PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK}

GAME_FIELDS = [
    "cohort",
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
    "target_mean",
    "control_mean",
    "delta_target_minus_control",
    "target_n",
    "control_n",
    "priority",
    "recommendation",
]

EVENT_FIELDS = [
    "kind",
    "key",
    "target_games",
    "control_games",
    "target_rate",
    "control_rate",
    "delta_target_minus_control",
    "target_count_per_game",
    "control_count_per_game",
    "delta_count_per_game",
    "priority",
    "example_target_game",
    "example_control_game",
    "recommendation",
]

OPPORTUNITY_FIELDS = [
    "key",
    "kind",
    "context",
    "turn_bucket",
    "option_type",
    "option_type_name",
    "card_id",
    "card_name",
    "target_available_games",
    "control_available_games",
    "target_availability_rate",
    "control_availability_rate",
    "target_choice_games",
    "control_choice_games",
    "target_choose_rate_when_available",
    "control_choose_rate_when_available",
    "delta_choose_rate",
    "target_available_count_per_game",
    "control_available_count_per_game",
    "target_choice_count_per_game",
    "control_choice_count_per_game",
    "priority",
    "example_target_game",
    "example_control_game",
    "recommendation",
]

RULE_FIELDS = [
    "source",
    "kind",
    "key",
    "priority",
    "support_target_games",
    "support_control_games",
    "delta",
    "recommendation",
    "rule_mode_hint",
    "teacher_status",
]


def clean_arch(name: str) -> str:
    return str(name).replace(" ", "_")


def slug(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return value[:max_len] or "x"


def lower_set(values: list[str]) -> set[str]:
    return {str(x).lower() for x in values if str(x)}


def str_set(values: list[str]) -> set[str]:
    return {str(x) for x in values if str(x)}


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


def card_name(card_id: int, card_names: dict[int, str]) -> str:
    if card_id <= 0:
        return ""
    return card_names.get(card_id) or str(card_id)


def card_label(card_id: int, card_names: dict[int, str]) -> str:
    if card_id <= 0:
        return "0"
    return f"{card_id}:{card_name(card_id, card_names)}"


def type_name(opt_type: int) -> str:
    return TYPE_NAMES.get(opt_type, str(opt_type))


def context_name(context: int) -> str:
    return CONTEXT_NAMES.get(context, str(context))


def discover_paths(corpus: str, archetype: str, score_bands: list[str]) -> list[str]:
    root = Path(corpus) / clean_arch(archetype)
    if not root.exists():
        return []
    if not score_bands:
        return sorted(glob.glob(str(root / "*" / "*.npz")))
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(str(root / band.replace(" ", "_") / "*.npz"))))
    return paths


def load_card_names(path: str) -> dict[int, str]:
    names = dict(CARD_NAMES)
    p = Path(path) if path else _REPO / "data" / "EN_Card_Data.csv"
    if not p.exists():
        return names
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw_id = (row.get("Card ID") or "").strip()
            raw_name = (row.get("Card Name") or "").strip()
            if raw_id.lstrip("-").isdigit() and raw_name:
                names[int(raw_id)] = raw_name
    return names


def parse_card_specs(specs: list[str], card_names: dict[int, str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs:
        if ":" in spec:
            raw_id, label = spec.split(":", 1)
        else:
            raw_id, label = spec, ""
        cid = int(raw_id)
        out[cid] = label or card_name(cid, card_names)
    return out


def default_track_cards(archetype: str, specs: list[str], card_names: dict[int, str]) -> dict[int, str]:
    out: dict[int, str] = {}
    plan = PLANS.get(archetype)
    if plan:
        for cid in sorted(plan.all_key_ids()):
            out[int(cid)] = card_name(int(cid), card_names)
    out.update(parse_card_specs(specs, card_names))
    return out


def outcome_at(data: dict[str, np.ndarray], i: int) -> str:
    if "won" in data and int(data["won"][i]) == 1:
        return "win"
    if "draw" in data and int(data["draw"][i]) == 1:
        return "draw"
    return "loss"


def outcome_allowed(outcome: str, selector: str) -> bool:
    return selector == "all" or outcome == selector


def turn_context(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int]:
    feats = np.asarray(data["feats"][i], dtype=np.float32)
    turn = inum(fnum(feats[0]) * 30.0) if len(feats) > 0 else 0
    tac = inum(fnum(feats[1]) * 50.0) if len(feats) > 1 else 0
    ctx = inum(fnum(feats[17]) * 64.0) if len(feats) > 17 else -1
    return turn, tac, ctx


def turn_bucket(turn: int) -> str:
    if turn <= 0:
        return "0"
    if turn <= 2:
        return "1-2"
    if turn <= 4:
        return "3-4"
    if turn <= 6:
        return "5-6"
    if turn <= 8:
        return "7-8"
    return "9+"


def first_action(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int, int]:
    action = np.asarray(data["action"][i], dtype=np.int64)
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    oc = np.asarray(data["oc"][i], dtype=np.int64)
    oc2 = np.asarray(data["oc2"][i], dtype=np.int64) if "oc2" in data else np.zeros_like(oc)
    if len(action) == 0:
        return -1, -1, -1, -1
    first = int(action[0])
    if first < 0 or first >= len(ot):
        return first, -1, -1, -1
    card = int(oc[first]) if first < len(oc) else 0
    card2 = int(oc2[first]) if first < len(oc2) else 0
    return first, int(ot[first]), card, card2


def option_type_set(data: dict[str, np.ndarray], i: int) -> set[int]:
    try:
        return {int(x) for x in np.asarray(data["ot"][i], dtype=np.int64).tolist()}
    except Exception:
        return set()


def board_cards(data: dict[str, np.ndarray], i: int) -> tuple[list[int], list[int]]:
    board = np.asarray(data["board"][i], dtype=np.int64)
    mine = [int(x) for x in board[:6].tolist()]
    opp = [int(x) for x in board[6:12].tolist()]
    return mine, opp


def row_global_match(
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
    if args.min_score and "score" in data and fnum(data["score"][i]) < args.min_score:
        return False
    if args.min_opponent_score and "opponent_score" in data and fnum(data["opponent_score"][i]) < args.min_opponent_score:
        return False
    return True


def target_match(data: dict[str, np.ndarray], i: int, args: argparse.Namespace) -> bool:
    team_ok = not args._target_team_names or str(data["team_name"][i]).lower() in args._target_team_names
    sig_ok = not args._target_deck_sigs or str(data["deck_sig"][i]) in args._target_deck_sigs
    return team_ok and sig_ok


def explicit_control_match(data: dict[str, np.ndarray], i: int, args: argparse.Namespace) -> bool:
    team_ok = not args._control_team_names or str(data["team_name"][i]).lower() in args._control_team_names
    sig_ok = not args._control_deck_sigs or str(data["deck_sig"][i]) in args._control_deck_sigs
    return team_ok and sig_ok


def cohort_for_row(data: dict[str, np.ndarray], i: int, args: argparse.Namespace) -> str:
    outcome = outcome_at(data, i)
    is_target = target_match(data, i, args)
    has_explicit_control = bool(args.control_team_name or args.control_deck_sig)
    is_control = explicit_control_match(data, i, args) if has_explicit_control else True
    if is_target and outcome_allowed(outcome, args.target_outcome):
        return "target"
    if not args.control_include_target and is_target and not has_explicit_control:
        return ""
    if is_control and outcome_allowed(outcome, args.control_outcome):
        return "control"
    return ""


def token_for_row(data: dict[str, np.ndarray], i: int, card_names: dict[int, str], *, with_turn: bool) -> str:
    turn, _tac, ctx = turn_context(data, i)
    _first, typ, card, card2 = first_action(data, i)
    parts: list[str] = []
    if with_turn:
        parts.append(f"t{min(turn, 12)}")
    parts.extend([context_name(ctx), type_name(typ)])
    if card > 0:
        label = card_name(card, card_names)
        parts.append(label)
    if card2 > 0:
        parts.append("to_" + card_name(card2, card_names))
    return ":".join(parts)


def option_rows(
    data: dict[str, np.ndarray],
    i: int,
    tracked_cards: set[int],
    card_names: dict[int, str],
) -> tuple[list[dict], tuple[int, int]]:
    turn, _tac, ctx = turn_context(data, i)
    tb = turn_bucket(turn)
    _first, chosen_type, chosen_card, _chosen_card2 = first_action(data, i)
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    oc = np.asarray(data["oc"][i], dtype=np.int64)
    seen: dict[str, dict] = {}
    type_seen = {int(x) for x in ot.tolist() if int(x) in TACTICAL_TYPES}
    for typ in sorted(type_seen):
        key = f"type|ctx={context_name(ctx)}|turn={tb}|available={type_name(typ)}"
        seen[key] = {
            "key": key,
            "kind": "type",
            "context": context_name(ctx),
            "turn_bucket": tb,
            "option_type": typ,
            "option_type_name": type_name(typ),
            "card_id": 0,
            "card_name": "",
            "chosen": int(chosen_type == typ),
        }
    for j, typ_raw in enumerate(ot.tolist()):
        typ = int(typ_raw)
        if typ not in TACTICAL_TYPES:
            continue
        card = int(oc[j]) if j < len(oc) else 0
        if card <= 0 or (tracked_cards and card not in tracked_cards):
            continue
        key = (
            f"card|ctx={context_name(ctx)}|turn={tb}|available="
            f"{type_name(typ)}:{card_label(card, card_names)}"
        )
        seen[key] = {
            "key": key,
            "kind": "card",
            "context": context_name(ctx),
            "turn_bucket": tb,
            "option_type": typ,
            "option_type_name": type_name(typ),
            "card_id": card,
            "card_name": card_name(card, card_names),
            "chosen": int(chosen_type == typ and chosen_card == card),
        }
    return list(seen.values()), (chosen_type, chosen_card)


def update_first(mapping: dict[str, int], key: str, turn: int) -> None:
    if key not in mapping or turn < mapping[key]:
        mapping[key] = turn


class GameAgg:
    def __init__(self, cohort: str, data: dict[str, np.ndarray], i: int, args: argparse.Namespace):
        self.cohort = cohort
        self.episode_id = str(data["episode_id"][i])
        self.player_index = int(data["player_index"][i])
        self.game_key = f"{self.episode_id}:{self.player_index}"
        self.archetype = args.archetype
        self.deck_sig = str(data["deck_sig"][i]) if "deck_sig" in data else ""
        self.team_name = str(data["team_name"][i]) if "team_name" in data else ""
        self.opponent_archetype = str(data["opponent_archetype"][i]) if "opponent_archetype" in data else ""
        self.opponent_deck_sig = str(data["opponent_deck_sig"][i]) if "opponent_deck_sig" in data else ""
        self.opponent_team_name = str(data["opponent_team_name"][i]) if "opponent_team_name" in data else ""
        self.outcome = outcome_at(data, i)
        self.score = fnum(data["score"][i]) if "score" in data else 0.0
        self.opponent_score = fnum(data["opponent_score"][i]) if "opponent_score" in data else 0.0
        self.counters: Counter = Counter()
        self.first_turns: dict[str, int] = {}
        self.events: Counter = Counter()
        self.opportunities: Counter = Counter()
        self.opportunity_choices: Counter = Counter()
        self.opportunity_meta: dict[str, dict] = {}
        self.tokens: list[str] = []
        self.max_turn = 0

    def add_row(
        self,
        data: dict[str, np.ndarray],
        i: int,
        args: argparse.Namespace,
        card_names: dict[int, str],
        track_cards: dict[int, str],
        track_opp_cards: dict[int, str],
    ) -> None:
        turn, _tac, ctx = turn_context(data, i)
        self.max_turn = max(self.max_turn, turn)
        _first, typ, card, card2 = first_action(data, i)
        my_board, opp_board = board_cards(data, i)
        active = my_board[0] if my_board else 0
        bench = [x for x in my_board[1:] if x]
        on_board = [x for x in my_board if x]
        opp_active = opp_board[0] if opp_board else 0
        opp_bench = [x for x in opp_board[1:] if x]
        opp_on_board = [x for x in opp_board if x]
        plan = PLANS.get(args.archetype)
        primary = set(int(x) for x in plan.primary_attackers) if plan else set()

        self.counters["decisions"] += 1
        if ctx == MAIN:
            self.counters["main_decisions"] += 1
            if typ == END and option_type_set(data, i) & PRESSING_TYPES:
                self.counters["early_end_count"] += 1
            self.events[f"choice|ctx=MAIN|turn={turn_bucket(turn)}|type={type_name(typ)}"] += 1
            if card > 0:
                self.events[
                    f"choice_card|ctx=MAIN|turn={turn_bucket(turn)}|type={type_name(typ)}|card={card_label(card, card_names)}"
                ] += 1
        if ctx == TO_ACTIVE and card > 0:
            self.events[f"to_active|turn={turn_bucket(turn)}|card={card_label(card, card_names)}"] += 1
        if ctx == ATTACH_FROM and card > 0:
            self.events[f"attach_from|turn={turn_bucket(turn)}|card={card_label(card, card_names)}"] += 1
        if ctx == ATTACH_TO and card > 0:
            self.events[f"attach_to|turn={turn_bucket(turn)}|card={card_label(card, card_names)}"] += 1
        if card2 > 0:
            self.events[
                f"target_card|ctx={context_name(ctx)}|turn={turn_bucket(turn)}|type={type_name(typ)}|card2={card_label(card2, card_names)}"
            ] += 1

        if typ == ATTACK:
            self.counters["attack_count"] += 1
            update_first(self.first_turns, "first_attack_turn", turn)
        elif typ == ATTACH:
            self.counters["attach_count"] += 1
        elif typ == EVOLVE:
            self.counters["evolve_count"] += 1
        elif typ == ABILITY:
            self.counters["ability_count"] += 1
        elif typ == PLAY:
            self.counters["play_count"] += 1
        elif typ == RETREAT:
            self.counters["retreat_count"] += 1
        elif typ == END:
            self.counters["end_count"] += 1

        if primary:
            if active in primary:
                self.counters["active_primary_turns"] += 1
                update_first(self.first_turns, "first_primary_active_turn", turn)
            if any(x in primary for x in bench):
                self.counters["bench_primary_turns"] += 1
            if any(x in primary for x in on_board):
                update_first(self.first_turns, "first_primary_board_turn", turn)

        for cid, label in track_cards.items():
            tag = slug(label, 30)
            if active == cid:
                self.counters[f"active_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_active_turn_{tag}", turn)
                self.events[f"active_board|turn={turn_bucket(turn)}|card={card_label(cid, card_names)}"] += 1
            if cid in bench:
                self.counters[f"bench_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_bench_turn_{tag}", turn)
            if cid in on_board:
                self.counters[f"board_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_board_turn_{tag}", turn)

        for cid, label in track_opp_cards.items():
            tag = slug(label, 30)
            if opp_active == cid:
                self.counters[f"opp_active_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_opp_active_turn_{tag}", turn)
                self.events[f"opp_active_board|turn={turn_bucket(turn)}|card={card_label(cid, card_names)}"] += 1
            if cid in opp_bench:
                self.counters[f"opp_bench_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_opp_bench_turn_{tag}", turn)
            if cid in opp_on_board:
                self.counters[f"opp_board_turns_{tag}"] += 1
                update_first(self.first_turns, f"first_opp_board_turn_{tag}", turn)

        tracked_for_options = set(track_cards) | set(track_opp_cards)
        for opp in option_rows(data, i, tracked_for_options, card_names)[0]:
            key = opp["key"]
            self.opportunities[key] += 1
            if opp["chosen"]:
                self.opportunity_choices[key] += 1
            self.opportunity_meta.setdefault(key, {k: v for k, v in opp.items() if k != "chosen"})

        if len(self.tokens) < args.max_tokens and ctx in {MAIN, TO_ACTIVE, ATTACH_FROM, ATTACH_TO, 3, 7, 8, 43}:
            self.tokens.append(token_for_row(data, i, card_names, with_turn=args.ngram_with_turn))

    def add_sequence_events(self, args: argparse.Namespace) -> None:
        for n in args.ngram:
            if n <= 0:
                continue
            for j in range(0, max(0, len(self.tokens) - n + 1)):
                seq = " > ".join(self.tokens[j:j + n])
                self.events[f"ngram{n}|{seq}"] += 1

    def finalize(self, args: argparse.Namespace, track_cards: dict[int, str], track_opp_cards: dict[int, str]) -> dict:
        row = {
            "cohort": self.cohort,
            "game_key": self.game_key,
            "episode_id": self.episode_id,
            "player_index": self.player_index,
            "archetype": self.archetype,
            "deck_sig": self.deck_sig,
            "team_name": self.team_name,
            "opponent_archetype": self.opponent_archetype,
            "opponent_deck_sig": self.opponent_deck_sig,
            "opponent_team_name": self.opponent_team_name,
            "outcome": self.outcome,
            "score": self.score,
            "opponent_score": self.opponent_score,
            "decisions": self.counters["decisions"],
            "max_turn": self.max_turn,
            "main_decisions": self.counters["main_decisions"],
            "attack_count": self.counters["attack_count"],
            "attach_count": self.counters["attach_count"],
            "evolve_count": self.counters["evolve_count"],
            "ability_count": self.counters["ability_count"],
            "play_count": self.counters["play_count"],
            "retreat_count": self.counters["retreat_count"],
            "end_count": self.counters["end_count"],
            "early_end_count": self.counters["early_end_count"],
            "first_attack_turn": self.first_turns.get("first_attack_turn", 999),
            "attack_by_4": int(self.first_turns.get("first_attack_turn", 999) <= 4),
            "attack_by_6": int(self.first_turns.get("first_attack_turn", 999) <= 6),
            "attack_by_8": int(self.first_turns.get("first_attack_turn", 999) <= 8),
            "active_primary_turns": self.counters["active_primary_turns"],
            "bench_primary_turns": self.counters["bench_primary_turns"],
            "first_primary_active_turn": self.first_turns.get("first_primary_active_turn", 999),
            "first_primary_board_turn": self.first_turns.get("first_primary_board_turn", 999),
            "primary_active_by_2": int(self.first_turns.get("first_primary_active_turn", 999) <= 2),
            "primary_active_by_4": int(self.first_turns.get("first_primary_active_turn", 999) <= 4),
            "primary_active_by_6": int(self.first_turns.get("first_primary_active_turn", 999) <= 6),
            "primary_board_by_2": int(self.first_turns.get("first_primary_board_turn", 999) <= 2),
            "primary_board_by_4": int(self.first_turns.get("first_primary_board_turn", 999) <= 4),
            "primary_board_by_6": int(self.first_turns.get("first_primary_board_turn", 999) <= 6),
            "first_token_sequence": " > ".join(self.tokens[: args.sequence_preview]),
        }
        for cid, label in track_cards.items():
            tag = slug(label, 30)
            row[f"active_turns_{tag}"] = self.counters[f"active_turns_{tag}"]
            row[f"bench_turns_{tag}"] = self.counters[f"bench_turns_{tag}"]
            row[f"board_turns_{tag}"] = self.counters[f"board_turns_{tag}"]
            row[f"first_active_turn_{tag}"] = self.first_turns.get(f"first_active_turn_{tag}", 999)
            row[f"first_bench_turn_{tag}"] = self.first_turns.get(f"first_bench_turn_{tag}", 999)
            row[f"first_board_turn_{tag}"] = self.first_turns.get(f"first_board_turn_{tag}", 999)
            row[f"board_by_4_{tag}"] = int(self.first_turns.get(f"first_board_turn_{tag}", 999) <= 4)
            row[f"active_by_4_{tag}"] = int(self.first_turns.get(f"first_active_turn_{tag}", 999) <= 4)
        for cid, label in track_opp_cards.items():
            tag = slug(label, 30)
            row[f"opp_active_turns_{tag}"] = self.counters[f"opp_active_turns_{tag}"]
            row[f"opp_bench_turns_{tag}"] = self.counters[f"opp_bench_turns_{tag}"]
            row[f"opp_board_turns_{tag}"] = self.counters[f"opp_board_turns_{tag}"]
            row[f"first_opp_active_turn_{tag}"] = self.first_turns.get(f"first_opp_active_turn_{tag}", 999)
            row[f"first_opp_bench_turn_{tag}"] = self.first_turns.get(f"first_opp_bench_turn_{tag}", 999)
            row[f"first_opp_board_turn_{tag}"] = self.first_turns.get(f"first_opp_board_turn_{tag}", 999)
            row[f"opp_board_by_4_{tag}"] = int(self.first_turns.get(f"first_opp_board_turn_{tag}", 999) <= 4)
            row[f"opp_active_by_4_{tag}"] = int(self.first_turns.get(f"first_opp_active_turn_{tag}", 999) <= 4)
        return row


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def metric_gaps(game_rows: list[dict], min_games: int) -> list[dict]:
    target = [r for r in game_rows if r["cohort"] == "target"]
    control = [r for r in game_rows if r["cohort"] == "control"]
    if len(target) < min_games or len(control) < min_games:
        return []
    skip = {
        "cohort",
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
        "first_token_sequence",
    }
    rows = []
    for key in game_rows[0]:
        if key in skip:
            continue
        try:
            tv = [float(r[key]) for r in target]
            cv = [float(r[key]) for r in control]
        except Exception:
            continue
        tm = mean(tv)
        cm = mean(cv)
        delta = tm - cm
        pooled = math.sqrt(max(1e-9, mean([(x - tm) ** 2 for x in tv] + [(x - cm) ** 2 for x in cv])))
        priority = abs(delta) / pooled
        recommendation = "trajectory_target" if abs(delta) > 0 else "none"
        rows.append({
            "metric": key,
            "target_mean": tm,
            "control_mean": cm,
            "delta_target_minus_control": delta,
            "target_n": len(target),
            "control_n": len(control),
            "priority": priority,
            "recommendation": recommendation,
        })
    rows.sort(key=lambda r: float(r["priority"]), reverse=True)
    return rows


def summarize_event_gaps(
    games: dict[tuple[str, str], GameAgg],
    game_rows: list[dict],
    *,
    min_games: int,
    min_rate_gap: float,
) -> list[dict]:
    target_keys = [(r["cohort"], r["game_key"]) for r in game_rows if r["cohort"] == "target"]
    control_keys = [(r["cohort"], r["game_key"]) for r in game_rows if r["cohort"] == "control"]
    if len(target_keys) < min_games or len(control_keys) < min_games:
        return []
    all_events = set()
    for game in games.values():
        all_events.update(game.events)
    rows = []
    for key in all_events:
        target_present = [gk for gk in target_keys if games[gk].events.get(key, 0) > 0]
        control_present = [gk for gk in control_keys if games[gk].events.get(key, 0) > 0]
        target_rate = len(target_present) / max(len(target_keys), 1)
        control_rate = len(control_present) / max(len(control_keys), 1)
        target_cpg = sum(games[gk].events.get(key, 0) for gk in target_keys) / max(len(target_keys), 1)
        control_cpg = sum(games[gk].events.get(key, 0) for gk in control_keys) / max(len(control_keys), 1)
        rate_gap = target_rate - control_rate
        cpg_gap = target_cpg - control_cpg
        if abs(rate_gap) < min_rate_gap and abs(cpg_gap) < min_rate_gap:
            continue
        if len(target_present) + len(control_present) < min_games:
            continue
        priority = abs(rate_gap) + 0.25 * abs(cpg_gap)
        recommendation = "prefer_or_preserve" if rate_gap > 0 else "avoid_or_delay"
        rows.append({
            "kind": key.split("|", 1)[0],
            "key": key,
            "target_games": len(target_present),
            "control_games": len(control_present),
            "target_rate": target_rate,
            "control_rate": control_rate,
            "delta_target_minus_control": rate_gap,
            "target_count_per_game": target_cpg,
            "control_count_per_game": control_cpg,
            "delta_count_per_game": cpg_gap,
            "priority": priority,
            "example_target_game": target_present[0][1] if target_present else "",
            "example_control_game": control_present[0][1] if control_present else "",
            "recommendation": recommendation,
        })
    rows.sort(key=lambda r: float(r["priority"]), reverse=True)
    return rows


def summarize_opportunity_gaps(
    games: dict[tuple[str, str], GameAgg],
    game_rows: list[dict],
    *,
    min_games: int,
    min_choose_gap: float,
) -> list[dict]:
    target_keys = [(r["cohort"], r["game_key"]) for r in game_rows if r["cohort"] == "target"]
    control_keys = [(r["cohort"], r["game_key"]) for r in game_rows if r["cohort"] == "control"]
    if len(target_keys) < min_games or len(control_keys) < min_games:
        return []
    meta: dict[str, dict] = {}
    all_keys = set()
    for game in games.values():
        all_keys.update(game.opportunities)
        meta.update(game.opportunity_meta)
    rows = []
    for key in all_keys:
        target_avail = [gk for gk in target_keys if games[gk].opportunities.get(key, 0) > 0]
        control_avail = [gk for gk in control_keys if games[gk].opportunities.get(key, 0) > 0]
        if len(target_avail) + len(control_avail) < min_games:
            continue
        target_choice = [gk for gk in target_avail if games[gk].opportunity_choices.get(key, 0) > 0]
        control_choice = [gk for gk in control_avail if games[gk].opportunity_choices.get(key, 0) > 0]
        target_rate = len(target_avail) / max(len(target_keys), 1)
        control_rate = len(control_avail) / max(len(control_keys), 1)
        target_choose = len(target_choice) / max(len(target_avail), 1)
        control_choose = len(control_choice) / max(len(control_avail), 1)
        choose_gap = target_choose - control_choose
        target_avail_cpg = sum(games[gk].opportunities.get(key, 0) for gk in target_keys) / max(len(target_keys), 1)
        control_avail_cpg = sum(games[gk].opportunities.get(key, 0) for gk in control_keys) / max(len(control_keys), 1)
        target_choice_cpg = sum(games[gk].opportunity_choices.get(key, 0) for gk in target_keys) / max(len(target_keys), 1)
        control_choice_cpg = sum(games[gk].opportunity_choices.get(key, 0) for gk in control_keys) / max(len(control_keys), 1)
        if abs(choose_gap) < min_choose_gap and abs(target_rate - control_rate) < min_choose_gap:
            continue
        priority = abs(choose_gap) * math.sqrt(max(len(target_avail) + len(control_avail), 1))
        info = meta.get(key, {})
        recommendation = "prefer_when_available" if choose_gap > 0 else "avoid_when_available"
        rows.append({
            "key": key,
            "kind": info.get("kind", key.split("|", 1)[0]),
            "context": info.get("context", ""),
            "turn_bucket": info.get("turn_bucket", ""),
            "option_type": info.get("option_type", ""),
            "option_type_name": info.get("option_type_name", ""),
            "card_id": info.get("card_id", ""),
            "card_name": info.get("card_name", ""),
            "target_available_games": len(target_avail),
            "control_available_games": len(control_avail),
            "target_availability_rate": target_rate,
            "control_availability_rate": control_rate,
            "target_choice_games": len(target_choice),
            "control_choice_games": len(control_choice),
            "target_choose_rate_when_available": target_choose,
            "control_choose_rate_when_available": control_choose,
            "delta_choose_rate": choose_gap,
            "target_available_count_per_game": target_avail_cpg,
            "control_available_count_per_game": control_avail_cpg,
            "target_choice_count_per_game": target_choice_cpg,
            "control_choice_count_per_game": control_choice_cpg,
            "priority": priority,
            "example_target_game": target_choice[0][1] if target_choice else (target_avail[0][1] if target_avail else ""),
            "example_control_game": control_avail[0][1] if control_avail else "",
            "recommendation": recommendation,
        })
    rows.sort(key=lambda r: float(r["priority"]), reverse=True)
    return rows


def build_rule_candidates(metric_rows: list[dict], event_rows: list[dict], opportunity_rows: list[dict], top: int) -> list[dict]:
    rows: list[dict] = []
    for row in opportunity_rows[: top * 2]:
        delta = fnum(row["delta_choose_rate"])
        if delta <= 0:
            continue
        typ = str(row.get("option_type_name", ""))
        card_id = str(row.get("card_id", ""))
        if card_id and card_id != "0":
            mode_hint = f"rerank {typ} card={card_id} in {row.get('context')} turn={row.get('turn_bucket')}"
        else:
            mode_hint = f"rerank type={typ} in {row.get('context')} turn={row.get('turn_bucket')}"
        rows.append({
            "source": "opportunity_gap",
            "kind": row.get("kind", ""),
            "key": row["key"],
            "priority": row["priority"],
            "support_target_games": row["target_available_games"],
            "support_control_games": row["control_available_games"],
            "delta": delta,
            "recommendation": "trace_then_narrow_rerank",
            "rule_mode_hint": mode_hint,
            "teacher_status": "rule_probe_candidate",
        })
    for row in event_rows[: top * 2]:
        delta = fnum(row["delta_target_minus_control"])
        if delta <= 0:
            continue
        rows.append({
            "source": "event_gap",
            "kind": row["kind"],
            "key": row["key"],
            "priority": row["priority"],
            "support_target_games": row["target_games"],
            "support_control_games": row["control_games"],
            "delta": delta,
            "recommendation": "inspect_trace_then_teacher_seed",
            "rule_mode_hint": "sequence_or_board_pattern",
            "teacher_status": "teacher_or_rule_probe_candidate",
        })
    for row in metric_rows[: top]:
        rows.append({
            "source": "metric_gap",
            "kind": "trajectory_metric",
            "key": row["metric"],
            "priority": row["priority"],
            "support_target_games": row["target_n"],
            "support_control_games": row["control_n"],
            "delta": row["delta_target_minus_control"],
            "recommendation": "trajectory_target_or_plan_condition",
            "rule_mode_hint": "use as target metric, not a direct single-action rule",
            "teacher_status": "trajectory_bc_candidate",
        })
    rows.sort(key=lambda r: fnum(r["priority"]), reverse=True)
    return rows[:top]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.6f}"
        else:
            out[key] = value
    return out


def write_game_keys(path: Path, rows: list[dict], cohort: str) -> None:
    keep = [r for r in rows if r["cohort"] == cohort]
    fields = [
        "game_key",
        "episode_id",
        "player_index",
        "outcome",
        "archetype",
        "deck_sig",
        "team_name",
        "opponent_archetype",
        "opponent_deck_sig",
        "opponent_team_name",
    ]
    write_csv(path, keep, fields)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    game_rows: list[dict],
    metric_rows: list[dict],
    event_rows: list[dict],
    opportunity_rows: list[dict],
    rule_rows: list[dict],
) -> None:
    target = [r for r in game_rows if r["cohort"] == "target"]
    control = [r for r in game_rows if r["cohort"] == "control"]
    target_wr = sum(r["outcome"] == "win" for r in target) / max(len(target), 1)
    control_wr = sum(r["outcome"] == "win" for r in control) / max(len(control), 1)
    lines = [
        "# Top Player Strategy Mining",
        "",
        f"corpus: `{args.corpus}`",
        f"archetype: `{args.archetype}`",
        f"target games: `{len(target)}` win_rate=`{target_wr:.3f}`",
        f"control games: `{len(control)}` win_rate=`{control_wr:.3f}`",
        f"target team filters: `{args.target_team_name}`",
        f"target deck filters: `{args.target_deck_sig}`",
        f"opponent archetype filters: `{args.opponent_archetype}`",
        f"opponent deck filters: `{args.opponent_deck_sig}`",
        "",
        "## Top Rule Candidates",
        "",
    ]
    if not rule_rows:
        lines.append("No rule candidates passed support thresholds.")
    for row in rule_rows[:20]:
        lines.append(
            f"- `{row['source']}` priority={fnum(row['priority']):.3f} "
            f"delta={fnum(row['delta']):+.3f}: {row['key']} -> {row['recommendation']}"
        )
    lines.extend(["", "## Top Opportunities", ""])
    for row in opportunity_rows[:15]:
        lines.append(
            f"- priority={fnum(row['priority']):.3f} choose_delta={fnum(row['delta_choose_rate']):+.3f} "
            f"{row['key']}"
        )
    lines.extend(["", "## Top Events", ""])
    for row in event_rows[:15]:
        lines.append(
            f"- priority={fnum(row['priority']):.3f} rate_delta={fnum(row['delta_target_minus_control']):+.3f} "
            f"{row['key']}"
        )
    lines.extend(["", "## Top Metrics", ""])
    for row in metric_rows[:15]:
        lines.append(
            f"- priority={fnum(row['priority']):.3f} delta={fnum(row['delta_target_minus_control']):+.3f} "
            f"{row['metric']}"
        )
    path.write_text("\n".join(lines) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if not args.target_team_name and not args.target_deck_sig:
        raise SystemExit("provide --target-team-name and/or --target-deck-sig")
    if args.control_include_target and args.control_outcome == args.target_outcome and not (args.control_team_name or args.control_deck_sig):
        raise SystemExit("--control-include-target with identical outcome selectors would overlap target/control")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="*", default=[])
    p.add_argument("--deck-sig", action="append", default=[], help="global filter before cohort split")
    p.add_argument("--team-name", action="append", default=[], help="global filter before cohort split")
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--opponent-team-name", action="append", default=[])
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-opponent-score", type=float, default=0.0)
    p.add_argument("--target-team-name", action="append", default=[])
    p.add_argument("--target-deck-sig", action="append", default=[])
    p.add_argument("--target-outcome", choices=["all", "win", "loss", "draw"], default="win")
    p.add_argument("--control-team-name", action="append", default=[])
    p.add_argument("--control-deck-sig", action="append", default=[])
    p.add_argument("--control-outcome", choices=["all", "win", "loss", "draw"], default="loss")
    p.add_argument("--control-include-target", action="store_true",
                   help="allow target team/sig rows into the implicit control cohort")
    p.add_argument("--track-card", action="append", default=[],
                   help="extra card id or id:label to track; deck plan cards are included by default")
    p.add_argument("--track-opponent-card", action="append", default=[],
                   help="opponent card id or id:label to track; opponent deck plan cards are included when one opponent archetype is supplied")
    p.add_argument("--no-default-track-cards", action="store_true")
    p.add_argument("--card-data", default="")
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--min-rate-gap", type=float, default=0.10)
    p.add_argument("--min-choose-gap", type=float, default=0.12)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--sequence-preview", type=int, default=24)
    p.add_argument("--ngram", nargs="*", type=int, default=[2, 3])
    p.add_argument("--ngram-with-turn", action="store_true")
    p.add_argument("--progress-every-files", type=int, default=4)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    validate_args(args)
    args._target_team_names = lower_set(args.target_team_name)
    args._target_deck_sigs = str_set(args.target_deck_sig)
    args._control_team_names = lower_set(args.control_team_name)
    args._control_deck_sigs = str_set(args.control_deck_sig)

    card_names = load_card_names(args.card_data)
    if args.no_default_track_cards:
        track_cards = parse_card_specs(args.track_card, card_names)
    else:
        track_cards = default_track_cards(args.archetype, args.track_card, card_names)
    track_opp_cards = parse_card_specs(args.track_opponent_card, card_names)
    opp_arches_raw = [x for x in args.opponent_archetype if x]
    if len(opp_arches_raw) == 1:
        opp_plan = PLANS.get(opp_arches_raw[0])
        if opp_plan:
            for cid in sorted(opp_plan.all_key_ids()):
                track_opp_cards.setdefault(int(cid), card_name(int(cid), card_names))

    paths = discover_paths(args.corpus, args.archetype, args.score_bands)
    if not paths:
        raise FileNotFoundError(f"no corpus files for {args.archetype} in {args.corpus}")

    deck_sigs = str_set(args.deck_sig)
    team_names = lower_set(args.team_name)
    opp_arches = lower_set(args.opponent_archetype)
    opp_sigs = str_set(args.opponent_deck_sig)
    opp_teams = lower_set(args.opponent_team_name)

    games: dict[tuple[str, str], GameAgg] = {}
    raw = matched = kept = 0
    t0 = time.time()
    for path_i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        required = {
            "board",
            "feats",
            "ot",
            "oc",
            "action",
            "episode_id",
            "player_index",
            "deck_sig",
            "team_name",
            "opponent_archetype",
            "opponent_deck_sig",
            "opponent_team_name",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path} missing metadata: {sorted(missing)}")
        n = len(data["board"])
        raw += n
        for i in range(n):
            if not row_global_match(data, i, args, deck_sigs, team_names, opp_arches, opp_sigs, opp_teams):
                continue
            matched += 1
            cohort = cohort_for_row(data, i, args)
            if not cohort:
                continue
            kept += 1
            game_key = f"{data['episode_id'][i]}:{data['player_index'][i]}"
            key = (cohort, game_key)
            agg = games.get(key)
            if agg is None:
                agg = GameAgg(cohort, data, i, args)
                games[key] = agg
            agg.add_row(data, i, args, card_names, track_cards, track_opp_cards)
        if args.progress_every_files and (
            path_i == 1 or path_i % args.progress_every_files == 0 or path_i == len(paths)
        ):
            rate = raw / max(time.time() - t0, 1e-9)
            print(
                f"scanned {path_i}/{len(paths)} raw={raw} matched={matched} kept={kept} "
                f"games={len(games)} rate={rate:.0f}/s",
                flush=True,
            )

    for game in games.values():
        game.add_sequence_events(args)

    game_rows = [
        game.finalize(args, track_cards, track_opp_cards)
        for game in games.values()
        if game.counters["decisions"] > 0
    ]
    game_rows.sort(key=lambda r: (r["cohort"], str(r["episode_id"]), int(r["player_index"])))
    target_n = sum(r["cohort"] == "target" for r in game_rows)
    control_n = sum(r["cohort"] == "control" for r in game_rows)
    if target_n < args.min_games or control_n < args.min_games:
        print(
            f"WARNING: low support target_games={target_n} control_games={control_n} "
            f"min_games={args.min_games}",
            flush=True,
        )

    metric_rows = metric_gaps(game_rows, args.min_games)
    event_rows = summarize_event_gaps(
        games,
        game_rows,
        min_games=args.min_games,
        min_rate_gap=args.min_rate_gap,
    )
    opportunity_rows = summarize_opportunity_gaps(
        games,
        game_rows,
        min_games=args.min_games,
        min_choose_gap=args.min_choose_gap,
    )
    rule_rows = build_rule_candidates(metric_rows, event_rows, opportunity_rows, args.top)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_game_fields = list(dict.fromkeys([*GAME_FIELDS, *sorted({k for r in game_rows for k in r})]))
    write_csv(out / "games.csv", [fmt_row(r) for r in game_rows], all_game_fields)
    write_game_keys(out / "target_game_keys.csv", game_rows, "target")
    write_game_keys(out / "control_game_keys.csv", game_rows, "control")
    write_csv(out / "metric_gaps.csv", [fmt_row(r) for r in metric_rows], METRIC_FIELDS)
    write_csv(out / "event_gaps.csv", [fmt_row(r) for r in event_rows], EVENT_FIELDS)
    write_csv(out / "opportunity_gaps.csv", [fmt_row(r) for r in opportunity_rows], OPPORTUNITY_FIELDS)
    write_csv(out / "rule_candidates.csv", [fmt_row(r) for r in rule_rows], RULE_FIELDS)
    write_summary(out / "summary.md", args, game_rows, metric_rows, event_rows, opportunity_rows, rule_rows)

    print(f"Corpus: {args.corpus}")
    print(f"Archetype: {args.archetype}")
    print(f"Rows: raw={raw} matched={matched} kept={kept}")
    print(f"Games: target={target_n} control={control_n}")
    print(f"Wrote: {out}")
    print("\nTop rule candidates:")
    for row in rule_rows[: min(args.top, 12)]:
        print(
            f"  {fnum(row['priority']):7.3f} {row['source']} "
            f"delta={fnum(row['delta']):+.3f} {row['key'][:180]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
