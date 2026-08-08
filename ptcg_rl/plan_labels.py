from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ptcg_rl.deck_plans import DeckPlan, get_plan


PLAN_LABELS = (
    "setup",
    "engine",
    "power",
    "attack",
    "disrupt",
    "preserve",
    "stall",
    "finish",
)
PLAN_DIM = len(PLAN_LABELS)

SETUP, ENGINE, POWER, ATTACK, DISRUPT, PRESERVE, STALL, FINISH = range(PLAN_DIM)

TYPE_PLAY = 7
TYPE_ATTACH = 8
TYPE_EVOLVE = 9
TYPE_ABILITY = 10
TYPE_DISCARD = 11
TYPE_RETREAT = 12
TYPE_ATTACK = 13
TYPE_END = 14

CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_HAND = 7
CTX_DISCARD = 8
CTX_DAMAGE = 15
CTX_HEAL = 17
CTX_EVOLVE = 37
CTX_ATTACH_TO = 22
CTX_DISCARD_ENERGY_CARD = 26
CTX_DISCARD_ENERGY = 30
CTX_ATTACK = 35
CTX_DRAW_COUNT = 38

_SETUP_CONTEXTS = {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_EVOLVE}
_ENGINE_CONTEXTS = {CTX_TO_HAND, CTX_DRAW_COUNT}
_POWER_CONTEXTS = {CTX_ATTACH_TO, CTX_DISCARD_ENERGY_CARD, CTX_DISCARD_ENERGY}
_PRESERVE_CONTEXTS = {CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_BENCH, CTX_HEAL}
_ATTACK_CONTEXTS = {CTX_ATTACK, CTX_DAMAGE}
_DISRUPT_CONTEXTS = {CTX_DISCARD, CTX_DISCARD_ENERGY, CTX_DISCARD_ENERGY_CARD}


def _as_int_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.int64).reshape(-1)
    except Exception:
        return np.zeros(0, dtype=np.int64)


def _as_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _first_selected(data: dict[str, np.ndarray], row_i: int) -> tuple[int, int, int, int, int]:
    action = _as_int_array(data.get("action", [])[row_i])
    if action.size <= 0:
        return -1, -1, 0, 0, 0
    first = int(action[0])
    ot = _as_int_array(data["ot"][row_i])
    if first < 0 or first >= ot.size:
        return first, -1, 0, 0, 0
    oc = _as_int_array(data["oc"][row_i])
    oc2 = _as_int_array(data["oc2"][row_i])
    oa = _as_int_array(data["oa"][row_i])
    of = np.asarray(data.get("of_arr", [])[row_i], dtype=np.float32)
    ctx = 0
    if of.ndim == 2 and first < of.shape[0] and of.shape[1] > 3:
        ctx = int(round(float(of[first, 3]) * 64.0))
    return (
        first,
        int(ot[first]),
        int(oc[first]) if first < oc.size else 0,
        int(oc2[first]) if first < oc2.size else 0,
        int(ctx),
    )


def _option_type_counts(data: dict[str, np.ndarray], row_i: int) -> Counter[int]:
    return Counter(int(x) for x in _as_int_array(data["ot"][row_i]))


def _deck_sets(plan: DeckPlan | None) -> dict[str, set[int]]:
    if not plan:
        return {k: set() for k in (
            "primary",
            "secondary",
            "setup",
            "evolution",
            "engine",
            "energy_accel",
            "draw_search",
            "stadium_tools",
            "disruption",
            "switching",
        )}
    return {
        "primary": {int(x) for x in plan.primary_attackers},
        "secondary": {int(x) for x in plan.secondary_attackers},
        "setup": {int(x) for x in plan.setup_basics},
        "evolution": {int(x) for x in plan.evolution_chain},
        "engine": {int(x) for x in plan.engine_cards},
        "energy_accel": {int(x) for x in plan.energy_accel},
        "draw_search": {int(x) for x in plan.draw_search},
        "stadium_tools": {int(x) for x in plan.stadium_tools},
        "disruption": {int(x) for x in plan.disruption},
        "switching": {int(x) for x in plan.switching},
    }


def _contains_any(values: np.ndarray, ids: set[int]) -> bool:
    return bool(ids) and any(int(x) in ids for x in values if int(x) > 0)


def label_decision_plan(
    data: dict[str, np.ndarray],
    row_i: int,
    *,
    archetype: str = "",
    position: int = 0,
    game_len: int = 1,
) -> np.ndarray:
    """Derive a multi-label plan target for one labeled decision.

    The labels are deliberately high level. They are not meant to prove optimal
    play; they give the policy a stable latent mode for a sequence of decisions:
    setup, build engine, power an attacker, attack, disrupt, preserve resources,
    stall/wall, or finish.
    """

    labels = np.zeros(PLAN_DIM, dtype=np.float32)
    plan = get_plan(archetype) if archetype else None
    sets = _deck_sets(plan)

    board = _as_int_array(data["board"][row_i])
    hand = _as_int_array(data["hand"][row_i]) if "hand" in data else np.zeros(0, dtype=np.int64)
    feats = _as_float_array(data["feats"][row_i]) if "feats" in data else np.zeros(0, dtype=np.float32)
    first, first_type, first_card, first_card2, ctx = _first_selected(data, row_i)
    opt_counts = _option_type_counts(data, row_i)

    game_len = max(int(game_len), 1)
    position = max(int(position), 0)
    progress = position / float(max(game_len - 1, 1))
    early = progress < 0.30 or position < 10
    late = progress > 0.72 or position >= 42

    my_board = board[:6] if board.size >= 6 else board
    opp_board = board[6:12] if board.size >= 12 else np.zeros(0, dtype=np.int64)
    active = int(my_board[0]) if my_board.size else 0
    opp_active = int(opp_board[0]) if opp_board.size else 0

    has_primary = _contains_any(my_board, sets["primary"])
    has_setup = _contains_any(my_board, sets["setup"])
    has_evolution_piece = _contains_any(my_board, sets["evolution"])
    has_engine = _contains_any(my_board, sets["engine"])
    hand_has_setup = _contains_any(hand, sets["setup"] | sets["evolution"])
    hand_has_engine = _contains_any(hand, sets["engine"] | sets["draw_search"] | sets["stadium_tools"])

    if early or (not has_primary and (not has_setup or hand_has_setup)):
        labels[SETUP] = 1.0
    if (not has_engine and (early or hand_has_engine)) or first_card in sets["engine"]:
        labels[ENGINE] = 1.0

    if first_type in {TYPE_PLAY, TYPE_EVOLVE} or ctx in _SETUP_CONTEXTS:
        if first_card in sets["setup"] or first_card in sets["evolution"] or first_card2 in sets["evolution"]:
            labels[SETUP] = 1.0
    if first_type == TYPE_EVOLVE or ctx == CTX_EVOLVE:
        labels[SETUP] = 1.0
        if has_setup or has_evolution_piece:
            labels[ENGINE] = max(labels[ENGINE], 1.0 if early else 0.5)

    if (
        first_card in sets["engine"]
        or first_card in sets["draw_search"]
        or first_card in sets["stadium_tools"]
        or ctx in _ENGINE_CONTEXTS
        or first_type == TYPE_ABILITY
    ):
        labels[ENGINE] = 1.0

    if (
        first_type == TYPE_ATTACH
        or ctx in _POWER_CONTEXTS
        or first_card in sets["energy_accel"]
        or first_card2 in sets["energy_accel"]
    ):
        labels[POWER] = 1.0
        if has_primary or active in sets["primary"]:
            labels[ATTACK] = max(labels[ATTACK], 0.5)

    if first_type == TYPE_ATTACK or ctx in _ATTACK_CONTEXTS:
        labels[ATTACK] = 1.0
        if late or (opp_active > 0 and feats.size > 60 and float(feats[60]) > 0.25):
            labels[FINISH] = 1.0

    if (
        first_card in sets["disruption"]
        or first_card2 in sets["disruption"]
        or first_type == TYPE_DISCARD
        or ctx in _DISRUPT_CONTEXTS
    ):
        labels[DISRUPT] = 1.0

    if (
        first_card in sets["switching"]
        or first_card2 in sets["switching"]
        or first_type == TYPE_RETREAT
        or ctx in _PRESERVE_CONTEXTS
    ):
        labels[PRESERVE] = 1.0

    arch_l = str(archetype).lower()
    wall_like = "crustle" in arch_l or "wall" in arch_l
    if wall_like:
        if first_type in {TYPE_END, TYPE_RETREAT} or ctx in _PRESERVE_CONTEXTS or late:
            labels[STALL] = 1.0
        if active in sets["primary"]:
            labels[STALL] = 1.0

    if first_type == TYPE_END:
        if opt_counts.get(TYPE_ATTACK, 0) <= 0:
            labels[STALL if wall_like else PRESERVE] = 1.0
        elif early:
            labels[SETUP] = max(labels[SETUP], 0.5)
        else:
            labels[PRESERVE] = max(labels[PRESERVE], 0.5)

    if late and (first_type == TYPE_ATTACK or has_primary):
        labels[FINISH] = 1.0

    if labels.sum() <= 0:
        if first_type == TYPE_ATTACK:
            labels[ATTACK] = 1.0
        elif first_type == TYPE_ATTACH:
            labels[POWER] = 1.0
        elif first_type in {TYPE_PLAY, TYPE_EVOLVE}:
            labels[SETUP] = 1.0
        elif first_type == TYPE_RETREAT:
            labels[PRESERVE] = 1.0
        elif first_type == TYPE_ABILITY:
            labels[ENGINE] = 1.0
        elif late:
            labels[FINISH] = 1.0
        else:
            labels[SETUP] = 1.0

    return labels

