from __future__ import annotations

import numpy as np

from ptcg_rl.v15.constants import TYPE_ABILITY, TYPE_CARD, TYPE_EVOLVE, TYPE_PLAY, TYPE_RETREAT


LINE_BY_ARCHETYPE: dict[str, dict[str, set[int] | int]] = {
    "dragapult": {
        "basic": {119},
        "stage1": {120},
        "stage2": {121},
        "desired_basic": 2,
    },
    "alakazam": {
        "basic": {741},
        "stage1": {742},
        "stage2": {743, 245},
        "desired_basic": 2,
    },
}

ROUTE_OPTION_TYPES = {TYPE_CARD, TYPE_PLAY, TYPE_EVOLVE}
ROUTE_BLOCKED_CONTEXTS = {
    8,   # DISCARD
    13,  # DAMAGE_COUNTER
    14,  # DAMAGE_COUNTER_ANY
    16,  # DAMAGE_COUNTER variants in older traces
}


def normalize_archetype(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def route_context_allowed(select_context: int | None) -> bool:
    if select_context is None:
        return True
    return int(select_context) not in ROUTE_BLOCKED_CONTEXTS


def route_targets(
    archetype: str,
    board_cards,
    opt_type,
    opt_card,
    opt_card2,
    *,
    select_context: int | None = None,
    max_options: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return explicit main-line route targets and route stage.

    Stage ids: 0 none, 1 basic setup, 2 stage1 route, 3 stage2 route.
    The helper intentionally uses only public/current observation fields so the
    same logic can be used in extraction diagnostics and live policy wrappers.
    """
    key = normalize_archetype(archetype)
    spec = LINE_BY_ARCHETYPE.get(key)
    n_opt = len(np.asarray(opt_type).reshape(-1))
    out_n = int(max_options) if max_options is not None else n_opt
    targets = np.zeros(out_n, dtype=np.float32)
    if not route_context_allowed(select_context):
        return targets, 0
    if not spec or n_opt <= 0:
        return targets, 0
    board = np.asarray(board_cards, dtype=np.int64).reshape(-1)
    own = board[:6]
    ot = np.asarray(opt_type, dtype=np.int64).reshape(-1)
    oc = np.asarray(opt_card, dtype=np.int64).reshape(-1)
    oc2 = np.asarray(opt_card2, dtype=np.int64).reshape(-1)
    basic = set(spec["basic"])  # type: ignore[arg-type]
    stage1 = set(spec["stage1"])  # type: ignore[arg-type]
    stage2 = set(spec["stage2"])  # type: ignore[arg-type]
    desired_basic = int(spec.get("desired_basic", 1))  # type: ignore[union-attr]

    own_basic = sum(1 for c in own if int(c) in basic)
    own_stage1 = sum(1 for c in own if int(c) in stage1)
    own_stage2 = sum(1 for c in own if int(c) in stage2)

    def mark_type(type_id: int, stage: int) -> tuple[bool, int]:
        hit = False
        for i in range(min(n_opt, out_n)):
            if int(ot[i]) == type_id:
                targets[i] = 1.0
                hit = True
        return hit, stage if hit else 0

    def mark(card_ids: set[int], stage: int) -> tuple[bool, int]:
        hit = False
        for i in range(min(n_opt, out_n)):
            if int(ot[i]) not in ROUTE_OPTION_TYPES:
                continue
            if int(oc[i]) in card_ids or int(oc2[i]) in card_ids:
                targets[i] = 1.0
                hit = True
        return hit, stage if hit else 0

    if key == "dragapult":
        active = int(own[0]) if own.size else 0
        stage2_benched = any(int(c) in stage2 for c in own[1:])
        if active not in stage2 and stage2_benched:
            hit, stage = mark_type(TYPE_RETREAT, 5)
            if hit:
                return targets, stage
        has_stage2_evolve = any(int(ot[i]) == TYPE_EVOLVE and (int(oc[i]) in stage2 or int(oc2[i]) in stage2) for i in range(n_opt))
        has_drakloak_ability = any(int(ot[i]) == TYPE_ABILITY and (int(oc[i]) in stage1 or int(oc2[i]) in stage1) for i in range(n_opt))
        if has_stage2_evolve and has_drakloak_ability:
            hit = False
            for i in range(min(n_opt, out_n)):
                if int(ot[i]) == TYPE_ABILITY and (int(oc[i]) in stage1 or int(oc2[i]) in stage1):
                    targets[i] = 1.0
                    hit = True
            if hit:
                return targets, 4

    if own_stage1 > 0 and own_stage2 <= 0:
        hit, stage = mark(stage2, 3)
        if hit:
            return targets, stage
    if own_basic > 0 and own_stage1 <= 0:
        hit, stage = mark(stage1, 2)
        if hit:
            return targets, stage
    if own_basic + own_stage1 + own_stage2 < desired_basic:
        hit, stage = mark(basic, 1)
        if hit:
            return targets, stage
    if own_stage2 <= 0:
        hit, stage = mark(stage2, 3)
        if hit:
            return targets, stage
    if own_stage1 <= 0:
        hit, stage = mark(stage1, 2)
        if hit:
            return targets, stage
    return targets, 0
