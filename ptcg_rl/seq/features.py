from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ptcg_rl.history_features import action_event_from_encoded
from ptcg_rl.seq.constants import (
    DAMAGE_COUNTER_ANY_CONTEXT,
    FUTURE_PLAN_DIM,
    LEDGER_FEAT_DIM,
    TYPE_ABILITY,
    TYPE_ATTACH,
    TYPE_ATTACK,
    TYPE_DISCARD,
    TYPE_END,
    TYPE_EVOLVE,
    TYPE_PLAY,
    TYPE_RETREAT,
    TYPE_SKILL,
)

PLANNED_TYPES = (
    TYPE_PLAY,
    TYPE_ATTACH,
    TYPE_EVOLVE,
    TYPE_ABILITY,
    TYPE_RETREAT,
    TYPE_ATTACK,
    TYPE_SKILL,
    TYPE_DISCARD,
    TYPE_END,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _first_action_event(encoded: Any, action: list[int] | np.ndarray) -> dict[str, Any]:
    ev = action_event_from_encoded(encoded, action)
    if ev is None:
        return {"type": 0, "card": 0, "card2": 0, "attack": 0, "context": 0, "select_type": 0, "count": 0.0}
    return ev


def selected_action_event(encoded: Any, action: list[int] | np.ndarray) -> dict[str, Any]:
    """Return a compact action event with engine type ids, not one-indexed ids."""
    ev = _first_action_event(encoded, action)
    typ = max(0, int(ev.get("type", 0)) - 1)
    return {
        "type": typ,
        "card": int(ev.get("card", 0) or 0),
        "card2": int(ev.get("card2", 0) or 0),
        "attack": int(ev.get("attack", 0) or 0),
        "context": max(0, int(ev.get("context", 0) or 0) - 1),
        "select_type": max(0, int(ev.get("select_type", 0) or 0) - 1),
        "count": float(ev.get("count", 0.0) or 0.0),
    }


def option_type_for_action(opt_type: np.ndarray, action: list[int] | np.ndarray) -> int:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    types = np.asarray(opt_type, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return TYPE_END
    first = int(arr[0])
    if first < 0 or first >= types.size:
        return 0
    return int(types[first])


@dataclass
class SequenceLedger:
    """Live-compatible prefix ledger for one player.

    This is deliberately explicit.  The model gets a compact account of what
    has already happened in this game instead of being forced to infer resource
    tempo from a raw list of previous actions.
    """

    decision_index: int = 0
    counts: dict[int, int] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
    last_event: dict[str, Any] = field(default_factory=dict)
    multi_select_count: int = 0
    card_repeat: dict[int, int] = field(default_factory=dict)
    dca_active: bool = False
    dca_steps: int = 0
    dca_last_remain: int = 0
    dca_target_repeat: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.decision_index = 0
        self.counts.clear()
        self.last_seen.clear()
        self.last_event.clear()
        self.multi_select_count = 0
        self.card_repeat.clear()
        self.dca_active = False
        self.dca_steps = 0
        self.dca_last_remain = 0
        self.dca_target_repeat.clear()

    def update(self, encoded: Any, action: list[int] | np.ndarray) -> dict[str, Any]:
        ev = selected_action_event(encoded, action)
        typ = int(ev["type"])
        self.counts[typ] = self.counts.get(typ, 0) + 1
        self.last_seen[typ] = self.decision_index
        if float(ev.get("count", 0.0)) > 1.0 / max(int(getattr(encoded, "max_count", 1) or 1), 1):
            self.multi_select_count += 1
        card = int(ev.get("card", 0) or 0)
        if card > 0:
            self.card_repeat[card] = self.card_repeat.get(card, 0) + 1
        self._update_damage_counter_any(encoded, ev)
        self.last_event = ev
        self.decision_index += 1
        return ev

    def features(self, encoded: Any) -> np.ndarray:
        out = np.zeros(LEDGER_FEAT_DIM, dtype=np.float32)
        n = max(self.decision_index, 1)
        feats = np.asarray(getattr(encoded, "state_feats", []), dtype=np.float32).reshape(-1)

        out[0] = min(self.decision_index / 64.0, 1.0)
        out[1] = _safe_feat(feats, 0)      # turn
        out[2] = _safe_feat(feats, 1)      # turn action count
        out[3] = _safe_feat(feats, 6)      # own prizes remaining
        out[4] = _safe_feat(feats, 7)      # opponent prizes remaining
        out[5] = _safe_feat(feats, 8)      # own deck count
        out[6] = _safe_feat(feats, 9)      # opp deck count
        out[7] = _safe_feat(feats, 12)     # own bench fullness
        out[8] = _safe_feat(feats, 13)     # opp bench fullness
        out[9] = _safe_feat(feats, 25)     # own active damage ratio
        out[10] = _safe_feat(feats, 27)    # opp active damage ratio
        out[11] = _safe_feat(feats, 29)    # own bench damage
        out[12] = _safe_feat(feats, 30)    # opp bench damage

        for j, typ in enumerate(PLANNED_TYPES):
            out[13 + j] = min(self.counts.get(typ, 0) / n, 1.0)
        for j, typ in enumerate(PLANNED_TYPES[:8]):
            out[22 + j] = self._age_feature(typ)

        last_type = int(self.last_event.get("type", TYPE_END)) if self.last_event else TYPE_END
        if 0 <= last_type < 17:
            out[30 + last_type] = 1.0
        out[47] = min(self.multi_select_count / n, 1.0)
        out[48] = min(max(self.card_repeat.values(), default=0) / 6.0, 1.0)
        out[49] = min(len(self.card_repeat) / 20.0, 1.0)

        # Availability and current phase are live state, duplicated here so the
        # sequence token can compare current opportunity against prior tempo.
        out[50] = _safe_feat(feats, 48)  # attack options
        out[51] = _safe_feat(feats, 49)  # attach options
        out[52] = _safe_feat(feats, 50)  # evolve options
        out[53] = _safe_feat(feats, 51)  # ability options
        out[54] = _safe_feat(feats, 52)  # play options
        out[55] = _safe_feat(feats, 55)  # legal option count
        out[56] = _safe_feat(feats, 56)  # own max energy
        out[57] = _safe_feat(feats, 57)  # opp max energy
        out[58] = _safe_feat(feats, 60)  # own evolved count
        out[59] = _safe_feat(feats, 62)  # opp damaged count
        out[60] = _safe_float(self.last_event.get("count", 0.0)) if self.last_event else 0.0
        out[61] = min(int(self.last_event.get("context", 0) or 0) / 64.0, 1.0) if self.last_event else 0.0
        out[62] = min(int(self.last_event.get("select_type", 0) or 0) / 16.0, 1.0) if self.last_event else 0.0
        out[63] = 1.0
        current_ctx = int(round(_safe_feat(feats, 17) * 64.0))
        current_remain = int(round(_safe_feat(feats, 21) * 30.0))
        if current_ctx == DAMAGE_COUNTER_ANY_CONTEXT:
            dca_steps = max(self.dca_steps, 0)
            unique_targets = len(self.dca_target_repeat)
            max_repeat = max(self.dca_target_repeat.values(), default=0)
            inferred_total = max(dca_steps + max(current_remain, 0), 1)
            out[64] = 1.0
            out[65] = min(current_remain / 10.0, 1.0)
            out[66] = min(dca_steps / 10.0, 1.0)
            out[67] = min(unique_targets / 6.0, 1.0)
            out[68] = min(max_repeat / 6.0, 1.0)
            out[69] = min(inferred_total / 10.0, 1.0)
            out[70] = min(dca_steps / inferred_total, 1.0)
            out[71] = min(max_repeat / max(dca_steps, 1), 1.0)
            out[72] = min(unique_targets / max(dca_steps, 1), 1.0)
            out[73] = 1.0 if int(self.last_event.get("context", -1)) == DAMAGE_COUNTER_ANY_CONTEXT else 0.0
            out[74] = _safe_feat(feats, 30)  # opponent bench damage sum
            out[75] = _safe_feat(feats, 62)  # opponent damaged Pokemon count
            out[76] = _safe_feat(feats, 63)  # max opponent in-play damage
            out[77] = _safe_feat(feats, 13)  # opponent bench fullness
            out[78] = _safe_feat(feats, 7)   # opponent prizes remaining
            out[79] = 1.0
        return out

    def _age_feature(self, typ: int) -> float:
        if typ not in self.last_seen:
            return 1.0
        age = max(self.decision_index - self.last_seen[typ], 0)
        return min(age / 16.0, 1.0)

    def _update_damage_counter_any(self, encoded: Any, ev: dict[str, Any]) -> None:
        ctx = int(ev.get("context", -1))
        if ctx != DAMAGE_COUNTER_ANY_CONTEXT:
            self.dca_active = False
            self.dca_steps = 0
            self.dca_last_remain = 0
            self.dca_target_repeat.clear()
            return

        feats = np.asarray(getattr(encoded, "state_feats", []), dtype=np.float32).reshape(-1)
        remain_before = int(round(_safe_feat(feats, 21) * 30.0))
        if (not self.dca_active) or (remain_before > self.dca_last_remain):
            self.dca_steps = 0
            self.dca_target_repeat.clear()
        self.dca_active = True
        self.dca_last_remain = max(remain_before - 1, 0)
        self.dca_steps += 1
        target_card = int(ev.get("card", 0) or 0)
        if target_card > 0:
            self.dca_target_repeat[target_card] = self.dca_target_repeat.get(target_card, 0) + 1


def _safe_feat(feats: np.ndarray, idx: int) -> float:
    if idx < 0 or idx >= feats.size:
        return 0.0
    return float(np.clip(feats[idx], 0.0, 1.0))


def future_plan_targets(
    action_types: list[int],
    actions: list[np.ndarray],
    rewards: list[float],
    won: int,
    position: int,
    game_len: int,
    *,
    horizon: int,
) -> np.ndarray:
    """Build dense future-behavior labels for every decision in a game.

    The targets are intentionally future-looking.  They force the sequence
    encoder to represent what the expert is setting up over the next few
    decisions, instead of only memorizing the current legal-option winner.
    """
    out = np.zeros(FUTURE_PLAN_DIM, dtype=np.float32)
    if game_len <= 0:
        return out
    lo = int(position)
    hi = min(game_len, lo + max(1, int(horizon)))
    window = action_types[lo:hi]
    denom = max(len(window), 1)

    for j, typ in enumerate(PLANNED_TYPES[:9]):
        out[j] = min(sum(1 for x in window if x == typ) / denom, 1.0)
    out[9] = _delay_signal(window, TYPE_ATTACK)
    out[10] = _delay_signal(window, TYPE_ATTACH)
    out[11] = _delay_signal(window, TYPE_EVOLVE)
    out[12] = _delay_signal(window, TYPE_ABILITY)
    out[13] = 1.0 if _before(window, TYPE_ATTACH, TYPE_ATTACK) else 0.0
    out[14] = 1.0 if _before(window, TYPE_EVOLVE, TYPE_ATTACK) else 0.0
    out[15] = 1.0 if _before(window, TYPE_ABILITY, TYPE_EVOLVE) else 0.0
    out[16] = min(sum(1 for a in actions[lo:hi] if np.asarray(a).size > 1) / denom, 1.0)
    out[17] = min((position + 1) / max(game_len, 1), 1.0)
    out[18] = 1.0 if position < game_len * 0.33 else 0.0
    out[19] = 1.0 if game_len * 0.33 <= position < game_len * 0.67 else 0.0
    out[20] = 1.0 if position >= game_len * 0.67 else 0.0
    out[21] = 1.0 if int(won) == 1 else 0.0
    out[22] = max(-1.0, min(1.0, float(rewards[position] if position < len(rewards) else 0.0)))
    out[23] = min((game_len - position) / max(horizon, 1), 1.0)
    return out


def _delay_signal(types: list[int], typ: int) -> float:
    for i, t in enumerate(types):
        if t == typ:
            return 1.0 / float(i + 1)
    return 0.0


def _before(types: list[int], first: int, second: int) -> bool:
    try:
        a = types.index(first)
        b = types.index(second)
    except ValueError:
        return False
    return a < b
