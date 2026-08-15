from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ptcg_rl.history_features import action_event_from_encoded
from ptcg_rl.history_features import log_event_from_raw
from ptcg_rl.v15.constants import (
    AREA_HAND,
    DEFAULT_HISTORY_K,
    EVENT_SOURCE_OWN_ACTION,
    EVENT_SOURCE_PUBLIC_LOG,
    HAND_RESET_TO_AREAS,
    KNOWN_OPP_CARDS,
    PUBLIC_TO_HAND_AREAS,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clip_int(value: Any, lo: int, hi: int, default: int = 0) -> int:
    return max(lo, min(hi, _safe_int(value, default)))


def _area(value: Any) -> int:
    return _clip_int(value, 0, 63)


def _log_type(value: Any) -> int:
    return _clip_int(value, 0, 63)


def action_event(encoded: Any, action: list[int] | np.ndarray, *, turn: int, decision: int) -> dict[str, Any]:
    """Build a live-compatible event from the selected action.

    The action type remains the engine option type id.  We do not subtract one:
    v15 deliberately keeps one canonical id space for options, labels, and
    event history.
    """
    ev = action_event_from_encoded(encoded, action)
    if ev is None:
        return {
            "event_type": 0,
            "source": EVENT_SOURCE_OWN_ACTION,
            "owner": 1,
            "card": 0,
            "card2": 0,
            "attack": 0,
            "context": 0,
            "select_type": 0,
            "from_area": 0,
            "to_area": 0,
            "value": 0.0,
            "turn": int(turn),
            "decision": int(decision),
        }
    return {
        "event_type": _clip_int(ev.get("type"), 0, 63),
        "source": EVENT_SOURCE_OWN_ACTION,
        "owner": 1,
        "card": _clip_int(ev.get("card"), 0, 4095),
        "card2": _clip_int(ev.get("card2"), 0, 4095),
        "attack": _clip_int(ev.get("attack"), 0, 4095),
        "context": _clip_int(ev.get("context"), 0, 127),
        "select_type": _clip_int(ev.get("select_type"), 0, 31),
        "from_area": 0,
        "to_area": 0,
        "value": float(ev.get("count", 0.0) or 0.0),
        "turn": int(turn),
        "decision": int(decision),
    }


def public_log_event(raw: dict[str, Any], *, you: int, turn: int, decision: int) -> dict[str, Any]:
    ev = log_event_from_raw(raw, you=you)
    player = _safe_int(raw.get("playerIndex"), -1)
    owner = 1 if player == you else 2 if player == 1 - you else 0
    return {
        "event_type": 64 + _log_type(raw.get("type")),
        "source": EVENT_SOURCE_PUBLIC_LOG,
        "owner": owner,
        "card": _clip_int(ev.get("card"), 0, 4095),
        "card2": _clip_int(ev.get("card2"), 0, 4095),
        "attack": _clip_int(ev.get("attack"), 0, 4095),
        "context": 0,
        "select_type": 0,
        "from_area": _area(raw.get("fromArea")),
        "to_area": _area(raw.get("toArea")),
        "value": float(ev.get("value", 0.0) or 0.0),
        "turn": int(turn),
        "decision": int(decision),
    }


def pack_event_history(
    events: list[dict[str, Any]],
    *,
    k: int = DEFAULT_HISTORY_K,
    current_turn: int = 0,
    current_decision: int = 0,
) -> dict[str, np.ndarray]:
    k = max(0, int(k))
    int_fields = (
        "event_type",
        "source",
        "owner",
        "card",
        "card2",
        "attack",
        "context",
        "select_type",
        "from_area",
        "to_area",
        "same_turn",
    )
    out: dict[str, np.ndarray] = {name: np.zeros(k, dtype=np.int16) for name in int_fields}
    out["value"] = np.zeros(k, dtype=np.float16)
    out["turn_delta"] = np.zeros(k, dtype=np.float16)
    out["step_delta"] = np.zeros(k, dtype=np.float16)
    out["mask"] = np.zeros(k, dtype=np.float16)
    if k <= 0:
        return out
    recent = events[-k:]
    start = k - len(recent)
    for pos, ev in enumerate(recent, start):
        ev_turn = _safe_int(ev.get("turn"), current_turn)
        ev_decision = _safe_int(ev.get("decision"), current_decision)
        for name in int_fields:
            if name == "same_turn":
                out[name][pos] = 1 if ev_turn == current_turn else 0
            else:
                out[name][pos] = _clip_int(ev.get(name), 0, 4095)
        out["value"][pos] = np.float16(float(ev.get("value", 0.0) or 0.0))
        out["turn_delta"][pos] = np.float16(min(max(current_turn - ev_turn, 0), 64) / 64.0)
        out["step_delta"][pos] = np.float16(min(max(current_decision - ev_decision, 0), 256) / 256.0)
        out["mask"][pos] = np.float16(1.0)
    return out


@dataclass
class V15Memory:
    """Per-perspective public memory used by extraction and later live policy."""

    events: list[dict[str, Any]] = field(default_factory=list)
    seen_logs: set[tuple[int, int, int, int, int, int, int]] = field(default_factory=set)
    known_opp_hand: dict[int, int] = field(default_factory=dict)
    known_seen_at: dict[int, int] = field(default_factory=dict)
    known_reveals: int = 0
    known_removes: int = 0
    known_resets: int = 0
    public_logs: int = 0

    def add_action(self, encoded: Any, action: list[int] | np.ndarray, *, turn: int, decision: int) -> dict[str, Any]:
        ev = action_event(encoded, action, turn=turn, decision=decision)
        self.events.append(ev)
        return ev

    def observe_logs(self, obs: dict[str, Any] | None, *, decision: int) -> None:
        if not isinstance(obs, dict):
            return
        cur = obs.get("current") or {}
        you = _safe_int(cur.get("yourIndex"), 0)
        turn = _safe_int(cur.get("turn"), 0)
        raw_logs = obs.get("logs") or []
        if not isinstance(raw_logs, list):
            return
        for raw in raw_logs:
            if not isinstance(raw, dict):
                continue
            player = _safe_int(raw.get("playerIndex"), -1)
            card = _clip_int(raw.get("cardId"), 0, 4095)
            serial = _safe_int(raw.get("serial"), 0)
            from_area = _area(raw.get("fromArea"))
            to_area = _area(raw.get("toArea"))
            typ = _log_type(raw.get("type"))
            key = (typ, player, card, serial, from_area, to_area, _safe_int(raw.get("cardIdTarget"), 0))
            if key in self.seen_logs:
                continue
            self.seen_logs.add(key)
            ev = public_log_event(raw, you=you, turn=turn, decision=decision)
            self.events.append(ev)
            self.public_logs += 1
            if player != 1 - you:
                continue
            if card > 0 and to_area == AREA_HAND and from_area in PUBLIC_TO_HAND_AREAS:
                self.known_opp_hand[card] = self.known_opp_hand.get(card, 0) + 1
                self.known_seen_at[card] = decision
                self.known_reveals += 1
            elif from_area == AREA_HAND and to_area != AREA_HAND:
                if card > 0:
                    cur_count = self.known_opp_hand.get(card, 0)
                    if cur_count > 1:
                        self.known_opp_hand[card] = cur_count - 1
                    elif cur_count == 1:
                        self.known_opp_hand.pop(card, None)
                        self.known_seen_at.pop(card, None)
                    self.known_removes += 1
                elif to_area in HAND_RESET_TO_AREAS:
                    self.known_opp_hand.clear()
                    self.known_seen_at.clear()
                    self.known_resets += 1

    def known_arrays(self, *, decision: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        items = sorted(
            self.known_opp_hand.items(),
            key=lambda kv: (-int(kv[1]), self.known_seen_at.get(int(kv[0]), -1), int(kv[0])),
        )[:KNOWN_OPP_CARDS]
        cards = np.zeros(KNOWN_OPP_CARDS, dtype=np.int16)
        counts = np.zeros(KNOWN_OPP_CARDS, dtype=np.float16)
        age = np.zeros(KNOWN_OPP_CARDS, dtype=np.float16)
        mask = np.zeros(KNOWN_OPP_CARDS, dtype=np.float16)
        for i, (card, count) in enumerate(items):
            cards[i] = int(card)
            counts[i] = np.float16(min(max(int(count), 0) / 4.0, 1.0))
            seen = self.known_seen_at.get(int(card), decision)
            age[i] = np.float16(min(max(decision - seen, 0), 64) / 64.0)
            mask[i] = np.float16(1.0)
        return cards, counts, age, mask

