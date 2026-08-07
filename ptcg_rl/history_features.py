from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_ACTION_HISTORY_K = 32
DEFAULT_LOG_HISTORY_K = 128
DEFAULT_BOARD_HISTORY_K = 12
BOARD_HISTORY_FEAT_DIM = 32

MAX_LOG_TYPE = 31
MAX_LOG_AREA = 31
MAX_LOG_PLAYER = 3
MAX_SERIAL = 2047


ACTION_FIELDS = ("type", "card", "card2", "attack", "context", "select_type", "count", "mask")
LOG_FIELDS = (
    "type",
    "player",
    "card",
    "card2",
    "attack",
    "serial",
    "serial2",
    "from_area",
    "to_area",
    "value",
    "mask",
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


def _fit_1d(arr: np.ndarray, dim: int, *, dtype=np.float32) -> np.ndarray:
    out = np.zeros(dim, dtype=dtype)
    if dim <= 0:
        return out
    x = np.asarray(arr, dtype=dtype).reshape(-1)
    n = min(len(x), dim)
    if n:
        out[:n] = x[:n]
    return out


def empty_action_history(k: int) -> dict[str, np.ndarray]:
    k = max(0, int(k))
    return {
        "type": np.zeros(k, dtype=np.int16),
        "card": np.zeros(k, dtype=np.int16),
        "card2": np.zeros(k, dtype=np.int16),
        "attack": np.zeros(k, dtype=np.int16),
        "context": np.zeros(k, dtype=np.int16),
        "select_type": np.zeros(k, dtype=np.int16),
        "count": np.zeros(k, dtype=np.float16),
        "mask": np.zeros(k, dtype=np.float16),
    }


def pack_action_history(events: list[dict[str, Any]], k: int) -> dict[str, np.ndarray]:
    out = empty_action_history(k)
    if k <= 0:
        return out
    recent = events[-k:]
    start = k - len(recent)
    for pos, ev in enumerate(recent, start):
        out["type"][pos] = _clip_int(ev.get("type"), 0, 18)
        out["card"][pos] = _clip_int(ev.get("card"), 0, 4095)
        out["card2"][pos] = _clip_int(ev.get("card2"), 0, 4095)
        out["attack"][pos] = _clip_int(ev.get("attack"), 0, 4095)
        out["context"][pos] = _clip_int(ev.get("context"), 0, 65)
        out["select_type"][pos] = _clip_int(ev.get("select_type"), 0, 17)
        out["count"][pos] = np.float16(float(ev.get("count", 0.0)))
        out["mask"][pos] = np.float16(1.0)
    return out


def action_event_from_encoded(encoded: Any, action: list[int] | np.ndarray) -> dict[str, Any] | None:
    action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
    max_count = max(_safe_int(getattr(encoded, "max_count", 1), 1), 1)
    if len(action_arr) == 0:
        state_feats = np.asarray(getattr(encoded, "state_feats", []), dtype=np.float32)
        ctx = int(round(float(state_feats[17]) * 64.0)) if len(state_feats) > 17 else 0
        return {
            "type": 15,
            "card": 0,
            "card2": 0,
            "attack": 0,
            "context": max(0, min(ctx, 64)) + 1,
            "select_type": 1,
            "count": 0.0,
        }
    first = int(action_arr[0])
    opt_type = np.asarray(getattr(encoded, "opt_type", []), dtype=np.int64)
    if first < 0 or first >= len(opt_type):
        return None
    opt_card = np.asarray(getattr(encoded, "opt_card", []), dtype=np.int64)
    opt_card2 = np.asarray(getattr(encoded, "opt_card2", []), dtype=np.int64)
    opt_attack = np.asarray(getattr(encoded, "opt_attack", []), dtype=np.int64)
    opt_feats = np.asarray(getattr(encoded, "opt_feats", []), dtype=np.float32)
    ctx = 0
    sel_type = 0
    if opt_feats.ndim == 2 and first < opt_feats.shape[0]:
        if opt_feats.shape[1] > 3:
            ctx = int(round(float(opt_feats[first, 3]) * 64.0))
        if opt_feats.shape[1] > 4:
            sel_type = int(round(float(opt_feats[first, 4]) * 16.0))
    valid = [int(a) for a in action_arr if 0 <= int(a) < len(opt_type)]
    return {
        "type": int(opt_type[first]) + 1,
        "card": int(opt_card[first]) if first < len(opt_card) else 0,
        "card2": int(opt_card2[first]) if first < len(opt_card2) else 0,
        "attack": int(opt_attack[first]) if first < len(opt_attack) else 0,
        "context": max(0, min(ctx, 64)) + 1,
        "select_type": max(0, min(sel_type, 16)) + 1,
        "count": min(len(valid), max_count) / float(max_count),
    }


def _log_card2(ev: dict[str, Any]) -> int:
    for key in ("cardIdTarget", "cardIdBench", "cardIdActive", "cardId2"):
        if key in ev:
            return _safe_int(ev.get(key))
    return 0


def _log_serial2(ev: dict[str, Any]) -> int:
    for key in ("serialTarget", "serialBench", "serialActive", "serial2"):
        if key in ev:
            return _safe_int(ev.get(key))
    return 0


def log_event_from_raw(ev: dict[str, Any], *, you: int) -> dict[str, Any]:
    pi = _safe_int(ev.get("playerIndex"), -1)
    if pi == you:
        player = 1
    elif pi >= 0:
        player = 2
    else:
        player = 0
    value = 0.0
    if "value" in ev:
        try:
            value = max(-5.0, min(5.0, float(ev.get("value") or 0.0) / 400.0))
        except Exception:
            value = 0.0
    if ev.get("putDamageCounter"):
        value = abs(value) if value else 1.0
    return {
        "type": _clip_int(ev.get("type"), 0, MAX_LOG_TYPE) + 1,
        "player": player,
        "card": _clip_int(ev.get("cardId"), 0, 4095),
        "card2": _clip_int(_log_card2(ev), 0, 4095),
        "attack": _clip_int(ev.get("attackId"), 0, 4095),
        "serial": _clip_int(ev.get("serial"), 0, MAX_SERIAL) + 1 if "serial" in ev else 0,
        "serial2": _clip_int(_log_serial2(ev), 0, MAX_SERIAL) + 1 if _log_serial2(ev) else 0,
        "from_area": _clip_int(ev.get("fromArea"), 0, MAX_LOG_AREA) + 1 if "fromArea" in ev else 0,
        "to_area": _clip_int(ev.get("toArea"), 0, MAX_LOG_AREA) + 1 if "toArea" in ev else 0,
        "value": value,
    }


def empty_log_history(k: int) -> dict[str, np.ndarray]:
    k = max(0, int(k))
    return {
        "type": np.zeros(k, dtype=np.int16),
        "player": np.zeros(k, dtype=np.int8),
        "card": np.zeros(k, dtype=np.int16),
        "card2": np.zeros(k, dtype=np.int16),
        "attack": np.zeros(k, dtype=np.int16),
        "serial": np.zeros(k, dtype=np.int16),
        "serial2": np.zeros(k, dtype=np.int16),
        "from_area": np.zeros(k, dtype=np.int8),
        "to_area": np.zeros(k, dtype=np.int8),
        "value": np.zeros(k, dtype=np.float16),
        "mask": np.zeros(k, dtype=np.float16),
    }


def pack_log_history_from_obs(obs: dict[str, Any], k: int) -> dict[str, np.ndarray]:
    out = empty_log_history(k)
    if k <= 0:
        return out
    cur = obs.get("current") or {}
    you = _safe_int(cur.get("yourIndex"), 0)
    raw_logs = obs.get("logs") or []
    events = [log_event_from_raw(ev, you=you) for ev in raw_logs if isinstance(ev, dict)]
    recent = events[-k:]
    start = k - len(recent)
    for pos, ev in enumerate(recent, start):
        for key in ("type", "card", "card2", "attack", "serial", "serial2"):
            out[key][pos] = _clip_int(ev.get(key), 0, 4095)
        out["player"][pos] = _clip_int(ev.get("player"), 0, MAX_LOG_PLAYER)
        out["from_area"][pos] = _clip_int(ev.get("from_area"), 0, MAX_LOG_AREA + 1)
        out["to_area"][pos] = _clip_int(ev.get("to_area"), 0, MAX_LOG_AREA + 1)
        out["value"][pos] = np.float16(float(ev.get("value", 0.0)))
        out["mask"][pos] = np.float16(1.0)
    return out


def empty_board_history(k: int, feat_dim: int = BOARD_HISTORY_FEAT_DIM) -> dict[str, np.ndarray]:
    k = max(0, int(k))
    feat_dim = max(0, int(feat_dim))
    return {
        "cards": np.zeros((k, 12), dtype=np.int16),
        "feats": np.zeros((k, feat_dim), dtype=np.float16),
        "mask": np.zeros(k, dtype=np.float16),
    }


def board_snapshot_from_encoded(encoded: Any, feat_dim: int = BOARD_HISTORY_FEAT_DIM) -> dict[str, np.ndarray]:
    cards = np.zeros(12, dtype=np.int16)
    board = np.asarray(getattr(encoded, "board_cards", []), dtype=np.int64).reshape(-1)
    n = min(len(board), 12)
    if n:
        cards[:n] = board[:n].astype(np.int16)
    feats = _fit_1d(np.asarray(getattr(encoded, "state_feats", []), dtype=np.float32), feat_dim, dtype=np.float32)
    return {"cards": cards, "feats": feats.astype(np.float16)}


def pack_board_history(snapshots: list[dict[str, np.ndarray]], k: int,
                       feat_dim: int = BOARD_HISTORY_FEAT_DIM) -> dict[str, np.ndarray]:
    out = empty_board_history(k, feat_dim)
    if k <= 0:
        return out
    recent = snapshots[-k:]
    start = k - len(recent)
    for pos, snap in enumerate(recent, start):
        cards = np.asarray(snap.get("cards", []), dtype=np.int64).reshape(-1)
        feats = np.asarray(snap.get("feats", []), dtype=np.float32).reshape(-1)
        n_cards = min(len(cards), 12)
        n_feats = min(len(feats), feat_dim)
        if n_cards:
            out["cards"][pos, :n_cards] = cards[:n_cards].astype(np.int16)
        if n_feats:
            out["feats"][pos, :n_feats] = feats[:n_feats].astype(np.float16)
        out["mask"][pos] = np.float16(1.0)
    return out
