#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.deck_registry import deck_signature
from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.v15.constants import (
    DAMAGE_COUNTER_ANY_CONTEXT,
    DEFAULT_HISTORY_K,
    DEFAULT_PLAN_STEPS,
    EVENT_FIELDS,
    FEATURE_VERSION,
    N_ACTION_TYPES,
    TYPE_ABILITY,
    TYPE_ATTACH,
    TYPE_ATTACK,
    TYPE_DISCARD,
    TYPE_END,
    TYPE_EVOLVE,
    TYPE_PLAY,
    TYPE_RETREAT,
    MODE_ATTACK,
    MODE_DAMAGE_PLAN,
    MODE_DISRUPT,
    MODE_END,
    MODE_RESOURCE,
    MODE_SETUP,
    MODE_SWITCH,
)
from ptcg_rl.v15.events import V15Memory, pack_event_history

EVENT_SAVE_NAMES = {
    "event_type": "event_type",
    "source": "event_source",
    "owner": "event_owner",
    "card": "event_card",
    "card2": "event_card2",
    "attack": "event_attack",
    "context": "event_context",
    "select_type": "event_select_type",
    "from_area": "event_from_area",
    "to_area": "event_to_area",
    "value": "event_value",
    "turn_delta": "event_turn_delta",
    "step_delta": "event_step_delta",
    "same_turn": "event_same_turn",
    "mask": "event_mask",
}

ARCHETYPES = {
    "Marnie Grimmsnarl": [648],
    "Alakazam": [743, 245, 741, 742],
    "Crustle Wall": [345, 344],
    "Dragapult": [121, 120, 119],
    "Mega Lucario": [678],
    "Archaludon": [190],
    "Cynthia Garchomp": [381],
    "Mega Lopunny": [849],
    "Teal Mask Ogerpon": [96],
    "Team Rocket Mewtwo": [431],
    "Festival Lead": [93],
    "Mega Starmie": [1031, 367],
    "Iono Bellibolt": [269],
    "Mega Abomasnow": [723],
    "N's Zoroark": [293, 320],
    "Hop Trevenant": [879],
    "Raging Bolt": [1065],
    "Metagross": [133, 134],
    "Beedrill": [15, 16],
    "Hydrapple": [1071],
    "Other": [],
}


def classify(deck: list[int]) -> str:
    cnt = Counter(deck)
    best, best_score = "Other", 0
    for name, ids in ARCHETYPES.items():
        score = sum(cnt.get(cid, 0) for cid in ids)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 2 else "Other"


def score_band(score: float) -> str:
    if score >= 1200:
        return "1200+"
    if score >= 1100:
        return "1100-1199"
    if score >= 1000:
        return "1000-1099"
    if score >= 900:
        return "900-999"
    if score >= 800:
        return "800-899"
    if score >= 700:
        return "700-799"
    return "600-699"


def load_leaderboard_scores(path: str = "") -> dict[str, float]:
    if not path:
        return {}
    out: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("TeamName") or row.get("team_name") or row.get("name") or ""
            raw = row.get("Score") or row.get("score") or ""
            if not name:
                continue
            try:
                out[str(name)] = float(raw)
            except Exception:
                out[str(name)] = 0.0
    return out


def valid_action(action: object, sel: dict[str, Any]) -> bool:
    if not isinstance(action, list) or len(action) == 60:
        return False
    n_opt = len(sel.get("option", []))
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if len(action) < mn or len(action) > mx:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(a, int) and 0 <= a < n_opt for a in action)


def action_type_for(encoded: Any, action: list[int]) -> int:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return TYPE_END
    opt_type = np.asarray(encoded.opt_type, dtype=np.int64).reshape(-1)
    first = int(arr[0])
    if first < 0 or first >= len(opt_type):
        return 0
    return int(opt_type[first])


def selected_option_fields(encoded: Any, action: list[int]) -> dict[str, int | float]:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    opt_type = np.asarray(encoded.opt_type, dtype=np.int64).reshape(-1)
    opt_card = np.asarray(encoded.opt_card, dtype=np.int64).reshape(-1)
    opt_card2 = np.asarray(encoded.opt_card2, dtype=np.int64).reshape(-1)
    opt_attack = np.asarray(encoded.opt_attack, dtype=np.int64).reshape(-1)
    opt_feats = np.asarray(encoded.opt_feats, dtype=np.float32)
    if arr.size == 0:
        feats = np.asarray(encoded.state_feats, dtype=np.float32)
        ctx = int(round(float(feats[17]) * 64.0)) if feats.size > 17 else 0
        return {"type": TYPE_END, "card": 0, "card2": 0, "attack": 0, "context": ctx, "select_type": 0, "count": 0.0}
    first = int(arr[0])
    if first < 0 or first >= len(opt_type):
        return {"type": 0, "card": 0, "card2": 0, "attack": 0, "context": 0, "select_type": 0, "count": 0.0}
    ctx = 0
    sel_type = 0
    if opt_feats.ndim == 2 and first < opt_feats.shape[0]:
        if opt_feats.shape[1] > 3:
            ctx = int(round(float(opt_feats[first, 3]) * 64.0))
        if opt_feats.shape[1] > 4:
            sel_type = int(round(float(opt_feats[first, 4]) * 16.0))
    return {
        "type": int(opt_type[first]),
        "card": int(opt_card[first]) if first < len(opt_card) else 0,
        "card2": int(opt_card2[first]) if first < len(opt_card2) else 0,
        "attack": int(opt_attack[first]) if first < len(opt_attack) else 0,
        "context": max(0, min(ctx, 127)),
        "select_type": max(0, min(sel_type, 31)),
        "count": min(len(arr), max(int(getattr(encoded, "max_count", 1) or 1), 1)) / float(max(int(getattr(encoded, "max_count", 1) or 1), 1)),
    }


def plan_mode_for(types: list[int], contexts: list[int]) -> int:
    if any(ctx == DAMAGE_COUNTER_ANY_CONTEXT for ctx in contexts):
        return MODE_DAMAGE_PLAN
    if any(t == TYPE_ATTACK for t in types):
        return MODE_ATTACK
    if any(t in (TYPE_ATTACH, TYPE_PLAY) for t in types):
        return MODE_RESOURCE
    if any(t == TYPE_EVOLVE for t in types):
        return MODE_SETUP
    if any(t in (TYPE_DISCARD,) for t in types):
        return MODE_DISRUPT
    if any(t == TYPE_RETREAT for t in types):
        return MODE_SWITCH
    if types and all(t == TYPE_END for t in types):
        return MODE_END
    return MODE_SETUP


def _first_action_index(row: dict[str, Any]) -> int:
    arr = np.asarray(row.get("action", []), dtype=np.int64).reshape(-1)
    return int(arr[0]) if arr.size else -1


def annotate_dca(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["dca_mask"] = 0
        row["dca_group_index"] = -1
        row["dca_pos"] = -1
        row["dca_len"] = 0
        row["dca_group_unique"] = 0
        row["dca_group_focus"] = 0.0
    group = 0
    i = 0
    while i < len(rows):
        if int(rows[i].get("act_context", -1)) != DAMAGE_COUNTER_ANY_CONTEXT:
            i += 1
            continue
        j = i
        while j < len(rows) and int(rows[j].get("act_context", -1)) == DAMAGE_COUNTER_ANY_CONTEXT:
            j += 1
        block = rows[i:j]
        slots = [_first_action_index(r) for r in block]
        counts: dict[int, int] = {}
        for slot in slots:
            if slot >= 0:
                counts[slot] = counts.get(slot, 0) + 1
        unique = len(counts)
        focus = max(counts.values(), default=0) / max(len(block), 1)
        for pos, row in enumerate(block):
            row["dca_mask"] = 1
            row["dca_group_index"] = group
            row["dca_pos"] = pos
            row["dca_len"] = len(block)
            row["dca_group_unique"] = unique
            row["dca_group_focus"] = focus
        group += 1
        i = j


def annotate_turn_blocks(rows: list[dict[str, Any]], *, plan_steps: int) -> None:
    by_turn: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_turn[int(row.get("turn_number", 0))].append(idx)
    for row in rows:
        row["plan_mask"] = np.zeros(plan_steps, dtype=np.float16)
        row["plan_type"] = np.zeros(plan_steps, dtype=np.int16)
        row["plan_card"] = np.zeros(plan_steps, dtype=np.int16)
        row["plan_card2"] = np.zeros(plan_steps, dtype=np.int16)
        row["plan_attack"] = np.zeros(plan_steps, dtype=np.int16)
        row["plan_context"] = np.zeros(plan_steps, dtype=np.int16)
        row["plan_mode"] = 0
        row["turn_continue"] = 0
        row["turn_remaining"] = 0
        row["block_pos"] = 0
        row["block_len"] = 1
        row["block_remaining"] = 0
        row["block_type_counts"] = np.zeros(N_ACTION_TYPES, dtype=np.float16)
    for indices in by_turn.values():
        indices.sort()
        block_types = [int(rows[j]["act_type"]) for j in indices]
        block_ctx = [int(rows[j]["act_context"]) for j in indices]
        type_counts = np.zeros(N_ACTION_TYPES, dtype=np.float32)
        for typ in block_types:
            if 0 <= typ < N_ACTION_TYPES:
                type_counts[typ] += 1.0
        if len(indices) > 0:
            type_counts = type_counts / max(float(len(indices)), 1.0)
        mode = plan_mode_for(block_types, block_ctx)
        for pos, idx in enumerate(indices):
            row = rows[idx]
            future = indices[pos + 1: pos + 1 + plan_steps]
            row["block_pos"] = pos
            row["block_len"] = len(indices)
            row["block_remaining"] = len(indices) - pos - 1
            row["turn_continue"] = 1 if future else 0
            row["turn_remaining"] = len(indices) - pos - 1
            row["block_type_counts"] = type_counts.astype(np.float16)
            row["plan_mode"] = mode
            for p, fidx in enumerate(future):
                f = rows[fidx]
                row["plan_mask"][p] = np.float16(1.0)
                row["plan_type"][p] = int(f["act_type"])
                row["plan_card"][p] = int(f["act_card"])
                row["plan_card2"][p] = int(f["act_card2"])
                row["plan_attack"][p] = int(f["act_attack"])
                row["plan_context"][p] = int(f["act_context"])


def make_row(
    *,
    encoded: Any,
    action: list[int],
    event_hist: dict[str, np.ndarray],
    known: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    deck: list[int],
    deck_sig: str,
    team_name: str,
    score: float,
    opponent_deck: list[int],
    opponent_deck_sig: str,
    opponent_team_name: str,
    opponent_score: float,
    episode_id: str,
    player_index: int,
    reward: float,
    won: int,
    draw: int,
    final_status: str,
    game_steps: int,
    step_index: int,
    decision_index: int,
    turn_number: int,
    turn_action_count: int,
) -> dict[str, Any]:
    fields = selected_option_fields(encoded, action)
    row = {
        "board": np.asarray(encoded.board_cards, dtype=np.int16),
        "hand": np.asarray(encoded.hand_cards, dtype=np.int16),
        "feats": np.asarray(encoded.state_feats, dtype=np.float16),
        "state_token_feats": np.asarray(encoded.state_token_feats, dtype=np.float16),
        "ot": np.asarray(encoded.opt_type, dtype=np.int16),
        "oc": np.asarray(encoded.opt_card, dtype=np.int16),
        "oc2": np.asarray(encoded.opt_card2, dtype=np.int16),
        "oa": np.asarray(encoded.opt_attack, dtype=np.int16),
        "of_arr": np.asarray(encoded.opt_feats, dtype=np.float16),
        "action": np.asarray(action, dtype=np.int16),
        "act_type": int(fields["type"]),
        "act_card": int(fields["card"]),
        "act_card2": int(fields["card2"]),
        "act_attack": int(fields["attack"]),
        "act_context": int(fields["context"]),
        "act_select_type": int(fields["select_type"]),
        "act_count": float(fields["count"]),
        "known_cards": np.asarray(known[0], dtype=np.int16),
        "known_counts": np.asarray(known[1], dtype=np.float16),
        "known_age": np.asarray(known[2], dtype=np.float16),
        "known_mask": np.asarray(known[3], dtype=np.float16),
        "min_c": int(encoded.min_count),
        "max_c": int(encoded.max_count),
        "deck_sig": deck_sig,
        "archetype": classify(deck),
        "team_name": team_name,
        "score": float(score),
        "score_band": score_band(score),
        "opponent_deck_sig": opponent_deck_sig,
        "opponent_archetype": classify(opponent_deck),
        "opponent_team_name": opponent_team_name,
        "opponent_score": float(opponent_score),
        "opponent_score_band": score_band(opponent_score),
        "episode_id": episode_id,
        "player_index": int(player_index),
        "game_key": f"{episode_id}:{player_index}",
        "reward": float(reward),
        "won": int(won),
        "draw": int(draw),
        "final_status": final_status,
        "game_steps": int(game_steps),
        "step_index": int(step_index),
        "decision_index": int(decision_index),
        "turn_number": int(turn_number),
        "turn_action_count": int(turn_action_count),
    }
    for key, val in event_hist.items():
        row[EVENT_SAVE_NAMES.get(key, f"event_{key}")] = val
    return row


def _save_rows(dest: Path, rows: list[dict[str, Any]], *, history_k: int, plan_steps: int) -> None:
    def obj(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in rows], dtype=object)

    def stack(name: str, dtype) -> np.ndarray:
        return np.stack([np.asarray(r[name]) for r in rows]).astype(dtype)

    save: dict[str, np.ndarray] = {
        "board": stack("board", np.int16),
        "hand": stack("hand", np.int16),
        "feats": stack("feats", np.float16),
        "state_token_feats": stack("state_token_feats", np.float16),
        "ot": obj("ot"),
        "oc": obj("oc"),
        "oc2": obj("oc2"),
        "oa": obj("oa"),
        "of_arr": obj("of_arr"),
        "action": obj("action"),
        "known_cards": stack("known_cards", np.int16),
        "known_counts": stack("known_counts", np.float16),
        "known_age": stack("known_age", np.float16),
        "known_mask": stack("known_mask", np.float16),
        "plan_mask": stack("plan_mask", np.float16),
        "plan_type": stack("plan_type", np.int16),
        "plan_card": stack("plan_card", np.int16),
        "plan_card2": stack("plan_card2", np.int16),
        "plan_attack": stack("plan_attack", np.int16),
        "plan_context": stack("plan_context", np.int16),
        "block_type_counts": stack("block_type_counts", np.float16),
    }
    for src, dest_name in EVENT_SAVE_NAMES.items():
        dtype = np.float16 if src in ("value", "turn_delta", "step_delta", "mask") else np.int16
        save[dest_name] = stack(dest_name, dtype)
    scalar_int = [
        "act_type",
        "act_card",
        "act_card2",
        "act_attack",
        "act_context",
        "act_select_type",
        "plan_mode",
        "turn_continue",
        "turn_remaining",
        "block_pos",
        "block_len",
        "block_remaining",
        "dca_mask",
        "dca_group_index",
        "dca_pos",
        "dca_len",
        "dca_group_unique",
        "min_c",
        "max_c",
        "player_index",
        "won",
        "draw",
        "game_steps",
        "step_index",
        "decision_index",
        "turn_number",
        "turn_action_count",
    ]
    for name in scalar_int:
        save[name] = np.asarray([r[name] for r in rows], dtype=np.int16)
    save["act_count"] = np.asarray([r["act_count"] for r in rows], dtype=np.float16)
    save["dca_group_focus"] = np.asarray([r["dca_group_focus"] for r in rows], dtype=np.float16)
    save["deck_sig"] = np.asarray([r["deck_sig"] for r in rows], dtype=object)
    save["archetype"] = np.asarray([r["archetype"] for r in rows], dtype=object)
    save["team_name"] = np.asarray([r["team_name"] for r in rows], dtype=object)
    save["score"] = np.asarray([r["score"] for r in rows], dtype=np.float32)
    save["score_band"] = np.asarray([r["score_band"] for r in rows], dtype=object)
    save["opponent_deck_sig"] = np.asarray([r["opponent_deck_sig"] for r in rows], dtype=object)
    save["opponent_archetype"] = np.asarray([r["opponent_archetype"] for r in rows], dtype=object)
    save["opponent_team_name"] = np.asarray([r["opponent_team_name"] for r in rows], dtype=object)
    save["opponent_score"] = np.asarray([r["opponent_score"] for r in rows], dtype=np.float32)
    save["opponent_score_band"] = np.asarray([r["opponent_score_band"] for r in rows], dtype=object)
    save["episode_id"] = np.asarray([r["episode_id"] for r in rows], dtype=object)
    save["game_key"] = np.asarray([r["game_key"] for r in rows], dtype=object)
    save["reward"] = np.asarray([r["reward"] for r in rows], dtype=np.float32)
    save["final_status"] = np.asarray([r["final_status"] for r in rows], dtype=object)
    save["feature_version"] = np.asarray(FEATURE_VERSION, dtype=object)
    save["state_feat_dim"] = np.asarray(STATE_FEAT_DIM, dtype=np.int16)
    save["opt_feat_dim"] = np.asarray(OPT_FEAT_DIM, dtype=np.int16)
    save["state_token_feat_dim"] = np.asarray(STATE_TOKEN_FEAT_DIM, dtype=np.int16)
    save["history_k"] = np.asarray(history_k, dtype=np.int16)
    save["plan_steps"] = np.asarray(plan_steps, dtype=np.int16)
    np.savez_compressed(dest, **save)


def process_zip(
    zip_path: str,
    out_dir: str,
    name_to_score: dict[str, float],
    *,
    history_k: int,
    plan_steps: int,
    min_rows: int,
    progress_every: int,
    max_episodes: int,
) -> int:
    from ptcg_rl.encoder import FastEncoder

    encoder = FastEncoder()
    zp = Path(zip_path)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    bad_actions = 0
    errors = 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if max_episodes > 0:
            names = names[:max_episodes]
        print(f"{zp.name}: {len(names)} eps", flush=True)
        for ep_i, name in enumerate(names, 1):
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                steps = data.get("steps") or []
                if len(steps) < 2:
                    continue
                decks_raw = steps[0][0].get("visualize", [{}])[0].get("action", [])
                if len(decks_raw) != 2:
                    continue
                decks = [list(decks_raw[0]), list(decks_raw[1])]
                if len(decks[0]) != 60 or len(decks[1]) != 60:
                    continue
                deck_sigs = [deck_signature(decks[0]), deck_signature(decks[1])]
                info = data.get("info", {})
                teams = list(info.get("TeamNames") or ["", ""])[:2]
                while len(teams) < 2:
                    teams.append("")
                scores = [float(name_to_score.get(teams[i], 0.0)) for i in range(2)]
                rewards_raw = data.get("rewards") or [0.0, 0.0]
                rewards = [float(rewards_raw[i] if i < len(rewards_raw) and rewards_raw[i] is not None else 0.0) for i in range(2)]
                statuses_raw = data.get("statuses") or ["", ""]
                statuses = [str(statuses_raw[i]) if i < len(statuses_raw) else "" for i in range(2)]
                wins = [int(rewards[i] > rewards[1 - i]) for i in range(2)]
                draws = [int(rewards[i] == rewards[1 - i]) for i in range(2)]
                episode_id = str(data.get("id") or info.get("EpisodeId") or Path(name).stem)
                memories = [V15Memory(), V15Memory()]
                pending: list[dict[str, Any] | None] = [None, None]
                player_rows: list[list[dict[str, Any]]] = [[], []]

                for step_index, step in enumerate(steps[1:], 1):
                    for pi, pd in enumerate(step[:2]):
                        if not isinstance(pd, dict):
                            continue
                        action = pd.get("action")
                        if pending[pi] is not None and isinstance(action, list) and len(action) != 60:
                            pend = pending[pi]
                            obs_prev = pend["obs"]
                            sel_prev = obs_prev.get("select") if isinstance(obs_prev, dict) else None
                            if isinstance(sel_prev, dict) and valid_action(action, sel_prev):
                                encoded = pend["encoded"]
                                row = make_row(
                                    encoded=encoded,
                                    action=action,
                                    event_hist=pend["event_hist"],
                                    known=pend["known"],
                                    deck=decks[pi],
                                    deck_sig=deck_sigs[pi],
                                    team_name=teams[pi],
                                    score=scores[pi],
                                    opponent_deck=decks[1 - pi],
                                    opponent_deck_sig=deck_sigs[1 - pi],
                                    opponent_team_name=teams[1 - pi],
                                    opponent_score=scores[1 - pi],
                                    episode_id=episode_id,
                                    player_index=pi,
                                    reward=rewards[pi],
                                    won=wins[pi],
                                    draw=draws[pi],
                                    final_status=statuses[pi],
                                    game_steps=len(steps),
                                    step_index=int(pend["step_index"]),
                                    decision_index=int(pend["decision_index"]),
                                    turn_number=int(pend["turn_number"]),
                                    turn_action_count=int(pend["turn_action_count"]),
                                )
                                player_rows[pi].append(row)
                                memories[pi].add_action(
                                    encoded,
                                    action,
                                    turn=int(pend["turn_number"]),
                                    decision=int(pend["decision_index"]),
                                )
                            else:
                                bad_actions += 1
                            pending[pi] = None

                        obs = pd.get("observation")
                        obs = obs if isinstance(obs, dict) else None
                        if obs:
                            memories[pi].observe_logs(obs, decision=len(player_rows[pi]))
                        sel = obs.get("select") if obs else None
                        if pd.get("status") == "ACTIVE" and isinstance(sel, dict) and len(sel.get("option", [])) > 0:
                            try:
                                encoded = encoder.encode(obs)
                                cur = obs.get("current") or {}
                                turn = int(cur.get("turn", 0) or 0)
                                decision = len(player_rows[pi])
                                event_hist = pack_event_history(
                                    memories[pi].events,
                                    k=history_k,
                                    current_turn=turn,
                                    current_decision=decision,
                                )
                                pending[pi] = {
                                    "obs": obs,
                                    "encoded": encoded,
                                    "event_hist": event_hist,
                                    "known": memories[pi].known_arrays(decision=decision),
                                    "step_index": step_index,
                                    "decision_index": decision,
                                    "turn_number": turn,
                                    "turn_action_count": int(cur.get("turnActionCount", 0) or 0),
                                }
                            except Exception:
                                errors += 1
                                pending[pi] = None

                for pi in (0, 1):
                    rows = player_rows[pi]
                    if len(rows) < 2:
                        continue
                    annotate_dca(rows)
                    annotate_turn_blocks(rows, plan_steps=plan_steps)
                    for row in rows:
                        key = f"{row['archetype']}|{row['score_band']}"
                        by_bucket[key].append(row)
                        total_rows += 1
            except Exception:
                errors += 1
            if progress_every and (ep_i == 1 or ep_i % progress_every == 0 or ep_i == len(names)):
                elapsed = time.time() - t0
                rate = ep_i / max(elapsed, 1e-9)
                eta = (len(names) - ep_i) / max(rate, 1e-9)
                print(
                    f"  {zp.name} {ep_i}/{len(names)} eps rows={total_rows} bad={bad_actions} "
                    f"err={errors} {rate:.1f} eps/s eta={eta:.0f}s",
                    flush=True,
                )

    written = 0
    for key, rows in sorted(by_bucket.items()):
        if len(rows) < min_rows:
            continue
        arch, band = key.split("|", 1)
        out_path = Path(out_dir) / arch.replace(" ", "_") / band.replace(" ", "_")
        out_path.mkdir(parents=True, exist_ok=True)
        dest = out_path / zp.name.replace(".zip", ".v15block.npz")
        _save_rows(dest, rows, history_k=history_k, plan_steps=plan_steps)
        written += len(rows)
        print(f"  {key}: {len(rows)} rows -> {dest} ({dest.stat().st_size / 1024**2:.1f}MB)", flush=True)
    print(f"  Done {zp.name}: written={written} total={total_rows} bad={bad_actions} err={errors}\n", flush=True)
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--lb-csv", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K)
    p.add_argument("--plan-steps", type=int, default=DEFAULT_PLAN_STEPS)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    args = p.parse_args()

    scores = load_leaderboard_scores(args.lb_csv)
    print(f"Leaderboard scores loaded: {len(scores)}" if scores else "WARNING: no leaderboard score CSV; scores default to 0", flush=True)
    zips = sorted(str(p) for p in Path(args.episodes_dir).glob("*.zip"))
    if args.date_from or args.date_to:
        zips = [z for z in zips if (not args.date_from or Path(z).name >= f"pokemon-tcg-ai-battle-episodes-{args.date_from}.zip") and (not args.date_to or Path(z).name <= f"pokemon-tcg-ai-battle-episodes-{args.date_to}.zip")]
    if not zips:
        raise FileNotFoundError(f"no episode zips found under {args.episodes_dir}")
    if args.workers <= 1:
        total = 0
        for zp in zips:
            total += process_zip(
                zp,
                args.out,
                scores,
                history_k=args.history_k,
                plan_steps=args.plan_steps,
                min_rows=args.min_rows,
                progress_every=args.progress_every,
                max_episodes=args.max_episodes,
            )
        print(f"All done: {total} rows", flush=True)
        return
    workers = min(int(args.workers), len(zips))
    print(f"Processing {len(zips)} zips with {workers} workers", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                process_zip,
                zp,
                args.out,
                scores,
                history_k=args.history_k,
                plan_steps=args.plan_steps,
                min_rows=args.min_rows,
                progress_every=args.progress_every,
                max_episodes=args.max_episodes,
            )
            for zp in zips
        ]
        for done, fut in enumerate(as_completed(futs), 1):
            total += int(fut.result())
            print(f"Finished {done}/{len(futs)} zips total_rows={total}", flush=True)
    print(f"All done: {total} rows", flush=True)


if __name__ == "__main__":
    main()
