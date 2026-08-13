#!/usr/bin/env python3
"""Generate search-improved action labels for weakness-state banks.

For each saved decision state, this tool enumerates a bounded set of legal root
actions, rolls each action forward inside the engine search API, and estimates
candidate-perspective win rate against fixed rollout policies. The resulting
CSV/JSONL is a planner-teacher target: it can drive rule design, hard-negative
analysis, or later scratch distillation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.encoder import FastEncoder
from ptcg_rl.numpy_policy import NumpyPolicy
from tools.eval_round_robin import Entry, parse_entry, read_deck
from tools.trace_matchup_decisions import card_name, type_name


ACTION_FIELDS = [
    "state_id",
    "candidate",
    "opponent",
    "outcome",
    "turn",
    "context",
    "option_count",
    "action_key",
    "action",
    "action_desc",
    "is_baseline",
    "is_policy_top1",
    "rollouts",
    "wins",
    "losses",
    "draws",
    "errors",
    "win_rate",
    "avg_score",
    "avg_steps",
    "baseline_action",
    "baseline_win_rate",
    "baseline_score",
    "best_action",
    "best_win_rate",
    "best_score",
    "delta_vs_baseline",
    "delta_score_vs_baseline",
]

BEST_FIELDS = [
    "state_id",
    "candidate",
    "opponent",
    "outcome",
    "turn",
    "context",
    "option_count",
    "baseline_action",
    "baseline_desc",
    "baseline_win_rate",
    "baseline_score",
    "best_action",
    "best_desc",
    "best_win_rate",
    "best_score",
    "delta_vs_baseline",
    "delta_score_vs_baseline",
    "evaluated_actions",
    "rollouts_per_action",
]

_CANDIDATE: Entry | None = None
_OPPONENT: Entry | None = None
_ARGS: argparse.Namespace | None = None


def has_cg_search() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("cg.api") is not None
    except ModuleNotFoundError:
        return False


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def open_text(path: Path, mode: str):
    if "w" in mode:
        path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_entry(spec: str) -> Entry:
    name, policy_path, deck_path = parse_entry(spec, default_deck="deck.csv")
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
    return Entry(name, policy_path, deck_path, policy, deck)


def reset_policy(entry: Entry) -> None:
    if entry.policy is not None and hasattr(entry.policy, "reset_history"):
        entry.policy.reset_history()


def obj_to_plain(obj: Any, depth: int = 0) -> Any:
    if depth > 64:
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): obj_to_plain(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [obj_to_plain(v, depth + 1) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj_to_plain(obj.model_dump(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj_to_plain(obj.dict(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): obj_to_plain(v, depth + 1)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(obj)


def legal_random(sel: dict[str, Any], rng: random.Random) -> list[int]:
    opts = sel.get("option", [])
    mn = safe_int(sel.get("minCount"))
    mx = safe_int(sel.get("maxCount"))
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = rng.randint(lo, hi)
    return rng.sample(range(len(opts)), k) if k > 0 else []


def valid_action(action: list[int], sel: dict[str, Any]) -> bool:
    opts = sel.get("option", [])
    mn = safe_int(sel.get("minCount"))
    mx = safe_int(sel.get("maxCount"))
    if len(action) < mn or len(action) > mx:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(i, int) and 0 <= i < len(opts) for i in action)


def action_key(action: list[int]) -> str:
    return ",".join(str(int(i)) for i in action) if action else "STOP"


def parse_action(value: Any) -> list[int]:
    if isinstance(value, str):
        value = value.strip()
        if not value or value == "STOP":
            return []
        try:
            return [int(x) for x in value.replace(";", ",").split(",") if x.strip()]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []


def policy_top_action(record: dict[str, Any], sel: dict[str, Any]) -> list[int]:
    ranking = record.get("policy_ranking") or []
    if not ranking:
        return []
    idx = safe_int(ranking[0].get("index"), -1)
    if idx < 0:
        return []
    action = [idx]
    if valid_action(action, sel):
        return action
    return []


def candidate_policy_action(policy: NumpyPolicy | None, obs: dict[str, Any]) -> list[int]:
    sel = obs.get("select") or {}
    if policy is None:
        return []
    try:
        action = policy.select(obs, greedy=True, update_history=False)
    except Exception:
        return []
    return action if valid_action(action, sel) else []


def enumerate_actions(
    record: dict[str, Any],
    policy: NumpyPolicy | None,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[list[int]]:
    obs = record.get("obs") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    n = len(opts)
    mn = safe_int(sel.get("minCount"))
    mx = safe_int(sel.get("maxCount"))
    if n <= 0 or mx <= 0:
        return []

    actions: list[list[int]] = []
    seen: set[str] = set()

    def add(action: list[int]) -> None:
        action = [int(i) for i in action if 0 <= int(i) < n]
        action = list(dict.fromkeys(action))
        if not valid_action(action, sel):
            return
        key = action_key(action)
        if key in seen:
            return
        seen.add(key)
        actions.append(action)

    add(parse_action(record.get("baseline_action", [])))
    add(policy_top_action(record, sel))
    add(candidate_policy_action(policy, obs))
    if mn <= 0:
        add([])

    ranking = record.get("policy_ranking") or []
    top_indices = [
        safe_int(r.get("index"), -1)
        for r in ranking
        if 0 <= safe_int(r.get("index"), -1) < n
    ]
    for i in range(n):
        if i not in top_indices:
            top_indices.append(i)
        if len(top_indices) >= max(args.root_top_options, args.combo_top_options):
            break

    for i in top_indices[: args.root_top_options]:
        add([i])

    if mx > 1:
        combo_pool = top_indices[: min(args.combo_top_options, n)]
        min_k = max(1, mn)
        max_k = min(mx, len(combo_pool), args.max_combo_size)
        for k in range(min_k, max_k + 1):
            for combo in itertools.combinations(combo_pool, k):
                add(list(combo))
                if len(actions) >= args.max_actions:
                    break
            if len(actions) >= args.max_actions:
                break

    for _ in range(args.random_actions):
        add(legal_random(sel, rng))
        if len(actions) >= args.max_actions:
            break

    return actions[: args.max_actions]


def option_card_from_raw(obs: dict[str, Any], option_index: int) -> int:
    try:
        encoded = FastEncoder().encode(obs)
        if 0 <= option_index < len(encoded.opt_card):
            return int(encoded.opt_card[option_index])
    except Exception:
        pass
    return 0


def describe_action(obs: dict[str, Any], action: list[int]) -> str:
    try:
        encoded = FastEncoder().encode(obs)
    except Exception:
        encoded = None
    parts = []
    for idx in action:
        opt = (obs.get("select") or {}).get("option", [])
        raw_type = safe_int(opt[idx].get("type")) if 0 <= idx < len(opt) else 0
        typ = int(encoded.opt_type[idx]) if encoded is not None and 0 <= idx < len(encoded.opt_type) else raw_type
        cid = int(encoded.opt_card[idx]) if encoded is not None and 0 <= idx < len(encoded.opt_card) else 0
        cid2 = int(encoded.opt_card2[idx]) if encoded is not None and 0 <= idx < len(encoded.opt_card2) else 0
        label = f"{idx}:{type_name(typ)}"
        if cid:
            label += f":{card_name(cid)}"
        if cid2 and cid2 != cid:
            label += f"->{card_name(cid2)}"
        parts.append(label)
    return " | ".join(parts) if parts else "STOP"


def visible_card_ids(obs: dict[str, Any], player_idx: int) -> list[int]:
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if not (0 <= player_idx < len(players)):
        return []
    player = players[player_idx] or {}
    out: list[int] = []

    def add_card(card: Any) -> None:
        if isinstance(card, dict):
            cid = safe_int(card.get("id"))
            if cid:
                out.append(cid)

    for zone in ("hand", "discard", "prize"):
        for card in player.get(zone) or []:
            add_card(card)
    for p in player.get("active") or []:
        add_card(p)
        for zone in ("tools", "energyCards", "energies"):
            for card in (p or {}).get(zone) or []:
                add_card(card)
    for p in player.get("bench") or []:
        add_card(p)
        for zone in ("tools", "energyCards", "energies"):
            for card in (p or {}).get(zone) or []:
                add_card(card)
    for card in cur.get("stadium") or []:
        add_card(card)
    for card in cur.get("looking") or []:
        add_card(card)
    return out


def remaining_pool(full_deck: list[int], obs: dict[str, Any], player_idx: int, count: int, rng: random.Random) -> list[int]:
    remaining = Counter(int(c) for c in full_deck)
    for cid in visible_card_ids(obs, player_idx):
        if remaining[cid] > 0:
            remaining[cid] -= 1
    pool: list[int] = []
    for cid, n in remaining.items():
        pool.extend([cid] * max(0, int(n)))
    if not pool:
        pool = list(full_deck) or [1]
    rng.shuffle(pool)
    if len(pool) < count:
        pad_src = list(full_deck) or pool or [1]
        while len(pool) < count:
            pool.append(rng.choice(pad_src))
    return pool[: max(0, count)]


def hidden_args(
    obs: dict[str, Any],
    candidate_side: int,
    candidate_deck: list[int],
    opponent_deck: list[int],
    rng: random.Random,
) -> dict[str, list[int]]:
    cur = obs.get("current") or {}
    players = cur.get("players") or [{}, {}]
    you = safe_int(cur.get("yourIndex"), candidate_side)
    opp_idx = 1 - you
    decks = {
        candidate_side: candidate_deck,
        1 - candidate_side: opponent_deck,
    }

    me = players[you] if you < len(players) else {}
    opp = players[opp_idx] if opp_idx < len(players) else {}
    my_deck = remaining_pool(decks.get(you, candidate_deck), obs, you, safe_int(me.get("deckCount")), rng)
    opp_deck = remaining_pool(decks.get(opp_idx, opponent_deck), obs, opp_idx, safe_int(opp.get("deckCount")), rng)
    my_prize_n = len(me.get("prize") or [])
    opp_prize_n = len(opp.get("prize") or [])
    opp_hand_n = safe_int(opp.get("handCount"))
    opp_active = []
    try:
        active = opp.get("active") or []
        if active and active[0] is None:
            opp_active = [rng.choice(decks.get(opp_idx, opponent_deck) or [1])]
    except Exception:
        pass
    return {
        "your_deck": my_deck,
        "your_prize": remaining_pool(decks.get(you, candidate_deck), obs, you, my_prize_n, rng),
        "opponent_deck": opp_deck,
        "opponent_prize": remaining_pool(decks.get(opp_idx, opponent_deck), obs, opp_idx, opp_prize_n, rng),
        "opponent_hand": remaining_pool(decks.get(opp_idx, opponent_deck), obs, opp_idx, max(1, opp_hand_n), rng)
        if opp_hand_n > 0 else [],
        "opponent_active": opp_active,
    }


def damage_ratio(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 0.0
    hp = float(pokemon.get("hp", 0) or 0)
    max_hp = float(pokemon.get("maxHp", 0) or 0)
    if max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, (max_hp - hp) / max_hp))


def energy_count(pokemon: dict[str, Any] | None) -> int:
    if not pokemon:
        return 0
    if pokemon.get("energies") is not None:
        return len(pokemon.get("energies") or [])
    return len(pokemon.get("energyCards") or [])


def in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    active = player.get("active") or []
    if active and active[0]:
        out.append(active[0])
    out.extend([p for p in (player.get("bench") or []) if p])
    return out


def active_pokemon(player: dict[str, Any]) -> dict[str, Any] | None:
    active = player.get("active") or []
    if active and active[0]:
        return active[0]
    return None


def heuristic_score(obs: dict[str, Any], candidate_side: int) -> float:
    """Candidate-perspective non-terminal score for horizon cutoffs."""
    cur = obs.get("current") or {}
    players = cur.get("players") or [{}, {}]
    if not (0 <= candidate_side < len(players)):
        return 0.0
    cand = players[candidate_side] or {}
    opp = players[1 - candidate_side] if 1 - candidate_side < len(players) else {}
    cand_active = active_pokemon(cand)
    opp_active = active_pokemon(opp)
    cand_in_play = in_play(cand)
    opp_in_play = in_play(opp)

    cand_prizes = len(cand.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    prize_term = 0.75 * ((opp_prizes - cand_prizes) / 6.0)
    active_damage = 0.30 * (damage_ratio(opp_active) - damage_ratio(cand_active))
    bench_damage = 0.10 * (
        sum(damage_ratio(p) for p in opp_in_play) - sum(damage_ratio(p) for p in cand_in_play)
    ) / 6.0
    setup_term = 0.08 * ((len(cand_in_play) - len(opp_in_play)) / 6.0)
    energy_term = 0.06 * (
        min(max(energy_count(p) for p in cand_in_play), 6) if cand_in_play else 0
    ) / 6.0
    hand_term = 0.04 * (
        (len(cand.get("hand") or []) if cand.get("hand") is not None else safe_int(cand.get("handCount")))
        - safe_int(opp.get("handCount"))
    ) / 10.0
    return float(np.clip(prize_term + active_damage + bench_damage + setup_term + energy_term + hand_term, -0.95, 0.95))


def remember_external_action(policy: NumpyPolicy | None, obs: dict[str, Any], action: list[int]) -> None:
    if policy is None:
        return
    try:
        policy.remember_decision(obs, action)
    except Exception:
        pass


def rollout_policy_action(entry: Entry, obs: dict[str, Any], rng: random.Random, *, temperature: float) -> list[int]:
    sel = obs.get("select") or {}
    if not sel.get("option"):
        return []
    if entry.policy is None:
        return legal_random(sel, rng)
    try:
        if temperature > 0:
            action = entry.policy.select(obs, greedy=False, temperature=temperature, top_k=0, update_history=False)
        else:
            action = entry.policy.select(obs, greedy=True, update_history=False)
    except Exception:
        action = legal_random(sel, rng)
    if not valid_action(action, sel):
        action = legal_random(sel, rng)
    try:
        entry.policy.remember_decision(obs, action)
    except Exception:
        pass
    return action


def rollout_once(record: dict[str, Any], root_action: list[int], seed: int, args: argparse.Namespace) -> tuple[str, int, float]:
    from cg.api import search_begin, search_end, search_step, to_observation_class

    assert _CANDIDATE is not None and _OPPONENT is not None
    candidate = _CANDIDATE
    opponent = _OPPONENT

    rng = random.Random(seed)
    obs0 = deepcopy(record.get("obs") or {})
    candidate_side = safe_int(record.get("candidate_side"))
    for entry in (candidate, opponent):
        reset_policy(entry)

    try:
        obs_cls = to_observation_class(obs0)
        ss = search_begin(obs_cls, **hidden_args(obs0, candidate_side, candidate.deck, opponent.deck, rng))
        root_sid = ss.searchId
        root_obs_plain = obj_to_plain(ss.observation)
        step_result = search_step(root_sid, root_action)
        sid = step_result.searchId
        obs_plain = obj_to_plain(step_result.observation)
        remember_external_action(candidate.policy, root_obs_plain if isinstance(root_obs_plain, dict) else obs0, root_action)

        for step in range(args.rollout_horizon):
            cur = obs_plain.get("current") if isinstance(obs_plain, dict) else {}
            res = safe_int((cur or {}).get("result"), -1)
            if res != -1:
                search_end()
                if res == 2:
                    return "draw", step, 0.0
                return ("win" if res == candidate_side else "loss"), step, (1.0 if res == candidate_side else -1.0)
            sel = obs_plain.get("select") if isinstance(obs_plain, dict) else None
            if not sel:
                search_end()
                return "draw", step, heuristic_score(obs_plain, candidate_side) if isinstance(obs_plain, dict) else 0.0
            side = safe_int((cur or {}).get("yourIndex"), candidate_side)
            entry = candidate if side == candidate_side else opponent
            temp = args.candidate_rollout_temperature if side == candidate_side else args.opponent_rollout_temperature
            action = rollout_policy_action(entry, obs_plain, rng, temperature=temp)
            try:
                nxt = search_step(sid, action)
            except Exception:
                search_end()
                return "error", step, -0.25
            sid = nxt.searchId
            obs_plain = obj_to_plain(nxt.observation)
        search_end()
        return "horizon", args.rollout_horizon, heuristic_score(obs_plain, candidate_side) if isinstance(obs_plain, dict) else 0.0
    except Exception:
        try:
            search_end()
        except Exception:
            pass
        return "error", 0, -0.25


def evaluate_action(record: dict[str, Any], action: list[int], action_seed: int, args: argparse.Namespace) -> dict[str, Any]:
    counts = Counter()
    steps_sum = 0
    score_sum = 0.0
    for i in range(args.rollouts_per_action):
        outcome, steps, score = rollout_once(record, action, action_seed + i * 104729, args)
        counts[outcome] += 1
        steps_sum += int(steps)
        score_sum += float(score)
    rollouts = max(1, args.rollouts_per_action)
    return {
        "action": [int(x) for x in action],
        "action_key": action_key(action),
        "rollouts": rollouts,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "errors": counts["error"],
        "win_rate": counts["win"] / rollouts,
        "avg_score": score_sum / rollouts,
        "avg_steps": steps_sum / rollouts,
    }


def process_record(task: tuple[int, dict[str, Any], int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx, record, seed = task
    assert _ARGS is not None and _CANDIDATE is not None
    args = _ARGS
    reset_policy(_CANDIDATE)
    if _OPPONENT is not None:
        reset_policy(_OPPONENT)
    rng = random.Random(seed)
    actions = enumerate_actions(record, _CANDIDATE.policy, rng, args)
    if not actions:
        return [], {
            "state_id": record.get("state_id", f"state_{idx}"),
            "candidate": record.get("candidate", ""),
            "opponent": record.get("opponent", ""),
            "evaluated_actions": 0,
            "rollouts_per_action": args.rollouts_per_action,
        }

    evaluated = []
    for ai, action in enumerate(actions):
        evaluated.append(evaluate_action(record, action, seed + ai * 1000003, args))

    baseline_action = parse_action(record.get("baseline_action", []))
    baseline_key = action_key(baseline_action)
    baseline_row = next((r for r in evaluated if r["action_key"] == baseline_key), None)
    if baseline_row is None:
        baseline_row = evaluated[0]
        baseline_key = baseline_row["action_key"]
    best = max(evaluated, key=lambda r: (float(r["avg_score"]), float(r["win_rate"]), int(r["wins"]), -len(r["action"])))
    baseline_wr = float(baseline_row["win_rate"])
    baseline_score = float(baseline_row["avg_score"])
    best_wr = float(best["win_rate"])
    best_score = float(best["avg_score"])
    obs = record.get("obs") or {}
    top_action = policy_top_action(record, obs.get("select") or {})
    top_key = action_key(top_action)

    action_rows: list[dict[str, Any]] = []
    for r in evaluated:
        action = r["action"]
        row = {
            "state_id": record.get("state_id", f"state_{idx}"),
            "candidate": record.get("candidate", ""),
            "opponent": record.get("opponent", ""),
            "outcome": record.get("outcome", ""),
            "turn": record.get("turn", ""),
            "context": record.get("context", ""),
            "option_count": record.get("option_count", ""),
            "action_key": r["action_key"],
            "action": action_key(action),
            "action_desc": describe_action(obs, action),
            "is_baseline": int(r["action_key"] == baseline_key),
            "is_policy_top1": int(r["action_key"] == top_key),
            "rollouts": r["rollouts"],
            "wins": r["wins"],
            "losses": r["losses"],
            "draws": r["draws"],
            "errors": r["errors"],
            "win_rate": f"{float(r['win_rate']):.6f}",
            "avg_score": f"{float(r['avg_score']):.6f}",
            "avg_steps": f"{float(r['avg_steps']):.3f}",
            "baseline_action": baseline_key,
            "baseline_win_rate": f"{baseline_wr:.6f}",
            "baseline_score": f"{baseline_score:.6f}",
            "best_action": best["action_key"],
            "best_win_rate": f"{best_wr:.6f}",
            "best_score": f"{best_score:.6f}",
            "delta_vs_baseline": f"{best_wr - baseline_wr:.6f}",
            "delta_score_vs_baseline": f"{best_score - baseline_score:.6f}",
        }
        action_rows.append(row)

    best_row = {
        "state_id": record.get("state_id", f"state_{idx}"),
        "candidate": record.get("candidate", ""),
        "opponent": record.get("opponent", ""),
        "outcome": record.get("outcome", ""),
        "turn": record.get("turn", ""),
        "context": record.get("context", ""),
        "option_count": record.get("option_count", ""),
        "baseline_action": baseline_key,
        "baseline_desc": describe_action(obs, baseline_action),
        "baseline_win_rate": f"{baseline_wr:.6f}",
        "baseline_score": f"{baseline_score:.6f}",
        "best_action": best["action_key"],
        "best_desc": describe_action(obs, best["action"]),
        "best_win_rate": f"{best_wr:.6f}",
        "best_score": f"{best_score:.6f}",
        "delta_vs_baseline": f"{best_wr - baseline_wr:.6f}",
        "delta_score_vs_baseline": f"{best_score - baseline_score:.6f}",
        "evaluated_actions": len(evaluated),
        "rollouts_per_action": args.rollouts_per_action,
    }
    return action_rows, best_row


def init_worker(candidate_spec: str, opponent_spec: str, args_dict: dict[str, Any]) -> None:
    global _CANDIDATE, _OPPONENT, _ARGS
    _CANDIDATE = load_entry(candidate_spec)
    _OPPONENT = load_entry(opponent_spec)
    _ARGS = argparse.Namespace(**args_dict)


def read_bank(path: Path, limit: int = 0, require_obs: bool = True) -> list[dict[str, Any]]:
    rows = []
    with open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if require_obs and "obs" not in row:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_teacher_jsonl(path: Path, best_rows: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]) -> None:
    with open_text(path, "w") as f:
        for best in best_rows:
            rec = records_by_id.get(str(best.get("state_id")), {})
            out = {
                "state_id": best.get("state_id"),
                "candidate": best.get("candidate"),
                "opponent": best.get("opponent"),
                "baseline_action": parse_action(best.get("baseline_action")),
                "best_action": parse_action(best.get("best_action")),
                "baseline_win_rate": float(best.get("baseline_win_rate") or 0.0),
                "best_win_rate": float(best.get("best_win_rate") or 0.0),
                "delta_vs_baseline": float(best.get("delta_vs_baseline") or 0.0),
                "baseline_score": float(best.get("baseline_score") or 0.0),
                "best_score": float(best.get("best_score") or 0.0),
                "delta_score_vs_baseline": float(best.get("delta_score_vs_baseline") or 0.0),
                "obs": rec.get("obs"),
            }
            f.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-jsonl", required=True)
    parser.add_argument("--candidate", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--opponent", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--limit-states", type=int, default=0)
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--rollout-horizon", type=int, default=160)
    parser.add_argument("--root-top-options", type=int, default=12)
    parser.add_argument("--combo-top-options", type=int, default=6)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--random-actions", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--candidate-rollout-temperature", type=float, default=0.0,
                        help="0=greedy, >0 stochastic temperature for candidate after root action")
    parser.add_argument("--opponent-rollout-temperature", type=float, default=0.0,
                        help="0=greedy, >0 stochastic temperature for opponent")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--out-actions-csv", required=True)
    parser.add_argument("--out-best-csv", required=True)
    parser.add_argument("--out-teacher-jsonl", default="")
    args = parser.parse_args()

    if not has_cg_search():
        parser.error("cg.api not found. Run this in the remote/Kaggle engine environment.")
    if args.rollouts_per_action <= 0:
        parser.error("--rollouts-per-action must be positive")
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")

    records = read_bank(Path(args.bank_jsonl), args.limit_states, require_obs=True)
    if not records:
        parser.error(f"no states with raw obs found in {args.bank_jsonl}")

    args_dict = vars(args).copy()
    action_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    t0 = time.time()

    if args.workers <= 1:
        init_worker(args.candidate, args.opponent, args_dict)
        for i, record in enumerate(records):
            rows, best = process_record((i, record, args.seed + i * 7919))
            action_rows.extend(rows)
            best_rows.append(best)
            done = i + 1
            if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == len(records)):
                deltas = [float(r.get("delta_score_vs_baseline") or 0.0) for r in best_rows if r.get("delta_score_vs_baseline") != ""]
                improved = sum(1 for d in deltas if d > 0.0)
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"  {done}/{len(records)} states action_rows={len(action_rows)} "
                    f"improved_score={improved}/{len(deltas)} mean_delta_score={np.mean(deltas) if deltas else 0.0:.3f} "
                    f"{rate:.2f} states/s",
                    flush=True,
                )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [(i, record, args.seed + i * 7919) for i, record in enumerate(records)]
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=init_worker,
            initargs=(args.candidate, args.opponent, args_dict),
        ) as ex:
            futs = [ex.submit(process_record, task) for task in tasks]
            for done, fut in enumerate(as_completed(futs), 1):
                rows, best = fut.result()
                action_rows.extend(rows)
                best_rows.append(best)
                if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == len(records)):
                    deltas = [float(r.get("delta_score_vs_baseline") or 0.0) for r in best_rows if r.get("delta_score_vs_baseline") != ""]
                    improved = sum(1 for d in deltas if d > 0.0)
                    rate = done / max(time.time() - t0, 1e-9)
                    print(
                        f"  {done}/{len(records)} states action_rows={len(action_rows)} "
                        f"improved_score={improved}/{len(deltas)} mean_delta_score={np.mean(deltas) if deltas else 0.0:.3f} "
                        f"{rate:.2f} states/s",
                        flush=True,
                    )

    best_rows.sort(key=lambda r: str(r.get("state_id", "")))
    action_rows.sort(key=lambda r: (str(r.get("state_id", "")), str(r.get("action_key", ""))))
    write_csv(Path(args.out_actions_csv), ACTION_FIELDS, action_rows)
    write_csv(Path(args.out_best_csv), BEST_FIELDS, best_rows)
    if args.out_teacher_jsonl:
        records_by_id = {str(r.get("state_id")): r for r in records}
        write_teacher_jsonl(Path(args.out_teacher_jsonl), best_rows, records_by_id)

    deltas = [float(r.get("delta_score_vs_baseline") or 0.0) for r in best_rows if r.get("delta_score_vs_baseline") != ""]
    improved = sum(1 for d in deltas if d > 0.0)
    strong = sum(1 for d in deltas if d >= 0.20)
    print(
        f"Wrote {len(action_rows)} action rows to {args.out_actions_csv}",
        flush=True,
    )
    print(
        f"Wrote {len(best_rows)} state best rows to {args.out_best_csv}; "
        f"improved_score={improved}/{len(deltas)} strong_score_delta>=0.20={strong} "
        f"mean_delta_score={np.mean(deltas) if deltas else 0.0:.4f}",
        flush=True,
    )
    if args.out_teacher_jsonl:
        print(f"Wrote teacher JSONL {args.out_teacher_jsonl}", flush=True)


if __name__ == "__main__":
    main()
