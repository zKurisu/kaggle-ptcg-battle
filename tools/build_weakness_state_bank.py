#!/usr/bin/env python3
"""Build replayable decision-state banks for weak matchup search.

The output JSONL keeps raw observations, baseline policy actions, policy
rankings, and compact board/option summaries. It is intentionally independent
from BC extraction: these states are meant for search/planner teachers, trace
audits, and later distillation targets.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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
from ptcg_rl.resource_planner import ResourcePlanner
from ptcg_rl.rule_overlay import RULE_MODES, apply_rule_overlay
from tools.eval_round_robin import Entry, parse_entry, read_deck
from tools.trace_matchup_decisions import ACTION_TYPES, card_name, type_name


def has_cg_engine() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("cg.game") is not None
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_entry(spec: str) -> Entry:
    name, policy_path, deck_path = parse_entry(spec, default_deck="deck.csv")
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else NumpyPolicy.load(policy_path)
    return Entry(name, policy_path, deck_path, policy, deck)


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


def sanitize_action(action: list[int], sel: dict[str, Any], rng: random.Random) -> list[int]:
    n = len(sel.get("option", []))
    out = [int(i) for i in action if isinstance(i, (int, np.integer)) and 0 <= int(i) < n]
    out = list(dict.fromkeys(out))
    if valid_action(out, sel):
        return out
    return legal_random(sel, rng)


def reset_entry(entry: Entry) -> None:
    if entry.policy is not None and hasattr(entry.policy, "reset_history"):
        entry.policy.reset_history()


def policy_ranking(policy: NumpyPolicy | None, obs: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if policy is None or not hasattr(policy, "first_step_ranking"):
        return []
    try:
        rows = policy.first_step_ranking(obs)[:limit]
    except Exception:
        return []
    return [
        {
            "index": int(r.get("index", -1)),
            "prob": float(r.get("prob", 0.0)),
            "logit": float(r.get("logit", 0.0)),
            "type": int(r.get("type", 0)),
            "type_name": type_name(int(r.get("type", 0))),
            "card": int(r.get("card", 0)),
            "card_name": card_name(int(r.get("card", 0))),
        }
        for r in rows
    ]


def choose_action(
    entry: Entry,
    obs: dict[str, Any],
    rng: random.Random,
    rules: str = "",
    planner: ResourcePlanner | None = None,
) -> tuple[list[int], str]:
    sel = obs.get("select") or {}
    if not sel.get("option"):
        return [], "empty"
    label = "random" if entry.policy is None else "greedy"
    try:
        if entry.policy is None:
            action = legal_random(sel, rng)
        else:
            action = entry.policy.select(obs, greedy=True, update_history=True)
        if rules == "resource_plan" and planner is not None:
            decision = planner.decide(obs, action, entry.deck)
            action = decision.action
            if decision.reason:
                label = f"{label}|rules:{decision.reason}"
        elif rules:
            decision = apply_rule_overlay(obs, action, entry.deck, mode=rules)
            action = decision.action
            if decision.reason:
                label = f"{label}|rules:{decision.reason}"
    except Exception as exc:
        action = legal_random(sel, rng)
        label = f"fallback_random:{type(exc).__name__}"
    return sanitize_action(action, sel, rng), label


def active_id(player: dict[str, Any]) -> int:
    active = player.get("active") or []
    if not active or not active[0]:
        return 0
    return safe_int(active[0].get("id"))


def bench_ids(player: dict[str, Any]) -> list[int]:
    out = []
    for p in player.get("bench") or []:
        if p:
            cid = safe_int(p.get("id"))
            if cid:
                out.append(cid)
    return out


def describe_action(encoded, action: list[int]) -> str:
    parts = []
    for idx in action:
        if not (0 <= idx < len(encoded.opt_type)):
            continue
        typ = int(encoded.opt_type[idx])
        cid = int(encoded.opt_card[idx])
        cid2 = int(encoded.opt_card2[idx])
        label = f"{idx}:{type_name(typ)}"
        if cid:
            label += f":{card_name(cid)}"
        if cid2 and cid2 != cid:
            label += f"->{card_name(cid2)}"
        parts.append(label)
    return " | ".join(parts) if parts else "STOP"


def action_types(encoded, action: list[int]) -> list[int]:
    return [int(encoded.opt_type[i]) for i in action if 0 <= i < len(encoded.opt_type)]


def action_cards(encoded, action: list[int]) -> list[int]:
    return [int(encoded.opt_card[i]) for i in action if 0 <= i < len(encoded.opt_card)]


def interesting_enough(encoded, obs: dict[str, Any], args: argparse.Namespace) -> bool:
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    turn = safe_int(cur.get("turn"))
    if turn < args.min_turn:
        return False
    if args.max_turn and turn > args.max_turn:
        return False
    option_count = len(sel.get("option") or [])
    if option_count < args.min_option_count:
        return False
    if args.context:
        ctx = str(safe_int(sel.get("context")))
        if ctx not in args.context:
            return False
    if args.available_type:
        available = {str(int(x)) for x in encoded.opt_type.tolist()}
        if not any(t in available for t in args.available_type):
            return False
    if not args.interesting_only:
        return True
    types = set(int(x) for x in encoded.opt_type.tolist())
    has_tempo = any(t in types for t in ACTION_TYPES.values())
    multi = safe_int(sel.get("maxCount")) > safe_int(sel.get("minCount"))
    return has_tempo or multi or option_count >= max(args.min_option_count, 8)


def make_state_record(
    *,
    encoder: FastEncoder,
    obs: dict[str, Any],
    candidate: Entry,
    opponent: Entry,
    game: int,
    seed: int,
    step: int,
    candidate_side: int,
    baseline_action: list[int],
    action_source: str,
    ranking: list[dict[str, Any]],
    policy_history: dict[str, Any] | None,
    include_obs: bool,
    state_index: int,
) -> dict[str, Any] | None:
    try:
        encoded = encoder.encode(obs)
    except Exception:
        return None
    cur = obs.get("current") or {}
    players = cur.get("players") or [{}, {}]
    you = safe_int(cur.get("yourIndex"))
    me = players[you] if you < len(players) else {}
    opp = players[1 - you] if 1 - you < len(players) else {}
    sel = obs.get("select") or {}
    counts = Counter(int(x) for x in encoded.opt_type.tolist())

    record: dict[str, Any] = {
        "state_id": f"{candidate.name}_vs_{opponent.name}_g{game:05d}_s{step:04d}_{state_index:06d}",
        "game": game,
        "seed": seed,
        "step": step,
        "candidate": candidate.name,
        "candidate_policy": candidate.policy_path,
        "candidate_deck": candidate.deck_path,
        "opponent": opponent.name,
        "opponent_policy": opponent.policy_path,
        "opponent_deck": opponent.deck_path,
        "candidate_side": candidate_side,
        "current_side": you,
        "turn": safe_int(cur.get("turn")),
        "turn_action_count": safe_int(cur.get("turnActionCount")),
        "context": safe_int(sel.get("context")),
        "select_type": safe_int(sel.get("type")),
        "min_count": safe_int(sel.get("minCount")),
        "max_count": safe_int(sel.get("maxCount")),
        "option_count": len(sel.get("option") or []),
        "available_type_counts": dict(sorted(counts.items())),
        "my_active": active_id(me),
        "my_active_name": card_name(active_id(me)),
        "opp_active": active_id(opp),
        "opp_active_name": card_name(active_id(opp)),
        "my_bench": bench_ids(me),
        "opp_bench": bench_ids(opp),
        "my_prizes": len(me.get("prize") or []),
        "opp_prizes": len(opp.get("prize") or []),
        "my_deck_count": safe_int(me.get("deckCount")),
        "opp_deck_count": safe_int(opp.get("deckCount")),
        "my_hand_count": len(me.get("hand") or []) if me.get("hand") is not None else safe_int(me.get("handCount")),
        "opp_hand_count": safe_int(opp.get("handCount")),
        "baseline_action": [int(x) for x in baseline_action],
        "baseline_action_desc": describe_action(encoded, baseline_action),
        "baseline_action_types": action_types(encoded, baseline_action),
        "baseline_action_cards": action_cards(encoded, baseline_action),
        "action_source": action_source,
        "policy_ranking": ranking,
        "policy_top1": ranking[0] if ranking else {},
        "outcome": "",
        "result": -1,
        "game_steps": 0,
    }
    if policy_history:
        record["policy_history"] = policy_history
    if include_obs:
        record["obs"] = deepcopy(obs)
    return record


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state_id",
        "game",
        "seed",
        "step",
        "candidate",
        "opponent",
        "outcome",
        "turn",
        "turn_action_count",
        "context",
        "select_type",
        "min_count",
        "max_count",
        "option_count",
        "baseline_action",
        "baseline_action_desc",
        "my_active_name",
        "opp_active_name",
        "my_prizes",
        "opp_prizes",
        "game_steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_game_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game",
        "seed",
        "candidate_side",
        "result",
        "outcome",
        "steps",
        "candidate_decisions",
        "saved_states",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def play_game(
    candidate: Entry,
    opponent: Entry,
    game: int,
    seed: int,
    args: argparse.Namespace,
    encoder: FastEncoder,
    state_counter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from cg.game import battle_finish, battle_select, battle_start

    rng = random.Random(seed)
    random.seed(seed)
    swapped = bool(game % 2)
    first, second = (opponent, candidate) if swapped else (candidate, opponent)
    candidate_side = 1 if swapped else 0
    for entry in (first, second):
        reset_entry(entry)
    candidate_planner = ResourcePlanner(candidate.deck) if args.candidate_rules == "resource_plan" else None
    opponent_planner = ResourcePlanner(opponent.deck) if args.opponent_rules == "resource_plan" else None

    obs, _ = battle_start(first.deck, second.deck)
    records: list[dict[str, Any]] = []
    result = 2
    steps = 0
    candidate_decisions = 0
    saved_this_game = 0

    if obs is None:
        return {
            "game": game,
            "seed": seed,
            "candidate_side": candidate_side,
            "result": result,
            "outcome": "draw",
            "steps": 0,
            "candidate_decisions": 0,
            "saved_states": 0,
        }, []

    try:
        for steps in range(args.max_turns):
            cur = obs.get("current") or {}
            res = safe_int(cur.get("result"), -1)
            if res != -1:
                result = res if res in (0, 1) else 2
                break
            sel = obs.get("select")
            if sel is None:
                result = 2
                break

            side = safe_int(cur.get("yourIndex"))
            entry = first if side == 0 else second
            if side == candidate_side:
                candidate_decisions += 1
                ranking = policy_ranking(entry.policy, obs, args.ranking_limit)
                pre_history = None
                if args.include_policy_history and entry.policy is not None and hasattr(entry.policy, "_history_arrays"):
                    try:
                        hist = entry.policy._history_arrays(obs)  # noqa: SLF001 - state bank needs exact live history.
                        if hist:
                            pre_history = {k: np.asarray(v).tolist() for k, v in hist.items()}
                    except Exception:
                        pre_history = None
                action, source = choose_action(entry, obs, rng, args.candidate_rules, candidate_planner)
                if saved_this_game < args.states_per_game:
                    try:
                        encoded = encoder.encode(obs)
                    except Exception:
                        encoded = None
                    if encoded is not None and interesting_enough(encoded, obs, args):
                        rec = make_state_record(
                            encoder=encoder,
                            obs=obs,
                            candidate=candidate,
                            opponent=opponent,
                            game=game,
                            seed=seed,
                            step=steps,
                            candidate_side=candidate_side,
                            baseline_action=action,
                            action_source=source,
                            ranking=ranking,
                            policy_history=pre_history,
                            include_obs=not args.no_obs,
                            state_index=state_counter + len(records),
                        )
                        if rec is not None:
                            records.append(rec)
                            saved_this_game += 1
            else:
                action, _ = choose_action(entry, obs, rng, args.opponent_rules, opponent_planner)

            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()

    if result == 2:
        outcome = "draw"
    elif result == candidate_side:
        outcome = "win"
    else:
        outcome = "loss"
    for rec in records:
        rec["outcome"] = outcome
        rec["result"] = result
        rec["game_steps"] = steps
    return {
        "game": game,
        "seed": seed,
        "candidate_side": candidate_side,
        "result": result,
        "outcome": outcome,
        "steps": steps,
        "candidate_decisions": candidate_decisions,
        "saved_states": len(records),
    }, records


def keep_for_outcome(record: dict[str, Any], outcome_filter: str) -> bool:
    if outcome_filter == "all":
        return True
    return record.get("outcome") == outcome_filter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--opponent", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=700)
    parser.add_argument("--states-per-game", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=1000)
    parser.add_argument("--out-jsonl", required=True, help="JSONL or JSONL.GZ path")
    parser.add_argument("--out-csv", default="", help="compact CSV index path")
    parser.add_argument("--out-games-csv", default="", help="per-game outcome CSV path")
    parser.add_argument("--outcome", choices=["all", "win", "loss", "draw"], default="loss")
    parser.add_argument("--interesting-only", action="store_true")
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--max-turn", type=int, default=0, help="0 disables upper turn filter")
    parser.add_argument("--min-option-count", type=int, default=2)
    parser.add_argument("--context", action="append", default=[], help="keep only select context id; repeatable")
    parser.add_argument("--available-type", action="append", default=[], help="keep if option type id is available")
    parser.add_argument("--ranking-limit", type=int, default=12)
    parser.add_argument("--candidate-rules", choices=["", *RULE_MODES], default="")
    parser.add_argument("--opponent-rules", choices=["", *RULE_MODES], default="")
    parser.add_argument("--include-policy-history", action="store_true")
    parser.add_argument("--no-obs", action="store_true", help="omit raw obs; search teacher requires obs")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if not has_cg_engine():
        parser.error("cg.game not found. Run this in the remote/Kaggle engine environment.")
    if args.max_states <= 0:
        parser.error("--max-states must be positive")
    if args.states_per_game <= 0:
        parser.error("--states-per-game must be positive")
    if args.no_obs:
        print("warning: --no-obs output cannot be consumed by search_action_teacher.py", file=sys.stderr)

    candidate = load_entry(args.candidate)
    opponent = load_entry(args.opponent)
    encoder = FastEncoder()
    out_path = Path(args.out_jsonl)

    kept_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    totals = Counter()
    t0 = time.time()
    state_counter = 0

    with open_text(out_path, "w") as f:
        for game in range(args.games):
            game_row, records = play_game(
                candidate,
                opponent,
                game,
                args.seed + game,
                args,
                encoder,
                state_counter,
            )
            game_rows.append(game_row)
            totals[game_row["outcome"]] += 1
            for rec in records:
                if not keep_for_outcome(rec, args.outcome):
                    continue
                if state_counter >= args.max_states:
                    break
                f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
                compact = {k: v for k, v in rec.items() if k not in {"obs", "policy_history", "policy_ranking"}}
                kept_rows.append(compact)
                state_counter += 1
            done = game + 1
            if args.progress_every and (done == 1 or done % args.progress_every == 0 or state_counter >= args.max_states):
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"  {done}/{args.games} games W/L/D={totals['win']}/{totals['loss']}/{totals['draw']} "
                    f"kept_states={state_counter} {rate:.2f} games/s",
                    flush=True,
                )
            if state_counter >= args.max_states:
                break

    if args.out_csv:
        write_csv_rows(Path(args.out_csv), kept_rows)
    if args.out_games_csv:
        write_game_csv(Path(args.out_games_csv), game_rows)

    print(
        f"Wrote {state_counter} states to {args.out_jsonl}; "
        f"games={len(game_rows)} W/L/D={totals['win']}/{totals['loss']}/{totals['draw']}",
        flush=True,
    )
    if args.out_csv:
        print(f"Wrote compact index {args.out_csv}", flush=True)
    if args.out_games_csv:
        print(f"Wrote game summary {args.out_games_csv}", flush=True)


if __name__ == "__main__":
    main()
