#!/usr/bin/env python3
"""Trace candidate decisions in local matchups.

This is a diagnostic companion to eval_round_robin.py. It plays a candidate
against one opponent, records every candidate-side decision, and summarizes
decision-type differences between wins and losses.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from ptcg_rl.deck_plans import CARD_NAMES
from ptcg_rl.encoder import FastEncoder
from tools.bc2_accuracy import CONTEXT_NAMES, OPT_NAMES
from tools.eval_round_robin import Entry, load_entries, policy_action


ACTION_TYPES = {
    "play": 7,
    "attach": 8,
    "evolve": 9,
    "ability": 10,
    "retreat": 12,
    "attack": 13,
    "end": 14,
}
PRESSING_TYPES = {7, 8, 9, 10, 12, 13}
_CARD_NAME_BY_ID: dict[int, str] | None = None


GAME_FIELDS = [
    "game",
    "seed",
    "candidate",
    "opponent",
    "candidate_side",
    "result",
    "outcome",
    "steps",
    "candidate_decisions",
    "candidate_attacks",
    "candidate_early_ends",
]

DECISION_FIELDS = [
    "game",
    "step",
    "outcome",
    "candidate_side",
    "turn",
    "turn_action_count",
    "context",
    "context_name",
    "select_type",
    "min_count",
    "max_count",
    "option_count",
    "chosen_len",
    "chosen_types",
    "chosen_type_names",
    "chosen_cards",
    "chosen_card_names",
    "my_active",
    "my_active_name",
    "opp_active",
    "opp_active_name",
    "my_bench_cards",
    "my_bench_card_names",
    "opp_bench_cards",
    "opp_bench_card_names",
    "my_bench_count",
    "opp_bench_count",
    "my_prizes",
    "opp_prizes",
    "my_deck",
    "opp_deck",
    "my_hand",
    "opp_hand",
    "attack_available",
    "attack_chosen",
    "ability_available",
    "ability_chosen",
    "attach_available",
    "attach_chosen",
    "evolve_available",
    "evolve_chosen",
    "play_available",
    "play_chosen",
    "retreat_available",
    "retreat_chosen",
    "end_chosen",
    "early_end",
    "short_optional_multi",
    "available_type_counts",
    "available_play_cards",
    "available_play_card_names",
    "available_attach_cards",
    "available_attach_card_names",
    "available_evolve_cards",
    "available_evolve_card_names",
    "available_ability_cards",
    "available_ability_card_names",
    "available_retreat_cards",
    "available_retreat_card_names",
    "available_attack_cards",
    "available_attack_card_names",
    "top_option_indices",
    "top_option_probs",
    "top_option_type_names",
    "top_option_card_names",
    "top1_type_name",
    "top1_card_name",
    "chosen_first_rank",
    "chosen_first_prob",
]

SUMMARY_FIELDS = [
    "table",
    "key",
    "decisions",
    "games",
    "wins",
    "losses",
    "draws",
    "avg_decisions_per_game",
    "avg_turn",
    "avg_option_count",
    "avg_chosen_len",
    "attack_available",
    "attack_chosen",
    "miss_attack_rate",
    "ability_available",
    "ability_chosen",
    "miss_ability_rate",
    "attach_available",
    "attach_chosen",
    "miss_attach_rate",
    "evolve_available",
    "evolve_chosen",
    "miss_evolve_rate",
    "play_available",
    "play_chosen",
    "miss_play_rate",
    "retreat_available",
    "retreat_chosen",
    "miss_retreat_rate",
    "end_chosen",
    "early_end",
    "early_end_rate",
    "short_optional_multi",
    "short_optional_multi_rate",
]

CHOICE_FIELDS = [
    "table",
    "key",
    "chosen_type",
    "chosen_type_name",
    "n",
    "rate",
]


def has_cg_engine() -> bool:
    try:
        return importlib.util.find_spec("cg.game") is not None
    except ModuleNotFoundError:
        return False


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def card_name(card_id: int) -> str:
    if not card_id:
        return ""
    global _CARD_NAME_BY_ID
    if _CARD_NAME_BY_ID is None:
        _CARD_NAME_BY_ID = dict(CARD_NAMES)
        try:
            from cg.api import all_card_data
            for card in all_card_data():
                name = getattr(card, "name", "") or getattr(card, "cardName", "")
                if name:
                    _CARD_NAME_BY_ID[int(card.cardId)] = str(name)
        except Exception:
            pass
    return _CARD_NAME_BY_ID.get(int(card_id), str(card_id))


def type_name(opt_type: int) -> str:
    return OPT_NAMES.get(int(opt_type), str(opt_type))


def context_name(context: int) -> str:
    return CONTEXT_NAMES.get(int(context), str(context))


def active_card(player: dict) -> int:
    active = player.get("active") or []
    if not active or not active[0]:
        return 0
    return safe_int(active[0].get("id"))


def bench_cards(player: dict) -> list[int]:
    cards = []
    for p in player.get("bench") or []:
        if p:
            cid = safe_int(p.get("id"))
            if cid:
                cards.append(cid)
    return cards


def in_play_count(player: dict) -> int:
    active = 1 if active_card(player) else 0
    return active + len([p for p in (player.get("bench") or []) if p])


def option_counts(types: list[int]) -> str:
    counts = Counter(types)
    return " ".join(f"{type_name(k)}:{v}" for k, v in sorted(counts.items()))


def join_limited(values: list, limit: int = 12) -> str:
    values = list(values)
    shown = values[:limit]
    suffix = ["..."] if len(values) > limit else []
    return " | ".join(str(x) for x in [*shown, *suffix] if str(x))


def cards_by_type(opt_types: list[int], opt_cards: list[int], opt_type: int) -> list[int]:
    out = []
    seen = set()
    for typ, card in zip(opt_types, opt_cards):
        if typ != opt_type:
            continue
        if card in seen:
            continue
        seen.add(card)
        out.append(card)
    return out


def ranking_fields(policy, obs: dict, chosen: list[int], *, top_n: int = 8) -> dict:
    if policy is None or not hasattr(policy, "first_step_ranking"):
        return {
            "top_option_indices": "",
            "top_option_probs": "",
            "top_option_type_names": "",
            "top_option_card_names": "",
            "top1_type_name": "",
            "top1_card_name": "",
            "chosen_first_rank": "",
            "chosen_first_prob": "",
        }
    try:
        ranking = policy.first_step_ranking(obs)
    except Exception as exc:
        return {
            "top_option_indices": "",
            "top_option_probs": "",
            "top_option_type_names": "",
            "top_option_card_names": f"rank_error:{exc}",
            "top1_type_name": "",
            "top1_card_name": "",
            "chosen_first_rank": "",
            "chosen_first_prob": "",
        }
    top = ranking[:top_n]
    chosen_idx = chosen[0] if chosen else -1
    rank_by_idx = {int(row["index"]): i + 1 for i, row in enumerate(ranking)}
    prob_by_idx = {int(row["index"]): float(row["prob"]) for row in ranking}
    top1 = top[0] if top else {}
    return {
        "top_option_indices": " ".join(str(int(row["index"])) for row in top),
        "top_option_probs": " ".join(f"{float(row['prob']):.4f}" for row in top),
        "top_option_type_names": " | ".join(type_name(int(row["type"])) for row in top),
        "top_option_card_names": " | ".join(card_name(int(row["card"])) for row in top),
        "top1_type_name": type_name(int(top1.get("type", 0))) if top1 else "",
        "top1_card_name": card_name(int(top1.get("card", 0))) if top1 else "",
        "chosen_first_rank": rank_by_idx.get(chosen_idx, ""),
        "chosen_first_prob": f"{prob_by_idx[chosen_idx]:.4f}" if chosen_idx in prob_by_idx else "",
    }


def choose_action(entry: Entry, obs: dict, args: argparse.Namespace) -> list[int]:
    return policy_action(
        entry,
        obs,
        use_mcts=args.mcts,
        sims=args.mcts_sims,
        time_budget=args.time_budget,
    )


def encode_decision(
    encoder: FastEncoder,
    obs: dict,
    action: list[int],
    game: int,
    step: int,
    candidate_side: int,
    policy=None,
) -> dict:
    cur = obs.get("current") or {}
    players = cur.get("players") or [{}, {}]
    you = safe_int(cur.get("yourIndex"))
    me = players[you] if you < len(players) else {}
    opp = players[1 - you] if 1 - you < len(players) else {}
    sel = obs.get("select") or {}

    encoded = encoder.encode(obs)
    opt_types = [int(x) for x in encoded.opt_type.tolist()]
    opt_cards = [int(x) for x in encoded.opt_card.tolist()]
    chosen = [i for i in action if 0 <= i < len(opt_types)]
    chosen_types = [opt_types[i] for i in chosen]
    chosen_cards = [opt_cards[i] for i in chosen]
    types_set = set(opt_types)
    context = safe_int(sel.get("context"))
    my_bench = bench_cards(me)
    opp_bench = bench_cards(opp)

    row = {
        "game": game,
        "step": step,
        "outcome": "",
        "candidate_side": candidate_side,
        "turn": safe_int(cur.get("turn")),
        "turn_action_count": safe_int(cur.get("turnActionCount")),
        "context": context,
        "context_name": context_name(context),
        "select_type": safe_int(sel.get("type")),
        "min_count": safe_int(sel.get("minCount")),
        "max_count": safe_int(sel.get("maxCount")),
        "option_count": len(opt_types),
        "chosen_len": len(chosen),
        "chosen_types": " ".join(map(str, chosen_types)),
        "chosen_type_names": " ".join(type_name(t) for t in chosen_types),
        "chosen_cards": " ".join(map(str, chosen_cards)),
        "chosen_card_names": " | ".join(card_name(c) for c in chosen_cards),
        "my_active": active_card(me),
        "my_active_name": card_name(active_card(me)),
        "opp_active": active_card(opp),
        "opp_active_name": card_name(active_card(opp)),
        "my_bench_cards": " ".join(map(str, my_bench)),
        "my_bench_card_names": " | ".join(card_name(c) for c in my_bench),
        "opp_bench_cards": " ".join(map(str, opp_bench)),
        "opp_bench_card_names": " | ".join(card_name(c) for c in opp_bench),
        "my_bench_count": max(0, in_play_count(me) - 1),
        "opp_bench_count": max(0, in_play_count(opp) - 1),
        "my_prizes": len(me.get("prize") or []),
        "opp_prizes": len(opp.get("prize") or []),
        "my_deck": safe_int(me.get("deckCount")),
        "opp_deck": safe_int(opp.get("deckCount")),
        "my_hand": len(me.get("hand") or []) if me.get("hand") is not None else safe_int(me.get("handCount")),
        "opp_hand": safe_int(opp.get("handCount")),
        "available_type_counts": option_counts(opt_types),
    }
    for label in ("play", "attach", "evolve", "ability", "retreat", "attack"):
        cards = cards_by_type(opt_types, opt_cards, ACTION_TYPES[label])
        row[f"available_{label}_cards"] = " ".join(map(str, cards))
        row[f"available_{label}_card_names"] = join_limited([card_name(c) for c in cards])
    row.update(ranking_fields(policy, obs, chosen))
    for label, typ in ACTION_TYPES.items():
        row[f"{label}_available"] = int(typ in types_set)
        row[f"{label}_chosen"] = int(typ in chosen_types)
    row["early_end"] = int(14 in chosen_types and bool(types_set & PRESSING_TYPES))
    row["short_optional_multi"] = int(
        safe_int(sel.get("maxCount")) > safe_int(sel.get("minCount"))
        and safe_int(sel.get("maxCount")) > 1
        and len(chosen) < min(safe_int(sel.get("maxCount")), len(opt_types))
    )
    return row


def play_traced_game(
    candidate: Entry,
    opponent: Entry,
    game: int,
    seed: int,
    args: argparse.Namespace,
    encoder: FastEncoder,
) -> tuple[dict, list[dict]]:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    swapped = bool(game % 2)
    first, second = (opponent, candidate) if swapped else (candidate, opponent)
    candidate_side = 1 if swapped else 0
    for entry in (first, second):
        if entry.policy is not None and hasattr(entry.policy, "reset_history"):
            entry.policy.reset_history()
    obs, sd = battle_start(first.deck, second.deck)
    decisions: list[dict] = []
    result = 2
    steps = 0

    if obs is None:
        return {
            "game": game,
            "seed": seed,
            "candidate": candidate.name,
            "opponent": opponent.name,
            "candidate_side": candidate_side,
            "result": result,
            "outcome": "draw",
            "steps": 0,
            "candidate_decisions": 0,
            "candidate_attacks": 0,
            "candidate_early_ends": 0,
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
            action = choose_action(entry, obs, args)
            if side == candidate_side:
                try:
                    decisions.append(encode_decision(
                        encoder,
                        obs,
                        action,
                        game,
                        steps,
                        candidate_side,
                        policy=entry.policy,
                    ))
                except Exception as exc:
                    decisions.append({
                        "game": game,
                        "step": steps,
                        "outcome": "",
                        "candidate_side": candidate_side,
                        "turn": safe_int(cur.get("turn")),
                        "turn_action_count": safe_int(cur.get("turnActionCount")),
                        "context": safe_int(sel.get("context")),
                        "context_name": context_name(safe_int(sel.get("context"))),
                        "select_type": safe_int(sel.get("type")),
                        "min_count": safe_int(sel.get("minCount")),
                        "max_count": safe_int(sel.get("maxCount")),
                        "option_count": len(sel.get("option") or []),
                        "chosen_len": len(action),
                        "chosen_types": "",
                        "chosen_type_names": "",
                        "chosen_cards": "",
                        "chosen_card_names": f"encode_error:{exc}",
                        "my_active": "",
                        "my_active_name": "",
                        "opp_active": "",
                        "opp_active_name": "",
                        "my_bench_count": "",
                        "opp_bench_count": "",
                        "my_prizes": "",
                        "opp_prizes": "",
                        "my_deck": "",
                        "opp_deck": "",
                        "my_hand": "",
                        "opp_hand": "",
                        "attack_available": 0,
                        "attack_chosen": 0,
                        "ability_available": 0,
                        "ability_chosen": 0,
                        "attach_available": 0,
                        "attach_chosen": 0,
                        "evolve_available": 0,
                        "evolve_chosen": 0,
                        "play_available": 0,
                        "play_chosen": 0,
                        "retreat_available": 0,
                        "retreat_chosen": 0,
                        "end_chosen": 0,
                        "early_end": 0,
                        "short_optional_multi": 0,
                        "available_type_counts": "",
                    })
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
    for row in decisions:
        row["outcome"] = outcome
    return {
        "game": game,
        "seed": seed,
        "candidate": candidate.name,
        "opponent": opponent.name,
        "candidate_side": candidate_side,
        "result": result,
        "outcome": outcome,
        "steps": steps,
        "candidate_decisions": len(decisions),
        "candidate_attacks": sum(int(row.get("attack_chosen") or 0) for row in decisions),
        "candidate_early_ends": sum(int(row.get("early_end") or 0) for row in decisions),
    }, decisions


def empty_summary() -> Counter:
    return Counter({
        "decisions": 0,
        "turn_sum": 0,
        "option_sum": 0,
        "chosen_len_sum": 0,
        "attack_available": 0,
        "attack_chosen": 0,
        "ability_available": 0,
        "ability_chosen": 0,
        "attach_available": 0,
        "attach_chosen": 0,
        "evolve_available": 0,
        "evolve_chosen": 0,
        "play_available": 0,
        "play_chosen": 0,
        "retreat_available": 0,
        "retreat_chosen": 0,
        "end_chosen": 0,
        "early_end": 0,
        "short_optional_multi": 0,
    })


def add_summary(row: dict, table: Counter) -> None:
    table["decisions"] += 1
    table["turn_sum"] += safe_int(row.get("turn"))
    table["option_sum"] += safe_int(row.get("option_count"))
    table["chosen_len_sum"] += safe_int(row.get("chosen_len"))
    for label in ("attack", "ability", "attach", "evolve", "play", "retreat"):
        table[f"{label}_available"] += safe_int(row.get(f"{label}_available"))
        table[f"{label}_chosen"] += safe_int(row.get(f"{label}_chosen"))
    table["end_chosen"] += safe_int(row.get("end_chosen"))
    table["early_end"] += safe_int(row.get("early_end"))
    table["short_optional_multi"] += safe_int(row.get("short_optional_multi"))


def finalize_summary(table_name: str, key: str, row: Counter, game_counts: Counter) -> dict:
    decisions = max(row["decisions"], 1)

    def miss(label: str) -> float:
        avail = row[f"{label}_available"]
        if avail <= 0:
            return 0.0
        return 1.0 - row[f"{label}_chosen"] / avail

    games = game_counts["win"] + game_counts["loss"] + game_counts["draw"]
    return {
        "table": table_name,
        "key": key,
        "decisions": row["decisions"],
        "games": games,
        "wins": game_counts["win"],
        "losses": game_counts["loss"],
        "draws": game_counts["draw"],
        "avg_decisions_per_game": row["decisions"] / max(games, 1),
        "avg_turn": row["turn_sum"] / decisions,
        "avg_option_count": row["option_sum"] / decisions,
        "avg_chosen_len": row["chosen_len_sum"] / decisions,
        "attack_available": row["attack_available"],
        "attack_chosen": row["attack_chosen"],
        "miss_attack_rate": miss("attack"),
        "ability_available": row["ability_available"],
        "ability_chosen": row["ability_chosen"],
        "miss_ability_rate": miss("ability"),
        "attach_available": row["attach_available"],
        "attach_chosen": row["attach_chosen"],
        "miss_attach_rate": miss("attach"),
        "evolve_available": row["evolve_available"],
        "evolve_chosen": row["evolve_chosen"],
        "miss_evolve_rate": miss("evolve"),
        "play_available": row["play_available"],
        "play_chosen": row["play_chosen"],
        "miss_play_rate": miss("play"),
        "retreat_available": row["retreat_available"],
        "retreat_chosen": row["retreat_chosen"],
        "miss_retreat_rate": miss("retreat"),
        "end_chosen": row["end_chosen"],
        "early_end": row["early_end"],
        "early_end_rate": row["early_end"] / decisions,
        "short_optional_multi": row["short_optional_multi"],
        "short_optional_multi_rate": row["short_optional_multi"] / decisions,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(prefix: Path, games: list[dict], decisions: list[dict]) -> None:
    game_counts_by_outcome = Counter(row["outcome"] for row in games)
    game_counts_by_context = defaultdict(Counter)
    summaries = []
    by = {
        "overall": defaultdict(empty_summary),
        "outcome": defaultdict(empty_summary),
        "context": defaultdict(empty_summary),
        "outcome_context": defaultdict(empty_summary),
    }
    choice_by = {
        "outcome": defaultdict(Counter),
        "context": defaultdict(Counter),
        "outcome_context": defaultdict(Counter),
    }
    for row in decisions:
        outcome = row["outcome"]
        context = f"{row['context_name']}({row['context']})"
        outcome_context = f"{outcome}:{context}"
        keys = {
            "overall": "overall",
            "outcome": outcome,
            "context": context,
            "outcome_context": outcome_context,
        }
        for table, key in keys.items():
            add_summary(row, by[table][key])
        for chosen in str(row.get("chosen_types", "")).split():
            typ = safe_int(chosen, -1)
            choice_by["outcome"][outcome][typ] += 1
            choice_by["context"][context][typ] += 1
            choice_by["outcome_context"][outcome_context][typ] += 1

    for table, rows in by.items():
        for key, row in rows.items():
            if table == "overall":
                counts = game_counts_by_outcome
            elif table == "outcome":
                counts = Counter({key: game_counts_by_outcome[key]})
            elif table == "context":
                counts = game_counts_by_context[key]
            else:
                outcome = str(key).split(":", 1)[0]
                counts = Counter({outcome: game_counts_by_outcome[outcome]})
            summaries.append(finalize_summary(table, key, row, counts))
    summaries.sort(key=lambda r: (r["table"], -int(r["decisions"]), r["key"]))
    write_csv(prefix.with_suffix(".summary.csv"), SUMMARY_FIELDS, summaries)

    choice_rows = []
    for table, rows in choice_by.items():
        for key, counts in rows.items():
            total = sum(counts.values())
            for typ, n in counts.most_common():
                choice_rows.append({
                    "table": table,
                    "key": key,
                    "chosen_type": typ,
                    "chosen_type_name": type_name(typ),
                    "n": n,
                    "rate": n / max(total, 1),
                })
    write_csv(prefix.with_suffix(".choice_types.csv"), CHOICE_FIELDS, choice_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="NAME=POLICY:DECK")
    parser.add_argument("--opponent", required=True, help="NAME=POLICY:DECK, or NAME=random:DECK")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=700)
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--mcts-sims", type=int, default=48)
    parser.add_argument("--time-budget", type=float, default=4.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    if not has_cg_engine():
        parser.error(
            "cg.game not found. Run this in the Kaggle/remote repo with the cg engine available."
        )

    candidate, opponent = load_entries(
        [args.candidate, args.opponent],
        default_deck="deck.csv",
        include_random=False,
    )
    encoder = FastEncoder()
    games: list[dict] = []
    decisions: list[dict] = []
    t0 = time.time()
    for game in range(args.games):
        game_row, decision_rows = play_traced_game(
            candidate, opponent, game, args.seed + game, args, encoder
        )
        games.append(game_row)
        decisions.extend(decision_rows)
        done = game + 1
        if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
            wins = sum(1 for row in games if row["outcome"] == "win")
            losses = sum(1 for row in games if row["outcome"] == "loss")
            draws = sum(1 for row in games if row["outcome"] == "draw")
            rate = done / max(time.time() - t0, 1e-9)
            eta = (args.games - done) / max(rate, 1e-9)
            print(
                f"  {done}/{args.games} {candidate.name} vs {opponent.name} "
                f"W/L/D={wins}/{losses}/{draws} wr={wins/done:.3f} "
                f"{rate:.2f} games/s eta={eta:.0f}s",
                flush=True,
            )

    prefix = Path(args.out_prefix)
    write_csv(prefix.with_suffix(".games.csv"), GAME_FIELDS, games)
    write_csv(prefix.with_suffix(".decisions.csv"), DECISION_FIELDS, decisions)
    write_summaries(prefix, games, decisions)

    wins = sum(1 for row in games if row["outcome"] == "win")
    losses = sum(1 for row in games if row["outcome"] == "loss")
    draws = sum(1 for row in games if row["outcome"] == "draw")
    print(
        f"Wrote {prefix.with_suffix('.games.csv')}, {prefix.with_suffix('.decisions.csv')}, "
        f"{prefix.with_suffix('.summary.csv')}, {prefix.with_suffix('.choice_types.csv')}",
        flush=True,
    )
    print(
        f"Final {candidate.name} vs {opponent.name}: {wins}-{losses}-{draws} "
        f"wr={wins / max(args.games, 1):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
