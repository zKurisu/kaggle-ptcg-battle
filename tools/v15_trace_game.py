#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.encoder import FastEncoder
from ptcg_rl.policy_loader import load_policy
from tools.trace_matchup_decisions import (
    ACTION_TYPES,
    active_card,
    bench_cards,
    card_name,
    encode_decision,
    pokemon_hp,
    safe_int,
    type_name,
)


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def legal_random(sel: dict) -> list[int]:
    opts = sel.get("option") or []
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []


def legalize(action: list[int], sel: dict) -> list[int]:
    n = len(sel.get("option") or [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    action = [int(x) for x in action if 0 <= int(x) < n]
    action = list(dict.fromkeys(action))
    if mn <= len(action) <= mx:
        return action[:mx]
    return legal_random(sel)


def choose(policy, obs: dict, deck: list[int], *, random_agent: bool) -> list[int]:
    sel = obs.get("select") or {}
    if random_agent or policy is None:
        return legal_random(sel)
    try:
        action = policy.select(obs, greedy=True, update_history=False)
    except Exception:
        action = []
    action = legalize(action, sel)
    try:
        policy.remember_decision(obs, action)
    except Exception:
        pass
    return action


def board_line(cur: dict, side: int) -> str:
    players = cur.get("players") or [{}, {}]
    p = players[side] if 0 <= side < len(players) else {}
    opp = players[1 - side] if 0 <= 1 - side < len(players) else {}

    def poke_text(poke: dict | None) -> str:
        if not poke:
            return ""
        cid = safe_int(poke.get("id"))
        hp, max_hp, dmg = pokemon_hp(poke)
        return f"{card_name(cid)}({cid}) hp={hp}/{max_hp} dmg={dmg}"

    active = poke_text((p.get("active") or [None])[0] if p.get("active") else None)
    opp_active = poke_text((opp.get("active") or [None])[0] if opp.get("active") else None)
    bench = " | ".join(f"{card_name(c)}({c})" for c in bench_cards(p)) or "-"
    opp_bench = " | ".join(f"{card_name(c)}({c})" for c in bench_cards(opp)) or "-"
    return (
        f"me active={active or '-'} bench=[{bench}] prizes={len(p.get('prize') or [])} "
        f"deck={safe_int(p.get('deckCount'))} hand={len(p.get('hand') or []) if p.get('hand') is not None else safe_int(p.get('handCount'))}; "
        f"opp active={opp_active or '-'} bench=[{opp_bench}] prizes={len(opp.get('prize') or [])} "
        f"deck={safe_int(opp.get('deckCount'))} hand={safe_int(opp.get('handCount'))}"
    )


def issue_tags(row: dict) -> list[str]:
    tags: list[str] = []
    if safe_int(row.get("early_end")):
        tags.append("early_end_with_actions")
    for label in ("attack", "ability", "attach", "evolve", "play", "retreat"):
        if safe_int(row.get(f"{label}_available")) and not safe_int(row.get(f"{label}_chosen")):
            tags.append(f"{label}_miss")
    if safe_int(row.get("drakloak_before_evolve_miss")):
        tags.append("drakloak_ability_before_dragapult_evolve_miss")
    if safe_int(row.get("damage_counter_context")) and not (
        safe_int(row.get("damage_counter_to_small_ko")) or safe_int(row.get("damage_counter_sets_up_200"))
    ):
        tags.append("dca_no_obvious_ko_or_200_setup")
    chosen_rank = row.get("chosen_first_rank")
    try:
        if chosen_rank != "" and int(chosen_rank) > 3:
            tags.append(f"low_rank_pick_{chosen_rank}")
    except Exception:
        pass
    return tags


def format_decision(row: dict, obs: dict) -> list[str]:
    cur = obs.get("current") or {}
    side = safe_int(cur.get("yourIndex"))
    tags = issue_tags(row)
    lines = [
        f"- step={row.get('step')} turn={row.get('turn')} tac={row.get('turn_action_count')} "
        f"context={row.get('context_name')} select_type={row.get('select_type')} "
        f"min/max={row.get('min_count')}/{row.get('max_count')} options={row.get('option_count')}",
        f"  board: {board_line(cur, side)}",
        f"  available: {row.get('available_type_counts')}",
        f"  chosen: {row.get('chosen_type_names')} {row.get('chosen_card_names')}",
        f"  top: {row.get('top_option_type_names')} :: {row.get('top_option_card_names')} :: p={row.get('top_option_probs')}",
    ]
    if row.get("chosen_target_name"):
        lines.append(
            f"  target: {row.get('chosen_target_name')} hp={row.get('chosen_target_hp')}/"
            f"{row.get('chosen_target_max_hp')} dmg={row.get('chosen_target_damage')} "
            f"area={row.get('chosen_target_area')} owner={row.get('chosen_target_owner')}"
        )
    if tags:
        lines.append("  flags: " + ", ".join(tags))
    return lines


def play_game(args, game: int, seed: int, encoder: FastEncoder, policy, opp_policy):
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(int(seed) & 0xFFFFFFFF)
    except Exception:
        pass
    candidate_side = game % 2
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    if opp_policy is not None and hasattr(opp_policy, "reset_history"):
        opp_policy.reset_history()
    first_deck = args.opponent_deck_cards if candidate_side == 1 else args.deck_cards
    second_deck = args.deck_cards if candidate_side == 1 else args.opponent_deck_cards
    obs, _ = battle_start(first_deck, second_deck)
    if obs is None:
        return "draw", [], 0
    trace: list[tuple[dict, dict]] = []
    result = 2
    steps = 0
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
            is_candidate = side == candidate_side
            if is_candidate:
                action = choose(policy, obs, args.deck_cards, random_agent=False)
                try:
                    row = encode_decision(encoder, obs, action, game, steps, candidate_side, policy=policy)
                    trace.append((row, obs))
                except Exception as exc:
                    trace.append(({"step": steps, "turn": safe_int(cur.get("turn")), "chosen_card_names": f"encode_error:{exc}"}, obs))
            else:
                action = choose(opp_policy, obs, args.opponent_deck_cards, random_agent=opp_policy is None)
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
    return outcome, trace, steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-policy", default="")
    p.add_argument("--opponent-deck", default="")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-outcome", choices=["loss", "win", "draw", "any"], default="loss")
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    args.deck_cards = load_deck(args.deck)
    args.opponent_deck_cards = load_deck(args.opponent_deck) if args.opponent_deck else list(args.deck_cards)
    policy = load_policy(args.policy, device="cpu")
    opp_policy = load_policy(args.opponent_policy, device="cpu") if args.opponent_policy else None
    encoder = FastEncoder()
    t0 = time.time()
    selected = None
    counts = {"win": 0, "loss": 0, "draw": 0}
    for g in range(args.games):
        outcome, trace, steps = play_game(args, g, args.seed + g, encoder, policy, opp_policy)
        counts[outcome] += 1
        if args.progress_every and (g == 0 or (g + 1) % args.progress_every == 0):
            print(
                f"{g + 1}/{args.games} win={counts['win']} loss={counts['loss']} draw={counts['draw']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if args.target_outcome == "any" or outcome == args.target_outcome:
            selected = (g, args.seed + g, outcome, trace, steps)
            break
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# v15 single-game trace\n\n")
        f.write(f"policy: `{args.policy}`\n\n")
        f.write(f"deck: `{args.deck}`\n\n")
        f.write(f"opponent: `{args.opponent_policy or 'legal_random'}` deck=`{args.opponent_deck or args.deck}`\n\n")
        f.write(f"searched_games={sum(counts.values())} counts={counts}\n\n")
        if selected is None:
            f.write(f"No game matched target_outcome={args.target_outcome}.\n")
        else:
            g, seed, outcome, trace, steps = selected
            f.write(f"selected_game={g} seed={seed} outcome={outcome} steps={steps} candidate_decisions={len(trace)}\n\n")
            for row, obs in trace:
                for line in format_decision(row, obs):
                    f.write(line + "\n")
                f.write("\n")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
