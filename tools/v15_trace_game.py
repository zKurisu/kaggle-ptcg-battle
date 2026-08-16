#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


AREA_NAMES = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    8: "energy",
    9: "tool",
    10: "evolution_stack",
    12: "look",
    13: "playing",
    14: "deck_bottom",
    24: "temporary",
}


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
        energies = []
        for card in poke.get("energyCards") or []:
            eid = safe_int(card.get("id"))
            if eid:
                energies.append(card_name(eid))
        if not energies:
            for eid in poke.get("energies") or []:
                eid = safe_int(eid)
                if eid:
                    energies.append(card_name(eid))
        energy_text = ",".join(energies) if energies else "-"
        return f"{card_name(cid)}({cid}) hp={hp}/{max_hp} dmg={dmg} en=[{energy_text}]"

    active = poke_text((p.get("active") or [None])[0] if p.get("active") else None)
    opp_active = poke_text((opp.get("active") or [None])[0] if opp.get("active") else None)
    bench = " | ".join(poke_text(c) for c in (p.get("bench") or []) if c) or "-"
    opp_bench = " | ".join(poke_text(c) for c in (opp.get("bench") or []) if c) or "-"
    stadium = ""
    stadium_cards = cur.get("stadium") or []
    if isinstance(stadium_cards, list) and stadium_cards:
        sid = safe_int((stadium_cards[0] or {}).get("id"))
        stadium = f" stadium={card_name(sid)}({sid})"
    return (
        f"me active={active or '-'} bench=[{bench}] prizes={len(p.get('prize') or [])} "
        f"deck={safe_int(p.get('deckCount'))} hand={len(p.get('hand') or []) if p.get('hand') is not None else safe_int(p.get('handCount'))}; "
        f"opp active={opp_active or '-'} bench=[{opp_bench}] prizes={len(opp.get('prize') or [])} "
        f"deck={safe_int(opp.get('deckCount'))} hand={safe_int(opp.get('handCount'))}{stadium}"
    )


def area_name(area) -> str:
    value = safe_int(area, -1)
    return AREA_NAMES.get(value, str(area))


def format_log_event(event: dict) -> str:
    typ = safe_int(event.get("type"), -1)
    pid = event.get("playerIndex")
    prefix = f"p{pid} " if pid is not None else ""
    cid = safe_int(event.get("cardId"))
    cname = card_name(cid) if cid else ""
    if typ == 0:
        return f"{prefix}shuffle"
    if typ == 4 and cid:
        return f"{prefix}draw/reveal {cname}({cid}) serial={event.get('serial', '')}"
    if typ == 5:
        return f"{prefix}opponent_draw"
    if typ == 6:
        card = f"{cname}({cid}) " if cid else ""
        return (
            f"{prefix}move {card}{area_name(event.get('fromArea'))}"
            f"->{area_name(event.get('toArea'))} serial={event.get('serial', '')}"
        )
    if typ == 7:
        return f"{prefix}move/unknown {area_name(event.get('fromArea'))}->{area_name(event.get('toArea'))}"
    if typ == 8:
        active = card_name(safe_int(event.get("cardIdActive")))
        bench = card_name(safe_int(event.get("cardIdBench")))
        return f"{prefix}switch active={active} bench={bench}"
    if typ == 9:
        before = card_name(safe_int(event.get("cardIdBefore")))
        after = card_name(safe_int(event.get("cardIdAfter")))
        return f"{prefix}change {before}({event.get('cardIdBefore', '')}) -> {after}({event.get('cardIdAfter', '')})"
    if typ == 10 and cid:
        return f"{prefix}play/ability/effect {cname}({cid}) serial={event.get('serial', '')}"
    if typ == 11 and cid:
        target = card_name(safe_int(event.get("cardIdTarget")))
        return f"{prefix}attach {cname}({cid}) -> {target}({event.get('cardIdTarget', '')})"
    if typ == 12 and cid:
        target = card_name(safe_int(event.get("cardIdTarget")))
        return f"{prefix}evolve {target}({event.get('cardIdTarget', '')}) -> {cname}({cid})"
    if typ == 13 and cid:
        target = card_name(safe_int(event.get("cardIdTarget")))
        return f"{prefix}devolve {target}({event.get('cardIdTarget', '')}) -> {cname}({cid})"
    if typ == 14 and cid:
        before = card_name(safe_int(event.get("cardIdBefore")))
        after = card_name(safe_int(event.get("cardIdAfter")))
        return (
            f"{prefix}move_attached {cname}({cid}) "
            f"{before}({event.get('cardIdBefore', '')})->{after}({event.get('cardIdAfter', '')})"
        )
    if typ == 15:
        return f"{prefix}attack card={cname}({cid}) attackId={event.get('attackId', '')}"
    if typ == 16 and cid:
        value = event.get("value", "")
        mode = "counter" if event.get("putDamageCounter") else "damage"
        return f"{prefix}{mode} {cname}({cid}) value={value}"
    if typ == 22:
        return f"{prefix}coin head={event.get('head')}"
    if typ == 23:
        return f"result winner={event.get('result')} reason={event.get('reason')}"
    if typ in (17, 18, 19, 20, 21):
        names = {
            17: "poisoned",
            18: "burned",
            19: "asleep",
            20: "paralyzed",
            21: "confused",
        }
        return f"{prefix}{names[typ]} recover={event.get('isRecover')} card={cname}({cid})"
    if typ == 2:
        return f"{prefix}turn_start"
    if typ == 3:
        return f"{prefix}turn_end"
    if "hasBasicPokemon" in event:
        return f"{prefix}hasBasicPokemon={event.get('hasBasicPokemon')}"
    return json.dumps(event, ensure_ascii=False, sort_keys=True)[:220]


def public_logs(obs: dict, limit: int = 5) -> list[str]:
    logs = obs.get("logs") or []
    if not isinstance(logs, list):
        return []
    out: list[str] = []
    for item in logs[-limit:]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            text = item.get("message") or item.get("text") or item.get("log") or ""
            out.append(str(text) if text else format_log_event(item))
        else:
            out.append(str(item))
    return out


def option_detail(obs: dict, idx: int) -> str:
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    if not (0 <= idx < len(options)):
        return f"{idx}:<missing>"
    opt = options[idx] or {}
    parts = [str(idx)]
    for key in (
        "type",
        "playerIndex",
        "area",
        "index",
        "inPlayArea",
        "inPlayIndex",
        "cardId",
        "attackId",
        "skillId",
        "param1",
        "param2",
        "damage",
        "remainDamageCounter",
        "remainEnergyCost",
    ):
        if key in opt and opt.get(key) is not None:
            value = opt.get(key)
            if key == "cardId":
                value = f"{value}:{card_name(safe_int(value))}"
            parts.append(f"{key}={value}")
    target = ""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    pid = safe_int(opt.get("playerIndex"), safe_int(cur.get("yourIndex")))
    area = safe_int(opt.get("area"), safe_int(opt.get("inPlayArea"), -1))
    pos = safe_int(opt.get("index"), safe_int(opt.get("inPlayIndex"), -1))
    poke = None
    if 0 <= pid < len(players):
        player = players[pid]
        if area == 4:
            active = player.get("active") or []
            poke = active[pos] if 0 <= pos < len(active) else None
        elif area == 5:
            bench = player.get("bench") or []
            poke = bench[pos] if 0 <= pos < len(bench) else None
    if poke:
        cid = safe_int(poke.get("id"))
        hp, max_hp, dmg = pokemon_hp(poke)
        target = f" target={card_name(cid)}({cid}) hp={hp}/{max_hp} dmg={dmg}"
    return " ".join(parts) + target


def option_details(obs: dict, indices_text: str, *, limit: int = 8) -> str:
    indices = []
    for tok in str(indices_text or "").split():
        try:
            indices.append(int(tok))
        except Exception:
            continue
    if not indices:
        return ""
    return " || ".join(option_detail(obs, i) for i in indices[:limit])


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
    chosen_details = option_details(obs, row.get("chosen_indices", ""))
    if chosen_details:
        lines.append(f"  chosen_options: {chosen_details}")
    top_details = option_details(obs, row.get("top_option_indices", ""))
    if top_details:
        lines.append(f"  top_options: {top_details}")
    if row.get("policy_rule_hits"):
        lines.append(f"  rules: {row.get('policy_rule_hits')}")
    if row.get("chosen_target_name"):
        lines.append(
            f"  target: {row.get('chosen_target_name')} hp={row.get('chosen_target_hp')}/"
            f"{row.get('chosen_target_max_hp')} dmg={row.get('chosen_target_damage')} "
            f"area={row.get('chosen_target_area')} owner={row.get('chosen_target_owner')}"
        )
    logs = public_logs(obs, limit=5)
    if logs:
        lines.append("  public_logs: " + " || ".join(logs))
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
    p.add_argument("--start-game", type=int, default=0,
                   help="first game index to scan; seed is computed as --seed + game")
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
    start_game = max(0, int(args.start_game))
    end_game = start_game + max(0, int(args.games))
    for scanned, g in enumerate(range(start_game, end_game), 1):
        outcome, trace, steps = play_game(args, g, args.seed + g, encoder, policy, opp_policy)
        counts[outcome] += 1
        if args.progress_every and (scanned == 1 or scanned % args.progress_every == 0):
            print(
                f"{scanned}/{args.games} game={g} win={counts['win']} loss={counts['loss']} draw={counts['draw']} "
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
