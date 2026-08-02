"""
Marnie Grimmsnarl ex — Rule-Based Agent

Strategy: Spread damage via Froslass + Munkidori, finish with Shadow Bullet.

Core combo:
  1. Froslass "Freezing Shroud" → chip all Ability Pokemon each checkup
  2. Munkidori "Adrena-Brain" → move damage counters to opponent
  3. Marnie's Grimmsnarl ex "Shadow Bullet" → 180 Active + 30 Bench
  4. "Punk Up" on evolve → attach 5 Dark Energy from deck

Deck (pkm pool_329, adapted for Kaggle card pool):
  Marnie's Impidimp x3 (646), Marnie's Morgrem x2 (647)
  Marnie's Grimmsnarl ex x3 (648)
  Munkidori x4 (112), Snorunt x2 (103), Froslass x2 (104)
  Budew x1 (235), Tatsugiri x1 (122), Yveltal x1 (689)
  Team Rocket's Petrel x4 (1219), Iris's Fighting Spirit x4 (1208)?
  Lillie's Determination x4 (1227), Boss's Orders x3 (1182)
  Poke Pad x4 (1152), Buddy-Buddy Poffin x3 (1086)
  Night Stretcher x3 (1097), Rare Candy x2 (1079)
  Energy Switch x1 (1116), Secret Box x1 (1092)
  Air Balloon x1 (1174), Spikemuth Gym x4 (1259)
  Basic Darkness Energy x10 (7)
"""

import os, sys, random
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from cg.api import (
    AreaType, CardType, Observation, SelectContext,
    OptionType, Card, Pokemon, all_card_data, to_observation_class,
)

# ── Load deck ────────────────────────────────────────────────────────────
DECK_PATH = os.path.join(os.path.dirname(_HERE), "deck.csv")
if not os.path.exists(DECK_PATH):
    DECK_PATH = os.path.join(_HERE, "deck.csv")
with open(DECK_PATH) as f:
    MY_DECK = [int(line.strip()) for line in f if line.strip()]

ALL_CARDS = all_card_data()
CARD_TABLE = {c.cardId: c for c in ALL_CARDS}

# ── Card IDs ─────────────────────────────────────────────────────────────
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
MUNKIDORI = 112
SNORUNT = 103
FROSLASS = 104
BUDEW = 235
TATSUGIRI = 122
YVELTAL = 689
TEAM_ROCKET_PETREL = 1219
IRIS_FIGHTING_SPIRIT = 1208
LILLIE_DETERMINATION = 1227
BOSS_ORDERS = 1182
POKE_PAD = 1152
BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
RARE_CANDY = 1079
ENERGY_SWITCH = 1116
SECRET_BOX = 1092
AIR_BALLOON = 1174
SPIKEMUTH_GYM = 1259
BASIC_DARK = 7

GRIMMSNARL_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL_EX}
FROSLASS_LINE = {SNORUNT, FROSLASS}

pre_turn = -1
ability_used_munkidori: set[int] = set()  # serials of Munkidori that used ability


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.HAND: return ps.hand[index] if ps.hand else None
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE: return ps.active[index]
        case AreaType.BENCH: return ps.bench[index]
        case AreaType.PRIZE: return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index] if obs.current.looking else None
        case _: return None


def agent(obs_dict: dict) -> list[int]:
    global pre_turn, ability_used_munkidori

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(MY_DECK)

    state = obs.current
    sel = obs.select
    ctx = sel.context
    you = state.yourIndex
    me = state.players[you]
    op = state.players[1 - you]

    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_munkidori.clear()

    # ── Count cards ────────────────────────────────────────────────────
    field_cnt = defaultdict(int)
    hand_cnt = defaultdict(int)
    discard_cnt = defaultdict(int)

    my_board = []  # (index, pokemon)
    if me.active and me.active[0]:
        my_board.append((0, me.active[0]))
        field_cnt[me.active[0].id] += 1
    for i, p in enumerate(me.bench):
        if p:
            my_board.append((i + 1, p))
            field_cnt[p.id] += 1

    for c in (me.hand or []):
        hand_cnt[c.id] += 1
    for c in me.discard:
        discard_cnt[c.id] += 1

    grimm_on_field = field_cnt[IMPIDIMP] + field_cnt[MORGREM] + field_cnt[GRIMMSNARL_EX]
    froslass_on_field = field_cnt[SNORUNT] + field_cnt[FROSLASS]
    munkidori_on_field = field_cnt[MUNKIDORI]
    bench_count = len(me.bench)
    bench_free = me.benchMax - bench_count
    stadium_id = state.stadium[0].id if state.stadium else 0

    # Energy state
    active = me.active[0] if me.active else None
    active_id = active.id if active else -1
    active_dark = sum(1 for ec in (active.energyCards or []) if ec.id == BASIC_DARK) if active else 0
    dark_in_hand = hand_cnt.get(BASIC_DARK, 0)

    # Munkidori with Dark energy for Adrena-Brain
    munkidori_dark = []
    for _, p in my_board:
        if p.id == MUNKIDORI:
            has_dark = any(ec.id == BASIC_DARK for ec in p.energyCards)
            if has_dark:
                munkidori_dark.append(p.serial)

    can_evolve_gr = (field_cnt[IMPIDIMP] > 0 or field_cnt[MORGREM] > 0)

    # Opponent
    op_active = op.active[0] if op.active else None
    op_board = []
    if op_active:
        op_board.append(op_active)
    for p in op.bench:
        if p: op_board.append(p)

    # ── Score each option ──────────────────────────────────────────────
    scores = []
    for o in sel.option:
        score = 0
        ot = o.type

        if ot == OptionType.YES:
            score = 1
        elif ot == OptionType.NO:
            score = 0
        elif ot == OptionType.END:
            score = 5 if (active_id == GRIMMSNARL_EX and active_dark >= 2) else 10

        elif ot == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, you)
            if card is None:
                scores.append(score); continue
            cid = card.id
            data = CARD_TABLE.get(cid)

            if data and data.cardType == CardType.POKEMON:
                score = 20000
                # Grimmsnarl line — highest priority
                if cid == IMPIDIMP:
                    if grimm_on_field < 2:
                        score += 500
                    elif bench_free <= 0:
                        score = -1
                elif cid == GRIMMSNARL_EX:
                    score = -1  # Can't play Stage2 directly — must evolve
                # Froslass line
                elif cid == SNORUNT:
                    score += 300 if froslass_on_field < 1 else (100 if froslass_on_field < 2 else -1)
                elif cid == FROSLASS:
                    score = -1  # Must evolve
                # Munkidori
                elif cid == MUNKIDORI:
                    if munkidori_on_field < 2:
                        score += 400
                    elif munkidori_on_field < 3:
                        score += 100
                    else:
                        score = -1
                # Budew
                elif cid == BUDEW:
                    score += 200 if bench_free >= 1 else -1
                # Don't fill last bench slot
                if bench_free <= 1 and score > 0:
                    score -= 5000
            else:
                # Trainers
                score = 10000
                if cid == SPIKEMUTH_GYM:
                    if stadium_id == 0:
                        score += 3000  # Need Spikemuth to search
                elif cid == TEAM_ROCKET_PETREL:
                    score += 2000  # Search any trainer
                elif cid == IRIS_FIGHTING_SPIRIT:
                    if len(me.hand or []) <= 3:
                        score += 2500  # Need draw
                elif cid == LILLIE_DETERMINATION:
                    if len(me.hand or []) <= 4:
                        score += 2000
                elif cid == BUDDY_BUDDY_POFFIN:
                    if grimm_on_field < 2 or munkidori_on_field < 1:
                        score += 3000
                    else:
                        score = -1
                elif cid == POKE_PAD:
                    if grimm_on_field < 2:
                        score += 1500
                elif cid == RARE_CANDY:
                    if field_cnt[IMPIDIMP] >= 1 and hand_cnt[GRIMMSNARL_EX] >= 1:
                        score += 5000  # Key combo!
                elif cid == NIGHT_STRETCHER:
                    if discard_cnt[GRIMMSNARL_EX] + discard_cnt[IMPIDIMP] >= 1:
                        score += 2500
                    elif discard_cnt[MUNKIDORI] >= 1:
                        score += 1500
                elif cid == BOSS_ORDERS:
                    if active_id == GRIMMSNARL_EX and active_dark >= 2:
                        score += 3000
                elif cid == SECRET_BOX:
                    score += 3500  # ACE SPEC — always good
                elif cid == AIR_BALLOON:
                    score += 1000
                elif cid == ENERGY_SWITCH:
                    if munkidori_dark:
                        score += 500
                    else:
                        score += 2000  # Move energy to Munkidori

        elif ot == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, you)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, you)
            if not isinstance(pokemon, Pokemon):
                scores.append(score); continue

            if card.id == BASIC_DARK:
                if pokemon.id == MUNKIDORI:
                    has_dark = any(ec.id == BASIC_DARK for ec in pokemon.energyCards)
                    score = 9000 if not has_dark else -1  # 1 is enough for Adrena-Brain
                elif pokemon.id == GRIMMSNARL_EX:
                    nrj = len(pokemon.energyCards)
                    score = 8000 if nrj < 2 else 2000
                elif pokemon.id in (IMPIDIMP, MORGREM):
                    score = -1  # Never attach to basics — Punk Up handles energy
                elif pokemon.id == FROSLASS:
                    score = 1000  # Only if needed for retreat
                else:
                    score = 500
            elif card.id == AIR_BALLOON:
                if pokemon.id in (TATSUGIRI, BUDEW):
                    score = 5000  # Free retreat for pivot
                else:
                    score = 3000

        elif ot == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, you)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, you)
            if not isinstance(pokemon, Pokemon):
                scores.append(score); continue
            score = 9000
            if card.id == GRIMMSNARL_EX:
                score += 5000  # Punk Up = 5 free energy, highest priority
                if o.inPlayArea == AreaType.ACTIVE:
                    score += 200  # Active Grimmsnarl can attack immediately
            elif card.id == MORGREM:
                score += 1500  # Step toward Grimmsnarl
            elif card.id == FROSLASS:
                score += 3000  # Freezing Shroud chip starts immediately
            score += len(pokemon.energyCards) * 10  # Preserve energy investment

        elif ot == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, you)
            if card is None:
                scores.append(score); continue
            if card.id == MUNKIDORI:
                if card.serial not in ability_used_munkidori:
                    # Use if we have damage counters on our board
                    our_dmg = sum(
                        p.hp for _, p in my_board
                        if p and hasattr(p, 'hp') and hasattr(p, 'maxHp')
                        and getattr(p, 'maxHp', 0) > 0
                        and p.hp < p.maxHp
                    )
                    if our_dmg > 0:
                        score = 30000
                    else:
                        score = 5000
                else:
                    score = -1
            elif card.id == SPIKEMUTH_GYM:
                score = 10000  # Use Spikemuth to search
            else:
                score = 28000

        elif ot == OptionType.ATTACK:
            score = 1000
            if o.attackId == 937:  # Shadow Bullet
                score += 3000
                if active_dark >= 2:
                    score += 2000

        elif ot == OptionType.RETREAT:
            if active_id in (BUDEW, TATSUGIRI):
                score = 8000  # Must get these out of active
            elif active_id == IMPIDIMP:
                # Impidimp can't attack well — retreat if bench has Grimmsnarl
                has_grimm_bench = any(p.id == GRIMMSNARL_EX for _, p in my_board)
                score = 7000 if has_grimm_bench else 2000
            elif active_id == GRIMMSNARL_EX and active_dark < 2:
                score = 1500
            elif active_id not in (GRIMMSNARL_EX, MUNKIDORI):
                score = 1000  # Get non-attackers out
            else:
                score = -1

        elif ot == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score); continue

            if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id == IMPIDIMP:
                    score = 10
                elif card.id == SNORUNT:
                    score = 5
                elif card.id == BUDEW:
                    score = 8 if state.firstPlayer != you else 3  # Budew going 2nd is strong

            elif ctx == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == IMPIDIMP:
                    score = 200 if grimm_on_field == 0 else 50
                elif card.id == MUNKIDORI:
                    score = 150
                elif card.id == SNORUNT:
                    score = 100

            elif ctx == SelectContext.TO_HAND:
                score = 200 - hand_cnt.get(card.id, 0) * 100
                if card.id == IMPIDIMP:
                    score += 50 if grimm_on_field < 2 else -50
                elif card.id == GRIMMSNARL_EX:
                    score += 80 if field_cnt[IMPIDIMP] >= 1 or field_cnt[MORGREM] >= 1 else -20
                elif card.id == MUNKIDORI:
                    score += 40 if munkidori_on_field < 3 else -100
                elif card.id == BASIC_DARK:
                    score += 30

            elif ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                if o.playerIndex == you:
                    if card.id == GRIMMSNARL_EX:
                        score = 100 if active_dark >= 2 else 50
                    elif card.id == MUNKIDORI:
                        score = 60
                    elif card.id == TATSUGIRI:
                        score = 30
                    else:
                        score = 10
                else:
                    # Opponent — for Boss's Orders
                    # Target high-value/ex Pokemon first
                    score = 1

            elif ctx == SelectContext.DISCARD:
                if card.id == BASIC_DARK:
                    score = 80  # Safe discard
                elif card.id in (BUDEW, TATSUGIRI):
                    score = 100 if field_cnt.get(card.id, 0) >= 1 else -50
                else:
                    score = 20

        scores.append(score)

    # ── Rank and return ────────────────────────────────────────────────
    ranked = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    # Track Munkidori ability usage
    if ctx == SelectContext.MAIN and ranked:
        top_o = sel.option[ranked[0]]
        if top_o.type == OptionType.ABILITY:
            card = get_card(obs, top_o.area, top_o.index, you)
            if card is not None and card.id == MUNKIDORI:
                ability_used_munkidori.add(card.serial)

    return ranked[:sel.maxCount]
