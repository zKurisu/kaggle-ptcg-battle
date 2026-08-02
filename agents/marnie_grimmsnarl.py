"""
Marnie Grimmsnarl ex — Rule-Based Agent v2

Strategy (from tournament guides, Bulbapedia, competitive forums):
  1. Golden Rule: Spikemuth Gym → Items → Evolve → Supporter → Attack → Munkidori → Attach/Retreat
  2. Grimmsnarl ex + Froslass + Munkidori spread-damage core
  3. Punk Up on evolve: ONLY grab 2 Darkness (leave rest in deck for Munkidori)
  4. Never attach energy to basics (Impidimp/Morgrem) — Punk Up handles it
  5. Munkidori needs exactly 1 Dark for Adrena-Brain
  6. Retreat non-attackers immediately (Budew 0-cost, use it!)
  7. Against slow decks: only 1 Froslass; vs aggressive: bench both Snorunt

Deck: pool_329_marnie_s_grimmsnarl_ex (Kaggle-adapted)
"""

import os, sys, random
from collections import defaultdict, Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from cg.api import (
    AreaType, CardType, Observation, SelectContext,
    OptionType, Card, Pokemon, all_card_data, to_observation_class,
)

# ── Deck loading ──────────────────────────────────────────────────────────
DECK_PATH = os.path.join(os.path.dirname(_HERE), "deck.csv")
if not os.path.exists(DECK_PATH):
    DECK_PATH = os.path.join(_HERE, "deck.csv")
with open(DECK_PATH) as f:
    MY_DECK = [int(line.strip()) for line in f if line.strip()]

ALL_CARDS = all_card_data()
CARD_TABLE = {c.cardId: c for c in ALL_CARDS}

# ── Card IDs ──────────────────────────────────────────────────────────────
IMP = 646; MORG = 647; GRIM = 648
MUNKI = 112; SNO = 103; FROS = 104
BUDEW = 235; TATSU = 122; YVEL = 689
PETREL = 1219; IRIS = 1208; LILLIE = 1227
BOSS = 1182; P_PAD = 1152; POFFIN = 1086
STRETCHER = 1097; CANDY = 1079; E_SWITCH = 1116
SECRET = 1092; BALLOON = 1174; SPIKE = 1259
DARK = 7

GRIM_LINE = {IMP, MORG, GRIM}
FROS_LINE = {SNO, FROS}
FREE_RETREAT = {BUDEW, TATSU}  # 0 or 1 retreat cost — cheap to get out of active

pre_turn = -1
munki_used: set[int] = set()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _card_at(obs, area, idx, player):
    ps = obs.current.players[player]
    try:
        match area:
            case AreaType.HAND: return ps.hand[idx] if ps.hand else None
            case AreaType.DISCARD: return ps.discard[idx]
            case AreaType.ACTIVE: return ps.active[idx] if ps.active else None
            case AreaType.BENCH: return ps.bench[idx]
            case AreaType.PRIZE: return ps.prize[idx]
            case AreaType.STADIUM: return obs.current.stadium[idx]
            case AreaType.LOOKING: return (obs.current.looking or [])[idx] if obs.current.looking else None
    except (IndexError, TypeError): pass
    return None


def _evo_target_card(obs, opt, you):
    """What card ID does this EVOLVE option evolve INTO?"""
    cid = getattr(opt, 'cardId', None)
    if cid and cid > 0: return cid
    area = getattr(opt, 'inPlayArea', None)
    idx = getattr(opt, 'inPlayIndex', 0)
    target = _card_at(obs, area, idx, you)
    if target:
        tid = target.id
        if tid == IMP: return MORG
        if tid == MORG: return GRIM
        if tid == SNO: return FROS
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════

def agent(obs_dict: dict) -> list[int]:
    global pre_turn, munki_used
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(MY_DECK)

    st = obs.current; sel = obs.select; ctx = sel.context
    you = st.yourIndex; me = st.players[you]; op = st.players[1 - you]

    if pre_turn != st.turn:
        pre_turn = st.turn; munki_used.clear()

    # ── Board state ───────────────────────────────────────────────────
    field = {}; fcnt = Counter()
    active = me.active[0] if me.active else None
    if active: field[0] = active; fcnt[active.id] += 1
    for i, p in enumerate(me.bench):
        if p: field[i + 1] = p; fcnt[p.id] += 1

    hcnt = Counter(c.id for c in (me.hand or []))
    dcnt = Counter(c.id for c in me.discard)

    grimm_chain = fcnt[IMP] + fcnt[MORG] + fcnt[GRIM]
    has_grimm = fcnt[GRIM] > 0
    has_munki = fcnt[MUNKI] > 0
    has_fros = fcnt[FROS] > 0
    bench_free = me.benchMax - len(me.bench)
    stadium_id = st.stadium[0].id if st.stadium else 0
    deck_low = me.deckCount < 8

    aid = active.id if active else -1
    a_dark = sum(1 for e in (active.energyCards or []) if e.id == DARK) if active else 0
    a_can_attack = (aid == GRIM and a_dark >= 2)
    a_is_weak = aid in FREE_RETREAT or aid in (IMP, SNO, MORG)
    a_has_energy = sum(1 for e in (active.energyCards or [])) if active else 0

    # Can evolve to Grimmsnarl this turn?
    can_rc = (aid == IMP and hcnt.get(CANDY, 0) > 0 and hcnt.get(GRIM, 0) > 0)
    can_evo = (aid == MORG and hcnt.get(GRIM, 0) > 0)
    grimm_ready = can_rc or can_evo

    # Munkidori state
    munki_dark = sum(1 for _, p in field.items()
                     if p and p.id == MUNKI and any(e.id == DARK for e in p.energyCards))
    our_dmg = sum(getattr(p, 'maxHp', 0) - getattr(p, 'hp', 0)
                  for _, p in field.items() if p and getattr(p, 'maxHp', 0) > 0)
    can_attach = not st.energyAttached

    # Opponent
    op_active = op.active[0] if op.active else None
    op_bench_count = len([p for p in op.bench if p])

    # ── Score ─────────────────────────────────────────────────────────
    scores = []
    for o in sel.option:
        score = 0; t = o.type

        # YES/NO/END
        if t in (OptionType.YES, OptionType.NO):
            score = 1 if t == OptionType.YES else 0
        elif t == OptionType.END:
            score = -500 if a_can_attack else 50

        # ── RETREAT: get non-attackers out of active ──────────────────
        elif t == OptionType.RETREAT:
            # CRITICAL: if Grimmsnarl ex is on bench, ALWAYS swap it in
            has_grimm_bench = any(p and p.id == GRIM for p in field.values() if getattr(p, 'id', 0) == GRIM and p != active)
            if aid in FREE_RETREAT:
                score = 50000  # Absolute highest priority — free retreat
            elif has_grimm_bench and aid != GRIM:
                score = 40000  # Swap in the attacker!
            elif aid == IMP and not grimm_ready:
                if has_grimm_bench:
                    score = 35000
                elif bench_free > 0:
                    score = 5000
            elif aid == GRIM and a_dark >= 2:
                score = -1  # Never retreat ready Grimmsnarl
            elif aid == SNO and fcnt.get(FROS, 0) == 0:
                score = 3000
            else:
                score = -1

        # ── EVOLVE: highest priority ─────────────────────────────────
        elif t == OptionType.EVOLVE:
            evo = _evo_target_card(obs, o, you)
            score = 9000
            if evo == GRIM:
                score += 25000  # Punk Up = free energy, GAME-CHANGING
            elif evo == MORG:
                score += 8000
            elif evo == FROS:
                score += 15000  # Freezing Shroud chip
            # Prefer evolving active Impidimp (it needs to get out anyway)
            target = _card_at(obs, getattr(o, 'inPlayArea', None),
                             getattr(o, 'inPlayIndex', 0), you)
            if target and getattr(target, 'id', 0) == aid:
                score += 2000

        # ── ATTACH energy (Golden Rule: LAST step) ──────────────────
        elif t == OptionType.ATTACH:
            card = _card_at(obs, AreaType.HAND, o.index, you)
            target = _card_at(obs, getattr(o, 'inPlayArea', None),
                             getattr(o, 'inPlayIndex', 0), you)
            if card is None or not isinstance(target, Pokemon):
                scores.append(score); continue

            if card.id == DARK:
                tid = target.id
                if tid == GRIM:
                    nd = len(target.energyCards)
                    score = 10000 if nd < 2 else 3000
                elif tid == MUNKI:
                    has_d = any(e.id == DARK for e in target.energyCards)
                    score = 9000 if not has_d else -1
                elif tid in (IMP, MORG):
                    score = -5000  # NEVER attach to basics
                elif tid == FROS:
                    score = 2000
                else:
                    score = 500
            elif card.id == BALLOON:
                score = 6000 if aid in FREE_RETREAT else 3000

        # ── PLAY from hand (Golden Rule: 1st-2nd step) ──────────────
        elif t == OptionType.PLAY:
            card = _card_at(obs, AreaType.HAND, o.index, you)
            if card is None: scores.append(score); continue
            data = CARD_TABLE.get(card.id); cid = card.id

            if data and data.cardType == CardType.POKEMON:
                score = 20000
                if cid in (IMP, MORG):
                    if grimm_chain < 2: score += 500
                    elif bench_free <= 0: score = -1
                elif cid == GRIM: score = -1
                elif cid == SNO:
                    if fcnt.get(FROS, 0) == 0: score += 400
                    elif fcnt.get(FROS, 0) < 2: score += 100
                    else: score = -1
                elif cid == FROS: score = -1
                elif cid == MUNKI:
                    if fcnt.get(MUNKI, 0) < 2: score += 500
                    elif fcnt.get(MUNKI, 0) < 3: score += 150
                    else: score = -1
                elif cid == BUDEW:
                    if st.turn <= 2: score += 300
                elif cid == TATSU: score += 200
                elif cid == YVEL: score += 100
                if bench_free <= 1 and score > 0: score -= 7000
            else:
                score = 10000
                if cid == SPIKE:
                    if stadium_id == 0: score += 5000
                    elif stadium_id != SPIKE: score += 2000
                elif cid == CANDY:
                    if (aid == IMP and hcnt.get(GRIM, 0) > 0): score += 10000
                    elif fcnt.get(IMP, 0) > 0 and hcnt.get(GRIM, 0) > 0: score += 4000
                    else: score = -1
                elif cid == PETREL: score += 3500
                elif cid == IRIS:
                    hc = len(me.hand) if me.hand else me.handCount
                    score += 4000 if hc <= 3 else 1000
                elif cid == LILLIE:
                    hc = len(me.hand) if me.hand else me.handCount
                    score += 3000 if hc <= 4 else 500
                    if deck_low: score = -1
                elif cid == POFFIN:
                    need = (grimm_chain < 2) or not has_munki or fcnt.get(SNO, 0) == 0
                    score += 4000 if need else -1
                elif cid == P_PAD:
                    score += 2500 if grimm_chain < 2 else 800
                elif cid == STRETCHER:
                    need = dcnt.get(GRIM, 0) + dcnt.get(IMP, 0) + dcnt.get(MUNKI, 0)
                    score += 3000 if need > 0 else 500
                elif cid == BOSS:
                    score += 4000 if a_can_attack else 800
                elif cid == SECRET: score += 4500
                elif cid == E_SWITCH:
                    score += 1500
                elif cid == BALLOON: score += 2000

        # ── ABILITY (Golden Rule: after attack) ─────────────────────
        elif t == OptionType.ABILITY:
            card = _card_at(obs, getattr(o, 'area', None),
                           getattr(o, 'index', 0), you)
            if card is None: scores.append(score); continue
            if card.id == MUNKI:
                if card.serial not in munki_used:
                    score = 35000 if our_dmg > 0 else 15000
                else: score = -1
            elif card.id == SPIKE:
                score = 20000  # Search Marnie's Pokemon
            elif card.id == TATSU:
                score = 15000  # Search supporter
            elif card.id == GRIM:
                score = 1  # Punk Up auto-triggered on evolve, not manual
            else: score = 30000

        # ── ATTACK ──────────────────────────────────────────────────
        elif t == OptionType.ATTACK:
            if o.attackId == 937:  # Shadow Bullet
                score = 60000 if a_can_attack else 500  # Top priority when ready
            elif o.attackId == 323:  # Itchy Pollen (Budew)
                score = 8000 if st.turn <= 3 else 3000
            elif o.attackId == 141:  # Mind Bend (Munkidori)
                score = 10000 if a_dark >= 1 else 500
            elif o.attackId == 155:  # Surf (Tatsugiri)
                score = 5000
            elif o.attackId == 997:  # Clutch (Yveltal)
                score = 5000
            elif o.attackId == 130:  # Astonish (Snorunt)
                score = 3000
            else: score = 2000

        # ── CARD selection ──────────────────────────────────────────
        elif t == OptionType.CARD:
            card = _card_at(obs, getattr(o, 'area', None),
                           getattr(o, 'index', 0), o.playerIndex)
            if card is None: scores.append(score); continue

            if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id == IMP: score = 10
                elif card.id == BUDEW and st.firstPlayer != you: score = 12
                elif card.id == SNO: score = 5

            elif ctx == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == IMP: score = 200 if grimm_chain == 0 else 60
                elif card.id == MUNKI: score = 150
                elif card.id == SNO: score = 100

            elif ctx == SelectContext.TO_HAND:
                score = 200 - hcnt.get(card.id, 0) * 100
                if card.id == IMP: score += 50 if grimm_chain < 2 else -50
                elif card.id == GRIM: score += 80 if (fcnt[IMP] + fcnt[MORG]) > 0 else -30
                elif card.id == MUNKI: score += 40 if fcnt[MUNKI] < 3 else -100
                elif card.id == SNO: score += 30 if (fcnt[SNO] + fcnt[FROS]) < 2 else -50
                elif card.id == DARK: score += 30

            elif ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                if o.playerIndex == you:
                    if card.id == GRIM: score = 100
                    elif card.id == MUNKI: score = 60
                    else: score = 15
                else: score = 5

            elif ctx == SelectContext.DISCARD:
                if card.id == DARK: score = 80
                elif card.id in (BUDEW, TATSU, YVEL):
                    score = 100 if fcnt.get(card.id, 0) >= 1 else -40
                else: score = 30

            elif ctx == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if card.id == MUNKI and not any(e.id == DARK for e in card.energyCards):
                        score = 200
                    elif card.id == GRIM and len(card.energyCards) < 2:
                        score = 300
                    else: score = 50

        scores.append(score)

    # ── Return top ───────────────────────────────────────────────────
    ranked = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    if ctx == SelectContext.MAIN and ranked:
        top_o = sel.option[ranked[0]]
        if top_o.type == OptionType.ABILITY:
            card = _card_at(obs, getattr(top_o, 'area', None),
                           getattr(top_o, 'index', 0), you)
            if card is not None and card.id == MUNKI:
                munki_used.add(card.serial)

    return ranked[:sel.maxCount]
