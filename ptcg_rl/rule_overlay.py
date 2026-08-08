from __future__ import annotations

from dataclasses import dataclass

from ptcg_rl.deck_plans import DeckPlan, infer_plan


PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14

RULE_MODES = (
    "conservative",
    "aggressive",
    "marnie_setup",
    "ogerpon_attach",
    "ogerpon_no_futile_crustle",
    "cynthia_spiritomb_crustle",
    "stage2_setup",
    "primary_active",
    "targeted",
    "counter_plan",
    "counter_plan_aggressive",
)

MARNIE_IMPIDIMP = 646
MARNIE_MORGREM = 647
MARNIE_GRIMMSNARL_EX = 648
MUNKIDORI = 112
SPIKEMUTH_GYM = 1259
OGERPON_EX = 96
DWEBBLE = 344
CRUSTLE = 345
CYNTHIA_GARCHOMP_EX = 381
CYNTHIA_SPIRITOMB = 387
TEAM_ROCKET_TAROUNTULA = 400
TEAM_ROCKET_SPIDOPS = 401
TEAM_ROCKET_MEWTWO_EX = 431
TEAM_ROCKET_MIMIKYU = 434
MEGA_LUCARIO_EX = 678

MARNIE_LINE = {MARNIE_IMPIDIMP, MARNIE_MORGREM, MARNIE_GRIMMSNARL_EX}
OGERPON_LINE = {OGERPON_EX}
CRUSTLE_LINE = {DWEBBLE, CRUSTLE}
CYNTHIA_LINE = {379, 380, CYNTHIA_GARCHOMP_EX, CYNTHIA_SPIRITOMB}
TEAM_ROCKET_LINE = {
    TEAM_ROCKET_TAROUNTULA,
    TEAM_ROCKET_SPIDOPS,
    TEAM_ROCKET_MEWTWO_EX,
    TEAM_ROCKET_MIMIKYU,
}
MEGA_LUCARIO_LINE = {333, 673, 674, 675, 676, 677, MEGA_LUCARIO_EX, 1141, 1142, 1152}
DRAGAPULT_LINE = {119, 120, 121}
ALAKAZAM_LINE = {109, 245, 742}
MEGA_LOPUNNY_LINE = {65, 66, 306, 858, 849}
FESTIVAL_LINE = {42, 89, 90, 93}
KNOWN_EX_ATTACKERS = {
    OGERPON_EX,
    CYNTHIA_GARCHOMP_EX,
    TEAM_ROCKET_MEWTWO_EX,
    MARNIE_GRIMMSNARL_EX,
    MEGA_LUCARIO_EX,
    121,
    272,
    306,
    381,
    648,
    756,
    849,
    1071,
}


@dataclass(frozen=True)
class RuleDecision:
    action: list[int]
    reason: str = ""


def _option_type(opt: dict) -> int:
    return int(opt.get("type", 0) or 0)


def _option_card(obs: dict, opt: dict) -> int:
    cur = obs.get("current") or {}
    you = cur.get("yourIndex", 0)
    pid = int(opt.get("playerIndex", you) if opt.get("playerIndex") is not None else you)
    area = opt.get("area")
    idx = opt.get("index")
    if area is None or idx is None:
        return int(opt.get("cardId") or 0)
    ps = (cur.get("players") or [{}, {}])[pid]
    idx = int(idx)
    if area == 2:
        cards = ps.get("hand") or []
    elif area == 3:
        cards = ps.get("discard") or []
    elif area == 4:
        cards = ps.get("active") or []
    elif area == 5:
        cards = ps.get("bench") or []
    elif area == 11:
        cards = cur.get("looking") or []
    else:
        cards = []
    if 0 <= idx < len(cards) and cards[idx]:
        return int(cards[idx].get("id") or 0)
    return int(opt.get("cardId") or 0)


def _find_type(options: list[dict], opt_type: int) -> list[int]:
    return [i for i, opt in enumerate(options) if _option_type(opt) == opt_type]


def _active_card_id(player: dict) -> int:
    active = player.get("active") or []
    if active and active[0]:
        return int(active[0].get("id") or 0)
    return 0


def _bench_card_ids(player: dict) -> set[int]:
    ids = set()
    for p in player.get("bench") or []:
        if p:
            ids.add(int(p.get("id") or 0))
    ids.discard(0)
    return ids


def _zone_card_ids(player: dict, *zones: str) -> set[int]:
    ids: set[int] = set()
    for zone in zones:
        for card in player.get(zone) or []:
            if card:
                ids.add(int(card.get("id") or 0))
    ids.discard(0)
    return ids


def _visible_card_ids(player: dict) -> set[int]:
    return _zone_card_ids(player, "active", "bench", "discard")


def _has_energy_attach_available(options: list[dict]) -> bool:
    return any(_option_type(opt) == ATTACH for opt in options)


def _first_card_tag(plan: DeckPlan, obs: dict, options: list[dict], opt_type: int,
                    card_ids: set[int]) -> int | None:
    for i, opt in enumerate(options):
        if _option_type(opt) == opt_type and _option_card(obs, opt) in card_ids:
            return i
    return None


def _first_exact_card(obs: dict, options: list[dict], opt_type: int, card_id: int) -> int | None:
    for i, opt in enumerate(options):
        if _option_type(opt) == opt_type and _option_card(obs, opt) == card_id:
            return i
    return None


def _first_card_any_type(obs: dict, options: list[dict], card_ids: set[int]) -> int | None:
    for i, opt in enumerate(options):
        if _option_card(obs, opt) in card_ids:
            return i
    return None


def _first_card_by_types(obs: dict, options: list[dict], card_ids: set[int],
                         opt_types: tuple[int, ...]) -> int | None:
    for opt_type in opt_types:
        for i, opt in enumerate(options):
            if _option_type(opt) == opt_type and _option_card(obs, opt) in card_ids:
                return i
    return None


def _first_non_attack_tempo(obs: dict, options: list[dict], *, avoid_cards: set[int] | None = None) -> int | None:
    avoid = avoid_cards or set()
    for opt_type in (RETREAT, EVOLVE, ATTACH, PLAY, ABILITY, END):
        for i, opt in enumerate(options):
            if _option_type(opt) != opt_type:
                continue
            if avoid and _option_card(obs, opt) in avoid:
                continue
            return i
    return None


def _visible_matchup_flags(opp: dict) -> dict[str, bool]:
    visible = _visible_card_ids(opp)
    active = _active_card_id(opp)
    return {
        "ogerpon": active in OGERPON_LINE or bool(visible & OGERPON_LINE),
        "crustle": active in CRUSTLE_LINE or bool(visible & CRUSTLE_LINE),
        "marnie": active in MARNIE_LINE or bool(visible & MARNIE_LINE),
        "cynthia": active in CYNTHIA_LINE or bool(visible & CYNTHIA_LINE),
        "trmewtwo": active in TEAM_ROCKET_LINE or bool(visible & TEAM_ROCKET_LINE),
        "lucario": active in MEGA_LUCARIO_LINE or bool(visible & MEGA_LUCARIO_LINE),
        "dragapult": active in DRAGAPULT_LINE or bool(visible & DRAGAPULT_LINE),
        "alakazam": active in ALAKAZAM_LINE or bool(visible & ALAKAZAM_LINE),
        "lopunny": active in MEGA_LOPUNNY_LINE or bool(visible & MEGA_LOPUNNY_LINE),
        "festival": active in FESTIVAL_LINE or bool(visible & FESTIVAL_LINE),
    }


def _counter_plan_overlay(
    *,
    plan: DeckPlan | None,
    obs: dict,
    action: list[int],
    options: list[dict],
    chosen_type: int,
    chosen_card: int,
    context: int,
    turn: int,
    turn_action_count: int,
    my_active: int,
    my_bench: set[int],
    opp: dict,
) -> RuleDecision | None:
    if not plan:
        return None

    flags = _visible_matchup_flags(opp)
    opp_active = _active_card_id(opp)
    plan_keys = set(plan.setup_basics) | set(plan.evolution_chain) | set(plan.engine_cards)

    # Active/bench selection is where bad BC often loses the entire game plan.
    # Prefer a deck's own setup/core cards early, and use known non-ex counters
    # into Crustle instead of letting an ex attacker become the default active.
    if context in (3, 4):
        if flags["crustle"] and plan.archetype == "Cynthia Garchomp" and chosen_card != CYNTHIA_SPIRITOMB:
            spiritomb = _first_card_any_type(obs, options, {CYNTHIA_SPIRITOMB})
            if spiritomb is not None:
                return RuleDecision([spiritomb], "counter:cynthia_spiritomb_active_vs_crustle")
        if flags["crustle"]:
            non_ex_counters = (set(plan.secondary_attackers) | set(plan.engine_cards)) - KNOWN_EX_ATTACKERS
            pick = _first_card_any_type(obs, options, non_ex_counters)
            if pick is not None and chosen_card in KNOWN_EX_ATTACKERS:
                return RuleDecision([pick], "counter:non_ex_active_vs_crustle")
        if turn <= 4 and plan_keys and chosen_card not in plan_keys:
            pick = _first_card_any_type(obs, options, plan_keys)
            if pick is not None:
                return RuleDecision([pick], "counter:early_plan_active")

    # Stage decks collapse if an available evolution is delayed by generic draw,
    # attach, or end choices. This is deliberately stronger than the old targeted
    # mode because the previous pilots showed small BC weights did not move
    # weak matchup behavior.
    if turn <= 12 and plan.evolution_chain and chosen_type in (PLAY, ATTACH, ABILITY, END):
        evo = _first_card_tag(plan, obs, options, EVOLVE, set(plan.evolution_chain))
        if evo is not None:
            return RuleDecision([evo], "counter:force_core_evolution")

    if plan.archetype == "Marnie Grimmsnarl" and flags["ogerpon"] and turn <= 12:
        delays_evolution = (
            chosen_type in (ATTACH, ABILITY, END)
            or (chosen_type == PLAY and chosen_card not in MARNIE_LINE)
        )
        if delays_evolution:
            if my_active == MARNIE_IMPIDIMP:
                morgrem = _first_exact_card(obs, options, EVOLVE, MARNIE_MORGREM)
                if morgrem is not None:
                    return RuleDecision([morgrem], "counter:marnie_morgrem_vs_ogerpon")
            if my_active == MARNIE_MORGREM:
                grimmsnarl = _first_exact_card(obs, options, EVOLVE, MARNIE_GRIMMSNARL_EX)
                if grimmsnarl is not None:
                    return RuleDecision([grimmsnarl], "counter:marnie_grimmsnarl_vs_ogerpon")
        punk_up = _first_exact_card(obs, options, ABILITY, MARNIE_GRIMMSNARL_EX)
        if punk_up is not None and chosen_type not in (ATTACK,):
            return RuleDecision([punk_up], "counter:marnie_punk_up_vs_ogerpon")

    if plan.archetype == "Teal Mask Ogerpon" and flags["crustle"]:
        if chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attaches = _find_type(options, ATTACH)
            if attaches:
                return RuleDecision([attaches[0]], "counter:ogerpon_attach_before_draw_vs_crustle")
        if my_active == OGERPON_EX and opp_active == CRUSTLE and chosen_type == ATTACK:
            # Do not feed blank ex attacks into Mysterious Rock Inn while there is
            # still useful setup left. After many actions, let the simulator end
            # rather than create infinite non-attack loops.
            if turn_action_count <= 7:
                pick = _first_non_attack_tempo(obs, options, avoid_cards={OGERPON_EX})
                if pick is not None:
                    return RuleDecision([pick], "counter:ogerpon_no_blank_attack_vs_crustle")

    if plan.archetype == "Cynthia Garchomp" and flags["crustle"]:
        if (
            my_active == CYNTHIA_GARCHOMP_EX
            and CYNTHIA_SPIRITOMB in my_bench
            and chosen_type == ATTACK
        ):
            retreats = _find_type(options, RETREAT)
            if retreats:
                return RuleDecision([retreats[0]], "counter:cynthia_retreat_to_spiritomb_vs_crustle")

    if plan.archetype == "Team Rocket Mewtwo" and turn <= 10 and chosen_type in (ATTACK, END):
        setup = _first_card_by_types(
            obs,
            options,
            TEAM_ROCKET_LINE,
            (PLAY, EVOLVE, ABILITY, ATTACH),
        )
        if setup is not None:
            return RuleDecision([setup], "counter:rocket_board_before_attack")

    if plan.archetype == "Mega Lucario" and turn <= 10 and chosen_type in (ATTACK, END, PLAY, ATTACH):
        setup = _first_card_by_types(
            obs,
            options,
            MEGA_LUCARIO_LINE,
            (EVOLVE, ABILITY, PLAY, ATTACH),
        )
        if setup is not None and _option_type(options[setup]) != chosen_type:
            return RuleDecision([setup], "counter:lucario_engine_before_attack")

    if chosen_type == END:
        for opt_type in (EVOLVE, ABILITY, ATTACH, ATTACK):
            candidates = _find_type(options, opt_type)
            if candidates:
                return RuleDecision([candidates[0]], f"counter:no_early_end:{opt_type}")

    if turn_action_count >= 5 and chosen_type in (END, PLAY, ABILITY):
        attacks = _find_type(options, ATTACK)
        if attacks and not _has_energy_attach_available(options):
            return RuleDecision([attacks[0]], "counter:take_late_attack_window")

    return None


def _counter_guard_overlay(
    *,
    plan: DeckPlan | None,
    obs: dict,
    options: list[dict],
    chosen_type: int,
    chosen_card: int,
    turn_action_count: int,
    my_active: int,
    opp: dict,
) -> RuleDecision | None:
    """High-precision matchup guards.

    Broad replacement rules were tested across top weak archetype pairs and
    hurt 29/36 pairs. Keep this mode narrow: only intercept decisions that are
    structurally invalid or almost always waste a turn.
    """
    if not plan:
        return None
    flags = _visible_matchup_flags(opp)
    opp_active = _active_card_id(opp)

    if plan.archetype == "Teal Mask Ogerpon" and flags["crustle"]:
        if chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attaches = _find_type(options, ATTACH)
            if attaches:
                return RuleDecision([attaches[0]], "guard:ogerpon_attach_before_draw_vs_crustle")
        if my_active == OGERPON_EX and opp_active == CRUSTLE and chosen_type == ATTACK:
            if turn_action_count <= 7:
                pick = _first_non_attack_tempo(obs, options, avoid_cards={OGERPON_EX})
                if pick is not None and _option_type(options[pick]) != END:
                    return RuleDecision([pick], "guard:ogerpon_no_blank_attack_vs_crustle")

    return None


def apply_rule_overlay(obs: dict, action: list[int], deck: list[int] | None = None,
                       *, mode: str = "conservative") -> RuleDecision:
    """Return a guarded action for local experiments.

    This is intentionally conservative and disabled by default in callers. It is
    a scaffold for measuring BC+rules; do not enable it for submission until
    round-robin proves a gain.
    """
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    if not options:
        return RuleDecision(action)
    mn = int(sel.get("minCount", 0) or 0)
    mc = int(sel.get("maxCount", 0) or 0)
    if mn != 1 or mc != 1:
        return RuleDecision(action)
    if not action or action[0] < 0 or action[0] >= len(options):
        action = []

    plan = infer_plan(deck or []) if deck else None
    cur = obs.get("current") or {}
    turn_action_count = int(cur.get("turnActionCount", 0) or 0)
    turn = int(cur.get("turn", 0) or 0)
    players = cur.get("players") or [{}, {}]
    you = int(cur.get("yourIndex", 0) or 0)
    me = players[you] if you < len(players) else {}
    opp = players[1 - you] if 1 - you < len(players) else {}
    my_active = _active_card_id(me)
    opp_active = _active_card_id(opp)
    my_bench = _bench_card_ids(me)
    chosen_type = _option_type(options[action[0]]) if action else END
    chosen_card = _option_card(obs, options[action[0]]) if action else 0
    context = int(sel.get("context", -1) if sel.get("context") is not None else -1)

    if mode == "counter_plan":
        decision = _counter_guard_overlay(
            plan=plan,
            obs=obs,
            options=options,
            chosen_type=chosen_type,
            chosen_card=chosen_card,
            turn_action_count=turn_action_count,
            my_active=my_active,
            opp=opp,
        )
        if decision is not None:
            return decision

    if mode == "counter_plan_aggressive":
        decision = _counter_plan_overlay(
            plan=plan,
            obs=obs,
            action=action,
            options=options,
            chosen_type=chosen_type,
            chosen_card=chosen_card,
            context=context,
            turn=turn,
            turn_action_count=turn_action_count,
            my_active=my_active,
            my_bench=my_bench,
            opp=opp,
        )
        if decision is not None:
            return decision

    if mode == "primary_active" and plan and context in (3, 4):
        primary = set(plan.primary_attackers)
        if primary and chosen_card not in primary:
            primary_pick = _first_card_any_type(obs, options, primary)
            if primary_pick is not None:
                return RuleDecision([primary_pick], "primary_active")

    if mode in ("marnie_setup", "targeted") and plan and plan.archetype == "Marnie Grimmsnarl":
        if opp_active == OGERPON_EX and turn <= 8 and chosen_type in (PLAY, ATTACH, ABILITY):
            # Trace showed Ogerpon losses often delaying the active Marnie line
            # while using utility abilities/bench setup. Keep this narrow and do
            # not override attacks, Rare Candy, or unrelated evolutions.
            delays_evolution = (
                chosen_type in (ATTACH, ABILITY)
                or (chosen_type == PLAY and chosen_card in (MUNKIDORI, SPIKEMUTH_GYM))
            )
            if my_active == MARNIE_IMPIDIMP and delays_evolution:
                morgrem = _first_exact_card(obs, options, EVOLVE, MARNIE_MORGREM)
                if morgrem is not None:
                    return RuleDecision([morgrem], "marnie_setup_morgrem")
            if my_active == MARNIE_MORGREM and delays_evolution:
                grimmsnarl = _first_exact_card(obs, options, EVOLVE, MARNIE_GRIMMSNARL_EX)
                if grimmsnarl is not None:
                    return RuleDecision([grimmsnarl], "marnie_setup_grimmsnarl")

    if mode in ("stage2_setup", "targeted") and plan and plan.evolution_chain:
        if turn <= 10 and chosen_type in (PLAY, ATTACH, ABILITY, END):
            evo = _first_card_tag(plan, obs, options, EVOLVE, set(plan.evolution_chain))
            if evo is not None:
                return RuleDecision([evo], "stage2_setup_evolve")
        if context in (3, 4) and turn <= 4:
            setup = set(plan.setup_basics) | set(plan.primary_attackers)
            if setup and chosen_card not in setup:
                pick = _first_card_any_type(obs, options, setup)
                if pick is not None:
                    return RuleDecision([pick], "stage2_setup_active")

    if mode in ("ogerpon_attach", "targeted") and plan and plan.archetype == "Teal Mask Ogerpon":
        if opp_active in (DWEBBLE, CRUSTLE) and chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attaches = _find_type(options, ATTACH)
            if attaches:
                return RuleDecision([attaches[0]], "ogerpon_attach_before_teal_dance_vs_crustle")

    if mode == "ogerpon_no_futile_crustle" and plan and plan.archetype == "Teal Mask Ogerpon":
        if my_active == OGERPON_EX and opp_active == CRUSTLE and chosen_type == ATTACK:
            # Probe only: an ex attack into Crustle's Mysterious Rock Inn is
            # usually blanked. Prefer any available setup/draw/retreat option
            # before accepting the attack; broad validation decides if this
            # creates harmful loops.
            for opt_type in (PLAY, ABILITY, ATTACH, RETREAT, END):
                picks = _find_type(options, opt_type)
                if picks:
                    return RuleDecision([picks[0]], "ogerpon_avoid_futile_ex_attack_into_crustle")

    if mode == "cynthia_spiritomb_crustle" and plan and plan.archetype == "Cynthia Garchomp":
        if opp_active == CRUSTLE:
            if context in (3, 4) and chosen_card != CYNTHIA_SPIRITOMB:
                spiritomb = _first_card_any_type(obs, options, {CYNTHIA_SPIRITOMB})
                if spiritomb is not None:
                    return RuleDecision([spiritomb], "cynthia_spiritomb_active_vs_crustle")
            if (
                my_active == CYNTHIA_GARCHOMP_EX
                and CYNTHIA_SPIRITOMB in my_bench
                and chosen_type == ATTACK
            ):
                retreats = _find_type(options, RETREAT)
                if retreats:
                    return RuleDecision([retreats[0]], "cynthia_retreat_to_spiritomb_vs_crustle")

    # Never end before taking a clearly available mandatory tempo action.
    if mode in ("conservative", "aggressive") and chosen_type == END:
        for opt_type in (ABILITY, EVOLVE, ATTACH, ATTACK):
            candidates = _find_type(options, opt_type)
            if candidates:
                return RuleDecision([candidates[0]], f"no_early_end:{opt_type}")

    if mode in ("conservative", "aggressive") and plan and plan.archetype == "Teal Mask Ogerpon":
        primary = set(plan.primary_attackers)
        # Teal Dance is central to the deck plan. Prefer it early unless the
        # model is already attacking after several actions.
        ability = _first_card_tag(plan, obs, options, ABILITY, primary)
        if ability is not None and chosen_type not in (ATTACK,):
            return RuleDecision([ability], "ogerpon_teal_dance")
        attacks = _find_type(options, ATTACK)
        if attacks and chosen_type in (PLAY, ABILITY) and turn_action_count >= 4 and not _has_energy_attach_available(options):
            return RuleDecision([attacks[0]], "ogerpon_take_attack_window")

    if mode in ("conservative", "aggressive") and plan and plan.archetype == "Marnie Grimmsnarl":
        evo = _first_card_tag(plan, obs, options, EVOLVE, set(plan.evolution_chain))
        if evo is not None:
            return RuleDecision([evo], "marnie_complete_evolution")
        ability = _first_card_tag(plan, obs, options, ABILITY, set(plan.primary_attackers))
        if ability is not None:
            return RuleDecision([ability], "marnie_punk_up")

    if mode == "aggressive":
        attacks = _find_type(options, ATTACK)
        if attacks and chosen_type in (END, PLAY) and turn_action_count >= 5:
            return RuleDecision([attacks[0]], "aggressive_attack_window")

    return RuleDecision(action)
