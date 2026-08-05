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
    "primary_active",
    "targeted",
)

MARNIE_IMPIDIMP = 646
MARNIE_MORGREM = 647
MARNIE_GRIMMSNARL_EX = 648
MUNKIDORI = 112
SPIKEMUTH_GYM = 1259
OGERPON_EX = 96
DWEBBLE = 344
CRUSTLE = 345


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
    chosen_type = _option_type(options[action[0]]) if action else END
    chosen_card = _option_card(obs, options[action[0]]) if action else 0
    context = int(sel.get("context", -1) if sel.get("context") is not None else -1)

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

    if mode in ("ogerpon_attach", "targeted") and plan and plan.archetype == "Teal Mask Ogerpon":
        if opp_active in (DWEBBLE, CRUSTLE) and chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attaches = _find_type(options, ATTACH)
            if attaches:
                return RuleDecision([attaches[0]], "ogerpon_attach_before_teal_dance_vs_crustle")

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
