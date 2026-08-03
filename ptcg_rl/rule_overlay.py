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


def _has_energy_attach_available(options: list[dict]) -> bool:
    return any(_option_type(opt) == ATTACH for opt in options)


def _first_card_tag(plan: DeckPlan, obs: dict, options: list[dict], opt_type: int,
                    card_ids: set[int]) -> int | None:
    for i, opt in enumerate(options):
        if _option_type(opt) == opt_type and _option_card(obs, opt) in card_ids:
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
    chosen_type = _option_type(options[action[0]]) if action else END

    # Never end before taking a clearly available mandatory tempo action.
    if chosen_type == END:
        for opt_type in (ABILITY, EVOLVE, ATTACH, ATTACK):
            candidates = _find_type(options, opt_type)
            if candidates:
                return RuleDecision([candidates[0]], f"no_early_end:{opt_type}")

    if plan and plan.archetype == "Teal Mask Ogerpon":
        primary = set(plan.primary_attackers)
        # Teal Dance is central to the deck plan. Prefer it early unless the
        # model is already attacking after several actions.
        ability = _first_card_tag(plan, obs, options, ABILITY, primary)
        if ability is not None and chosen_type not in (ATTACK,):
            return RuleDecision([ability], "ogerpon_teal_dance")
        attacks = _find_type(options, ATTACK)
        if attacks and chosen_type in (PLAY, ABILITY) and turn_action_count >= 4 and not _has_energy_attach_available(options):
            return RuleDecision([attacks[0]], "ogerpon_take_attack_window")

    if plan and plan.archetype == "Marnie Grimmsnarl":
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
