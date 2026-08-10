from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ptcg_rl.deck_plans import DeckPlan, infer_plan
from ptcg_rl.rule_overlay import (
    ABILITY,
    ATTACH,
    ATTACK,
    BOSS_ORDERS,
    CRUSTLE,
    CYNTHIA_GARCHOMP_EX,
    DWEBBLE,
    END,
    EVOLVE,
    FIGHTING_GONG,
    JUDGE,
    LUCARIO_LUNATONE,
    LUCARIO_SOLROCK,
    MARNIE_GRIMMSNARL_EX,
    MARNIE_IMPIDIMP,
    MARNIE_MORGREM,
    MUNKIDORI,
    OGERPON_EX,
    PLAY,
    RETREAT,
    SPIKEMUTH_GYM,
    TEAM_ROCKET_LINE,
    TEAM_ROCKET_MEWTWO_EX,
    RuleDecision,
    ULTRA_BALL,
    apply_rule_overlay,
    _active_card_id,
    _bench_card_ids,
    _find_type,
    _first_card_by_types,
    _first_card_any_type,
    _first_exact_card,
    _option_card,
    _option_type,
    _visible_matchup_flags,
)


SECONDARY_OGERPON_ROUTE = {756, 1071, 272}
RAGING_BOLT_EX = 63
CORNERSTONE_OGERPON_EX = 117
WELLSPRING_OGERPON_EX = 108
CORNERSTONE_OGERPON = 386
FEZANDIPITI_EX = 140
LATIAS_EX = 184
PASSIMIAN = 978
MEGA_KANGASKHAN_EX = 756
MEOWTH_EX = 1071
LILLIE_CLEFAIRY_EX = 272
ENERGY_SWITCH = 1116
GLASS_TRUMPET = 1098
TERA_ORB = 1127
CRISPIN = 1198
CYRANO = 1205
AREA_ZERO_UNDERDEPTHS = 1250
PRIME_CATCHER = 1088
FIGHTING_ENERGIES = {6, 16, 20}
CORNERSTONE_ROUTE = {CORNERSTONE_OGERPON_EX, CORNERSTONE_OGERPON, ENERGY_SWITCH, TERA_ORB, ULTRA_BALL, CRISPIN}
WELLSPRING_ROUTE = {WELLSPRING_OGERPON_EX, ULTRA_BALL, CRISPIN}
PASSIMIAN_BOARD_BASICS = {
    OGERPON_EX,
    PASSIMIAN,
    RAGING_BOLT_EX,
    LATIAS_EX,
    MEGA_KANGASKHAN_EX,
    MEOWTH_EX,
    LILLIE_CLEFAIRY_EX,
    FEZANDIPITI_EX,
    WELLSPRING_OGERPON_EX,
    CORNERSTONE_OGERPON_EX,
    CORNERSTONE_OGERPON,
}
PASSIMIAN_ROUTE = {
    PASSIMIAN,
    AREA_ZERO_UNDERDEPTHS,
    ULTRA_BALL,
    CYRANO,
    CRISPIN,
    ENERGY_SWITCH,
    GLASS_TRUMPET,
    PRIME_CATCHER,
    RAGING_BOLT_EX,
    LATIAS_EX,
    MEGA_KANGASKHAN_EX,
} | PASSIMIAN_BOARD_BASICS
OGERPON_CRUSTLE_ROUTE_CARDS = SECONDARY_OGERPON_ROUTE | CORNERSTONE_ROUTE | WELLSPRING_ROUTE | PASSIMIAN_ROUTE | {ULTRA_BALL}
LUCARIO_ENGINE_ROUTE = {LUCARIO_LUNATONE, LUCARIO_SOLROCK, FIGHTING_GONG, ULTRA_BALL}
EX_HEAVY_FLAGS = ("ogerpon", "marnie", "cynthia", "trmewtwo", "lucario", "dragapult", "lopunny")
STAGE_ROUTE_ARCHES = {
    "Alakazam",
    "Cynthia Garchomp",
    "Dragapult",
    "Festival Lead",
    "Mega Lopunny",
    "Mega Starmie",
}


@dataclass
class ResourceSnapshot:
    turn: int = 0
    turn_action_count: int = 0
    my_deck_count: int = 0
    opp_deck_count: int = 0
    opp_hand_count: int = 0
    my_prizes_left: int = 0
    opp_prizes_left: int = 0
    my_active: int = 0
    opp_active: int = 0
    my_board: set[int] = field(default_factory=set)
    opp_board: set[int] = field(default_factory=set)
    my_board_count: int = 0
    opp_board_count: int = 0
    my_bench_count: int = 0
    opp_bench_count: int = 0
    my_discard: set[int] = field(default_factory=set)
    opp_discard: set[int] = field(default_factory=set)
    my_board_energy: dict[int, int] = field(default_factory=dict)
    opp_board_energy: dict[int, int] = field(default_factory=dict)
    my_board_energy_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)
    my_damaged_board: set[int] = field(default_factory=set)
    opp_damaged_board: set[int] = field(default_factory=set)
    flags: dict[str, bool] = field(default_factory=dict)
    known_self: Counter[int] = field(default_factory=Counter)
    estimated_unseen: Counter[int] = field(default_factory=Counter)


def _iter_card_ids(obj) -> list[int]:
    ids: list[int] = []
    if isinstance(obj, dict):
        cid = obj.get("id")
        if cid:
            try:
                ids.append(int(cid))
            except Exception:
                pass
        for key in (
            "cards",
            "hand",
            "active",
            "bench",
            "discard",
            "prize",
            "tools",
            "energyCards",
            "energies",
            "attached",
            "evolutions",
            "evolution",
        ):
            if key in obj:
                ids.extend(_iter_card_ids(obj.get(key)))
    elif isinstance(obj, list):
        for item in obj:
            ids.extend(_iter_card_ids(item))
    return ids


def _zone_ids(player: dict, zone: str) -> set[int]:
    out: set[int] = set()
    for card in player.get(zone) or []:
        if card:
            try:
                out.add(int(card.get("id") or 0))
            except Exception:
                pass
    out.discard(0)
    return out


def _visible_self_counts(cur: dict, me: dict) -> Counter[int]:
    counts: Counter[int] = Counter()
    for zone in ("hand", "active", "bench", "discard"):
        counts.update(_iter_card_ids(me.get(zone) or []))
    counts.update(_iter_card_ids(cur.get("stadium") or []))
    counts.update(_iter_card_ids(cur.get("looking") or []))
    return counts


def _looking_ids(obs: dict) -> set[int]:
    return set(_iter_card_ids((obs.get("current") or {}).get("looking") or []))


def _board_ids(player: dict) -> set[int]:
    return _zone_ids(player, "active") | _zone_ids(player, "bench")


def _board_count(player: dict) -> int:
    return len([p for p in (player.get("active") or []) + (player.get("bench") or []) if p])


def _bench_count(player: dict) -> int:
    return len([p for p in (player.get("bench") or []) if p])


def _energy_count(pokemon: dict | None) -> int:
    if not pokemon:
        return 0
    total = 0
    for key in ("energyCards", "energies"):
        cards = pokemon.get(key) or []
        if isinstance(cards, list):
            total += len([c for c in cards if c])
    return total


def _energy_ids(pokemon: dict | None) -> tuple[int, ...]:
    if not pokemon:
        return ()
    ids: list[int] = []
    for key in ("energyCards", "energies"):
        cards = pokemon.get(key) or []
        if not isinstance(cards, list):
            continue
        for card in cards:
            if isinstance(card, dict):
                cid = int(card.get("id") or 0)
            else:
                try:
                    cid = int(card or 0)
                except Exception:
                    cid = 0
            if cid:
                ids.append(cid)
    return tuple(ids)


def _board_energy_by_card(player: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for p in (player.get("active") or []) + (player.get("bench") or []):
        if not p:
            continue
        cid = int(p.get("id") or 0)
        if cid:
            out[cid] = max(out.get(cid, 0), _energy_count(p))
    return out


def _board_energy_ids_by_card(player: dict) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for p in (player.get("active") or []) + (player.get("bench") or []):
        if not p:
            continue
        cid = int(p.get("id") or 0)
        if cid:
            out[cid] = _energy_ids(p)
    return out


def _pokemon_from_area(player: dict, area, index) -> dict | None:
    try:
        idx = int(index)
    except Exception:
        return None
    if area == 4:
        cards = player.get("active") or []
    elif area == 5:
        cards = player.get("bench") or []
    else:
        return None
    if 0 <= idx < len(cards):
        return cards[idx] or None
    return None


def _option_target_card(obs: dict, opt: dict) -> int:
    cur = obs.get("current") or {}
    players = cur.get("players") or [{}, {}]
    you = int(cur.get("yourIndex", 0) or 0)
    pid = int(opt.get("playerIndex", you) if opt.get("playerIndex") is not None else you)
    if pid < 0 or pid >= len(players):
        pid = you
    target = _pokemon_from_area(players[pid], opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if target is None:
        target = _pokemon_from_area(players[pid], opt.get("area"), opt.get("index"))
    if not target:
        return 0
    return int(target.get("id") or 0)


def _first_target_card_by_types(obs: dict, options: list[dict], card_ids: set[int],
                                opt_types: tuple[int, ...]) -> int | None:
    for opt_type in opt_types:
        for i, opt in enumerate(options):
            if _option_type(opt) == opt_type and _option_target_card(obs, opt) in card_ids:
                return i
    return None


def _first_target_card_any_type(obs: dict, options: list[dict], card_ids: set[int]) -> int | None:
    for i, opt in enumerate(options):
        if _option_target_card(obs, opt) in card_ids:
            return i
    return None


def _first_attack_by_card(obs: dict, options: list[dict], card_id: int) -> int | None:
    for i, opt in enumerate(options):
        if _option_type(opt) == ATTACK and _option_card(obs, opt) == card_id:
            return i
    return None


def _damage_count(pokemon: dict | None) -> int:
    if not pokemon:
        return 0
    for key in (
        "damage",
        "damageCount",
        "damageCounter",
        "damageCounters",
        "damage_counter",
        "damage_counters",
    ):
        value = pokemon.get(key)
        if value is None:
            continue
        try:
            return max(int(value), 0)
        except Exception:
            pass
    return 0


def _damaged_board_ids(player: dict) -> set[int]:
    out: set[int] = set()
    for p in (player.get("active") or []) + (player.get("bench") or []):
        if not p:
            continue
        cid = int(p.get("id") or 0)
        if cid and _damage_count(p) > 0:
            out.add(cid)
    return out


class ResourcePlanner:
    """Stateful explicit planner for resource and route-aware rule overlays.

    The stateless rule overlay can only react to the current option list. This
    planner persists across turns, estimates which key cards are still unseen,
    tracks whether a matchup route has been committed, and limits repeated
    overrides that otherwise create no-progress loops.
    """

    def __init__(self, deck: list[int] | None = None):
        self.deck: list[int] = []
        self.deck_counts: Counter[int] = Counter()
        self.plan: DeckPlan | None = None
        self.reset(deck)

    def reset(self, deck: list[int] | None = None) -> None:
        if deck is not None:
            self.deck = [int(c) for c in deck]
            self.deck_counts = Counter(self.deck)
            self.plan = infer_plan(self.deck)
        self.route = ""
        self.phase = "unknown"
        self.last_turn = -1
        self.turns_on_route = 0
        self.override_counts: Counter[str] = Counter()
        self.route_attempts: Counter[str] = Counter()
        self.last_snapshot = ResourceSnapshot()

    def _snapshot(self, obs: dict) -> ResourceSnapshot:
        cur = obs.get("current") or {}
        players = cur.get("players") or [{}, {}]
        you = int(cur.get("yourIndex", 0) or 0)
        me = players[you] if you < len(players) else {}
        opp = players[1 - you] if 1 - you < len(players) else {}
        known = _visible_self_counts(cur, me)
        estimated = Counter({
            cid: max(n - known.get(cid, 0), 0)
            for cid, n in self.deck_counts.items()
        })
        snap = ResourceSnapshot(
            turn=int(cur.get("turn", 0) or 0),
            turn_action_count=int(cur.get("turnActionCount", 0) or 0),
            my_deck_count=int(me.get("deckCount", 0) or 0),
            opp_deck_count=int(opp.get("deckCount", 0) or 0),
            opp_hand_count=int(opp.get("handCount", 0) or 0),
            my_prizes_left=len(me.get("prize") or []),
            opp_prizes_left=len(opp.get("prize") or []),
            my_active=_active_card_id(me),
            opp_active=_active_card_id(opp),
            my_board=_board_ids(me),
            opp_board=_board_ids(opp),
            my_board_count=_board_count(me),
            opp_board_count=_board_count(opp),
            my_bench_count=_bench_count(me),
            opp_bench_count=_bench_count(opp),
            my_discard=_zone_ids(me, "discard"),
            opp_discard=_zone_ids(opp, "discard"),
            my_board_energy=_board_energy_by_card(me),
            opp_board_energy=_board_energy_by_card(opp),
            my_board_energy_ids=_board_energy_ids_by_card(me),
            my_damaged_board=_damaged_board_ids(me),
            opp_damaged_board=_damaged_board_ids(opp),
            flags=_visible_matchup_flags(opp),
            known_self=known,
            estimated_unseen=estimated,
        )
        self.last_snapshot = snap
        if snap.turn != self.last_turn:
            self.last_turn = snap.turn
            self.turns_on_route += 1 if self.route else 0
        return snap

    def _remaining_any(self, snap: ResourceSnapshot, cards: set[int]) -> int:
        return sum(max(int(snap.estimated_unseen.get(cid, 0)), 0) for cid in cards)

    def _commit_route(self, route: str) -> None:
        if self.route != route:
            self.route = route
            self.phase = "enter"
            self.turns_on_route = 0

    def _update_route(self, snap: ResourceSnapshot) -> None:
        if not self.plan:
            self.route = ""
            self.phase = "unknown"
            return
        if self.plan.archetype == "Teal Mask Ogerpon" and snap.flags.get("crustle"):
            self._commit_route("ogerpon_secondary_vs_crustle")
            if snap.my_board & SECONDARY_OGERPON_ROUTE:
                self.phase = "secondary_online"
            elif self._remaining_any(snap, SECONDARY_OGERPON_ROUTE) > 0:
                self.phase = "find_secondary"
            else:
                self.phase = "disrupt_fallback"
            return
        if self.plan.archetype == "Marnie Grimmsnarl" and snap.flags.get("ogerpon"):
            self._commit_route("marnie_race_ogerpon")
            if MARNIE_GRIMMSNARL_EX in snap.my_board:
                self.phase = "spread_pressure"
            elif MARNIE_MORGREM in snap.my_board or MARNIE_IMPIDIMP in snap.my_board:
                self.phase = "complete_line"
            else:
                self.phase = "find_line"
            return
        if self.plan.archetype == "Mega Lucario" and (
            snap.flags.get("marnie") or snap.flags.get("crustle") or snap.flags.get("ogerpon")
        ):
            self._commit_route("lucario_engine_resource")
            if {LUCARIO_LUNATONE, LUCARIO_SOLROCK}.issubset(snap.my_board):
                self.phase = "engine_online"
            elif self._remaining_any(snap, {LUCARIO_LUNATONE, LUCARIO_SOLROCK}) > 0:
                self.phase = "assemble_engine"
            else:
                self.phase = "payoff_fallback"
            return
        if self.plan.archetype == "Crustle Wall" and any(snap.flags.get(k) for k in EX_HEAVY_FLAGS):
            self._commit_route("crustle_wall_vs_ex")
            if CRUSTLE in snap.my_board:
                self.phase = "wall_online"
            elif DWEBBLE in snap.my_board:
                self.phase = "evolve_wall"
            else:
                self.phase = "find_dwebble"
            return
        if self.plan.archetype == "Team Rocket Mewtwo":
            self._commit_route("rocket_board_count")
            if TEAM_ROCKET_MEWTWO_EX in snap.my_board and len(snap.my_board & set(TEAM_ROCKET_LINE)) >= 3:
                self.phase = "mewtwo_ready"
            else:
                self.phase = "build_rocket_board"
            return
        if self.plan.archetype in STAGE_ROUTE_ARCHES:
            self._commit_route("stage_route")
            primary = set(self.plan.primary_attackers)
            setup = set(self.plan.setup_basics)
            if primary and snap.my_board & primary:
                self.phase = "primary_online"
            elif snap.my_board & (set(self.plan.evolution_chain) - setup):
                self.phase = "complete_line"
            else:
                self.phase = "find_basic"
            return
        if self.route:
            self.phase = "inactive"

    def _take_once_or_limited(self, reason: str, pick: int, limit: int) -> RuleDecision | None:
        if self.override_counts[reason] >= limit:
            return None
        self.override_counts[reason] += 1
        return RuleDecision([pick], f"resource:{reason}:route={self.route}:phase={self.phase}")

    def decide(self, obs: dict, action: list[int], deck: list[int] | None = None) -> RuleDecision:
        if deck is not None and Counter(int(c) for c in deck) != self.deck_counts:
            self.reset(deck)
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

        snap = self._snapshot(obs)
        self._update_route(snap)
        chosen_type = _option_type(options[action[0]]) if action else END
        chosen_card = _option_card(obs, options[action[0]]) if action else 0

        if self.route == "ogerpon_secondary_vs_crustle":
            decision = self._decide_ogerpon_vs_crustle(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        if self.route == "marnie_race_ogerpon":
            decision = self._decide_marnie_vs_ogerpon(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        if self.route == "lucario_engine_resource":
            decision = self._decide_lucario_engine(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        if self.route == "crustle_wall_vs_ex":
            decision = self._decide_crustle_wall(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        if self.route == "rocket_board_count":
            decision = self._decide_rocket_board(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        if self.route == "stage_route":
            decision = self._decide_stage_route(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        return RuleDecision(action)

    def _decide_ogerpon_vs_crustle(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        route_pick = _first_card_by_types(obs, options, OGERPON_CRUSTLE_ROUTE_CARDS, (PLAY, ABILITY))
        disrupt_pick = _first_card_by_types(obs, options, {BOSS_ORDERS, JUDGE}, (PLAY,))
        route_available = self._remaining_any(snap, SECONDARY_OGERPON_ROUTE) > 0
        secondary_online = bool(snap.my_board & SECONDARY_OGERPON_ROUTE)

        if self.phase == "find_secondary" and route_available and not secondary_online:
            if route_pick is not None and chosen_card not in OGERPON_CRUSTLE_ROUTE_CARDS:
                if chosen_type in (PLAY, ABILITY, ATTACH, END):
                    return self._take_once_or_limited("ogerpon_find_secondary_vs_crustle", route_pick, 6)

        if chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attaches = _find_type(options, ATTACH)
            if attaches and snap.my_deck_count > 4:
                return self._take_once_or_limited("ogerpon_attach_before_draw", attaches[0], 5)

        if self.phase == "disrupt_fallback":
            boss = _first_exact_card(obs, options, PLAY, BOSS_ORDERS)
            if boss is not None and snap.opp_active == CRUSTLE and len(snap.opp_board - {CRUSTLE, DWEBBLE}) > 0:
                if chosen_type in (ABILITY, END, 13):
                    return self._take_once_or_limited("ogerpon_boss_around_crustle", boss, 2)
            judge = _first_exact_card(obs, options, PLAY, JUDGE)
            if judge is not None and snap.opp_hand_count >= 5 and chosen_type in (ABILITY, END):
                return self._take_once_or_limited("ogerpon_judge_large_hand", judge, 2)

        if snap.my_active == OGERPON_EX and snap.opp_active == CRUSTLE and chosen_type == 13:
            retreats = _find_type(options, RETREAT)
            if retreats and secondary_online:
                return self._take_once_or_limited("ogerpon_pivot_to_secondary", retreats[0], 3)
            if route_pick is not None and route_available and self.override_counts["ogerpon_delay_blank_attack"] < 4:
                return self._take_once_or_limited("ogerpon_delay_blank_attack", route_pick, 4)
            if disrupt_pick is not None and self.override_counts["ogerpon_disrupt_before_blank_attack"] < 2:
                return self._take_once_or_limited("ogerpon_disrupt_before_blank_attack", disrupt_pick, 2)
        return None

    def _decide_marnie_vs_ogerpon(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if self.phase == "find_line" and self._remaining_any(snap, {MARNIE_IMPIDIMP}) > 0:
            impidimp = _first_exact_card(obs, options, PLAY, MARNIE_IMPIDIMP)
            if impidimp is not None and chosen_card != MARNIE_IMPIDIMP:
                return self._take_once_or_limited("marnie_find_impidimp", impidimp, 3)
        if self.phase == "complete_line" and chosen_type in (PLAY, ATTACH, ABILITY, END):
            if snap.my_active == MARNIE_IMPIDIMP:
                morgrem = _first_exact_card(obs, options, EVOLVE, MARNIE_MORGREM)
                if morgrem is not None:
                    return self._take_once_or_limited("marnie_evolve_morgrem", morgrem, 2)
            if snap.my_active == MARNIE_MORGREM:
                grimmsnarl = _first_exact_card(obs, options, EVOLVE, MARNIE_GRIMMSNARL_EX)
                if grimmsnarl is not None:
                    return self._take_once_or_limited("marnie_evolve_grimmsnarl", grimmsnarl, 2)
        if self.phase == "spread_pressure":
            punk = _first_exact_card(obs, options, ABILITY, MARNIE_GRIMMSNARL_EX)
            if punk is not None and chosen_type not in (13,):
                return self._take_once_or_limited("marnie_punk_up_once_online", punk, 4)
        return None

    def _decide_lucario_engine(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if self.phase == "assemble_engine":
            if LUCARIO_LUNATONE not in snap.my_board and self._remaining_any(snap, {LUCARIO_LUNATONE}) > 0:
                lunatone = _first_exact_card(obs, options, PLAY, LUCARIO_LUNATONE)
                if lunatone is not None and chosen_card != LUCARIO_LUNATONE:
                    return self._take_once_or_limited("lucario_find_lunatone", lunatone, 2)
            if LUCARIO_SOLROCK not in snap.my_board and self._remaining_any(snap, {LUCARIO_SOLROCK}) > 0:
                solrock = _first_exact_card(obs, options, PLAY, LUCARIO_SOLROCK)
                if solrock is not None and chosen_card != LUCARIO_SOLROCK:
                    return self._take_once_or_limited("lucario_find_solrock", solrock, 2)
            search = _first_card_by_types(obs, options, LUCARIO_ENGINE_ROUTE, (PLAY,))
            if search is not None and chosen_type in (END, 13):
                return self._take_once_or_limited("lucario_search_engine", search, 3)
        if self.phase == "engine_online":
            lunar = _first_exact_card(obs, options, ABILITY, LUCARIO_LUNATONE)
            if lunar is not None and chosen_type in (PLAY, ATTACH, END, 13):
                return self._take_once_or_limited("lucario_lunar_cycle", lunar, 3)
        return None

    def _decide_crustle_wall(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if self.phase == "find_dwebble":
            dwebble = _first_exact_card(obs, options, PLAY, DWEBBLE)
            if dwebble is not None and chosen_card != DWEBBLE:
                return self._take_once_or_limited("crustle_find_dwebble", dwebble, 4)
        if self.phase in ("find_dwebble", "evolve_wall"):
            crustle = _first_exact_card(obs, options, EVOLVE, CRUSTLE)
            if crustle is not None and chosen_card != CRUSTLE:
                return self._take_once_or_limited("crustle_evolve_wall", crustle, 4)
            if snap.my_active == DWEBBLE and CRUSTLE not in snap.my_board:
                dwebble_attacks = [
                    i for i, opt in enumerate(options)
                    if _option_type(opt) == ATTACK and _option_card(obs, opt) == DWEBBLE
                ]
                if dwebble_attacks and chosen_type in (PLAY, ABILITY, ATTACH, END):
                    return self._take_once_or_limited("crustle_ascension_window", dwebble_attacks[0], 2)
        if self.phase == "wall_online":
            if snap.my_active == CRUSTLE and chosen_type == RETREAT:
                attacks = [
                    i for i, opt in enumerate(options)
                    if _option_type(opt) == ATTACK and _option_card(obs, opt) == CRUSTLE
                ]
                if attacks:
                    return self._take_once_or_limited("crustle_keep_wall_active", attacks[0], 3)
            if snap.my_active == CRUSTLE and chosen_type == END:
                for opt_type in (ATTACK, ATTACH, ABILITY):
                    picks = _find_type(options, opt_type)
                    if picks:
                        return self._take_once_or_limited(f"crustle_no_idle_wall:{opt_type}", picks[0], 4)
        return None

    def _decide_rocket_board(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if self.phase == "build_rocket_board":
            setup = _first_card_by_types(obs, options, set(TEAM_ROCKET_LINE), (PLAY, EVOLVE, ABILITY, ATTACH))
            if setup is not None and chosen_card not in TEAM_ROCKET_LINE:
                if chosen_type in (ATTACK, END, PLAY, ATTACH, ABILITY):
                    return self._take_once_or_limited("rocket_build_board_before_payoff", setup, 8)
            if chosen_type == END:
                for opt_type in (EVOLVE, ABILITY, ATTACH):
                    picks = _find_type(options, opt_type)
                    if picks:
                        return self._take_once_or_limited(f"rocket_no_idle_setup:{opt_type}", picks[0], 4)
        return None

    def _decide_stage_route(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if not self.plan:
            return None
        setup = set(self.plan.setup_basics)
        evolution = set(self.plan.evolution_chain)
        engine = set(self.plan.engine_cards) | set(self.plan.draw_search)
        if self.phase == "find_basic" and setup:
            missing_setup = {cid for cid in setup if cid not in snap.my_board}
            pick = _first_card_by_types(obs, options, missing_setup or setup, (PLAY,))
            if pick is not None and chosen_card not in setup:
                return self._take_once_or_limited("stage_find_basic", pick, 4)
        if self.phase in ("find_basic", "complete_line") and evolution:
            evo = _first_card_by_types(obs, options, evolution, (EVOLVE,))
            if evo is not None and chosen_type in (PLAY, ATTACH, ABILITY, END):
                return self._take_once_or_limited("stage_complete_core_evolution", evo, 8)
        if engine:
            eng = _first_card_by_types(obs, options, engine, (ABILITY, PLAY))
            if eng is not None and chosen_type in (END, ATTACK):
                return self._take_once_or_limited("stage_use_engine_before_commit", eng, 5)
        if chosen_type == END and snap.turn_action_count <= 6:
            for opt_type in (EVOLVE, ABILITY, ATTACH):
                picks = _find_type(options, opt_type)
                if picks:
                    return self._take_once_or_limited(f"stage_no_idle_setup:{opt_type}", picks[0], 4)
        return None


class OpportunityPlanner(ResourcePlanner):
    """Stateful matchup-opportunity planner.

    ``ResourcePlanner`` commits to a broad matchup route as soon as the opponent
    archetype is visible. That helped expose bad BC habits, but broad routes
    often override too many ordinary decisions. This planner only switches when
    a concrete public-information window is visible, then keeps a short TTL so
    the policy can execute a small sequence instead of one isolated action.
    """

    def reset(self, deck: list[int] | None = None) -> None:
        super().reset(deck)
        self.active_window = ""
        self.window_ttl = 0
        self.window_last_turn = -1
        self.window_counts: Counter[str] = Counter()

    def _tick_window(self, snap: ResourceSnapshot) -> None:
        if snap.turn != self.window_last_turn:
            if self.window_last_turn >= 0 and self.window_ttl > 0:
                self.window_ttl -= 1
            self.window_last_turn = snap.turn
        if self.window_ttl <= 0:
            self.active_window = ""

    def _activate_window(self, name: str, ttl: int) -> None:
        if self.active_window != name:
            self.active_window = name
            self.window_ttl = ttl
            return
        self.window_ttl = max(self.window_ttl, ttl)

    def _take_window_limited(self, reason: str, pick: int, limit: int) -> RuleDecision | None:
        if self.window_counts[reason] >= limit:
            return None
        self.window_counts[reason] += 1
        window = self.active_window or "instant"
        return RuleDecision([pick], f"opportunity:{reason}:window={window}:ttl={self.window_ttl}")

    def _detect_windows(self, snap: ResourceSnapshot) -> None:
        if not self.plan:
            return
        arch = self.plan.archetype
        if arch == "Teal Mask Ogerpon" and snap.flags.get("crustle"):
            if self.deck_counts.get(PASSIMIAN, 0) > 0 and (CRUSTLE in snap.opp_board or DWEBBLE in snap.opp_board):
                self._activate_window("ogerpon_passimian_break_wall", 6)
                return
            if self.deck_counts.get(CORNERSTONE_OGERPON_EX, 0) > 0 and CRUSTLE in snap.opp_board:
                self._activate_window("ogerpon_cornerstone_break_wall", 5)
                return
            if DWEBBLE in snap.opp_board and CRUSTLE not in snap.opp_board:
                self._activate_window("ogerpon_dwebble_punish", 3)
                return
            if snap.my_board & SECONDARY_OGERPON_ROUTE and snap.opp_active == CRUSTLE:
                self._activate_window("ogerpon_secondary_escape", 2)
                return
            if snap.opp_active == CRUSTLE:
                self._activate_window("ogerpon_wall_disrupt", 2)
                return

        if arch == "Marnie Grimmsnarl" and snap.flags.get("ogerpon"):
            if MARNIE_GRIMMSNARL_EX in snap.my_board:
                self._activate_window("marnie_spread_convert", 3)
                return
            if snap.my_board & {MARNIE_IMPIDIMP, MARNIE_MORGREM} or MARNIE_IMPIDIMP not in snap.my_discard:
                self._activate_window("marnie_setup_race", 3)
                return

        if arch == "Crustle Wall" and any(snap.flags.get(k) for k in EX_HEAVY_FLAGS):
            if CRUSTLE in snap.my_board:
                self._activate_window("crustle_wall_pressure", 3)
                return
            if DWEBBLE in snap.my_board:
                self._activate_window("crustle_ascension_window", 3)
                return

        if arch == "Cynthia Garchomp" and snap.flags.get("crustle"):
            if 387 in snap.my_board or snap.my_active == CYNTHIA_GARCHOMP_EX:
                self._activate_window("cynthia_spiritomb_counter", 2)
                return

        if arch == "Mega Lucario" and (
            snap.flags.get("marnie") or snap.flags.get("crustle") or snap.flags.get("ogerpon")
        ):
            if not {LUCARIO_LUNATONE, LUCARIO_SOLROCK}.issubset(snap.my_board):
                self._activate_window("lucario_engine_gap", 3)
                return

    def decide(self, obs: dict, action: list[int], deck: list[int] | None = None) -> RuleDecision:
        if deck is not None and Counter(int(c) for c in deck) != self.deck_counts:
            self.reset(deck)
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

        snap = self._snapshot(obs)
        self._tick_window(snap)
        self._detect_windows(snap)
        chosen_type = _option_type(options[action[0]]) if action else END
        chosen_card = _option_card(obs, options[action[0]]) if action else 0

        if self.active_window == "ogerpon_passimian_break_wall":
            decision = self._opp_ogerpon_passimian_break_wall(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "ogerpon_cornerstone_break_wall":
            decision = self._opp_ogerpon_cornerstone_break_wall(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "ogerpon_dwebble_punish":
            decision = self._opp_ogerpon_dwebble_punish(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window in ("ogerpon_secondary_escape", "ogerpon_wall_disrupt"):
            decision = self._opp_ogerpon_wall_window(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "marnie_setup_race":
            decision = self._opp_marnie_setup_race(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "marnie_spread_convert":
            decision = self._opp_marnie_spread_convert(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window in ("crustle_ascension_window", "crustle_wall_pressure"):
            decision = self._opp_crustle_window(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "cynthia_spiritomb_counter":
            decision = self._opp_cynthia_spiritomb(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision
        if self.active_window == "lucario_engine_gap":
            decision = self._opp_lucario_engine(obs, options, chosen_type, chosen_card, snap)
            if decision is not None:
                return decision

        return RuleDecision(action)

    def _opp_ogerpon_passimian_break_wall(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        passimian_online = PASSIMIAN in snap.my_board
        passimian_energy = snap.my_board_energy.get(PASSIMIAN, 0)
        passimian_energy_ids = set(snap.my_board_energy_ids.get(PASSIMIAN, ()))
        passimian_has_fighting = bool(passimian_energy_ids & FIGHTING_ENERGIES)
        early = snap.turn <= 6
        looking = _looking_ids(obs)

        if looking:
            if not passimian_online and PASSIMIAN in looking:
                pick = _first_card_any_type(obs, options, {PASSIMIAN})
                if pick is not None:
                    return self._take_window_limited("ogerpon_choose_passimian_from_search", pick, 3)
            if snap.my_board_count < 6:
                pick = _first_card_any_type(obs, options, PASSIMIAN_BOARD_BASICS & looking)
                if pick is not None:
                    return self._take_window_limited("ogerpon_choose_board_basic_from_search", pick, 5)
            if passimian_online and passimian_energy < 2:
                pick = _first_card_any_type(obs, options, FIGHTING_ENERGIES & looking)
                if pick is not None:
                    return self._take_window_limited("ogerpon_choose_fighting_from_search", pick, 2)

        play_area_zero = _first_exact_card(obs, options, PLAY, AREA_ZERO_UNDERDEPTHS)
        if play_area_zero is not None and snap.my_board_count >= 3 and chosen_card != AREA_ZERO_UNDERDEPTHS:
            if chosen_type in (END, ATTACK):
                return self._take_window_limited("ogerpon_passimian_play_area_zero", play_area_zero, 2)

        if not passimian_online:
            play_passimian = _first_exact_card(obs, options, PLAY, PASSIMIAN)
            if play_passimian is not None and chosen_card != PASSIMIAN:
                if chosen_type in (PLAY, ATTACH, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_play_passimian_vs_crustle", play_passimian, 4)
            search = _first_exact_card(obs, options, PLAY, ULTRA_BALL)
            if search is not None and chosen_card != ULTRA_BALL:
                if chosen_type in (END, ATTACK, ABILITY):
                    return self._take_window_limited("ogerpon_search_passimian_vs_crustle", search, 8)

        if snap.my_board_count < 7:
            filler = _first_card_by_types(obs, options, PASSIMIAN_BOARD_BASICS, (PLAY,))
            if filler is not None and chosen_card not in PASSIMIAN_BOARD_BASICS:
                if chosen_type in (END, ATTACK):
                    return self._take_window_limited("ogerpon_fill_basic_board_for_passimian", filler, 5)

        if passimian_online:
            target_passimian_attach = _first_target_card_by_types(obs, options, {PASSIMIAN}, (ATTACH,))
            if target_passimian_attach is not None:
                attach_card = _option_card(obs, options[target_passimian_attach])
                needs_fighting = not passimian_has_fighting and attach_card in FIGHTING_ENERGIES
                if needs_fighting or passimian_energy < 2:
                    return self._take_window_limited("ogerpon_attach_to_passimian", target_passimian_attach, 8)

            if passimian_energy < 2:
                fighting_attach = _first_card_by_types(obs, options, FIGHTING_ENERGIES, (ATTACH,))
                if fighting_attach is not None and chosen_type in (END, ATTACK, ABILITY, PLAY, ATTACH):
                    return self._take_window_limited("ogerpon_select_fighting_for_passimian", fighting_attach, 5)
                any_attach = _find_type(options, ATTACH)
                if any_attach and chosen_type in (END, ATTACK, ABILITY, PLAY):
                    return self._take_window_limited("ogerpon_select_energy_for_passimian", any_attach[0], 8)
                accel = _first_card_by_types(obs, options, {CRISPIN, ENERGY_SWITCH, GLASS_TRUMPET}, (PLAY,))
                if accel is not None and chosen_card not in {CRISPIN, ENERGY_SWITCH, GLASS_TRUMPET}:
                    if chosen_type in (END, ATTACK, ABILITY, PLAY):
                        return self._take_window_limited("ogerpon_accelerate_passimian", accel, 6)

            if snap.opp_active == DWEBBLE and snap.my_active == PASSIMIAN:
                attack = _first_attack_by_card(obs, options, PASSIMIAN)
                if attack is None:
                    attacks = _find_type(options, ATTACK)
                    attack = attacks[0] if attacks else None
                if attack is not None and chosen_type in (END, PLAY, ABILITY, ATTACH):
                    return self._take_window_limited("ogerpon_passimian_attack_dwebble_before_wall", attack, 4)

            target_passimian = _first_target_card_any_type(obs, options, {PASSIMIAN})
            if target_passimian is not None and chosen_type in (END, ATTACK, RETREAT, PLAY, ABILITY, ATTACH):
                if passimian_energy < 2 or snap.my_active != PASSIMIAN:
                    return self._take_window_limited("ogerpon_choose_passimian_target_prompt", target_passimian, 8)

            if snap.my_active != PASSIMIAN and passimian_energy >= 2:
                retreats = _find_type(options, RETREAT)
                if retreats and chosen_type in (END, ATTACK, ABILITY, PLAY):
                    return self._take_window_limited("ogerpon_pivot_to_passimian_wall_breaker", retreats[0], 4)
                switch = _first_exact_card(obs, options, PLAY, PRIME_CATCHER)
                if switch is not None and chosen_type in (END, ATTACK, ABILITY):
                    return self._take_window_limited("ogerpon_prime_catcher_to_unlock_passimian", switch, 2)

            if snap.my_active == PASSIMIAN:
                attack = _first_attack_by_card(obs, options, PASSIMIAN)
                if attack is None:
                    attacks = _find_type(options, ATTACK)
                    attack = attacks[0] if attacks else None
                if attack is not None and chosen_type in (END, PLAY, ABILITY, ATTACH, RETREAT):
                    return self._take_window_limited("ogerpon_passimian_coordinated_throwing", attack, 10)

        if early:
            kang = _first_exact_card(obs, options, ABILITY, MEGA_KANGASKHAN_EX)
            if kang is not None and chosen_type in (END, ATTACK):
                return self._take_window_limited("ogerpon_early_kangaskhan_engine_for_passimian", kang, 2)
            bolt = _first_exact_card(obs, options, PLAY, RAGING_BOLT_EX)
            if bolt is not None and snap.my_board_count < 6 and chosen_card != RAGING_BOLT_EX:
                if chosen_type in (END, ATTACK, ATTACH, ABILITY):
                    return self._take_window_limited("ogerpon_play_raging_bolt_for_wall_route", bolt, 2)

        if snap.my_active == OGERPON_EX and snap.opp_active == CRUSTLE and chosen_type == ATTACK:
            route_pick = _first_card_by_types(obs, options, PASSIMIAN_ROUTE, (PLAY, ABILITY, ATTACH))
            if route_pick is not None:
                return self._take_window_limited("ogerpon_skip_blank_teal_attack_for_passimian_route", route_pick, 8)
            retreats = _find_type(options, RETREAT)
            if retreats and passimian_online:
                return self._take_window_limited("ogerpon_retreat_blank_teal_into_passimian_plan", retreats[0], 4)

        if snap.opp_active == DWEBBLE and chosen_type == END:
            attacks = _find_type(options, ATTACK)
            if attacks:
                return self._take_window_limited("ogerpon_attack_dwebble_inside_passimian_route", attacks[0], 4)
            attach = _find_type(options, ATTACH)
            if attach:
                return self._take_window_limited("ogerpon_attach_before_dwebble_attack_inside_passimian_route", attach[0], 4)

        if chosen_type == END:
            route_pick = _first_card_by_types(obs, options, {PASSIMIAN, AREA_ZERO_UNDERDEPTHS}, (PLAY,))
            if route_pick is None and passimian_online:
                route_pick = _first_card_by_types(obs, options, {PASSIMIAN}, (ATTACK, ATTACH))
            if route_pick is not None:
                return self._take_window_limited("ogerpon_no_idle_in_passimian_wall_route", route_pick, 4)

        return None

    def _opp_ogerpon_cornerstone_break_wall(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        cornerstone_online = CORNERSTONE_OGERPON_EX in snap.my_board
        cornerstone_energy = snap.my_board_energy.get(CORNERSTONE_OGERPON_EX, 0)
        cornerstone_energy_ids = set(snap.my_board_energy_ids.get(CORNERSTONE_OGERPON_EX, ()))
        cornerstone_has_fighting = bool(cornerstone_energy_ids & FIGHTING_ENERGIES)

        if not cornerstone_online:
            play_cornerstone = _first_exact_card(obs, options, PLAY, CORNERSTONE_OGERPON_EX)
            if play_cornerstone is not None and chosen_card != CORNERSTONE_OGERPON_EX:
                if chosen_type in (PLAY, ATTACH, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_play_cornerstone_vs_crustle", play_cornerstone, 3)
            search = _first_card_by_types(obs, options, {TERA_ORB, ULTRA_BALL}, (PLAY,))
            if search is not None and chosen_card not in {TERA_ORB, ULTRA_BALL}:
                if chosen_type in (PLAY, ATTACH, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_search_cornerstone_vs_crustle", search, 5)
            # If no Cornerstone is currently reachable, do not burn the turn on
            # blank Teal Mask attacks into an established wall.
            if snap.my_active == OGERPON_EX and snap.opp_active == CRUSTLE and chosen_type == ATTACK:
                boss = _first_exact_card(obs, options, PLAY, BOSS_ORDERS)
                if boss is not None and len(snap.opp_board - {CRUSTLE, DWEBBLE}) > 0:
                    return self._take_window_limited("ogerpon_boss_while_finding_cornerstone", boss, 2)
                judge = _first_exact_card(obs, options, PLAY, JUDGE)
                if judge is not None and snap.opp_hand_count >= 5:
                    return self._take_window_limited("ogerpon_judge_while_finding_cornerstone", judge, 2)
            return None

        attach_to_cornerstone = _first_target_card_by_types(
            obs, options, {CORNERSTONE_OGERPON_EX}, (ATTACH,)
        )
        if attach_to_cornerstone is not None:
            attach_card = _option_card(obs, options[attach_to_cornerstone])
            need_fighting = not cornerstone_has_fighting and attach_card in FIGHTING_ENERGIES
            need_energy = cornerstone_energy < 3
            if need_fighting or need_energy:
                target_card = _option_target_card(obs, options[attach_to_cornerstone])
                if chosen_type in (PLAY, ABILITY, END, ATTACK, ATTACH) and target_card == CORNERSTONE_OGERPON_EX:
                    return self._take_window_limited("ogerpon_attach_to_cornerstone", attach_to_cornerstone, 5)

        if cornerstone_energy < 3:
            energy_switch = _first_exact_card(obs, options, PLAY, ENERGY_SWITCH)
            if energy_switch is not None and chosen_card != ENERGY_SWITCH:
                if chosen_type in (PLAY, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_energy_switch_to_cornerstone_route", energy_switch, 4)
            crispin = _first_exact_card(obs, options, PLAY, CRISPIN)
            if crispin is not None and chosen_card != CRISPIN:
                if chosen_type in (PLAY, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_crispin_cornerstone_route", crispin, 3)

        target_cornerstone = _first_target_card_any_type(obs, options, {CORNERSTONE_OGERPON_EX})
        if target_cornerstone is not None and chosen_type in (END, ATTACK, RETREAT, PLAY, ABILITY, ATTACH):
            return self._take_window_limited("ogerpon_choose_cornerstone_target_prompt", target_cornerstone, 6)

        if snap.my_active != CORNERSTONE_OGERPON_EX and cornerstone_online and cornerstone_energy >= 3:
            retreats = _find_type(options, RETREAT)
            if retreats and chosen_type in (ATTACK, END, ABILITY, PLAY):
                return self._take_window_limited("ogerpon_pivot_to_ready_cornerstone", retreats[0], 3)
            prime = _first_exact_card(obs, options, PLAY, PRIME_CATCHER)
            if prime is not None and chosen_type in (ATTACK, END, ABILITY):
                return self._take_window_limited("ogerpon_prime_catcher_cornerstone_pivot", prime, 2)

        if snap.my_active == CORNERSTONE_OGERPON_EX:
            attack = _first_attack_by_card(obs, options, CORNERSTONE_OGERPON_EX)
            if attack is None:
                attacks = _find_type(options, ATTACK)
                attack = attacks[0] if attacks else None
            if attack is not None and chosen_type in (END, PLAY, ABILITY, ATTACH):
                return self._take_window_limited("ogerpon_demolish_crustle", attack, 6)

        if snap.my_active == OGERPON_EX and snap.opp_active == CRUSTLE and chosen_type == ATTACK:
            pick = _first_target_card_any_type(obs, options, {CORNERSTONE_OGERPON_EX})
            if pick is not None:
                return self._take_window_limited("ogerpon_take_cornerstone_step_over_blank_attack", pick, 4)
        return None

    def _opp_ogerpon_dwebble_punish(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if snap.opp_active != DWEBBLE and DWEBBLE in snap.opp_board:
            boss = _first_exact_card(obs, options, PLAY, BOSS_ORDERS)
            if boss is not None and chosen_card != BOSS_ORDERS:
                if chosen_type in (PLAY, ATTACH, ABILITY, END, ATTACK):
                    return self._take_window_limited("ogerpon_boss_dwebble_before_wall", boss, 2)

        if snap.opp_active == DWEBBLE:
            attacks = _find_type(options, ATTACK)
            if attacks and chosen_type in (END, PLAY, ABILITY):
                return self._take_window_limited("ogerpon_attack_dwebble_window", attacks[0], 3)
            attach = _find_type(options, ATTACH)
            if attach and chosen_type in (ABILITY, END):
                return self._take_window_limited("ogerpon_attach_for_dwebble_window", attach[0], 3)

        if chosen_type == ABILITY and chosen_card == OGERPON_EX:
            attach = _find_type(options, ATTACH)
            if attach:
                return self._take_window_limited("ogerpon_attach_before_draw_in_punish_window", attach[0], 3)

        return None

    def _opp_ogerpon_wall_window(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        secondary_online = bool(snap.my_board & SECONDARY_OGERPON_ROUTE)
        if snap.my_active == OGERPON_EX and snap.opp_active == CRUSTLE and chosen_type == ATTACK:
            retreats = _find_type(options, RETREAT)
            if retreats and secondary_online:
                return self._take_window_limited("ogerpon_pivot_to_secondary_when_wall_online", retreats[0], 2)
            boss = _first_exact_card(obs, options, PLAY, BOSS_ORDERS)
            if boss is not None and len(snap.opp_board - {CRUSTLE, DWEBBLE}) > 0:
                return self._take_window_limited("ogerpon_boss_around_established_wall", boss, 2)
            judge = _first_exact_card(obs, options, PLAY, JUDGE)
            if judge is not None and snap.opp_hand_count >= 5:
                return self._take_window_limited("ogerpon_judge_wall_large_hand", judge, 2)

        if not secondary_online and self._remaining_any(snap, SECONDARY_OGERPON_ROUTE) > 0:
            route_pick = _first_card_by_types(obs, options, OGERPON_CRUSTLE_ROUTE_CARDS, (PLAY, ABILITY))
            if route_pick is not None and chosen_card not in OGERPON_CRUSTLE_ROUTE_CARDS:
                if chosen_type in (END, ABILITY, PLAY):
                    return self._take_window_limited("ogerpon_build_secondary_after_wall_seen", route_pick, 4)
        return None

    def _opp_marnie_setup_race(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if MARNIE_IMPIDIMP not in snap.my_board:
            impidimp = _first_exact_card(obs, options, PLAY, MARNIE_IMPIDIMP)
            if impidimp is not None and chosen_card != MARNIE_IMPIDIMP:
                return self._take_window_limited("marnie_find_impidimp_in_race", impidimp, 3)
        if snap.my_active == MARNIE_IMPIDIMP and chosen_type in (PLAY, ATTACH, ABILITY, END):
            morgrem = _first_exact_card(obs, options, EVOLVE, MARNIE_MORGREM)
            if morgrem is not None:
                return self._take_window_limited("marnie_evolve_morgrem_in_race", morgrem, 2)
        if snap.my_active == MARNIE_MORGREM and chosen_type in (PLAY, ATTACH, ABILITY, END):
            grimmsnarl = _first_exact_card(obs, options, EVOLVE, MARNIE_GRIMMSNARL_EX)
            if grimmsnarl is not None:
                return self._take_window_limited("marnie_evolve_grimmsnarl_in_race", grimmsnarl, 2)
        return None

    def _opp_marnie_spread_convert(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        punk = _first_exact_card(obs, options, ABILITY, MARNIE_GRIMMSNARL_EX)
        if punk is not None and chosen_type in (END, PLAY, ATTACH, ABILITY):
            return self._take_window_limited("marnie_punk_up_when_online", punk, 3)
        convert_cards = {MUNKIDORI, SPIKEMUTH_GYM}
        convert = _first_card_by_types(obs, options, convert_cards, (PLAY, ABILITY))
        if convert is not None and chosen_type in (END, ATTACH):
            return self._take_window_limited("marnie_convert_spread_window", convert, 3)
        if chosen_type == END:
            attacks = _find_type(options, ATTACK)
            if attacks and not _has_any_attach(options):
                return self._take_window_limited("marnie_attack_after_engine_window", attacks[0], 3)
        judge = _first_exact_card(obs, options, PLAY, JUDGE)
        if judge is not None and snap.opp_hand_count >= 6 and chosen_type in (END, ABILITY):
            return self._take_window_limited("marnie_disrupt_ogerpon_large_hand", judge, 2)
        return None

    def _opp_crustle_window(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if CRUSTLE not in snap.my_board:
            crustle = _first_exact_card(obs, options, EVOLVE, CRUSTLE)
            if crustle is not None and chosen_card != CRUSTLE:
                return self._take_window_limited("crustle_evolve_wall_in_ex_window", crustle, 4)
            if snap.my_active == DWEBBLE:
                attacks = [
                    i for i, opt in enumerate(options)
                    if _option_type(opt) == ATTACK and _option_card(obs, opt) == DWEBBLE
                ]
                if attacks and chosen_type in (END, PLAY, ATTACH, ABILITY):
                    return self._take_window_limited("crustle_ascend_before_ex_attacks", attacks[0], 2)

        if snap.my_active == CRUSTLE:
            if chosen_type == RETREAT:
                attacks = _find_type(options, ATTACK)
                if attacks:
                    return self._take_window_limited("crustle_keep_wall_active_in_window", attacks[0], 3)
            if chosen_type == END:
                for opt_type in (ATTACK, ATTACH, ABILITY):
                    picks = _find_type(options, opt_type)
                    if picks:
                        return self._take_window_limited(f"crustle_no_idle_wall_window:{opt_type}", picks[0], 4)
        return None

    def _opp_cynthia_spiritomb(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if snap.my_active == CYNTHIA_GARCHOMP_EX and snap.opp_active == CRUSTLE and chosen_type == ATTACK:
            retreats = _find_type(options, RETREAT)
            if retreats and 387 in snap.my_board:
                return self._take_window_limited("cynthia_pivot_spiritomb_vs_wall", retreats[0], 2)
        spiritomb = _first_exact_card(obs, options, PLAY, 387)
        if spiritomb is not None and chosen_card != 387 and snap.opp_active == CRUSTLE:
            return self._take_window_limited("cynthia_play_spiritomb_counter", spiritomb, 2)
        return None

    def _opp_lucario_engine(
        self,
        obs: dict,
        options: list[dict],
        chosen_type: int,
        chosen_card: int,
        snap: ResourceSnapshot,
    ) -> RuleDecision | None:
        if LUCARIO_LUNATONE not in snap.my_board:
            lunatone = _first_exact_card(obs, options, PLAY, LUCARIO_LUNATONE)
            if lunatone is not None and chosen_type in (END, ATTACK, ATTACH):
                return self._take_window_limited("lucario_take_lunatone_engine_window", lunatone, 2)
        if LUCARIO_SOLROCK not in snap.my_board:
            solrock = _first_exact_card(obs, options, PLAY, LUCARIO_SOLROCK)
            if solrock is not None and chosen_type in (END, ATTACK, ATTACH):
                return self._take_window_limited("lucario_take_solrock_engine_window", solrock, 2)
        lunar = _first_exact_card(obs, options, ABILITY, LUCARIO_LUNATONE)
        if lunar is not None and chosen_type in (END, ATTACK):
            return self._take_window_limited("lucario_lunar_cycle_window", lunar, 2)
        search = _first_card_by_types(obs, options, LUCARIO_ENGINE_ROUTE, (PLAY,))
        if search is not None and chosen_type in (END, ATTACK):
            return self._take_window_limited("lucario_engine_search_window", search, 3)
        return None


def _has_any_attach(options: list[dict]) -> bool:
    return bool(_find_type(options, ATTACH))


STATEFUL_RULE_MODES = ("resource_plan", "opportunity_plan")


def make_rule_planner(mode: str, deck: list[int] | None = None) -> ResourcePlanner | OpportunityPlanner | None:
    if mode == "resource_plan":
        return ResourcePlanner(deck)
    if mode == "opportunity_plan":
        return OpportunityPlanner(deck)
    return None


def apply_rule_decision(
    obs: dict,
    action: list[int],
    deck: list[int] | None,
    *,
    mode: str = "",
    planner: ResourcePlanner | OpportunityPlanner | None = None,
) -> RuleDecision:
    if mode in STATEFUL_RULE_MODES and planner is not None:
        return planner.decide(obs, action, deck)
    if mode:
        return apply_rule_overlay(obs, action, deck, mode=mode)
    return RuleDecision(action)
