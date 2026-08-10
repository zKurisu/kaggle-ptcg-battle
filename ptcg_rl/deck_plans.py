from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ptcg_rl.deck_registry import deck_signature, read_deck


@dataclass(frozen=True)
class DeckPlan:
    archetype: str
    signature_ids: tuple[int, ...]
    primary_attackers: tuple[int, ...] = ()
    secondary_attackers: tuple[int, ...] = ()
    setup_basics: tuple[int, ...] = ()
    evolution_chain: tuple[int, ...] = ()
    engine_cards: tuple[int, ...] = ()
    energy_accel: tuple[int, ...] = ()
    draw_search: tuple[int, ...] = ()
    stadium_tools: tuple[int, ...] = ()
    disruption: tuple[int, ...] = ()
    switching: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    def all_key_ids(self) -> set[int]:
        out: set[int] = set()
        for field in (
            self.signature_ids,
            self.primary_attackers,
            self.secondary_attackers,
            self.setup_basics,
            self.evolution_chain,
            self.engine_cards,
            self.energy_accel,
            self.draw_search,
            self.stadium_tools,
            self.disruption,
            self.switching,
        ):
            out.update(int(x) for x in field)
        return out


CARD_NAMES = {
    42: "Applin",
    65: "Dunsparce",
    66: "Dudunsparce",
    96: "Teal Mask Ogerpon ex",
    103: "Snorunt",
    104: "Froslass",
    109: "Abra",
    112: "Munkidori",
    119: "Dreepy",
    120: "Drakloak",
    121: "Dragapult ex",
    245: "Alakazam",
    272: "Lillie's Clefairy ex",
    306: "Dudunsparce ex",
    333: "Riolu",
    344: "Dwebble",
    345: "Crustle",
    379: "Cynthia's Gible",
    380: "Cynthia's Gabite",
    381: "Cynthia's Garchomp ex",
    400: "Team Rocket's Tarountula",
    401: "Team Rocket's Spidops",
    431: "Team Rocket's Mewtwo ex",
    434: "Team Rocket's Mimikyu",
    646: "Marnie's Impidimp",
    647: "Marnie's Morgrem",
    648: "Marnie's Grimmsnarl ex",
    673: "Makuhita",
    674: "Hariyama",
    675: "Lunatone",
    676: "Solrock",
    677: "Riolu",
    678: "Mega Lucario ex",
    742: "Kadabra",
    756: "Mega Kangaskhan ex",
    849: "Mega Lopunny ex",
    858: "Psyduck",
    1030: "Staryu",
    1031: "Mega Starmie ex",
    1071: "Meowth ex",
    1086: "Buddy-Buddy Poffin",
    1121: "Ultra Ball",
    1123: "Switch",
    1141: "Premium Power Pro",
    1142: "Fighting Gong",
    1152: "Poke Pad",
    1159: "Hero's Cape",
    1182: "Boss's Orders",
    1213: "Judge",
    1227: "Lillie's Determination",
    1229: "Wally's Compassion",
    1249: "Grand Tree",
    1259: "Spikemuth Gym",
}


PLANS: dict[str, DeckPlan] = {
    "Marnie Grimmsnarl": DeckPlan(
        archetype="Marnie Grimmsnarl",
        signature_ids=(648,),
        primary_attackers=(648,),
        setup_basics=(646,),
        evolution_chain=(646, 647, 648),
        engine_cards=(112, 103, 104),
        energy_accel=(648,),
        draw_search=(1259,),
        stadium_tools=(1259,),
        notes=(
            "Stage-2 plan. Prioritize finding Impidimp/Morgrem/Grimmsnarl and timing Punk Up.",
            "After Punk Up, attach dark energy to the attacker line rather than spreading randomly.",
        ),
    ),
    "Teal Mask Ogerpon": DeckPlan(
        archetype="Teal Mask Ogerpon",
        signature_ids=(96,),
        primary_attackers=(96,),
        secondary_attackers=(756, 1071, 272),
        setup_basics=(96, 756, 1071, 272),
        engine_cards=(96, 1071),
        energy_accel=(96,),
        draw_search=(1071,),
        notes=(
            "Tempo basic-ex plan. Teal Dance converts grass energy into draw and acceleration.",
            "Keep Ogerpon powered while using support basics for draw/search and matchup coverage.",
        ),
    ),
    "Mega Lopunny": DeckPlan(
        archetype="Mega Lopunny",
        signature_ids=(849,),
        primary_attackers=(849,),
        secondary_attackers=(306, 858),
        setup_basics=(65, 858),
        evolution_chain=(65, 66, 306, 849),
        engine_cards=(65, 66, 306),
        notes=(
            "Mega attacker plan backed by Dudunsparce-style consistency.",
            "Avoid mixed signatures unless round-robin proves the game plan remains coherent.",
        ),
    ),
    "Mega Lucario": DeckPlan(
        archetype="Mega Lucario",
        signature_ids=(678,),
        primary_attackers=(678,),
        secondary_attackers=(674, 676, 675),
        setup_basics=(333, 677, 673, 676, 675),
        evolution_chain=(333, 677, 678, 673, 674),
        engine_cards=(676, 675, 1141, 1142, 1152),
        draw_search=(1121, 1152, 1227),
        disruption=(1182, 1213),
        switching=(1123, 1229),
        notes=(
            "Sample-limited specialist. Correct deck signature and score bands are mandatory.",
            "The 43d6d8b0fce9 build uses Riolu 677, not only Riolu 333.",
            "Failure reports show MAIN, ATTACH_FROM, TO_HAND, and active-selection tempo drive most errors.",
        ),
    ),
    "Mega Starmie": DeckPlan(
        archetype="Mega Starmie",
        signature_ids=(1031,),
        primary_attackers=(1031,),
        secondary_attackers=(133, 132),
        setup_basics=(1030, 131),
        evolution_chain=(1030, 1031, 131, 132, 133),
        engine_cards=(131, 132, 133, 1249),
        draw_search=(1086, 1122, 1152, 1225, 1194),
        disruption=(1182,),
        switching=(1229,),
        notes=(
            "Mega Starmie route. Set up Staryu/Starmie while building the Duskull line.",
            "Use Hilda/Grand Tree/Poke Pad/Poffin-style search to assemble evolution pressure.",
            "Dusknoir/Dusclops damage counters can create prize maps before Mega Starmie attacks.",
        ),
    ),
    "Alakazam": DeckPlan(
        archetype="Alakazam",
        signature_ids=(245, 742),
        primary_attackers=(245,),
        setup_basics=(109,),
        evolution_chain=(109, 742, 245),
        engine_cards=(742,),
        draw_search=(742,),
        notes=(
            "Stage-2 control/bench attack plan. Correct policy/deck pairing is critical.",
            "Always evaluate with registry auto-deck because wrong Alakazam lists collapse vs random.",
        ),
    ),
    "Dragapult": DeckPlan(
        archetype="Dragapult",
        signature_ids=(121,),
        primary_attackers=(121,),
        setup_basics=(119,),
        evolution_chain=(119, 120, 121),
        engine_cards=(120,),
        draw_search=(120,),
        notes=(
            "Stage-2 spread plan. Needs deck-sig training and damage-counter/discard diagnostics.",
            "Do not treat high random win rate as sufficient; core matchups have been weak.",
        ),
    ),
    "Festival Lead": DeckPlan(
        archetype="Festival Lead",
        signature_ids=(93,),
        primary_attackers=(93,),
        setup_basics=(89, 42),
        evolution_chain=(89, 90, 93),
        engine_cards=(90, 93),
        draw_search=(90,),
        stadium_tools=(93,),
        notes=(
            "Festival Grounds/Dipplin plan. The main goal is enabling repeated attacks.",
            "Top1 specialist has been strong vs random; mixed is a population baseline only.",
        ),
    ),
    "Crustle Wall": DeckPlan(
        archetype="Crustle Wall",
        signature_ids=(345,),
        primary_attackers=(345,),
        setup_basics=(344,),
        evolution_chain=(344, 345),
        engine_cards=(756,),
        notes=(
            "Anti-ex wall plan. Its value is matchup-specific, especially into ex-heavy decks.",
            "Keep top-k specialists for Ogerpon/meta counter testing.",
        ),
    ),
    "Cynthia Garchomp": DeckPlan(
        archetype="Cynthia Garchomp",
        signature_ids=(381,),
        primary_attackers=(381,),
        setup_basics=(379,),
        evolution_chain=(379, 380, 381),
        engine_cards=(380,),
        draw_search=(380,),
        notes=(
            "Linear Stage-2 plan. Gabite's search ability should support Garchomp setup.",
            "Mixed is acceptable for population, but top1 is safer for candidate submission.",
        ),
    ),
    "Team Rocket Mewtwo": DeckPlan(
        archetype="Team Rocket Mewtwo",
        signature_ids=(431,),
        primary_attackers=(431,),
        secondary_attackers=(434,),
        setup_basics=(400, 434, 431),
        evolution_chain=(400, 401),
        engine_cards=(400, 401, 434),
        energy_accel=(401,),
        notes=(
            "Mewtwo cannot attack until enough Team Rocket Pokemon are in play.",
            "Random win rate has not translated to Kaggle; keep as population until matchups improve.",
        ),
    ),
}


ALIASES = {
    "marnie": "Marnie Grimmsnarl",
    "marnie grimmsnarl": "Marnie Grimmsnarl",
    "teal mask ogerpon": "Teal Mask Ogerpon",
    "ogerpon": "Teal Mask Ogerpon",
    "mega lopunny": "Mega Lopunny",
    "lopunny": "Mega Lopunny",
    "mega lucario": "Mega Lucario",
    "lucario": "Mega Lucario",
    "mega starmie": "Mega Starmie",
    "starmie": "Mega Starmie",
    "alakazam": "Alakazam",
    "dragapult": "Dragapult",
    "festival": "Festival Lead",
    "festival lead": "Festival Lead",
    "crustle": "Crustle Wall",
    "crustle wall": "Crustle Wall",
    "cynthia": "Cynthia Garchomp",
    "cynthia garchomp": "Cynthia Garchomp",
    "mewtwo": "Team Rocket Mewtwo",
    "team rocket mewtwo": "Team Rocket Mewtwo",
}


def canonical_archetype(name: str) -> str | None:
    if name in PLANS:
        return name
    return ALIASES.get(name.strip().lower())


def get_plan(name: str) -> DeckPlan | None:
    arch = canonical_archetype(name)
    return PLANS.get(arch) if arch else None


def infer_plan(cards: Iterable[int]) -> DeckPlan | None:
    counts = {int(c): 0 for c in cards}
    for c in cards:
        counts[int(c)] = counts.get(int(c), 0) + 1
    best: tuple[int, str, DeckPlan] | None = None
    for name, plan in PLANS.items():
        score = sum(counts.get(cid, 0) for cid in plan.signature_ids)
        if score <= 0:
            continue
        cand = (score, name, plan)
        if best is None or cand > best:
            best = cand
    return best[2] if best else None


def infer_plan_for_deck(path: str | Path) -> DeckPlan | None:
    return infer_plan(read_deck(path))


def tag_card(plan: DeckPlan | None, card_id: int) -> str:
    if not plan or not card_id:
        return "other"
    cid = int(card_id)
    tags = []
    for label, ids in (
        ("primary", plan.primary_attackers),
        ("secondary", plan.secondary_attackers),
        ("setup", plan.setup_basics),
        ("evolution", plan.evolution_chain),
        ("engine", plan.engine_cards),
        ("energy_accel", plan.energy_accel),
        ("draw_search", plan.draw_search),
        ("stadium_tool", plan.stadium_tools),
        ("disruption", plan.disruption),
        ("switching", plan.switching),
    ):
        if cid in ids:
            tags.append(label)
    return "|".join(tags) if tags else "other"


def tag_cards(plan: DeckPlan | None, card_ids: Iterable[int]) -> list[str]:
    return [tag_card(plan, int(cid)) for cid in card_ids]


def card_name(card_id: int) -> str:
    return CARD_NAMES.get(int(card_id), str(int(card_id)))


def plan_score(plan: DeckPlan, cards: Iterable[int]) -> dict[str, int | str]:
    cards = [int(c) for c in cards]
    present = set(cards)
    counts = {cid: cards.count(cid) for cid in plan.all_key_ids()}
    missing = sorted(cid for cid in plan.signature_ids if cid not in present)
    return {
        "archetype": plan.archetype,
        "deck_sig": deck_signature(cards),
        "signature_hits": sum(counts.get(cid, 0) for cid in plan.signature_ids),
        "key_hits": sum(counts.values()),
        "missing_signature": " ".join(str(x) for x in missing),
    }
