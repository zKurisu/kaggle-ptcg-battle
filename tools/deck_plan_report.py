#!/usr/bin/env python3
"""Inspect deck signatures against the built-in game-plan registry."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_plans import PLANS, card_name, get_plan, infer_plan, plan_score
from ptcg_rl.deck_registry import read_deck


def _card_lines(cards: list[int], ids: list[int]) -> list[str]:
    counts = Counter(cards)
    lines = []
    for cid in ids:
        lines.append(f"{cid:4d} x{counts.get(cid, 0):d}  {card_name(cid)}")
    return lines


def _print_plan(cards: list[int], requested: str | None) -> None:
    plan = get_plan(requested) if requested else infer_plan(cards)
    if not plan:
        print("Plan: unknown")
        return
    score = plan_score(plan, cards)
    print(f"Plan: {plan.archetype}")
    print(f"Deck signature: {score['deck_sig']}")
    print(f"Signature hits: {score['signature_hits']}  key hits: {score['key_hits']}")
    if score["missing_signature"]:
        print(f"Missing signature cards: {score['missing_signature']}")
    for title, ids in (
        ("Primary attackers", plan.primary_attackers),
        ("Secondary attackers", plan.secondary_attackers),
        ("Setup basics", plan.setup_basics),
        ("Evolution chain", plan.evolution_chain),
        ("Engine cards", plan.engine_cards),
        ("Energy acceleration", plan.energy_accel),
        ("Draw/search", plan.draw_search),
        ("Stadium/tools", plan.stadium_tools),
    ):
        if ids:
            print(f"\n{title}:")
            for line in _card_lines(cards, list(ids)):
                print(f"  {line}")
    if plan.notes:
        print("\nPlan notes:")
        for note in plan.notes:
            print(f"  - {note}")


def _report_registry(path: Path, requested: str | None) -> None:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Registry: {path} rows={len(rows)}")
    for row in rows:
        deck_path = row.get("deck_path") or row.get("deck") or ""
        policy = row.get("policy_path") or row.get("policy") or ""
        if not deck_path:
            continue
        try:
            cards = read_deck(deck_path)
        except Exception as e:
            print(f"\n{policy}\n  deck={deck_path}\n  ERROR: {e}")
            continue
        plan = get_plan(requested) if requested else infer_plan(cards)
        score = plan_score(plan, cards) if plan else {}
        print(
            f"{Path(policy).name or '-':56s} "
            f"plan={(plan.archetype if plan else 'unknown'):20s} "
            f"sig={score.get('deck_sig', '')} key={score.get('key_hits', '')} "
            f"deck={deck_path}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deck", help="60-line deck csv")
    p.add_argument("--registry", help="policy/deck registry csv")
    p.add_argument("--archetype", help="force a deck-plan archetype")
    p.add_argument("--list", action="store_true", help="list known plans")
    args = p.parse_args()

    if args.list:
        for name in sorted(PLANS):
            plan = PLANS[name]
            print(f"{name}: signature={','.join(map(str, plan.signature_ids))}")
        return
    if args.registry:
        _report_registry(Path(args.registry), args.archetype)
        return
    if not args.deck:
        p.error("provide --deck, --registry, or --list")
    _print_plan(read_deck(args.deck), args.archetype)


if __name__ == "__main__":
    main()
