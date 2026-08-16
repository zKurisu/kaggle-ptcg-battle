#!/usr/bin/env python3
"""Summarize human-readable teacher traces into route-level signals.

The trace files are intentionally verbose.  This script extracts the parts we
need for matchup plan work: setup choices, resource route cards, disruption,
attack/no-damage patterns, damage-counter targets, and deck-out indicators.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


HEADER_RE = re.compile(r"^([a-zA-Z_]+): `([^`]*)`")
DECISION_RE = re.compile(r"^### Decision (\d+) step=(\d+) turn=(\d+) tac=(\d+)")
CHOSEN_RE = re.compile(r"^\s+chosen: ([A-Z_]+)(?:\s+(.+))?$")
BOARD_RE = re.compile(
    r"^\s+(?:board|board_snapshot): .*?prizes=(\d+) deck=(\d+) hand=(\d+); "
    r"opp .*?prizes=(\d+) deck=(\d+) hand=(\d+)"
)

ROUTE_CARDS = {
    "Budew",
    "Dreepy",
    "Drakloak",
    "Dragapult ex",
    "Munkidori",
    "Fezandipiti ex",
    "Crispin",
    "Dawn",
    "Lillie's Determination",
    "Jamming Tower",
    "Crushing Hammer",
    "Boss's Orders",
    "Boss’s Orders",
    "Judge",
    "Ultra Ball",
    "Buddy-Buddy Poffin",
    "Night Stretcher",
    "Chi-Yu",
    "Torchic",
    "Combusken",
    "Blaziken ex",
    "Rare Candy",
    "Area Zero Underdepths",
    "Team Rocket's Watchtower",
    "Lillie’s Clefairy ex",
    "Shaymin",
}


def clean_card(text: str) -> str:
    text = (text or "").strip()
    text = text.split(" | ", 1)[0].strip()
    return re.sub(r"\(\d+\)", "", text).strip()


def parse_trace(path: Path) -> dict[str, str | int]:
    header: dict[str, str] = {}
    counts: Counter[str] = Counter()
    chosen_type_counts: Counter[str] = Counter()
    chosen_card_counts: Counter[str] = Counter()
    first_decision: dict[str, int] = {}
    active_setup = ""
    bench_setup = ""
    last_decision = -1
    last_step = -1
    last_turn = -1
    min_opp_deck = 999
    min_my_deck = 999
    final_my_prizes = -1
    final_opp_prizes = -1
    final_opp_deck = -1

    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                header[m.group(1)] = m.group(2)
                continue
            m = DECISION_RE.match(line)
            if m:
                last_decision = int(m.group(1))
                last_step = int(m.group(2))
                last_turn = int(m.group(3))
                continue
            m = CHOSEN_RE.match(line)
            if m:
                typ = m.group(1)
                card = clean_card(m.group(2) or "")
                chosen_type_counts[typ] += 1
                if card:
                    chosen_card_counts[card] += 1
                if last_decision == 0 and not active_setup:
                    active_setup = card
                if last_decision == 1 and not bench_setup:
                    bench_setup = card
                slug = card.lower().replace(" ", "_").replace("’", "").replace("'", "")
                key = f"first_{typ.lower()}_{slug}"
                first_decision.setdefault(key, last_decision)
                if card in ROUTE_CARDS:
                    counts[f"chosen_{card}"] += 1
                continue
            m = BOARD_RE.match(line)
            if m:
                my_prizes, my_deck, _my_hand, opp_prizes, opp_deck, _opp_hand = map(int, m.groups())
                min_my_deck = min(min_my_deck, my_deck)
                min_opp_deck = min(min_opp_deck, opp_deck)
                final_my_prizes = my_prizes
                final_opp_prizes = opp_prizes
                final_opp_deck = opp_deck

            if "attack card=Dragapult ex" in line:
                counts["dragapult_ex_attacks"] += 1
            if "attack card=Drakloak" in line:
                counts["drakloak_attacks"] += 1
            if "damage Crustle(345) value=0" in line:
                counts["zero_damage_into_crustle"] += 1
            if "counter Crustle(345)" in line:
                counts["counter_crustle"] += 1
            if "counter Dwebble(344)" in line:
                counts["counter_dwebble"] += 1
            if "damage Dwebble(344)" in line:
                counts["damage_dwebble"] += 1
            if "move Crustle(345) bench->discard" in line or "move Dwebble(344) evolution_stack->discard" in line:
                counts["bench_crustle_ko"] += 1
            if "move Dwebble(344) bench->discard" in line:
                counts["bench_dwebble_ko"] += 1
            if "coin head=True" in line and "Crushing Hammer" in line:
                counts["hammer_heads"] += 1
            if "coin head=False" in line and "Crushing Hammer" in line:
                counts["hammer_tails"] += 1
            if "play/ability/effect Boss" in line:
                counts["boss_played"] += 1
            if "play/ability/effect Jamming Tower" in line:
                counts["jamming_played"] += 1
            if "play/ability/effect Crushing Hammer" in line:
                counts["hammer_played"] += 1
            if "evolve Dreepy(119) -> Drakloak" in line:
                counts["evolve_drakloak"] += 1
            if "evolve Drakloak(120) -> Dragapult ex" in line:
                counts["evolve_dragapult"] += 1
            if "attach Basic {D} Energy" in line and "Munkidori" in line:
                counts["dark_to_munkidori"] += 1
            if "deck=0" in line or "opp active=" in line and " deck=0 " in line:
                counts["deckout_seen"] += 1

    row: dict[str, str | int] = {
        "trace_path": str(path),
        "date": header.get("date", ""),
        "episode_id": header.get("episode_id", ""),
        "team_name": header.get("team_name", ""),
        "deck_sig": header.get("deck_sig", ""),
        "archetype": header.get("archetype", ""),
        "score": header.get("score", ""),
        "opponent_team_name": header.get("opponent_team_name", ""),
        "opponent_deck_sig": header.get("opponent_deck_sig", ""),
        "opponent_archetype": header.get("opponent_archetype", ""),
        "opponent_score": header.get("opponent_score", ""),
        "won": header.get("won", ""),
        "steps": header.get("steps", ""),
        "decisions_seen": last_decision + 1,
        "last_step": last_step,
        "last_turn": last_turn,
        "active_setup": active_setup,
        "bench_setup": bench_setup,
        "min_my_deck": min_my_deck if min_my_deck != 999 else "",
        "min_opp_deck": min_opp_deck if min_opp_deck != 999 else "",
        "final_my_prizes": final_my_prizes,
        "final_opp_prizes": final_opp_prizes,
        "final_opp_deck": final_opp_deck,
    }
    for key in sorted(counts):
        row[key] = counts[key]
    for key in sorted(ROUTE_CARDS):
        row[f"chosen_{key}"] = chosen_card_counts[key]
    for typ in sorted(chosen_type_counts):
        row[f"type_{typ}"] = chosen_type_counts[typ]
    for key, value in first_decision.items():
        row[key] = value
    return row


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    preferred = [
        "trace_path",
        "date",
        "episode_id",
        "team_name",
        "deck_sig",
        "score",
        "opponent_team_name",
        "opponent_deck_sig",
        "opponent_score",
        "won",
        "steps",
        "decisions_seen",
        "active_setup",
        "bench_setup",
        "min_opp_deck",
        "final_my_prizes",
        "final_opp_prizes",
        "final_opp_deck",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Teacher Trace Route Summary\n\n")
        f.write(f"traces: {len(rows)}\n\n")
        if not rows:
            return
        for row in rows:
            f.write(
                "- "
                f"{row.get('date')} {row.get('team_name')} {row.get('deck_sig')} "
                f"vs {row.get('opponent_team_name')} {row.get('opponent_deck_sig')} "
                f"won={row.get('won')} setup={row.get('active_setup')}/{row.get('bench_setup')} "
                f"opp_deck_min={row.get('min_opp_deck')} final_prizes="
                f"{row.get('final_my_prizes')}-{row.get('final_opp_prizes')} "
                f"hammer={row.get('hammer_played', 0)} jamming={row.get('jamming_played', 0)} "
                f"munkidori={row.get('chosen_Munkidori', 0)} "
                f"dca_crustle={row.get('counter_crustle', 0)} "
                f"dca_dwebble={row.get('counter_dwebble', 0)} "
                f"zero_wall={row.get('zero_damage_into_crustle', 0)} "
                f"path={row.get('trace_path')}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    root = Path(args.trace_dir)
    paths = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    rows = [parse_trace(path) for path in paths]
    write_csv(Path(args.out_csv), rows)
    if args.out_md:
        write_md(Path(args.out_md), rows)
    print(f"summarized {len(rows)} traces -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
