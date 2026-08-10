#!/usr/bin/env python3
"""Build public-source search tasks for matchup strategy mining."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ARCH_ALIASES = {
    "Teal Mask Ogerpon": [
        "Teal Mask Ogerpon",
        "Ogerpon Box",
        "Ogerpon Meganium Arboliva",
        "オーガポン",
    ],
    "Crustle Wall": [
        "Crustle",
        "Crustle Wall",
        "Mysterious Rock Inn",
        "イワパレス",
    ],
    "Marnie Grimmsnarl": [
        "Marnie's Grimmsnarl ex",
        "Marnie Grimmsnarl",
        "Grimmsnarl Froslass Munkidori",
        "マリィのオーロンゲ",
    ],
    "Mega Lucario": [
        "Mega Lucario ex",
        "Lucario Solrock Lunatone",
        "メガルカリオ",
    ],
    "Mega Lopunny": [
        "Mega Lopunny ex",
        "Lopunny Dudunsparce",
        "メガミミロップ",
    ],
    "Dragapult": [
        "Dragapult ex",
        "Drakloak Dragapult",
        "ドラパルト",
    ],
    "Alakazam": [
        "Alakazam",
        "Alakazam ex",
        "フーディン",
    ],
    "Cynthia Garchomp": [
        "Cynthia's Garchomp ex",
        "Cynthia Garchomp",
        "シロナのガブリアス",
    ],
    "Team Rocket Mewtwo": [
        "Team Rocket's Mewtwo ex",
        "Rocket Mewtwo",
        "ロケット団のミュウツー",
    ],
    "Festival Lead": [
        "Festival Lead",
        "Dipplin Festival Grounds",
        "おまつりおんど",
    ],
}

EN_SUFFIXES = (
    "matchup guide",
    "how to beat",
    "counterplay",
    "deck guide",
    "tournament report",
    "matchup spread",
)
JA_SUFFIXES = (
    "対面",
    "立ち回り",
    "デッキ解説",
    "有利不利",
)


def read_sources(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aliases(name: str) -> list[str]:
    return DEFAULT_ARCH_ALIASES.get(name, [name])


def build_queries(candidate: str, opponent: str, source: dict[str, str]) -> list[str]:
    cand = aliases(candidate)
    opp = aliases(opponent)
    lang = source.get("language", "")
    queries: list[str] = []
    if "ja" in lang:
        for a in cand[:2]:
            for b in opp[:2]:
                for suffix in JA_SUFFIXES:
                    queries.append(f"{a} {b} {suffix}")
    else:
        for a in cand[:3]:
            for b in opp[:3]:
                for suffix in EN_SUFFIXES:
                    queries.append(f"{a} vs {b} {suffix}")
                    queries.append(f"{a} {suffix} {b}")
    return queries[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-archetype", required=True)
    p.add_argument("--opponent-archetype", required=True)
    p.add_argument("--sources", default="data/public_strategy_sources_v1.csv")
    p.add_argument("--priority", action="append", default=["high", "medium"])
    p.add_argument("--out-csv", default="logs/public_strategy_search/tasks.csv")
    args = p.parse_args()

    priorities = set(args.priority)
    rows: list[dict[str, str]] = []
    for src in read_sources(Path(args.sources)):
        if priorities and src.get("priority") not in priorities:
            continue
        source_type = src.get("source_type", "")
        if source_type in {"card_text", "paid_guides", "tracker_stats"}:
            continue
        for query in build_queries(args.candidate_archetype, args.opponent_archetype, src):
            rows.append({
                "candidate_archetype": args.candidate_archetype,
                "opponent_archetype": args.opponent_archetype,
                "source_name": src.get("source_name", ""),
                "source_type": source_type,
                "access": src.get("access", ""),
                "language": src.get("language", ""),
                "url": src.get("url", ""),
                "query": query,
                "notes": src.get("notes", ""),
            })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = [
            "candidate_archetype",
            "opponent_archetype",
            "source_name",
            "source_type",
            "access",
            "language",
            "url",
            "query",
            "notes",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
