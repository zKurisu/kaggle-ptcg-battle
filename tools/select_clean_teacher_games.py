#!/usr/bin/env python3
"""Select clean teacher wins from matchup quality audits."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_pair(value: str) -> tuple[str, str]:
    if "=>" not in value:
        raise ValueError(f"pair must be ARCHETYPE=>OPPONENT, got {value!r}")
    left, right = value.split("=>", 1)
    return left.strip().lower(), right.strip().lower()


def slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "unknown"


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except Exception:
        return 0


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0)
    except Exception:
        return 0.0


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quality-csv", required=True)
    p.add_argument("--game-quality-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pair", action="append", default=[],
                   help="optional ARCHETYPE=>OPPONENT filter; repeatable")
    p.add_argument("--recommendation", action="append", default=["clean_teacher"])
    p.add_argument("--min-clean-wins", type=int, default=10)
    p.add_argument("--min-clean-share", type=float, default=0.15)
    p.add_argument("--max-brick-share", type=float, default=0.10)
    p.add_argument("--max-teachers-per-pair", type=int, default=0)
    args = p.parse_args()

    quality_fields, quality_rows = read_csv(Path(args.quality_csv))
    game_fields, game_rows = read_csv(Path(args.game_quality_csv))
    pair_filter = set(parse_pair(x) for x in args.pair)
    recs = set(args.recommendation)

    selected: list[dict[str, str]] = []
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in quality_rows:
        pair = (row.get("archetype", "").lower(), row.get("opponent_archetype", "").lower())
        if pair_filter and pair not in pair_filter:
            continue
        if row.get("recommendation", "") not in recs:
            continue
        if as_int(row, "clean_wins") < args.min_clean_wins:
            continue
        if as_float(row, "clean_win_share") < args.min_clean_share:
            continue
        if as_float(row, "brick_win_share") > args.max_brick_share:
            continue
        by_pair[pair].append(row)

    for pair, rows in sorted(by_pair.items()):
        rows = sorted(
            rows,
            key=lambda r: (
                as_float(r, "quality_score"),
                as_int(r, "clean_wins"),
                as_float(r, "game_wr"),
            ),
            reverse=True,
        )
        if args.max_teachers_per_pair > 0:
            rows = rows[: args.max_teachers_per_pair]
        selected.extend(rows)

    selected_keys = {
        (
            r.get("archetype", ""),
            r.get("opponent_archetype", ""),
            r.get("deck_sig", ""),
            r.get("team_name", ""),
        )
        for r in selected
    }
    selected_games = [
        r for r in game_rows
        if r.get("clean_win", "") in {"1", "1.0", "true", "True"}
        and (
            r.get("archetype", ""),
            r.get("opponent_archetype", ""),
            r.get("deck_sig", ""),
            r.get("team_name", ""),
        ) in selected_keys
    ]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "selected_clean_teachers.csv", quality_fields, selected)
    write_csv(out_dir / "selected_clean_teacher_games.csv", game_fields, selected_games)

    pair_games: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    arch_games: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_games:
        pair = (row.get("archetype", ""), row.get("opponent_archetype", ""))
        pair_games[pair].append(row)
        arch_games[row.get("archetype", "")].append(row)

    summary_fields = [
        "scope", "archetype", "opponent_archetype", "teachers",
        "games", "clean_games",
    ]
    summary_rows: list[dict[str, str]] = []
    for pair, rows in sorted(pair_games.items()):
        arch, opp = pair
        pair_teachers = {
            (r.get("deck_sig", ""), r.get("team_name", ""))
            for r in selected
            if r.get("archetype", "") == arch and r.get("opponent_archetype", "") == opp
        }
        filename = f"{slug(arch)}__vs__{slug(opp)}.csv"
        write_csv(out_dir / "pairs" / filename, game_fields, rows)
        summary_rows.append({
            "scope": "pair",
            "archetype": arch,
            "opponent_archetype": opp,
            "teachers": str(len(pair_teachers)),
            "games": str(len({r.get("game_key", "") for r in rows})),
            "clean_games": str(len(rows)),
        })

    for arch, rows in sorted(arch_games.items()):
        filename = f"{slug(arch)}.csv"
        write_csv(out_dir / "archetypes" / filename, game_fields, rows)
        summary_rows.append({
            "scope": "archetype",
            "archetype": arch,
            "opponent_archetype": "*",
            "teachers": "",
            "games": str(len({r.get("game_key", "") for r in rows})),
            "clean_games": str(len(rows)),
        })

    write_csv(out_dir / "summary.csv", summary_fields, summary_rows)
    print(
        f"selected_teachers={len(selected)} selected_clean_games={len(selected_games)} "
        f"pairs={len(pair_games)} archetypes={len(arch_games)} out={out_dir}",
        flush=True,
    )
    for row in summary_rows:
        print(
            f"{row['scope']} {row['archetype']} vs {row['opponent_archetype']} "
            f"teachers={row['teachers']} games={row['games']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
