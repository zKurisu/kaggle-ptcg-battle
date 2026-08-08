#!/usr/bin/env python3
"""Build trajectory filters for aggressive weak-matchup counter training."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTRA_FIELDS = [
    "counter_weak",
    "counter_clean",
    "counter_bad",
    "counter_dirty_win",
    "counter_loss",
    "counter_status",
]


def norm(value: str) -> str:
    return str(value or "").strip().lower()


def truthy(value: str) -> bool:
    return str(value).strip() in {"1", "1.0", "true", "True", "yes", "YES"}


def outcome(row: dict[str, str]) -> str:
    raw = norm(row.get("outcome", ""))
    if raw in {"win", "loss", "draw"}:
        return raw
    if truthy(row.get("outcome_win", "0")):
        return "win"
    if truthy(row.get("outcome_draw", "0")):
        return "draw"
    if truthy(row.get("outcome_loss", "0")):
        return "loss"
    return ""


def game_key(row: dict[str, str]) -> str:
    key = str(row.get("game_key", "")).strip()
    if key:
        return key
    episode = str(row.get("episode_id", "")).strip()
    player = str(row.get("player_index", "")).strip()
    if episode and player:
        return f"{episode}:{player}"
    return ""


def read_clean_keys(path: str, archetype: str, opponents: set[str]) -> set[str]:
    clean: set[str] = set()
    if not path:
        return clean
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if norm(row.get("archetype", "")) != archetype:
                continue
            if norm(row.get("opponent_archetype", "")) not in opponents:
                continue
            if not truthy(row.get("clean_win", "0")):
                continue
            key = game_key(row)
            if key:
                clean.add(key)
    return clean


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-csv", required=True)
    p.add_argument("--teacher-games-csv", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--opponent-archetype", action="append", default=[], required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument(
        "--dirty-win-policy",
        choices=["keep", "drop_nonclean"],
        default="drop_nonclean",
        help="whether weak matchup wins not selected as clean teachers are kept or marked counter_bad",
    )
    args = p.parse_args()

    arch = norm(args.archetype)
    opponents = {norm(x) for x in args.opponent_archetype if norm(x)}
    if not opponents:
        raise ValueError("at least one --opponent-archetype is required")
    clean_keys = read_clean_keys(args.teacher_games_csv, arch, opponents)

    with open(args.trajectory_csv, newline="") as f:
        reader = csv.DictReader(f)
        base_fields = list(reader.fieldnames or [])
        if not base_fields:
            raise ValueError(f"{args.trajectory_csv} has no header")
        rows: list[dict[str, str]] = []
        stats = {
            "rows": 0,
            "weak": 0,
            "clean": 0,
            "bad": 0,
            "dirty_win": 0,
            "loss": 0,
        }
        for row in reader:
            stats["rows"] += 1
            key = game_key(row)
            weak = norm(row.get("opponent_archetype", "")) in opponents
            clean = key in clean_keys
            out = outcome(row)
            loss = weak and out != "win"
            dirty_win = weak and out == "win" and not clean
            bad = loss or (dirty_win and args.dirty_win_policy == "drop_nonclean")
            if weak:
                stats["weak"] += 1
            if clean:
                stats["clean"] += 1
            if loss:
                stats["loss"] += 1
            if dirty_win:
                stats["dirty_win"] += 1
            if bad:
                stats["bad"] += 1
            if clean:
                status = "clean_teacher"
            elif loss:
                status = "weak_loss_or_draw"
            elif dirty_win:
                status = "weak_dirty_win"
            elif weak:
                status = "weak_other"
            else:
                status = "anchor"
            row.update({
                "counter_weak": "1" if weak else "0",
                "counter_clean": "1" if clean else "0",
                "counter_bad": "1" if bad else "0",
                "counter_dirty_win": "1" if dirty_win else "0",
                "counter_loss": "1" if loss else "0",
                "counter_status": status,
            })
            rows.append(row)

    fields = list(base_fields)
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    write_rows(Path(args.out_csv), fields, rows)
    print(
        f"wrote {args.out_csv} rows={stats['rows']} weak={stats['weak']} "
        f"clean={stats['clean']} bad={stats['bad']} loss={stats['loss']} "
        f"dirty_win={stats['dirty_win']} opponents={sorted(opponents)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
