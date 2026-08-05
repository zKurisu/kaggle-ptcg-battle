#!/usr/bin/env python3
"""Mine outcome-skewed decision patterns from trace_matchup_decisions CSVs."""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


OUT_FIELDS = [
    "source",
    "view",
    "key",
    "loss_n",
    "win_n",
    "draw_n",
    "loss_games",
    "win_games",
    "loss_per_game",
    "win_per_game",
    "delta_per_game",
    "loss_share",
    "win_share",
    "delta_share",
    "ratio_per_game",
    "priority",
    "example_loss",
    "example_win",
]

MISS_FLAGS = [
    ("miss_attack", "attack_available", "attack_chosen", "available_attack_card_names"),
    ("miss_ability", "ability_available", "ability_chosen", "available_ability_card_names"),
    ("miss_attach", "attach_available", "attach_chosen", "available_attach_card_names"),
    ("miss_evolve", "evolve_available", "evolve_chosen", "available_evolve_card_names"),
    ("miss_play", "play_available", "play_chosen", "available_play_card_names"),
    ("miss_retreat", "retreat_available", "retreat_chosen", "available_retreat_card_names"),
]


def clean_source(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".decisions.csv", ".csv"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sibling_games(path: str) -> str:
    if path.endswith(".decisions.csv"):
        return path[: -len(".decisions.csv")] + ".games.csv"
    return ""


def game_counts(path: str, rows: list[dict]) -> Counter:
    games = sibling_games(path)
    counts: Counter = Counter()
    if games and os.path.exists(games):
        with open(games, newline="") as f:
            for row in csv.DictReader(f):
                counts[row.get("outcome", "")] += 1
        return counts
    seen = set()
    for row in rows:
        key = (row.get("game", ""), row.get("outcome", ""))
        if key in seen:
            continue
        seen.add(key)
        counts[row.get("outcome", "")] += 1
    return counts


def truth(row: dict, key: str) -> bool:
    try:
        return int(float(row.get(key, 0) or 0)) != 0
    except Exception:
        return False


def fnum(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def bucket_int(value, cuts: list[int]) -> str:
    try:
        x = int(float(value or 0))
    except Exception:
        x = 0
    lo = 0
    for hi in cuts:
        if x <= hi:
            return f"{lo}-{hi}"
        lo = hi + 1
    return f"{lo}+"


def context(row: dict) -> str:
    name = row.get("context_name", "")
    raw = row.get("context", "")
    return f"{name}({raw})" if raw != "" else name


def compact(value: str, max_len: int = 180) -> str:
    value = " ".join(str(value or "").split())
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def row_summary(row: dict) -> str:
    parts = [
        f"turn={row.get('turn')}",
        f"tac={row.get('turn_action_count')}",
        f"ctx={context(row)}",
        f"avail={row.get('available_type_counts')}",
        f"chosen={row.get('chosen_type_names')}:{row.get('chosen_card_names')}",
    ]
    if row.get("top1_type_name"):
        parts.append(f"top1={row.get('top1_type_name')}:{row.get('top1_card_name')} r={row.get('chosen_first_rank')}")
    if row.get("my_active_name") or row.get("opp_active_name"):
        parts.append(f"active={row.get('my_active_name')} vs {row.get('opp_active_name')}")
    return compact(" | ".join(parts), 500)


def add_pattern(patterns: dict, view: str, key: str, row: dict) -> None:
    outcome = row.get("outcome", "")
    if outcome not in ("loss", "win", "draw"):
        return
    entry = patterns[(view, key)]
    entry["counts"][outcome] += 1
    if not entry.get(f"example_{outcome}"):
        entry[f"example_{outcome}"] = row_summary(row)


def build_patterns(rows: list[dict], views: set[str]) -> dict:
    patterns = defaultdict(lambda: {"counts": Counter()})
    for row in rows:
        ctx = context(row)
        opt_bucket = bucket_int(row.get("option_count"), [1, 2, 5, 10])
        turn_bucket = bucket_int(row.get("turn"), [0, 1, 2, 4, 7])
        action_bucket = bucket_int(row.get("turn_action_count"), [1, 2, 4, 7])
        active_pair = f"{row.get('my_active_name')} vs {row.get('opp_active_name')}"
        chosen = f"{row.get('chosen_type_names')}:{row.get('chosen_card_names')}"
        top1 = f"{row.get('top1_type_name')}:{row.get('top1_card_name')}"

        if "main_choice" in views and row.get("context_name") == "MAIN":
            key = (
                f"turn={turn_bucket} tac={action_bucket} opts={opt_bucket} "
                f"avail={row.get('available_type_counts')} chosen={chosen}"
            )
            add_pattern(patterns, "main_choice", key, row)

        if "main_top1_gap" in views and row.get("context_name") == "MAIN" and row.get("top1_type_name"):
            key = (
                f"turn={turn_bucket} opts={opt_bucket} avail={row.get('available_type_counts')} "
                f"top1={top1} chosen={chosen} chosen_rank={row.get('chosen_first_rank')}"
            )
            add_pattern(patterns, "main_top1_gap", key, row)

        if "active_choice" in views:
            key = f"ctx={ctx} active={active_pair} avail={row.get('available_type_counts')} chosen={chosen}"
            add_pattern(patterns, "active_choice", key, row)

        if "multi_select" in views:
            try:
                max_count = int(float(row.get("max_count", 0) or 0))
                min_count = int(float(row.get("min_count", 0) or 0))
            except Exception:
                max_count = min_count = 0
            if max_count > min_count or int(float(row.get("chosen_len", 0) or 0)) > 1:
                key = (
                    f"ctx={ctx} opts={opt_bucket} min={min_count} max={max_count} "
                    f"chosen_len={row.get('chosen_len')} chosen={chosen}"
                )
                add_pattern(patterns, "multi_select", key, row)

        if "miss_flags" in views:
            for label, available_key, chosen_key, card_key in MISS_FLAGS:
                if truth(row, available_key) and not truth(row, chosen_key):
                    key = (
                        f"ctx={ctx} flag={label} turn={turn_bucket} tac={action_bucket} "
                        f"available={row.get(card_key)} chosen={chosen} active={active_pair}"
                    )
                    add_pattern(patterns, "miss_flags", key, row)

        if "miss_card_summary" in views:
            for label, available_key, chosen_key, card_key in MISS_FLAGS:
                if truth(row, available_key) and not truth(row, chosen_key):
                    key = f"ctx={ctx} flag={label} available={row.get(card_key)} chosen={chosen}"
                    add_pattern(patterns, "miss_card_summary", key, row)

        if "top_card" in views and row.get("top1_card_name"):
            key = f"ctx={ctx} top1={top1} chosen={chosen} active={active_pair}"
            add_pattern(patterns, "top_card", key, row)
    return patterns


def finalize_source(path: str, args: argparse.Namespace, views: set[str]) -> list[dict]:
    rows = read_rows(path)
    counts = game_counts(path, rows)
    decision_counts = Counter(row.get("outcome", "") for row in rows)
    patterns = build_patterns(rows, views)
    out = []
    src = clean_source(path)
    loss_games = max(counts.get("loss", 0), 1)
    win_games = max(counts.get("win", 0), 1)
    loss_decisions = max(decision_counts.get("loss", 0), 1)
    win_decisions = max(decision_counts.get("win", 0), 1)

    for (view, key), payload in patterns.items():
        c = payload["counts"]
        loss_n = int(c.get("loss", 0))
        win_n = int(c.get("win", 0))
        draw_n = int(c.get("draw", 0))
        if loss_n < args.min_loss:
            continue
        loss_pg = loss_n / loss_games
        win_pg = win_n / win_games
        delta_pg = loss_pg - win_pg
        loss_share = loss_n / loss_decisions
        win_share = win_n / win_decisions
        delta_share = loss_share - win_share
        if delta_pg < args.min_delta_per_game and delta_share < args.min_delta_share:
            continue
        priority = max(delta_pg, 0.0) * math.sqrt(max(loss_n, 1))
        ratio = loss_pg / max(win_pg, 1e-9)
        out.append({
            "source": src,
            "view": view,
            "key": key,
            "loss_n": loss_n,
            "win_n": win_n,
            "draw_n": draw_n,
            "loss_games": counts.get("loss", 0),
            "win_games": counts.get("win", 0),
            "loss_per_game": loss_pg,
            "win_per_game": win_pg,
            "delta_per_game": delta_pg,
            "loss_share": loss_share,
            "win_share": win_share,
            "delta_share": delta_share,
            "ratio_per_game": ratio,
            "priority": priority,
            "example_loss": payload.get("example_loss", ""),
            "example_win": payload.get("example_win", ""),
        })
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("decisions", nargs="+", help="trace .decisions.csv files or glob patterns")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--view", action="append", default=[],
                   choices=[
                       "main_choice",
                       "main_top1_gap",
                       "active_choice",
                       "multi_select",
                       "miss_flags",
                       "miss_card_summary",
                       "top_card",
                   ],
                   help="view to emit; repeatable. Defaults to all views.")
    p.add_argument("--min-loss", type=int, default=8)
    p.add_argument("--min-delta-per-game", type=float, default=0.05)
    p.add_argument("--min-delta-share", type=float, default=0.002)
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()

    paths: list[str] = []
    for spec in args.decisions:
        matches = sorted(glob.glob(spec))
        paths.extend(matches or [spec])
    paths = [path for path in paths if os.path.exists(path)]
    if not paths:
        raise SystemExit("no decision files found")

    views = set(args.view or [
        "main_choice",
        "main_top1_gap",
        "active_choice",
        "multi_select",
        "miss_flags",
        "miss_card_summary",
        "top_card",
    ])
    rows: list[dict] = []
    for path in paths:
        rows.extend(finalize_source(path, args, views))
    rows.sort(key=lambda r: (-float(r["priority"]), -int(r["loss_n"]), r["source"], r["view"], r["key"]))
    write_csv(args.out_csv, rows)

    print(f"Wrote {args.out_csv} rows={len(rows)} from files={len(paths)}")
    for row in rows[:args.top]:
        print(
            f"{float(row['priority']):8.3f} {row['source']} {row['view']} "
            f"loss={row['loss_n']} win={row['win_n']} "
            f"dpg={float(row['delta_per_game']):+.3f} key={row['key'][:220]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
