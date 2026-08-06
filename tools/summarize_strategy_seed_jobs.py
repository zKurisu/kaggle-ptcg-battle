#!/usr/bin/env python3
"""Summarize seed-driven trace/rule-probe job results."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


MATCHUP_FIELDS = [
    "task_id",
    "seed_id",
    "archetype",
    "opponent_archetype",
    "intervention",
    "teacher_status",
    "candidate_name",
    "candidate_deck_sig",
    "opponent_name",
    "opponent_archetype_entry",
    "opponent_deck_sig",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "rule_mode",
    "rule_plain_wr",
    "rule_wr",
    "rule_delta",
    "top_gap_priority",
    "top_gap_view",
    "top_gap_key",
    "top_gap_loss_n",
    "top_gap_win_n",
    "top_gap_delta_per_game",
    "trace_prefix",
    "action_hint",
]

SEED_FIELDS = [
    "seed_id",
    "archetype",
    "opponent_archetype",
    "intervention",
    "teacher_status_counts",
    "tasks",
    "completed",
    "avg_wr",
    "min_wr",
    "max_wr",
    "weak_tasks",
    "strong_tasks",
    "worst_task_id",
    "worst_opponent",
    "worst_wr",
    "best_task_id",
    "best_opponent",
    "best_wr",
    "max_gap_priority",
    "max_gap_task_id",
    "max_gap_view",
    "max_gap_key",
    "rule_probe_count",
    "avg_rule_delta",
    "recommendation",
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fmt(value: float, digits: int = 6) -> str:
    if value != value:
        return ""
    return f"{value:.{digits}f}"


def clean_name(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def basename_prefix(trace_prefix: str) -> str:
    return os.path.basename(str(trace_prefix or "").rstrip("/"))


def game_counts(path: Path) -> tuple[int, int, int, int, float]:
    counts: Counter[str] = Counter()
    for row in read_rows(path):
        counts[str(row.get("outcome", ""))] += 1
    games = sum(counts.values())
    wins = counts.get("win", 0)
    losses = counts.get("loss", 0)
    draws = counts.get("draw", 0)
    wr = wins / games if games else math.nan
    return games, wins, losses, draws, wr


def top_gap(path: Path) -> dict:
    rows = read_rows(path)
    if not rows:
        return {}
    rows.sort(key=lambda r: fnum(r.get("priority"), -1.0), reverse=True)
    return rows[0]


def parse_rule_probe(path: Path, opponent_name: str, rule_mode: str) -> tuple[float, float, float]:
    if not path.exists() or not rule_mode:
        return math.nan, math.nan, math.nan
    opponent_key = clean_name(opponent_name)
    plain_wr = math.nan
    rule_wr = math.nan
    for row in read_rows(path):
        row_name = clean_name(row.get("row", ""))
        col_name = clean_name(row.get("column", ""))
        if col_name != opponent_key:
            continue
        wr = fnum(row.get("row_win_rate"))
        if row_name.endswith("_plain"):
            plain_wr = wr
        elif row_name.endswith(f"_{clean_name(rule_mode)}"):
            rule_wr = wr
    delta = rule_wr - plain_wr if plain_wr == plain_wr and rule_wr == rule_wr else math.nan
    return plain_wr, rule_wr, delta


def action_hint(row: dict, args: argparse.Namespace) -> str:
    wr = fnum(row.get("win_rate"))
    teacher_status = row.get("teacher_status", "")
    intervention = row.get("intervention", "")
    rule_delta = fnum(row.get("rule_delta"))
    top_key = row.get("top_gap_key", "")
    if rule_delta == rule_delta and rule_delta < args.min_rule_delta:
        return "rule_probe_weak__prefer_teacher_or_data"
    if wr == wr and wr < args.weak_wr and teacher_status == "needs_teacher_policy":
        return "construct_success_rollouts"
    if wr == wr and wr < args.weak_wr and teacher_status == "trace_then_matchup_bc":
        return "build_matchup_conditioned_bc_after_gap_review"
    if wr == wr and wr < args.weak_wr and teacher_status == "needs_rule_implementation":
        return "implement_narrow_rule_then_probe"
    if wr == wr and wr < args.weak_wr:
        return "structural_weakness_trace_first"
    if "Crustle" in top_key and "ATTACK:" in top_key:
        return "inspect_crustle_wall_loop"
    if wr == wr and wr >= args.strong_wr:
        return "low_priority_or_validation_opponent"
    if intervention == "rerank_guard":
        return "review_gap_before_rule"
    return "review_trace_gap"


def summarize_seed(seed_id: str, rows: list[dict], args: argparse.Namespace) -> dict:
    completed = [r for r in rows if str(r.get("games", ""))]
    wrs = [fnum(r.get("win_rate")) for r in completed]
    wrs = [x for x in wrs if x == x]
    weak = [r for r in completed if fnum(r.get("win_rate"), 1.0) < args.weak_wr]
    strong = [r for r in completed if fnum(r.get("win_rate"), 0.0) >= args.strong_wr]
    worst = min(completed, key=lambda r: fnum(r.get("win_rate"), 1.0), default={})
    best = max(completed, key=lambda r: fnum(r.get("win_rate"), -1.0), default={})
    gap_owner = max(completed, key=lambda r: fnum(r.get("top_gap_priority"), -1.0), default={})
    statuses = Counter(r.get("teacher_status", "") for r in rows)
    rule_deltas = [fnum(r.get("rule_delta")) for r in rows]
    rule_deltas = [x for x in rule_deltas if x == x]

    recommendation = "review_trace_gap"
    avg_wr = sum(wrs) / len(wrs) if wrs else math.nan
    if weak and any(r.get("teacher_status") == "needs_teacher_policy" for r in weak):
        recommendation = "prioritize_teacher_rollout_success_data"
    elif weak and any(r.get("teacher_status") == "trace_then_matchup_bc" for r in weak):
        recommendation = "prioritize_matchup_conditioned_bc"
    elif weak and any(r.get("teacher_status") == "needs_rule_implementation" for r in weak):
        recommendation = "implement_narrow_rule_probe"
    elif rule_deltas and max(rule_deltas) < args.min_rule_delta:
        recommendation = "do_not_scale_current_rule"
    elif strong and len(strong) == len(completed):
        recommendation = "low_priority_for_fix_use_as_validator"

    first = rows[0] if rows else {}
    return {
        "seed_id": seed_id,
        "archetype": first.get("archetype", ""),
        "opponent_archetype": first.get("opponent_archetype", ""),
        "intervention": first.get("intervention", ""),
        "teacher_status_counts": " ".join(f"{k}:{v}" for k, v in statuses.items()),
        "tasks": len(rows),
        "completed": len(completed),
        "avg_wr": fmt(avg_wr),
        "min_wr": fmt(min(wrs) if wrs else math.nan),
        "max_wr": fmt(max(wrs) if wrs else math.nan),
        "weak_tasks": len(weak),
        "strong_tasks": len(strong),
        "worst_task_id": worst.get("task_id", ""),
        "worst_opponent": worst.get("opponent_name", ""),
        "worst_wr": worst.get("win_rate", ""),
        "best_task_id": best.get("task_id", ""),
        "best_opponent": best.get("opponent_name", ""),
        "best_wr": best.get("win_rate", ""),
        "max_gap_priority": gap_owner.get("top_gap_priority", ""),
        "max_gap_task_id": gap_owner.get("task_id", ""),
        "max_gap_view": gap_owner.get("top_gap_view", ""),
        "max_gap_key": gap_owner.get("top_gap_key", ""),
        "rule_probe_count": len(rule_deltas),
        "avg_rule_delta": fmt(sum(rule_deltas) / len(rule_deltas) if rule_deltas else math.nan),
        "recommendation": recommendation,
    }


def write_markdown(path: Path, seed_rows: list[dict], matchup_rows: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# Strategy Seed Job Summary",
        "",
        f"Job dir: `{args.job_dir}`",
        f"Weak WR threshold: `{args.weak_wr}`",
        "",
        "## Seed Summary",
        "",
        "| seed | completed | avg_wr | min_wr | max_wr | recommendation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in seed_rows:
        lines.append(
            f"| `{row['seed_id']}` | {row['completed']}/{row['tasks']} | "
            f"{row['avg_wr']} | {row['min_wr']} | {row['max_wr']} | {row['recommendation']} |"
        )
    lines.extend(["", "## Weakest Matchups", ""])
    weak = sorted(
        [r for r in matchup_rows if fnum(r.get("win_rate")) == fnum(r.get("win_rate"))],
        key=lambda r: fnum(r.get("win_rate")),
    )[: args.top]
    for row in weak:
        lines.append(
            f"- `{row['seed_id']}` `{row['candidate_deck_sig']}` vs "
            f"`{row['opponent_archetype_entry']}:{row['opponent_deck_sig']}` "
            f"WR={row['win_rate']} action={row['action_hint']}"
        )
        if row.get("top_gap_key"):
            lines.append(f"  top_gap: {row['top_gap_view']} `{row['top_gap_key'][:220]}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-dir", required=True)
    p.add_argument("--out-prefix", default="")
    p.add_argument("--weak-wr", type=float, default=0.35)
    p.add_argument("--strong-wr", type=float, default=0.85)
    p.add_argument("--min-rule-delta", type=float, default=0.05)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    job_dir = Path(args.job_dir)
    out_prefix = Path(args.out_prefix) if args.out_prefix else job_dir / "strategy_seed_summary"
    tasks = read_rows(job_dir / "strategy_seed_tasks.csv")
    matchup_rows: list[dict] = []

    for task in tasks:
        prefix = Path(task.get("trace_prefix", ""))
        if not prefix.is_absolute():
            prefix = Path(str(prefix))
        name = basename_prefix(task.get("trace_prefix", ""))
        if not prefix.exists():
            prefix = job_dir / "trace" / name
        games, wins, losses, draws, wr = game_counts(prefix.with_suffix(".games.csv"))
        gap = top_gap(prefix.with_suffix(".gap.csv"))
        rule_plain, rule_wr, rule_delta = parse_rule_probe(
            prefix.with_suffix(".rule_probe.csv"),
            task.get("opponent_name", ""),
            task.get("rule_mode", ""),
        )
        row = {
            **task,
            "games": games or "",
            "wins": wins if games else "",
            "losses": losses if games else "",
            "draws": draws if games else "",
            "win_rate": fmt(wr),
            "rule_plain_wr": fmt(rule_plain),
            "rule_wr": fmt(rule_wr),
            "rule_delta": fmt(rule_delta),
            "top_gap_priority": fmt(fnum(gap.get("priority"))),
            "top_gap_view": gap.get("view", ""),
            "top_gap_key": gap.get("key", ""),
            "top_gap_loss_n": gap.get("loss_n", ""),
            "top_gap_win_n": gap.get("win_n", ""),
            "top_gap_delta_per_game": fmt(fnum(gap.get("delta_per_game"))),
        }
        row["action_hint"] = action_hint(row, args)
        matchup_rows.append(row)

    by_seed: dict[str, list[dict]] = defaultdict(list)
    for row in matchup_rows:
        by_seed[row.get("seed_id", "")].append(row)
    seed_rows = [summarize_seed(seed, rows, args) for seed, rows in sorted(by_seed.items())]
    seed_rows.sort(key=lambda r: (fnum(r.get("min_wr"), 1.0), r.get("seed_id", "")))

    write_csv(out_prefix.with_name(out_prefix.name + "_matchups.csv"), MATCHUP_FIELDS, matchup_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_seeds.csv"), SEED_FIELDS, seed_rows)
    write_markdown(out_prefix.with_name(out_prefix.name + ".md"), seed_rows, matchup_rows, args)

    print(
        f"Wrote {out_prefix.with_name(out_prefix.name + '_matchups.csv')}, "
        f"{out_prefix.with_name(out_prefix.name + '_seeds.csv')}, "
        f"{out_prefix.with_name(out_prefix.name + '.md')}",
        flush=True,
    )
    for row in seed_rows[: args.top]:
        print(
            f"{row['seed_id']:38s} avg={row['avg_wr']} min={row['min_wr']} "
            f"weak={row['weak_tasks']}/{row['completed']} {row['recommendation']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
