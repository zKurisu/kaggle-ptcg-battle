#!/usr/bin/env python3
"""Plan weak-matchup rollout-teacher jobs from RR or baseline-delta CSVs."""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tools.plan_strategy_seed_jobs import (
    ManifestEntry,
    clean_entry_name,
    entry_rest,
    normalize_arch,
    read_manifest_entries,
    shell_cmd,
    slugify,
)
from ptcg_rl.rule_overlay import RULE_MODES


PLAN_FIELDS = [
    "task_id",
    "source",
    "format",
    "metric",
    "win_rate",
    "candidate",
    "candidate_archetype",
    "candidate_deck_sig",
    "candidate_entry",
    "opponent",
    "opponent_archetype",
    "opponent_deck_sig",
    "opponent_entry",
    "rule_mode",
    "actor_scope",
    "games",
    "workers",
    "out_root",
    "out_band",
    "tag",
    "summary_csv",
    "log_path",
    "command",
]

SKIP_FIELDS = [
    "source",
    "format",
    "candidate",
    "opponent",
    "reason",
    "details",
]


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def entry_maps(entries: list[ManifestEntry]) -> dict[str, ManifestEntry]:
    out: dict[str, ManifestEntry] = {}
    for entry in entries:
        keys = {
            entry.name,
            clean_entry_name(entry.name),
            entry.team_name,
            clean_entry_name(entry.team_name),
            entry.deck_sig,
        }
        if entry.eval_entry and "=" in entry.eval_entry:
            raw = entry.eval_entry.split("=", 1)[0]
            keys.add(raw)
            keys.add(clean_entry_name(raw))
        for key in keys:
            key = str(key or "").strip()
            if key and key not in out:
                out[key] = entry
    return out


def find_entry(name: str, mapping: dict[str, ManifestEntry]) -> ManifestEntry | None:
    keys = [name, clean_entry_name(name)]
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def infer_format(path: str, rows: list[dict]) -> str:
    if not rows:
        return "empty"
    fields = set(rows[0].keys())
    if {"row", "column", "row_win_rate"}.issubset(fields):
        return "round_robin"
    if {"candidate", "opponent", "candidate_wr"}.issubset(fields):
        return "baseline_delta"
    return "unknown"


def weakness_rows(path: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows = read_csv(path)
    fmt = infer_format(path, rows)
    out: list[dict] = []
    skipped: list[dict] = []
    if fmt == "round_robin":
        for row in rows:
            cand = str(row.get("row", "")).strip()
            opp = str(row.get("column", "")).strip()
            if args.ignore_random and normalize_arch(opp) == "random":
                continue
            wr = fnum(row.get("row_win_rate"), 1.0)
            if wr > args.max_win_rate:
                continue
            out.append({
                "source": path,
                "format": fmt,
                "candidate": cand,
                "opponent": opp,
                "metric": "row_win_rate",
                "win_rate": wr,
            })
    elif fmt == "baseline_delta":
        for row in rows:
            cand = str(row.get("candidate", "")).strip()
            opp = str(row.get("opponent", "")).strip()
            wr = fnum(row.get("candidate_wr"), 1.0)
            if wr > args.max_win_rate:
                continue
            out.append({
                "source": path,
                "format": fmt,
                "candidate": cand,
                "opponent": opp,
                "metric": "candidate_wr",
                "win_rate": wr,
            })
    else:
        skipped.append({
            "source": path,
            "format": fmt,
            "candidate": "",
            "opponent": "",
            "reason": "unsupported_format",
            "details": "expected RR fields row,column,row_win_rate or delta fields candidate,opponent,candidate_wr",
        })
    return out, skipped


def dedupe_keep_hardest(rows: list[dict]) -> list[dict]:
    by_pair: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (clean_entry_name(row["candidate"]), clean_entry_name(row["opponent"]))
        prev = by_pair.get(key)
        if prev is None or float(row["win_rate"]) < float(prev["win_rate"]):
            by_pair[key] = row
    return sorted(by_pair.values(), key=lambda r: (float(r["win_rate"]), r["candidate"], r["opponent"]))


def enforce_coverage(rows: list[dict], args: argparse.Namespace,
                     candidate_map: dict[str, ManifestEntry],
                     opponent_map: dict[str, ManifestEntry]) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    skipped: list[dict] = []
    by_arch_count: defaultdict[str, int] = defaultdict(int)
    by_candidate_count: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        cand = find_entry(row["candidate"], candidate_map)
        opp = find_entry(row["opponent"], opponent_map)
        if cand is None:
            skipped.append({**row, "reason": "missing_candidate_manifest_entry", "details": row["candidate"]})
            continue
        if opp is None:
            skipped.append({**row, "reason": "missing_opponent_manifest_entry", "details": row["opponent"]})
            continue
        arch = cand.archetype or "unknown"
        if args.max_per_archetype and by_arch_count[arch] >= args.max_per_archetype:
            skipped.append({**row, "reason": "max_per_archetype", "details": arch})
            continue
        if args.max_per_candidate and by_candidate_count[cand.name] >= args.max_per_candidate:
            skipped.append({**row, "reason": "max_per_candidate", "details": cand.name})
            continue
        row = dict(row)
        row["_candidate_entry"] = cand
        row["_opponent_entry"] = opp
        selected.append(row)
        by_arch_count[arch] += 1
        by_candidate_count[cand.name] += 1
        if args.max_jobs and len(selected) >= args.max_jobs:
            break
    return selected, skipped


def task_tag(candidate: ManifestEntry, opponent: ManifestEntry, index: int) -> str:
    return slugify(f"{index:03d}_{candidate.name}_vs_{opponent.name}", 110)


def make_command(args: argparse.Namespace, row: dict, index: int) -> dict:
    cand: ManifestEntry = row["_candidate_entry"]
    opp: ManifestEntry = row["_opponent_entry"]
    tag = task_tag(cand, opp, index)
    summary_csv = str(Path(args.log_dir) / f"{tag}.csv")
    log_path = str(Path(args.log_dir) / f"{tag}.log")
    parts = [
        args.python,
        "tools/generate_rollout_bc.py",
        "--candidate",
        f"{clean_entry_name(cand.name)}={entry_rest(cand.eval_entry)}",
        "--opponent",
        f"{clean_entry_name(opp.name)}={entry_rest(opp.eval_entry)}",
        "--actor-scope",
        args.actor_scope,
        "--games",
        str(args.games),
        "--workers",
        str(args.workers),
        "--max-turns",
        str(args.max_turns),
        "--keep-outcomes",
        args.keep_outcomes,
        "--epsilon-random",
        str(args.epsilon_random),
        "--worker-progress-every",
        str(args.worker_progress_every),
        "--flush-every-games",
        str(args.flush_every_games),
        "--out-root",
        args.out_root,
        "--out-band",
        args.out_band,
        "--tag",
        tag,
        "--summary-csv",
        summary_csv,
    ]
    for actor in args.actor:
        parts.extend(["--actor", actor])
    cmd = shell_cmd(parts)
    return {
        "task_id": tag,
        "source": row["source"],
        "format": row["format"],
        "metric": row["metric"],
        "win_rate": f"{float(row['win_rate']):.6f}",
        "candidate": cand.name,
        "candidate_archetype": cand.archetype,
        "candidate_deck_sig": cand.deck_sig,
        "candidate_entry": cand.eval_entry,
        "opponent": opp.name,
        "opponent_archetype": opp.archetype,
        "opponent_deck_sig": opp.deck_sig,
        "opponent_entry": opp.eval_entry,
        "rule_mode": args.rule_mode,
        "actor_scope": args.actor_scope,
        "games": args.games,
        "workers": args.workers,
        "out_root": args.out_root,
        "out_band": args.out_band,
        "tag": tag,
        "summary_csv": summary_csv,
        "log_path": log_path,
        "command": cmd,
    }


def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_shell(path: str, plan_rows: list[dict], args: argparse.Namespace) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(args.log_dir)} {shlex.quote(args.out_root)}",
        f"parallel_jobs={int(args.parallel_jobs)}",
        "running=0",
        "",
        "launch_job() {",
        "  local name=\"$1\"",
        "  local log=\"$2\"",
        "  shift 2",
        "  echo \"[$(date '+%F %T')] launch ${name}\"",
        "  \"$@\" >\"${log}\" 2>&1 &",
        "  running=$((running + 1))",
        "  if (( running >= parallel_jobs )); then",
        "    wait -n",
        "    running=$((running - 1))",
        "  fi",
        "}",
        "",
    ]
    for row in plan_rows:
        cmd_parts = shlex.split(row["command"])
        cmd = " ".join(shlex.quote(x) for x in cmd_parts)
        lines.append(f"launch_job {shlex.quote(row['task_id'])} {shlex.quote(row['log_path'])} {cmd}")
    lines.extend([
        "wait",
        f"{shlex.quote(args.python)} tools/summarize_rollout_bc.py "
        f"--summary-glob {shlex.quote(str(Path(args.log_dir) / '*.csv'))} "
        f"--corpus {shlex.quote(args.out_root)} "
        f"--out-csv {shlex.quote(str(Path(args.log_dir) / 'rollout_teacher_summary.csv'))}",
        "",
    ])
    out.write_text("\n".join(lines) + "\n")
    out.chmod(0o755)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weakness-csv", action="append", required=True,
                   help="RR CSV or baseline-delta CSV; repeatable")
    p.add_argument("--candidate-manifest", action="append", required=True)
    p.add_argument("--opponent-manifest", action="append", default=[])
    p.add_argument("--max-win-rate", type=float, default=0.45)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--max-per-archetype", type=int, default=4)
    p.add_argument("--max-per-candidate", type=int, default=2)
    p.add_argument("--ignore-random", action="store_true", default=True)
    p.add_argument("--games", type=int, default=600)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--parallel-jobs", type=int, default=6)
    p.add_argument("--max-turns", type=int, default=420)
    p.add_argument("--keep-outcomes", choices=["win", "nonloss", "all"], default="win")
    p.add_argument("--epsilon-random", type=float, default=0.01)
    p.add_argument("--actor-scope", choices=["game", "decision"], default="game")
    p.add_argument("--rule-mode", default="targeted")
    p.add_argument("--actor", action="append", default=None)
    p.add_argument("--worker-progress-every", type=int, default=10)
    p.add_argument("--flush-every-games", type=int, default=20)
    p.add_argument("--python", default="python3")
    p.add_argument("--out-root", default="data/generated_rollout_bc_teacher")
    p.add_argument("--out-band", default="weak_win_search")
    p.add_argument("--log-dir", default="logs/rollout_teacher")
    p.add_argument("--out-csv", default="logs/rollout_teacher/rollout_teacher_plan.csv")
    p.add_argument("--skipped-csv", default="logs/rollout_teacher/rollout_teacher_skipped.csv")
    p.add_argument("--out-sh", default="logs/rollout_teacher/run_rollout_teacher_jobs.sh")
    args = p.parse_args()
    if args.rule_mode not in RULE_MODES:
        p.error(f"--rule-mode must be one of: {', '.join(RULE_MODES)}")
    if args.actor is None:
        rm = args.rule_mode
        args.actor = [
            f"greedy+rules:{rm}=1",
            f"topk@2+rules:{rm}=2",
            f"topk@3+rules:{rm}=2",
            f"sample@1.25+rules:{rm}=1",
            f"sample@1.75+rules:{rm}=1",
            f"random+rules:{rm}=0.08",
        ]

    candidate_entries = read_manifest_entries(args.candidate_manifest, sort_by="weight")
    opponent_entries = read_manifest_entries(args.opponent_manifest or args.candidate_manifest, sort_by="weight")
    candidate_map = entry_maps(candidate_entries)
    opponent_map = entry_maps(opponent_entries)

    raw: list[dict] = []
    skipped: list[dict] = []
    for path in args.weakness_csv:
        rows, skips = weakness_rows(path, args)
        raw.extend(rows)
        skipped.extend(skips)
    raw = dedupe_keep_hardest(raw)
    selected, more_skips = enforce_coverage(raw, args, candidate_map, opponent_map)
    skipped.extend(more_skips)
    plan_rows = [make_command(args, row, i) for i, row in enumerate(selected, 1)]

    write_csv(args.out_csv, PLAN_FIELDS, plan_rows)
    write_csv(args.skipped_csv, SKIP_FIELDS, skipped)
    write_shell(args.out_sh, plan_rows, args)

    by_arch = defaultdict(int)
    for row in plan_rows:
        by_arch[row["candidate_archetype"]] += 1
    print(f"Wrote {args.out_csv} jobs={len(plan_rows)} skipped={len(skipped)}")
    print(f"Wrote {args.out_sh}")
    for arch, n in sorted(by_arch.items()):
        print(f"  {arch or 'unknown'}: {n}")
    for row in plan_rows[:20]:
        print(
            f"{row['win_rate']} {row['candidate']} -> {row['opponent']} "
            f"({row['candidate_archetype']} vs {row['opponent_archetype']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
