#!/usr/bin/env python3
"""Plan adaptive date windows for BC deck-signature training.

The BC corpus is partitioned by archetype/score band/date, but specialist
training usually filters again by deck signature. This tool scans the extracted
corpus, counts usable rows per day for each deck signature, and chooses the
newest suffix of dates that reaches a target row count. Sparse signatures get
older dates added; huge signatures keep only recent data.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2.data import discover_npz_paths


DEFAULT_ARCHETYPES = [
    "Marnie Grimmsnarl",
    "Alakazam",
    "Crustle Wall",
    "Mega Lucario",
    "Mega Abomasnow",
    "Mega Starmie",
    "Archaludon",
    "Hop Trevenant",
    "Team Rocket Mewtwo",
    "Teal Mask Ogerpon",
    "Mega Lopunny",
    "Dragapult",
    "Festival Lead",
    "Cynthia Garchomp",
]

DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

FIELDS = [
    "count_mode",
    "archetype",
    "rank",
    "deck_sig",
    "date_from",
    "date_to",
    "train_date_args",
    "status",
    "target_decisions",
    "min_decisions",
    "max_decisions",
    "selected_kept",
    "selected_raw",
    "selected_share",
    "total_kept",
    "total_raw",
    "total_episodes",
    "n_dates",
    "first_date",
    "last_date",
    "decision_win_rate",
    "decision_draw_rate",
    "avg_score",
    "max_score",
    "top_team",
    "teams",
    "estimated_rss_gb",
    "date_kept",
    "date_raw",
]


def date_from_path(path: str | Path) -> str:
    m = DATE_RE.search(str(path))
    return m.group(1) if m else ""


def label_status(data: dict[str, np.ndarray], i: int, include_empty: bool) -> str:
    action = np.asarray(data["action"][i], dtype=np.int64)
    n_opt = len(data["ot"][i])
    mn = int(data["min_c"][i])
    mx = int(data["max_c"][i])
    if len(action) == 0:
        return "keep" if include_empty and mn == 0 else "empty"
    if len(action) < mn or len(action) > mx:
        return "bad"
    if len(set(action.tolist())) != len(action):
        return "bad"
    if not ((action >= 0) & (action < n_opt)).all():
        return "bad"
    return "keep"


def parse_int_specs(specs: list[str], *, what: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{what} must be NAME=INT, got {spec!r}")
        name, value = spec.split("=", 1)
        name = name.strip().lower()
        if not name:
            raise ValueError(f"{what} has empty name: {spec!r}")
        out[name] = int(value)
    return out


def split_sigs(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[\s,]+", str(text or "")) if x.strip()]


def read_requested_sigs(path: str) -> dict[str, set[str]]:
    if not path:
        return {}
    requested: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = str(row.get("archetype") or "").strip().lower()
            sigs = split_sigs(row.get("deck_sigs") or row.get("deck_sig") or "")
            if not sigs:
                continue
            requested[arch].update(sigs)
    return dict(requested)


def format_counts(counter: Counter[str]) -> str:
    return " ".join(f"{day}:{int(counter[day])}" for day in sorted(counter))


@dataclass
class SigStats:
    raw_by_date: Counter[str] = field(default_factory=Counter)
    kept_by_date: Counter[str] = field(default_factory=Counter)
    episodes: set[str] = field(default_factory=set)
    teams: Counter[str] = field(default_factory=Counter)
    wins: int = 0
    draws: int = 0
    score_sum: float = 0.0
    score_n: int = 0
    max_score: float = 0.0

    @property
    def total_raw(self) -> int:
        return int(sum(self.raw_by_date.values()))

    @property
    def total_kept(self) -> int:
        return int(sum(self.kept_by_date.values()))


def row_matches_request(archetype: str, sig: str, requested: dict[str, set[str]], deck_sigs: set[str]) -> bool:
    if deck_sigs and sig not in deck_sigs:
        return False
    if not requested:
        return True
    arch_key = archetype.lower()
    if sig in requested.get(arch_key, set()):
        return True
    if sig in requested.get("", set()):
        return True
    return False


def scan(args: argparse.Namespace) -> dict[tuple[str, str], SigStats]:
    requested = read_requested_sigs(args.input_csv)
    deck_sigs = {str(x).strip() for x in args.deck_sig if str(x).strip()}
    archetypes = args.archetype or DEFAULT_ARCHETYPES
    stats: dict[tuple[str, str], SigStats] = defaultdict(SigStats)
    file_count = 0
    row_count = 0
    kept_count = 0
    t0 = time.time()

    for arch in archetypes:
        if requested:
            arch_key = arch.lower()
            if arch_key not in requested and "" not in requested:
                continue
        paths = discover_npz_paths(
            args.corpus,
            arch,
            args.score_bands,
            date_from=args.scan_date_from,
            date_to=args.scan_date_to,
        )
        for path in paths:
            file_count += 1
            day = date_from_path(path)
            with np.load(path, allow_pickle=True) as z:
                if "deck_sig" not in z.files:
                    raise ValueError(f"{path} lacks deck_sig")
                if args.count_mode == "raw" and not args.raw_metadata:
                    sig_arr = np.asarray(z["deck_sig"]).astype(str)
                    n = len(sig_arr)
                    row_count += n
                    unique_sigs, counts = np.unique(sig_arr, return_counts=True)
                    for sig, count in zip(unique_sigs.tolist(), counts.tolist()):
                        if not row_matches_request(arch, str(sig), requested, deck_sigs):
                            continue
                        row = stats[(arch, str(sig))]
                        row.raw_by_date[day] += int(count)
                        row.kept_by_date[day] += int(count)
                        kept_count += int(count)
                    if args.progress_every_files and (
                        file_count == 1 or file_count % args.progress_every_files == 0
                    ):
                        elapsed = max(time.time() - t0, 1e-9)
                        print(
                            f"  files={file_count} rows={row_count} kept={kept_count} "
                            f"sigs={len(stats)} rate={row_count/elapsed:.0f} rows/s",
                            flush=True,
                        )
                    continue
                data = {k: z[k] for k in z.files}
            n = len(data["board"])
            row_count += n
            for i in range(n):
                sig = str(data["deck_sig"][i])
                if not row_matches_request(arch, sig, requested, deck_sigs):
                    continue
                row = stats[(arch, sig)]
                row.raw_by_date[day] += 1
                if "episode_id" in data:
                    row.episodes.add(str(data["episode_id"][i]))
                if "team_name" in data:
                    team = str(data["team_name"][i])
                    if team:
                        row.teams[team] += 1
                if "won" in data:
                    row.wins += int(data["won"][i])
                if "draw" in data:
                    row.draws += int(data["draw"][i])
                if "score" in data:
                    score = float(data["score"][i])
                    row.score_sum += score
                    row.score_n += 1
                    row.max_score = max(row.max_score, score)
                if args.count_mode == "raw" or label_status(data, i, args.include_empty) == "keep":
                    row.kept_by_date[day] += 1
                    kept_count += 1
            if args.progress_every_files and (
                file_count == 1 or file_count % args.progress_every_files == 0
            ):
                elapsed = max(time.time() - t0, 1e-9)
                print(
                    f"  files={file_count} rows={row_count} kept={kept_count} "
                    f"sigs={len(stats)} rate={row_count/elapsed:.0f} rows/s",
                    flush=True,
                )
    return stats


def choose_value(archetype: str, sig: str, default: int, by_arch: dict[str, int], by_sig: dict[str, int]) -> int:
    if sig in by_sig:
        return int(by_sig[sig])
    return int(by_arch.get(archetype.lower(), default))


def select_window(
    kept_by_date: Counter[str],
    raw_by_date: Counter[str],
    *,
    target: int,
    min_decisions: int,
    max_decisions: int,
) -> tuple[str, int, int, str]:
    total_kept = int(sum(kept_by_date.values()))
    if total_kept == 0:
        return "", 0, 0, "empty"
    dates = sorted(kept_by_date.keys(), reverse=True)
    selected_kept = 0
    selected_raw = 0
    date_from = dates[-1]
    for day in dates:
        selected_kept += int(kept_by_date[day])
        selected_raw += int(raw_by_date[day])
        date_from = day
        if target <= 0 or selected_kept >= target:
            break
    if total_kept < min_decisions:
        status = "low_data_all_dates"
        date_from = dates[-1]
        selected_kept = total_kept
        selected_raw = int(sum(raw_by_date.values()))
    elif selected_kept < target:
        status = "below_target_all_dates"
    elif max_decisions > 0 and selected_kept > max_decisions:
        status = "above_max_recent_bucket"
    else:
        status = "ok"
    return date_from, selected_kept, selected_raw, status


def make_rows(args: argparse.Namespace, stats: dict[tuple[str, str], SigStats]) -> list[dict]:
    target_by_arch = parse_int_specs(args.target_by_archetype, what="--target-by-archetype")
    target_by_sig = parse_int_specs(args.target_by_deck_sig, what="--target-by-deck-sig")
    min_by_arch = parse_int_specs(args.min_by_archetype, what="--min-by-archetype")
    min_by_sig = parse_int_specs(args.min_by_deck_sig, what="--min-by-deck-sig")
    max_by_arch = parse_int_specs(args.max_by_archetype, what="--max-by-archetype")
    max_by_sig = parse_int_specs(args.max_by_deck_sig, what="--max-by-deck-sig")

    grouped: dict[str, list[tuple[str, SigStats]]] = defaultdict(list)
    for (arch, sig), row in stats.items():
        if row.total_kept < args.min_total_decisions:
            continue
        grouped[arch].append((sig, row))

    rows: list[dict] = []
    for arch in sorted(grouped):
        sig_rows = sorted(grouped[arch], key=lambda x: (x[1].total_kept, x[1].total_raw), reverse=True)
        if args.top_per_archetype > 0:
            sig_rows = sig_rows[: args.top_per_archetype]
        for rank, (sig, row) in enumerate(sig_rows, 1):
            target = choose_value(arch, sig, args.target_decisions, target_by_arch, target_by_sig)
            min_decisions = choose_value(arch, sig, args.min_decisions, min_by_arch, min_by_sig)
            max_decisions = choose_value(arch, sig, args.max_decisions, max_by_arch, max_by_sig)
            date_from, selected_kept, selected_raw, status = select_window(
                row.kept_by_date,
                row.raw_by_date,
                target=target,
                min_decisions=min_decisions,
                max_decisions=max_decisions,
            )
            total_kept = row.total_kept
            total_raw = row.total_raw
            dates = sorted(row.kept_by_date)
            avg_score = row.score_sum / max(row.score_n, 1)
            win_rate = row.wins / max(total_raw, 1)
            draw_rate = row.draws / max(total_raw, 1)
            top_team = row.teams.most_common(1)[0][0] if row.teams else ""
            date_to = args.date_to or ""
            train_date_args = ""
            if date_from:
                train_date_args += f"--date-from {date_from}"
            if date_to:
                train_date_args += (" " if train_date_args else "") + f"--date-to {date_to}"
            rows.append(
                {
                    "count_mode": args.count_mode,
                    "archetype": arch,
                    "rank": rank,
                    "deck_sig": sig,
                    "date_from": date_from,
                    "date_to": date_to,
                    "train_date_args": train_date_args,
                    "status": status,
                    "target_decisions": target,
                    "min_decisions": min_decisions,
                    "max_decisions": max_decisions,
                    "selected_kept": selected_kept,
                    "selected_raw": selected_raw,
                    "selected_share": selected_kept / max(total_kept, 1),
                    "total_kept": total_kept,
                    "total_raw": total_raw,
                    "total_episodes": len(row.episodes),
                    "n_dates": len(dates),
                    "first_date": dates[0] if dates else "",
                    "last_date": dates[-1] if dates else "",
                    "decision_win_rate": win_rate,
                    "decision_draw_rate": draw_rate,
                    "avg_score": avg_score,
                    "max_score": row.max_score,
                    "top_team": top_team,
                    "teams": len(row.teams),
                    "estimated_rss_gb": selected_kept * args.rss_gb_per_100k / 100000.0,
                    "date_kept": format_counts(row.kept_by_date),
                    "date_raw": format_counts(row.raw_by_date),
                }
            )
    return rows


def write_rows(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            out_row = dict(row)
            for key in ("selected_share", "decision_win_rate", "decision_draw_rate"):
                out_row[key] = f"{float(out_row[key]):.6f}"
            for key in ("avg_score", "max_score", "estimated_rss_gb"):
                out_row[key] = f"{float(out_row[key]):.4f}"
            w.writerow(out_row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", action="append", default=[],
                   help="repeatable; omitted scans common archetypes")
    p.add_argument("--score-bands", nargs="+",
                   default=["1200+", "1100-1199", "1000-1099", "900-999"])
    p.add_argument("--deck-sig", action="append", default=[],
                   help="optional global deck-signature filter; repeatable")
    p.add_argument("--input-csv", default="",
                   help="optional CSV containing archetype and deck_sig/deck_sigs columns to restrict planning")
    p.add_argument("--scan-date-from", default="",
                   help="optional earliest date to scan before planning")
    p.add_argument("--scan-date-to", default="",
                   help="optional latest date to scan before planning")
    p.add_argument("--date-to", default="",
                   help="emit this date_to in train args; normally left empty")
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--count-mode", choices=["kept", "raw"], default="kept",
                   help="kept validates BC labels; raw is much faster and is good enough for broad window planning")
    p.add_argument("--raw-metadata", action="store_true",
                   help="with --count-mode raw, still scan rows for team/score/outcome metadata; slower")
    p.add_argument("--min-total-decisions", type=int, default=1000,
                   help="drop sigs with fewer kept rows across all scanned dates")
    p.add_argument("--top-per-archetype", type=int, default=0,
                   help="keep only the largest N deck signatures per archetype; 0 keeps all")
    p.add_argument("--target-decisions", type=int, default=160000,
                   help="newest date suffix should reach at least this many kept rows")
    p.add_argument("--min-decisions", type=int, default=80000,
                   help="flag as low_data_all_dates below this many all-date kept rows")
    p.add_argument("--max-decisions", type=int, default=500000,
                   help="flag selected suffix above this many kept rows; 0 disables")
    p.add_argument("--target-by-archetype", action="append", default=[],
                   help="override target, e.g. 'Alakazam=350000'; repeatable")
    p.add_argument("--target-by-deck-sig", action="append", default=[],
                   help="override target, e.g. 7f9a538936e3=350000")
    p.add_argument("--min-by-archetype", action="append", default=[])
    p.add_argument("--min-by-deck-sig", action="append", default=[])
    p.add_argument("--max-by-archetype", action="append", default=[])
    p.add_argument("--max-by-deck-sig", action="append", default=[])
    p.add_argument("--rss-gb-per-100k", type=float, default=1.5,
                   help="rough resident-memory estimate after compact row storage")
    p.add_argument("--progress-every-files", type=int, default=20)
    p.add_argument("--out-csv", default="logs/bc_date_windows.csv")
    args = p.parse_args()

    stats = scan(args)
    rows = make_rows(args, stats)
    write_rows(args.out_csv, rows)
    print(f"Wrote {args.out_csv}: {len(rows)} rows")
    for row in rows[: min(len(rows), 40)]:
        print(
            f"  {row['archetype']:<22} sig={row['deck_sig'][:12]} "
            f"rank={row['rank']:<2} date_from={row['date_from'] or 'all':<10} "
            f"selected={int(row['selected_kept']):7d}/{int(row['total_kept']):7d} "
            f"target={int(row['target_decisions']):7d} status={row['status']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
