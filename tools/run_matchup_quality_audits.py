#!/usr/bin/env python3
"""Run paired win-quality audits for selected weak matchups."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return out or "unknown"


def parse_weak_pair(raw: str) -> tuple[str, str]:
    if "=>" in raw:
        left, right = raw.split("=>", 1)
    elif ":" in raw:
        left, right = raw.split(":", 1)
    else:
        raise ValueError(f"weak pair must use 'A=>B': {raw}")
    return left.strip(), right.strip()


def read_selected_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in args.weak_pair:
        pair = parse_weak_pair(raw)
        if pair not in seen:
            selected.append(pair)
            seen.add(pair)

    if args.pairs_csv:
        with open(args.pairs_csv, newline="") as f:
            for row in csv.DictReader(f):
                games = int(float(row.get("games", 0) or 0))
                wins = int(float(row.get("wins", 0) or 0))
                wr = float(row.get("game_wr", 0.0) or 0.0)
                if games < args.min_pair_games:
                    continue
                if wins < args.min_pair_wins:
                    continue
                if wr > args.max_pair_wr:
                    continue
                pair = (str(row["archetype"]), str(row["opponent_archetype"]))
                if pair not in seen:
                    selected.append(pair)
                    seen.add(pair)
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("no matchup pairs selected")
    return selected


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def build_targets(args: argparse.Namespace, archetype: str, opponent: str, out_csv: Path) -> None:
    if out_csv.exists() and not args.force:
        print(f"reuse {out_csv}", flush=True)
        return
    cmd = [
        sys.executable,
        "tools/build_trajectory_targets.py",
        "--corpus",
        args.corpus,
        "--archetype",
        archetype,
        "--opponent-archetype",
        opponent,
        "--out-csv",
        str(out_csv),
    ]
    if args.score_bands:
        cmd.extend(["--score-bands", *args.score_bands])
    cmd.extend(["--progress-every", str(args.progress_every)])
    run(cmd, dry_run=args.dry_run)


def audit_pair(
    args: argparse.Namespace,
    archetype: str,
    opponent: str,
    cand_csv: Path,
    opp_csv: Path,
    out_csv: Path,
    out_game_csv: Path,
) -> None:
    if out_csv.exists() and not args.force:
        print(f"reuse {out_csv}", flush=True)
        return
    cmd = [
        sys.executable,
        "tools/audit_teacher_win_quality.py",
        "--candidate-csv",
        str(cand_csv),
        "--opponent-csv",
        str(opp_csv),
        "--min-games",
        str(args.min_teacher_games),
        "--min-wins",
        str(args.min_teacher_wins),
        "--min-clean-wins",
        str(args.min_clean_wins),
        "--min-clean-share",
        str(args.min_clean_share),
        "--max-brick-share",
        str(args.max_brick_share),
        "--out-csv",
        str(out_csv),
        "--out-game-csv",
        str(out_game_csv),
    ]
    if args.top:
        cmd.extend(["--top", str(args.top)])
    run(cmd, dry_run=args.dry_run)


def merge_csv(out_path: Path, paths: list[Path]) -> None:
    rows: list[dict] = []
    fields: list[str] | None = None
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
    if not fields:
        return
    if "quality_score" in fields:
        rows.sort(
            key=lambda r: (
                str(r.get("archetype", "")),
                str(r.get("opponent_archetype", "")),
                -float(r.get("quality_score", 0.0) or 0.0),
            )
        )
    else:
        rows.sort(
            key=lambda r: (
                str(r.get("archetype", "")),
                str(r.get("opponent_archetype", "")),
                str(r.get("episode_id", "")),
                int(float(r.get("player_index", 0) or 0)),
            )
        )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--pairs-csv", default="")
    p.add_argument("--weak-pair", action="append", default=[])
    p.add_argument("--score-bands", nargs="*", default=[])
    p.add_argument("--max-pair-wr", type=float, default=0.45)
    p.add_argument("--min-pair-games", type=int, default=80)
    p.add_argument("--min-pair-wins", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-teacher-games", type=int, default=1)
    p.add_argument("--min-teacher-wins", type=int, default=1)
    p.add_argument("--min-clean-wins", type=int, default=5)
    p.add_argument("--min-clean-share", type=float, default=0.20)
    p.add_argument("--max-brick-share", type=float, default=0.55)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    pairs = read_selected_pairs(args)
    out_dir = Path(args.out_dir)
    target_dir = out_dir / "targets"
    quality_dir = out_dir / "quality"
    game_dir = out_dir / "game_quality"
    target_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)
    print(f"selected pairs={len(pairs)}", flush=True)

    quality_files: list[Path] = []
    game_files: list[Path] = []
    for idx, (arch, opp) in enumerate(pairs, 1):
        stem = f"{safe_name(arch)}__vs__{safe_name(opp)}"
        rev_stem = f"{safe_name(opp)}__vs__{safe_name(arch)}"
        cand_csv = target_dir / f"{stem}.csv"
        opp_csv = target_dir / f"{rev_stem}.csv"
        out_csv = quality_dir / f"{stem}.csv"
        out_game_csv = game_dir / f"{stem}.games.csv"
        print(f"[{idx}/{len(pairs)}] {arch} vs {opp}", flush=True)
        build_targets(args, arch, opp, cand_csv)
        build_targets(args, opp, arch, opp_csv)
        audit_pair(args, arch, opp, cand_csv, opp_csv, out_csv, out_game_csv)
        quality_files.append(out_csv)
        game_files.append(out_game_csv)

    if not args.dry_run:
        merge_csv(out_dir / "quality_all_pairs.csv", quality_files)
        merge_csv(out_dir / "game_quality_all_pairs.csv", game_files)


if __name__ == "__main__":
    main()
