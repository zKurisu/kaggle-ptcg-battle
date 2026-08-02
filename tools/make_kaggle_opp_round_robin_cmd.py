#!/usr/bin/env python3
"""Build an eval_round_robin.py command against cached Kaggle opponent decks."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def deck_files(dirs: list[str], limit: int) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        for path in sorted(Path(d).glob("*.csv")):
            sig = path.name.split("_", 2)[1] if path.name.startswith("opp_") else path.stem
            if sig in seen:
                continue
            seen.add(sig)
            files.append(path)
            if limit and len(files) >= limit:
                return files
    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-name", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--opp-dir", action="append", required=True,
                   help="directory containing opponent deck CSV files; repeatable")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-csv", default="logs/round_robin_vs_kaggle_opp.csv")
    args = p.parse_args()

    files = deck_files(args.opp_dir, args.limit)
    if not files:
        raise SystemExit("no opponent deck CSV files found")

    cmd = [
        "python3", "tools/eval_round_robin.py",
        "--entry", f"{args.policy_name}={args.policy}:{args.deck}",
    ]
    for i, path in enumerate(files, 1):
        name = path.stem
        if len(name) > 42:
            name = f"opp{i:02d}_{path.stem.split('_')[1] if '_' in path.stem else path.stem[:12]}"
        cmd.extend(["--entry", f"{name}=random:{path}"])
    cmd.extend([
        "--games", str(args.games),
        "--progress-every", str(args.progress_every),
        "--out-csv", args.out_csv,
    ])
    print(" ".join(shlex.quote(str(x)) for x in cmd))


if __name__ == "__main__":
    main()
