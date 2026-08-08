#!/usr/bin/env python3
"""Generate subset/training commands for weak-matchup clean teacher specialists."""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
from pathlib import Path


def clean_name(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "entry"


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if row.get(name):
            return str(row[name]).strip().replace("_", " ")
    return ""


def load_weak_pairs(path: str, limit: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = first_present(row, ("cand_arch", "archetype", "candidate_archetype", "row_archetype", "row"))
            opp = first_present(row, ("opp_arch", "opponent_archetype", "target_archetype", "column_archetype", "column", "opponent"))
            if not arch or not opp:
                continue
            key = (arch, opp)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if limit and len(out) >= limit:
                break
    return out


def load_clean_teacher_counts(path: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], set[str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("clean_win", "0")).strip() not in ("1", "1.0", "true", "True"):
                continue
            arch = first_present(row, ("archetype", "candidate_archetype"))
            opp = first_present(row, ("opponent_archetype", "target_archetype"))
            game_key = str(row.get("game_key", "")).strip()
            if not game_key and row.get("episode_id") and row.get("player_index"):
                game_key = f"{row['episode_id']}:{row['player_index']}"
            if not arch or not opp or not game_key:
                continue
            counts.setdefault((arch, opp), set()).add(game_key)
    return {k: len(v) for k, v in counts.items()}


def q(value: str) -> str:
    return shlex.quote(str(value))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--weak-pairs-csv", required=True)
    p.add_argument("--teacher-games-csv", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--out-corpus", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-sh", required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--width", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--cuda-memory-gb", type=float, default=24.0)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--min-clean-games", type=int, default=5,
                   help="skip weak pairs whose teacher CSV has fewer clean-win games")
    args = p.parse_args()

    pairs = load_weak_pairs(args.weak_pairs_csv, args.limit)
    clean_counts = load_clean_teacher_counts(args.teacher_games_csv)
    script = Path(args.out_sh)
    script.parent.mkdir(parents=True, exist_ok=True)
    out_corpus = Path(args.out_corpus)
    ckpt_dir = Path(args.checkpoint_dir)

    rows = []
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    score_bands = " ".join(q(x) for x in args.score_bands)
    with script.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"mkdir -p {q(str(out_corpus))} {q(str(ckpt_dir))}\n")
        train_cmds: list[str] = []
        for idx, (arch, opp) in enumerate(pairs, 1):
            clean_games = int(clean_counts.get((arch, opp), 0))
            if clean_games < args.min_clean_games:
                rows.append({
                    "idx": str(idx),
                    "archetype": arch,
                    "opponent_archetype": opp,
                    "status": "skipped_low_clean_games",
                    "clean_games": str(clean_games),
                })
                continue
            stem = f"{clean_name(arch)}_vs_{clean_name(opp)}"
            band = f"clean_{clean_name(opp)}"
            subset_name = f"{stem}_clean_teacher"
            log_subset = f"logs/pair_teacher_pipeline/{stem}_subset.log"
            f.write("mkdir -p logs/pair_teacher_pipeline\n")
            f.write(" ".join([
                "python3", "tools/build_bc_subset.py",
                "--corpus", q(args.corpus),
                "--archetype", q(arch),
                "--score-bands", score_bands,
                "--game-key-csv", q(args.teacher_games_csv),
                "--where", q(f"archetype={arch}"),
                "--where", q(f"opponent_archetype={opp}"),
                "--where", q("clean_win=1"),
                "--out", q(str(out_corpus)),
                "--out-band", q(band),
                "--name", q(subset_name),
                "2>&1", "|", "tee", q(log_subset),
            ]) + "\n")
            if args.train:
                gpu = gpu_ids[(idx - 1) % max(len(gpu_ids), 1)] if gpu_ids else "0"
                ckpt = ckpt_dir / f"bc2_{stem}_clean_teacher_w{args.width:g}.npz"
                log_train = f"logs/pair_teacher_pipeline/{stem}_train_w{args.width:g}.log"
                train_cmds.append(" ".join([
                    f"CUDA_VISIBLE_DEVICES={q(gpu)}",
                    "python3", "tools/bc2_train.py",
                    "--corpus", q(str(out_corpus)),
                    "--archetype", q(arch),
                    "--score-bands", q(band),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--width", str(args.width),
                    "--arch", "cross_attn",
                    "--state-layers", "2",
                    "--history-k", "8",
                    "--log-history-k", "32",
                    "--board-history-k", "4",
                    "--hierarchical-plan",
                    "--step-plan",
                    "--step-plan-loss-weight", "0.6",
                    "--step-plan-teacher-forcing", "0.8",
                    "--first-action-weight", "2.0",
                    "--set-loss-weight", "0.25",
                    "--multi-select-weight", "2.0",
                    "--cuda-memory-gb", str(args.cuda_memory_gb),
                    "--device", "cuda:0",
                    "--split-by-game",
                    "--save", q(str(ckpt)),
                    "2>&1", "|", "tee", q(log_train),
                ]))
                rows.append({
                    "idx": str(idx),
                    "archetype": arch,
                    "opponent_archetype": opp,
                    "subset_band": band,
                    "subset_name": subset_name,
                    "checkpoint_path": str(ckpt),
                    "log_path": log_train,
                    "gpu": gpu,
                    "clean_games": str(clean_games),
                    "status": "planned_train",
                })
            else:
                rows.append({
                    "idx": str(idx),
                    "archetype": arch,
                    "opponent_archetype": opp,
                    "subset_band": band,
                    "subset_name": subset_name,
                    "clean_games": str(clean_games),
                    "status": "planned_subset",
                })
        if train_cmds:
            f.write("\n")
            f.write("cat > /tmp/pair_teacher_train_commands.txt <<'CMDS'\n")
            for cmd in train_cmds:
                f.write(cmd + "\n")
            f.write("CMDS\n")
            f.write(f"xargs -P {max(1, args.max_parallel)} -I CMD bash -lc CMD < /tmp/pair_teacher_train_commands.txt\n")

    os.chmod(script, 0o755)
    manifest = script.with_suffix(".manifest.csv")
    fieldnames = sorted({k for row in rows for k in row})
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {script} pairs={len(pairs)} train={args.train} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
