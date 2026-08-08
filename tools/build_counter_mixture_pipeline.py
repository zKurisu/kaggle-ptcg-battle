#!/usr/bin/env python3
"""Generate aggressive counter-mixture training scripts by archetype.

The generated models are intended to be submit-capable single policies:

* base corpus is still used as an anchor;
* selected weak-matchup losses and non-clean wins are removed from the base pass;
* clean teacher wins for those weak matchups are mixed back as repeated aux data;
* a trajectory plan head learns explicit counter_clean/setup/tempo labels.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
from collections import defaultdict
from pathlib import Path


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return value or "entry"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value.replace("_", " ")
    return ""


def load_weak_pairs(path: str, limit: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = first_present(row, ("cand_arch", "archetype", "candidate_archetype", "row"))
            opp = first_present(row, ("opp_arch", "opponent_archetype", "target_archetype", "column"))
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


def load_clean_counts(path: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("clean_win", "")).strip() not in {"1", "1.0", "true", "True"}:
                continue
            arch = first_present(row, ("archetype", "candidate_archetype"))
            opp = first_present(row, ("opponent_archetype", "target_archetype"))
            key = str(row.get("game_key", "")).strip()
            if not key and row.get("episode_id") and row.get("player_index"):
                key = f"{row['episode_id']}:{row['player_index']}"
            if arch and opp and key:
                counts[(arch, opp)].add(key)
    return {k: len(v) for k, v in counts.items()}


def select_pairs(args: argparse.Namespace) -> dict[str, list[tuple[str, int]]]:
    wanted_arch = {x.lower() for x in args.archetype}
    pairs = load_weak_pairs(args.weak_pairs_csv, args.limit_pairs)
    clean_counts = load_clean_counts(args.teacher_games_csv)
    by_arch: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for arch, opp in pairs:
        if wanted_arch and arch.lower() not in wanted_arch:
            continue
        clean = int(clean_counts.get((arch, opp), 0))
        if clean < args.min_clean_games:
            continue
        by_arch[arch].append((opp, clean))
    for arch in list(by_arch):
        by_arch[arch] = sorted(by_arch[arch], key=lambda x: x[1], reverse=True)
        if args.max_opponents_per_arch > 0:
            by_arch[arch] = by_arch[arch][: args.max_opponents_per_arch]
    return dict(sorted(by_arch.items()))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--weak-pairs-csv", required=True)
    p.add_argument("--teacher-games-csv", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099", "900-999"])
    p.add_argument("--limit-pairs", type=int, default=80)
    p.add_argument("--min-clean-games", type=int, default=10)
    p.add_argument("--max-opponents-per-arch", type=int, default=4)
    p.add_argument("--archetype", action="append", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subset-corpus", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-sh", required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=3072)
    p.add_argument("--width", type=float, default=4.0)
    p.add_argument("--state-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.2e-4)
    p.add_argument("--aux-repeat", type=int, default=24)
    p.add_argument("--cuda-memory-gb", type=float, default=24.0)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--dirty-win-policy", choices=["keep", "drop_nonclean"], default="drop_nonclean")
    args = p.parse_args()

    by_arch = select_pairs(args)
    if not by_arch:
        raise RuntimeError("no archetypes selected; check clean teacher counts and filters")

    out_dir = Path(args.out_dir)
    subset_root = Path(args.subset_corpus)
    ckpt_dir = Path(args.checkpoint_dir)
    script = Path(args.out_sh)
    script.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    score_bands = " ".join(q(x) for x in args.score_bands)
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]

    rows: list[dict[str, str]] = []
    train_cmds: list[tuple[str, str, str]] = []
    with script.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"mkdir -p {q(out_dir)} {q(subset_root)} {q(ckpt_dir)} logs/counter_mixture\n")
        for arch_i, (arch, opps) in enumerate(by_arch.items(), 1):
            arch_slug = slug(arch)
            traj_csv = out_dir / f"{arch_slug}_trajectory_games.csv"
            filter_csv = out_dir / f"{arch_slug}_counter_filter.csv"
            f.write("\n")
            f.write(f"# {arch}: {', '.join(f'{opp}({clean})' for opp, clean in opps)}\n")
            f.write(" ".join([
                "python3", "tools/build_trajectory_targets.py",
                "--corpus", q(args.corpus),
                "--archetype", q(arch),
                "--score-bands", score_bands,
                "--out-csv", q(traj_csv),
                "--progress-every", "4",
                "2>&1", "|", "tee", q(f"logs/counter_mixture/{arch_slug}_trajectory.log"),
            ]) + "\n")
            filter_cmd = [
                "python3", "tools/build_counter_filter_csv.py",
                "--trajectory-csv", q(traj_csv),
                "--teacher-games-csv", q(args.teacher_games_csv),
                "--archetype", q(arch),
                "--out-csv", q(filter_csv),
                "--dirty-win-policy", args.dirty_win_policy,
            ]
            for opp, _clean in opps:
                filter_cmd.extend(["--opponent-archetype", q(opp)])
            f.write(" ".join(filter_cmd) + "\n")

            aux_bands: list[str] = []
            pair_names: list[str] = []
            for opp, clean in opps:
                opp_slug = slug(opp)
                stem = f"{arch_slug}_vs_{opp_slug}"
                band = f"clean_{opp_slug}"
                name = f"{stem}_clean_teacher"
                aux_bands.append(band)
                pair_names.append(f"{opp}:{clean}")
                f.write(" ".join([
                    "python3", "tools/build_bc_subset.py",
                    "--corpus", q(args.corpus),
                    "--archetype", q(arch),
                    "--score-bands", score_bands,
                    "--game-key-csv", q(args.teacher_games_csv),
                    "--where", q(f"archetype={arch}"),
                    "--where", q(f"opponent_archetype={opp}"),
                    "--where", q("clean_win=1"),
                    "--out", q(subset_root),
                    "--out-band", q(band),
                    "--name", q(name),
                    "2>&1", "|", "tee", q(f"logs/counter_mixture/{stem}_subset.log"),
                ]) + "\n")

            ckpt = ckpt_dir / f"bc2_{arch_slug}_counter_mix_w{args.width:g}.npz"
            log = f"logs/counter_mixture/{arch_slug}_counter_mix_w{args.width:g}.log"
            if args.train:
                gpu = gpu_ids[(arch_i - 1) % max(len(gpu_ids), 1)] if gpu_ids else "0"
                aux_band_args = " ".join(q(x) for x in aux_bands)
                cmd = " ".join([
                    "python3", "tools/bc2_train.py",
                    "--corpus", q(args.corpus),
                    "--archetype", q(arch),
                    "--score-bands", score_bands,
                    "--aux-corpus", q(subset_root),
                    "--aux-score-bands", aux_band_args,
                    "--aux-repeat", str(args.aux_repeat),
                    "--trajectory-csv", q(filter_csv),
                    "--trajectory-drop", "counter_bad",
                    "--trajectory-filter-missing-policy", "keep",
                    "--trajectory-weight", "counter_clean=32",
                    "--trajectory-weight", "strategy_success=2.0",
                    "--trajectory-weight", "setup_success=1.4",
                    "--trajectory-weight", "tempo_success=1.4",
                    "--trajectory-weight", "no_early_end=1.2",
                    "--trajectory-weight-cap", "96",
                    "--trajectory-target", "counter_clean",
                    "--trajectory-target", "strategy_success",
                    "--trajectory-target", "setup_success",
                    "--trajectory-target", "tempo_success",
                    "--trajectory-target", "no_early_end",
                    "--trajectory-target-loss-weight", "0.8",
                    "--hierarchical-plan",
                    "--step-plan",
                    "--step-plan-loss-weight", "0.7",
                    "--step-plan-teacher-forcing", "0.85",
                    "--arch", "cross_attn",
                    "--state-layers", str(args.state_layers),
                    "--history-k", "8",
                    "--log-history-k", "32",
                    "--board-history-k", "4",
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--width", str(args.width),
                    "--win-weight", "1.8",
                    "--loss-weight", "0.15",
                    "--draw-weight", "0.6",
                    "--first-action-weight", "2.2",
                    "--set-loss-weight", "0.35",
                    "--multi-select-weight", "2.5",
                    "--split-by-game",
                    "--cuda-memory-gb", str(args.cuda_memory_gb),
                    "--device", "cuda:0",
                    "--save", q(ckpt),
                    "2>&1", "|", "tee", q(log),
                ])
                train_cmds.append((arch_slug, gpu, cmd))
                rows.append({
                    "archetype": arch,
                    "opponents": ";".join(pair_names),
                    "checkpoint_path": str(ckpt),
                    "log_path": log,
                    "filter_csv": str(filter_csv),
                    "trajectory_csv": str(traj_csv),
                    "aux_bands": ";".join(aux_bands),
                    "gpu": gpu,
                    "status": "planned_train",
                })
            else:
                rows.append({
                    "archetype": arch,
                    "opponents": ";".join(pair_names),
                    "checkpoint_path": str(ckpt),
                    "filter_csv": str(filter_csv),
                    "trajectory_csv": str(traj_csv),
                    "aux_bands": ";".join(aux_bands),
                    "status": "planned_inputs",
                })

        if train_cmds:
            f.write("\n")
            f.write("job_dir=/tmp/counter_mixture_train_jobs\n")
            f.write("mkdir -p \"$job_dir\"\n")
            f.write(": > /tmp/counter_mixture_train_jobs.list\n")
            for job_i, (stem, gpu, cmd) in enumerate(train_cmds, 1):
                job_path = f"/tmp/counter_mixture_train_jobs/{job_i:03d}_{slug(stem)}.sh"
                f.write(f"cat > {q(job_path)} <<'JOB'\n")
                f.write("#!/usr/bin/env bash\n")
                f.write("set -euo pipefail\n")
                f.write(f"export CUDA_VISIBLE_DEVICES={q(gpu)}\n")
                f.write("export PTCG_DISABLE_CUDNN=1\n")
                f.write(cmd + "\n")
                f.write("JOB\n")
                f.write(f"chmod +x {q(job_path)}\n")
                f.write(f"printf '%s\\n' {q(job_path)} >> /tmp/counter_mixture_train_jobs.list\n")
            f.write(
                f"xargs -P {max(1, args.max_parallel)} -n 1 bash "
                "< /tmp/counter_mixture_train_jobs.list\n"
            )

    os.chmod(script, 0o755)
    manifest = script.with_suffix(".manifest.csv")
    write_manifest(manifest, rows)
    print(
        f"wrote {script} archetypes={len(by_arch)} train={args.train} "
        f"manifest={manifest}",
        flush=True,
    )
    for row in rows:
        print(f"{row['archetype']}: {row['opponents']}", flush=True)


if __name__ == "__main__":
    main()
