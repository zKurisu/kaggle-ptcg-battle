#!/usr/bin/env python3
"""Launch BC2 training jobs for a population of archetypes."""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import discover_npz_paths


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


def slugify(text: str) -> str:
    text = text.lower().replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "policy"


@dataclass
class Job:
    archetype: str
    save: Path
    log: Path
    acc_log: Path
    corpus_files: int = 0
    decisions: int = 0
    gpu: str | None = None
    proc: subprocess.Popen | None = None
    start_time: float = 0.0
    status: str = "pending"
    returncode: int | None = None


def score_tag(score_bands: list[str]) -> str:
    lows = []
    for band in score_bands:
        m = re.search(r"\d+", band)
        if m:
            lows.append(int(m.group(0)))
    return str(min(lows)) if lows else "all"


def build_train_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "tools/bc2_train.py",
        "--corpus", args.corpus,
        "--archetype", job.archetype,
        "--score-bands", *args.score_bands,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--width", str(args.width),
        "--device", "cuda:0" if job.gpu is not None else args.device,
        "--first-action-weight", str(args.first_action_weight),
        "--option-weight", str(args.option_weight),
        "--checkpoint-every", str(args.checkpoint_every),
        "--save", str(job.save),
    ]
    if args.include_empty:
        cmd.append("--include-empty")
    if args.legacy_state_pool:
        cmd.append("--legacy-state-pool")
    return cmd


def count_decisions(paths: list[str]) -> int:
    total = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            total += int(len(z["board"]))
    return total


def preflight_corpus(args: argparse.Namespace, archetype: str) -> tuple[int, int]:
    paths = discover_npz_paths(args.corpus, archetype, args.score_bands)
    return len(paths), count_decisions(paths) if paths else 0


def build_accuracy_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    return [
        sys.executable,
        "tools/bc2_accuracy.py",
        str(job.save),
        "--corpus", args.corpus,
        "--archetype", job.archetype,
        "--score-bands", *args.score_bands,
        "--max-samples", str(args.accuracy_samples),
        "--batch-size", str(args.accuracy_batch_size),
        "--progress-every", str(args.accuracy_progress_every),
        "--device", "cuda:0" if job.gpu is not None else args.device,
    ]


def launch_job(args: argparse.Namespace, job: Job, gpu: str | None) -> None:
    job.gpu = gpu
    job.start_time = time.time()
    job.status = "running"
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.save.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = build_train_cmd(args, job)
    with job.log.open("w") as f:
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        f.write("$ " + prefix + shlex.join(cmd) + "\n\n")
    log_f = job.log.open("a")
    job.proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    # Keep the file handle alive through the child process.
    job.proc._bc_log_handle = log_f  # type: ignore[attr-defined]


def finish_job(args: argparse.Namespace, job: Job) -> None:
    assert job.proc is not None
    handle = getattr(job.proc, "_bc_log_handle", None)
    if handle is not None:
        handle.close()
    job.returncode = job.proc.returncode
    job.status = "failed" if job.returncode else "done"
    elapsed = time.time() - job.start_time
    print(
        f"Finished {job.archetype}: status={job.status} rc={job.returncode} "
        f"elapsed={elapsed/60:.1f}m log={job.log}",
        flush=True,
    )
    if job.returncode or args.accuracy_samples <= 0:
        return

    cmd = build_accuracy_cmd(args, job)
    print(f"  accuracy {job.archetype}: log={job.acc_log}", flush=True)
    env = os.environ.copy()
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
    with job.acc_log.open("w") as f:
        prefix = f"CUDA_VISIBLE_DEVICES={job.gpu} " if job.gpu is not None else ""
        f.write("$ " + prefix + shlex.join(cmd) + "\n\n")
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        print(f"  accuracy failed {job.archetype}: rc={rc}", flush=True)
    else:
        print(f"  accuracy done {job.archetype}", flush=True)


def make_jobs(args: argparse.Namespace) -> list[Job]:
    tag = args.tag or f"{score_tag(args.score_bands)}_w{args.width:g}"
    jobs = []
    for arch in args.archetype:
        files, decisions = preflight_corpus(args, arch)
        if files == 0:
            print(f"Skip {arch}: no corpus files for bands={args.score_bands}", flush=True)
            continue
        if decisions < args.min_decisions:
            print(
                f"Skip {arch}: only {decisions} decisions in {files} files "
                f"(< --min-decisions {args.min_decisions})",
                flush=True,
            )
            continue
        slug = slugify(arch)
        save = Path(args.checkpoint_dir) / f"bc2_{slug}_{tag}.npz"
        log = Path(args.log_dir) / f"bc2_{slug}_{tag}.log"
        acc_log = Path(args.log_dir) / f"bc2_{slug}_{tag}_accuracy.log"
        if args.skip_existing and save.exists():
            print(f"Skip existing {arch}: {save}", flush=True)
            continue
        jobs.append(Job(arch, save, log, acc_log, corpus_files=files, decisions=decisions))
    return jobs


def print_status(pending: list[Job], running: list[Job], completed: list[Job], failed: list[Job]) -> None:
    now = time.time()
    parts = [
        f"pending={len(pending)}",
        f"running={len(running)}",
        f"done={len(completed)}",
        f"failed={len(failed)}",
    ]
    print("Status: " + " ".join(parts), flush=True)
    for job in running:
        elapsed = (now - job.start_time) / 60.0
        print(f"  gpu={job.gpu} {job.archetype} {elapsed:.1f}m log={job.log}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    p.add_argument("--archetype", action="append", default=[],
                   help="archetype to train; repeat. Defaults to common population.")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--min-decisions", type=int, default=20000,
                   help="skip archetypes with fewer available corpus decisions; 0 disables")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--device", default="cuda:0", help="used only when --gpus is empty")
    p.add_argument("--gpus", default="0,1,2,3", help="comma-separated physical GPU ids; empty for one local device")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--first-action-weight", type=float, default=1.5)
    p.add_argument("--option-weight", type=float, default=0.15)
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--legacy-state-pool", action="store_true",
                   help="use old pooled board encoder instead of slot-aware active/bench encoder")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--tag", default="", help="checkpoint/log suffix; default is SCORE_wWIDTH")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--accuracy-samples", type=int, default=50000, help="0 disables post-train accuracy")
    p.add_argument("--accuracy-batch-size", type=int, default=4096)
    p.add_argument("--accuracy-progress-every", type=int, default=5000)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.archetype:
        args.archetype = list(DEFAULT_ARCHETYPES)

    jobs = make_jobs(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = gpus * max(args.jobs_per_gpu, 1) if gpus else [None]
    if not slots:
        slots = [None]

    print(
        f"BC population: jobs={len(jobs)} slots={slots} corpus={args.corpus} "
        f"score_bands={args.score_bands} min_decisions={args.min_decisions}",
        flush=True,
    )
    for i, job in enumerate(jobs):
        slot = slots[i % len(slots)] if slots else None
        cmd = build_train_cmd(args, job)
        prefix = f"CUDA_VISIBLE_DEVICES={slot} " if slot is not None else ""
        print(
            f"  {job.archetype}: files={job.corpus_files} decisions={job.decisions} "
            f"save={job.save}",
            flush=True,
        )
        if args.dry_run:
            print("    " + prefix + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not jobs:
        print("No trainable jobs after corpus preflight", flush=True)
        return

    pending = list(jobs)
    running: list[Job] = []
    completed: list[Job] = []
    failed: list[Job] = []
    free_slots = list(slots)
    last_status = 0.0

    while pending or running:
        while pending and free_slots:
            gpu = free_slots.pop(0)
            job = pending.pop(0)
            launch_job(args, job, gpu)
            running.append(job)
            print(f"Launched {job.archetype} on gpu={gpu} log={job.log}", flush=True)

        still_running = []
        for job in running:
            assert job.proc is not None
            rc = job.proc.poll()
            if rc is None:
                still_running.append(job)
                continue
            finish_job(args, job)
            free_slots.append(job.gpu)
            if rc:
                failed.append(job)
            else:
                completed.append(job)
        running = still_running

        if time.time() - last_status >= args.poll_seconds or not running:
            print_status(pending, running, completed, failed)
            last_status = time.time()
        if pending or running:
            time.sleep(min(args.poll_seconds, 5.0))

    if failed:
        print("Failed jobs: " + ", ".join(job.archetype for job in failed), flush=True)
        sys.exit(1)
    print("All BC population jobs completed", flush=True)


if __name__ == "__main__":
    main()
