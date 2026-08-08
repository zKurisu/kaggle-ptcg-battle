#!/usr/bin/env python3
"""Run shadow BC training commands from a build_shadow_pool manifest."""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    rank: int
    name: str
    archetype: str
    checkpoint: Path
    train_cmd: list[str]
    log: Path
    gpu: str | None = None
    proc: subprocess.Popen | None = None
    start_time: float = 0.0
    returncode: int | None = None


def _read_jobs(args: argparse.Namespace) -> list[Job]:
    wanted_arch = {x.lower() for x in args.archetype}
    date_windows = _read_date_windows(args.date_window_csv)
    jobs: list[Job] = []
    queued_checkpoints: set[Path] = set()
    with open(args.manifest, newline="") as f:
        for raw in csv.DictReader(f):
            if args.limit and len(jobs) >= args.limit:
                break
            archetype = str(raw.get("archetype") or "")
            if wanted_arch and archetype.lower() not in wanted_arch:
                continue
            cmd_text = str(raw.get("train_cmd") or "").strip()
            checkpoint_text = str(raw.get("checkpoint_path") or "").strip()
            if not cmd_text or not checkpoint_text:
                continue
            checkpoint = Path(checkpoint_text)
            if checkpoint in queued_checkpoints:
                print(f"Skip duplicate checkpoint {raw.get('shadow_name')}: {checkpoint}", flush=True)
                continue
            if args.skip_existing and checkpoint.exists():
                print(f"Skip existing {raw.get('shadow_name')}: {checkpoint}", flush=True)
                continue
            queued_checkpoints.add(checkpoint)
            name = str(raw.get("shadow_name") or checkpoint.stem)
            rank = int(raw.get("rank") or len(jobs) + 1)
            log = Path(args.log_dir) / f"{rank:03d}_{name}.log"
            train_cmd = shlex.split(cmd_text)
            if args.cuda_memory_gb > 0:
                train_cmd.extend(["--cuda-memory-gb", str(args.cuda_memory_gb)])
            if args.cuda_memory_fraction > 0:
                train_cmd.extend(["--cuda-memory-fraction", str(args.cuda_memory_fraction)])
            if args.batch_size > 0:
                train_cmd = _replace_or_append(train_cmd, "--batch-size", str(args.batch_size))
            train_cmd = _apply_date_window(train_cmd, raw, date_windows)
            jobs.append(Job(rank, name, archetype, checkpoint, train_cmd, log))
    return jobs


def _read_date_windows(path: str) -> dict[tuple[str, str], tuple[str, str]]:
    if not path:
        return {}
    windows: dict[tuple[str, str], tuple[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = str(row.get("archetype") or "").strip().lower()
            sig = str(row.get("deck_sig") or "").strip()
            date_from = str(row.get("date_from") or "").strip()
            date_to = str(row.get("date_to") or "").strip()
            if arch and sig and (date_from or date_to):
                windows[(arch, sig)] = (date_from, date_to)
    return windows


def _apply_date_window(
    cmd: list[str],
    manifest_row: dict[str, str],
    windows: dict[tuple[str, str], tuple[str, str]],
) -> list[str]:
    if not windows:
        return cmd
    archetype = str(manifest_row.get("archetype") or "").strip().lower()
    deck_sig = str(manifest_row.get("deck_sig") or "").strip()
    if not deck_sig:
        deck_sig = _cmd_value(cmd, "--deck-sig")
    date_from, date_to = windows.get((archetype, deck_sig), ("", ""))
    out = list(cmd)
    if date_from:
        out = _replace_or_append(out, "--date-from", date_from)
    if date_to:
        out = _replace_or_append(out, "--date-to", date_to)
    return out


def _cmd_value(cmd: list[str], flag: str) -> str:
    for i, token in enumerate(cmd):
        if token == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return ""


def _replace_or_append(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    for i, token in enumerate(out):
        if token == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
    out.extend([flag, value])
    return out


def _launch(job: Job, gpu: str | None) -> None:
    job.gpu = gpu
    job.start_time = time.time()
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = ""
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} "
    with job.log.open("w") as f:
        f.write("$ " + prefix + shlex.join(job.train_cmd) + "\n\n")
    log_f = job.log.open("a")
    job.proc = subprocess.Popen(job.train_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    job.proc._bc_log_handle = log_f  # type: ignore[attr-defined]


def _finish(job: Job) -> None:
    assert job.proc is not None
    handle = getattr(job.proc, "_bc_log_handle", None)
    if handle is not None:
        handle.close()
    job.returncode = job.proc.returncode
    elapsed = (time.time() - job.start_time) / 60.0
    status = "done" if job.returncode == 0 else "failed"
    print(
        f"Finished rank={job.rank} {job.name}: {status} rc={job.returncode} "
        f"gpu={job.gpu} elapsed={elapsed:.1f}m log={job.log}",
        flush=True,
    )


def _print_status(pending: list[Job], running: list[Job], done: list[Job], failed: list[Job]) -> None:
    print(
        f"Status: pending={len(pending)} running={len(running)} "
        f"done={len(done)} failed={len(failed)}",
        flush=True,
    )
    now = time.time()
    for job in running:
        elapsed = (now - job.start_time) / 60.0
        print(f"  gpu={job.gpu} rank={job.rank} {job.name} {elapsed:.1f}m log={job.log}", flush=True)
    for job in failed[-8:]:
        print(
            f"  failed rank={job.rank} {job.name} rc={job.returncode} "
            f"log={job.log}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--archetype", action="append", default=[],
                   help="optional exact archetype filter; repeatable")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gpus", default="0,1,2,3",
                   help="comma-separated physical GPU ids; empty runs without CUDA_VISIBLE_DEVICES")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--cuda-memory-gb", type=float, default=0.0,
                   help="append an approximate GiB CUDA allocator cap to manifest commands")
    p.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                   help="append a visible-GPU CUDA allocator fraction cap to manifest commands")
    p.add_argument("--batch-size", type=int, default=0,
                   help="override --batch-size in manifest commands; 0 keeps manifest value")
    p.add_argument("--date-window-csv", default="",
                   help="CSV from tools/plan_bc_date_windows.py; matched by archetype+deck_sig and appended as --date-from/--date-to")
    p.add_argument("--log-dir", default="logs/shadow_train")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = _read_jobs(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots: list[str | None] = gpus * max(args.jobs_per_gpu, 1) if gpus else [None]

    print(f"Shadow manifest train: jobs={len(jobs)} slots={slots} manifest={args.manifest}", flush=True)
    for i, job in enumerate(jobs):
        gpu = slots[i % len(slots)]
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        print(
            f"  rank={job.rank} {job.name}: archetype={job.archetype} "
            f"save={job.checkpoint} log={job.log}",
            flush=True,
        )
        if args.dry_run:
            print("    " + prefix + shlex.join(job.train_cmd), flush=True)
    if args.dry_run or not jobs:
        return

    pending = list(jobs)
    running: list[Job] = []
    done: list[Job] = []
    failed: list[Job] = []
    free_slots = list(slots)
    last_status = 0.0
    while pending or running:
        while pending and free_slots:
            gpu = free_slots.pop(0)
            job = pending.pop(0)
            _launch(job, gpu)
            running.append(job)
            print(f"Launched rank={job.rank} {job.name} on gpu={gpu} log={job.log}", flush=True)

        still_running: list[Job] = []
        for job in running:
            assert job.proc is not None
            rc = job.proc.poll()
            if rc is None:
                still_running.append(job)
                continue
            _finish(job)
            free_slots.append(job.gpu)
            if rc:
                failed.append(job)
            else:
                done.append(job)
        running = still_running

        if time.time() - last_status >= args.poll_seconds or not running:
            _print_status(pending, running, done, failed)
            last_status = time.time()
        if pending or running:
            time.sleep(min(args.poll_seconds, 5.0))

    if failed:
        print("Failed jobs: " + ", ".join(job.name for job in failed), flush=True)
        sys.exit(1)
    print("All shadow jobs completed", flush=True)


if __name__ == "__main__":
    main()
