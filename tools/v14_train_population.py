#!/usr/bin/env python3
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
    name: str
    archetype: str
    deck_sig: str
    team_name: str
    score_bands: str
    out: str
    deck_path: str = ""
    gpu: str = ""
    proc: subprocess.Popen | None = None
    log_path: str = ""
    start_time: float = 0.0
    status: str = "pending"


def clean_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def read_jobs(path: str, *, default_out_dir: str, default_score_bands: str) -> list[Job]:
    jobs: list[Job] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = (row.get("archetype") or row.get("arch") or "").strip()
            if not arch:
                continue
            deck_sig = (row.get("deck_sig") or "").strip()
            team = (row.get("team_name") or "").strip()
            name = (row.get("name") or row.get("job") or "").strip()
            if not name:
                parts = [clean_name(arch), deck_sig[:8] if deck_sig else "", clean_name(team)[:24] if team else ""]
                name = "_".join(x for x in parts if x)
            out = (row.get("out") or "").strip()
            if not out:
                out = str(Path(default_out_dir) / f"{name}.pt")
            bands = (row.get("score_bands") or default_score_bands).strip()
            deck_path = (row.get("deck_path") or row.get("deck") or row.get("deck_file") or "").strip()
            jobs.append(Job(
                name=name,
                archetype=arch,
                deck_sig=deck_sig,
                team_name=team,
                score_bands=bands,
                out=out,
                deck_path=deck_path,
            ))
    if not jobs:
        raise ValueError(f"manifest has no jobs: {path}")
    return jobs


def build_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    cmd = [
        sys.executable,
        "tools/v14_train_sequence_policy.py",
        "--corpus",
        args.corpus,
        "--archetype",
        job.archetype,
        "--score-bands",
        *job.score_bands.split(),
        "--seq-len",
        str(args.seq_len),
        "--stride",
        str(args.stride),
        "--width",
        str(args.width),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--win-weight",
        str(args.win_weight),
        "--loss-weight",
        str(args.loss_weight),
        "--draw-weight",
        str(args.draw_weight),
        "--progress-every",
        str(args.progress_every),
        "--action-weight",
        str(args.action_weight),
        "--current-action-weight",
        str(args.current_action_weight),
        "--prefix-action-weight",
        str(args.prefix_action_weight),
        "--order-weight",
        str(args.order_weight),
        "--multi-weight",
        str(args.multi_weight),
        "--count-weight",
        str(args.count_weight),
        "--plan-weight",
        str(args.plan_weight),
        "--next-type-weight",
        str(args.next_type_weight),
        "--dca-plan-weight",
        str(args.dca_plan_weight),
        "--known-action-weight",
        str(args.known_action_weight),
        "--turn-plan-weight",
        str(args.turn_plan_weight),
        "--turn-terminal-weight",
        str(args.turn_terminal_weight),
        "--turn-next-plan-weight",
        str(args.turn_next_plan_weight),
        "--turn-next-type-weight",
        str(args.turn_next_type_weight),
        "--turn-next-card-weight",
        str(args.turn_next_card_weight),
        "--turn-next-attack-weight",
        str(args.turn_next_attack_weight),
        "--turn-next-context-weight",
        str(args.turn_next_context_weight),
        "--opportunity-type-weight",
        str(args.opportunity_type_weight),
        "--opportunity-margin-weight",
        str(args.opportunity_margin_weight),
        "--opportunity-margin",
        str(args.opportunity_margin),
        "--current-rank-margin-weight",
        str(args.current_rank_margin_weight),
        "--current-rank-margin",
        str(args.current_rank_margin),
        "--current-rank-margin-min-options",
        str(args.current_rank_margin_min_options),
        "--history-condition-scale",
        str(args.history_condition_scale),
        "--plan-condition-scale",
        str(args.plan_condition_scale),
        "--next-type-condition-scale",
        str(args.next_type_condition_scale),
        "--dca-condition-scale",
        str(args.dca_condition_scale),
        "--known-condition-scale",
        str(args.known_condition_scale),
        "--known-logit-scale",
        str(args.known_logit_scale),
        "--turn-condition-scale",
        str(args.turn_condition_scale),
        "--turn-next-condition-scale",
        str(args.turn_next_condition_scale),
        "--type-prior-scale",
        str(args.type_prior_scale),
        "--current-complexity-weight",
        str(args.current_complexity_weight),
        "--multi-target-weight",
        str(args.multi_target_weight),
        "--damage-counter-weight",
        str(args.damage_counter_weight),
        "--device",
        "cuda" if job.gpu else args.device,
        "--out",
        job.out,
    ]
    if args.date_from:
        cmd += ["--date-from", args.date_from]
    if args.date_to:
        cmd += ["--date-to", args.date_to]
    if args.amp:
        cmd.append("--amp")
    if args.diagnostic_ablation:
        cmd.append("--diagnostic-ablation")
    if args.winner_only:
        cmd.append("--winner-only")
    if args.min_score:
        cmd += ["--min-score", str(args.min_score)]
    if job.deck_sig:
        cmd += ["--deck-sig", job.deck_sig]
    if job.team_name:
        cmd += ["--team-name", job.team_name]
    smoke_deck = args.random_smoke_deck or job.deck_path
    if args.random_smoke_games > 0 and smoke_deck:
        cmd += [
            "--random-smoke-deck",
            smoke_deck,
            "--random-smoke-games",
            str(args.random_smoke_games),
            "--random-smoke-workers",
            str(args.random_smoke_workers),
            "--random-smoke-every",
            str(args.random_smoke_every),
            "--random-smoke-max-turns",
            str(args.random_smoke_max_turns),
            "--random-smoke-device",
            args.random_smoke_device,
            "--random-smoke-progress-every",
            str(args.random_smoke_progress_every),
            "--random-smoke-min-wr",
            str(args.random_smoke_min_wr),
        ]
    return cmd


def _last_progress_line(path: str) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 20000), os.SEEK_SET)
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "batch=" in line or line.startswith("done epoch") or line.startswith("Training complete"):
            return line[-500:]
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="CSV with archetype, optional deck_sig/team_name/name/out/score_bands")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out-dir", default="checkpoints/v14_sequence_population")
    p.add_argument("--log-dir", default="logs/v14_sequence_population")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--score-bands", default="900-999 1000-1099 1100-1199 1200+")
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--win-weight", type=float, default=1.5)
    p.add_argument("--loss-weight", type=float, default=0.5)
    p.add_argument("--draw-weight", type=float, default=0.8)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--current-action-weight", type=float, default=1.0)
    p.add_argument("--prefix-action-weight", type=float, default=0.10)
    p.add_argument("--order-weight", type=float, default=0.15)
    p.add_argument("--multi-weight", type=float, default=0.15)
    p.add_argument("--count-weight", type=float, default=0.20)
    p.add_argument("--plan-weight", type=float, default=0.35)
    p.add_argument("--next-type-weight", type=float, default=0.25)
    p.add_argument("--dca-plan-weight", type=float, default=0.25)
    p.add_argument("--known-action-weight", type=float, default=0.0)
    p.add_argument("--turn-plan-weight", type=float, default=0.0)
    p.add_argument("--turn-terminal-weight", type=float, default=0.0)
    p.add_argument("--turn-next-plan-weight", type=float, default=0.0)
    p.add_argument("--turn-next-type-weight", type=float, default=1.0)
    p.add_argument("--turn-next-card-weight", type=float, default=0.25)
    p.add_argument("--turn-next-attack-weight", type=float, default=0.25)
    p.add_argument("--turn-next-context-weight", type=float, default=0.10)
    p.add_argument("--opportunity-type-weight", type=float, default=0.0)
    p.add_argument("--opportunity-margin-weight", type=float, default=0.0)
    p.add_argument("--opportunity-margin", type=float, default=0.25)
    p.add_argument("--current-rank-margin-weight", type=float, default=0.0)
    p.add_argument("--current-rank-margin", type=float, default=0.25)
    p.add_argument("--current-rank-margin-min-options", type=int, default=2)
    p.add_argument("--history-condition-scale", type=float, default=0.0)
    p.add_argument("--plan-condition-scale", type=float, default=0.0)
    p.add_argument("--next-type-condition-scale", type=float, default=0.0)
    p.add_argument("--dca-condition-scale", type=float, default=0.0)
    p.add_argument("--known-condition-scale", type=float, default=0.0)
    p.add_argument("--known-logit-scale", type=float, default=0.0)
    p.add_argument("--turn-condition-scale", type=float, default=0.0)
    p.add_argument("--turn-next-condition-scale", type=float, default=0.0)
    p.add_argument("--type-prior-scale", type=float, default=0.0)
    p.add_argument("--current-complexity-weight", type=float, default=0.0)
    p.add_argument("--multi-target-weight", type=float, default=1.0)
    p.add_argument("--damage-counter-weight", type=float, default=1.0)
    p.add_argument("--random-smoke-deck", default="",
                   help="optional global deck CSV for per-epoch random smoke; manifest deck_path is used when omitted")
    p.add_argument("--random-smoke-games", type=int, default=0)
    p.add_argument("--random-smoke-workers", type=int, default=4)
    p.add_argument("--random-smoke-every", type=int, default=1)
    p.add_argument("--random-smoke-max-turns", type=int, default=700)
    p.add_argument("--random-smoke-device", default="cpu")
    p.add_argument("--random-smoke-progress-every", type=int, default=20)
    p.add_argument("--random-smoke-min-wr", type=float, default=0.90)
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--diagnostic-ablation", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--poll", type=float, default=30.0)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    jobs = read_jobs(args.manifest, default_out_dir=args.out_dir, default_score_bands=args.score_bands)
    gpu_slots: list[str] = []
    for gpu in [x.strip() for x in args.gpus.split(",") if x.strip()]:
        gpu_slots.extend([gpu] * max(1, args.jobs_per_gpu))
    if not gpu_slots:
        gpu_slots = [""]

    pending = list(jobs)
    running: list[Job] = []
    done: list[Job] = []
    failed: list[Job] = []
    print(f"v14 population jobs={len(jobs)} slots={gpu_slots}", flush=True)
    while pending or running:
        free = list(gpu_slots)
        for job in running:
            if job.gpu in free:
                free.remove(job.gpu)
        while pending and free:
            job = pending.pop(0)
            job.gpu = free.pop(0)
            job.log_path = str(Path(args.log_dir) / f"{job.name}.log")
            cmd = build_cmd(args, job)
            env = os.environ.copy()
            if job.gpu:
                env["CUDA_VISIBLE_DEVICES"] = job.gpu
            with open(job.log_path, "w") as log:
                log.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
                log.flush()
                job.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            job.start_time = time.time()
            job.status = "running"
            running.append(job)
            print(f"start gpu={job.gpu or '-'} {job.name} log={job.log_path}", flush=True)

        still: list[Job] = []
        for job in running:
            rc = job.proc.poll() if job.proc else 1
            if rc is None:
                still.append(job)
            elif rc == 0:
                job.status = "done"
                done.append(job)
                print(f"done {job.name} {((time.time()-job.start_time)/60):.1f}m", flush=True)
            else:
                job.status = "failed"
                failed.append(job)
                print(f"failed rc={rc} {job.name} log={job.log_path}", flush=True)
        running = still
        print(f"Status: pending={len(pending)} running={len(running)} done={len(done)} failed={len(failed)}", flush=True)
        for job in running:
            tail = _last_progress_line(job.log_path)
            suffix = f" | {tail}" if tail else ""
            print(f"  gpu={job.gpu or '-'} {job.name} {(time.time()-job.start_time)/60:.1f}m log={job.log_path}{suffix}", flush=True)
        if pending or running:
            time.sleep(args.poll)
    if failed:
        raise SystemExit("Failed jobs: " + ", ".join(job.name for job in failed))


if __name__ == "__main__":
    main()
