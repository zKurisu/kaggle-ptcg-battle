#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def run(cmd: list[str], *, log: Path | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        return int(proc.wait())


def parse_wr(log: Path) -> tuple[int, int]:
    text = log.read_text(errors="replace")
    marker = "Win rate vs Random:"
    pos = text.rfind(marker)
    if pos < 0:
        return 0, 0
    tail = text[pos:].splitlines()[0]
    if "(" not in tail or "/" not in tail:
        return 0, 0
    frac = tail.split("(", 1)[1].split(")", 1)[0]
    wins, games = frac.split("/", 1)
    return int(wins), int(games)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--deck", required=True)
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_log = out_dir / "random_eval.log"
    rc = run(
        [
            sys.executable,
            "tools/eval_bc.py",
            args.policy,
            "--deck",
            args.deck,
            "--games",
            str(args.games),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--max-turns",
            str(args.max_turns),
            "--progress-every",
            str(args.progress_every),
        ],
        log=eval_log,
    )
    if rc != 0:
        raise SystemExit(rc)
    wins, games = parse_wr(eval_log)
    passed = bool(games and wins == games)
    (out_dir / "gate_status.txt").write_text(f"passed={int(passed)} wins={wins} games={games}\n")
    if passed:
        print(f"V15_RANDOM_GATE_PASS wins={wins} games={games}", flush=True)
        return
    trace = out_dir / "first_loss_trace.md"
    script = out_dir / "first_loss_random_script.json"
    rc = run(
        [
            sys.executable,
            "tools/v15_scripted_random_trace.py",
            args.policy,
            "--deck",
            args.deck,
            "--games",
            str(max(args.games, 80)),
            "--seed",
            str(args.seed),
            "--target-outcome",
            "loss",
            "--max-turns",
            str(args.max_turns),
            "--progress-every",
            str(max(5, args.progress_every // 10)),
            "--out-md",
            str(trace),
            "--script-out",
            str(script),
        ],
        log=out_dir / "first_loss_trace.log",
    )
    if rc != 0:
        raise SystemExit(rc)
    print(f"V15_RANDOM_GATE_FAIL wins={wins} games={games} trace={trace} script={script}", flush=True)


if __name__ == "__main__":
    os.chdir(_REPO)
    main()
