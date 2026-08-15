#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from tools.eval_bc import _init_worker, _worker_play_one, load_deck


def _fmt_eta(done: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = max(total - done, 0) / max(rate, 1e-9)
    return f"{rate:.2f} games/s eta={eta:.0f}s"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--deck", required=True)
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--start-game", type=int, default=0,
                   help="first game index; seed for a game is --seed + game")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--fresh-workers", action="store_true",
                   help="start a fresh worker process for each game; slower but avoids engine state leakage")
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    deck = load_deck(args.deck)
    start_game = max(0, int(args.start_game))
    tasks = [(g, int(args.seed) + g) for g in range(start_game, start_game + int(args.games))]
    wins = 0
    losses: list[dict[str, int]] = []
    t0 = time.time()
    workers = max(1, min(int(args.workers), int(args.games)))
    print(
        f"Finding random losses: policy={args.policy} deck={args.deck} "
        f"games={args.games} workers={workers} seed={args.seed}",
        flush=True,
    )
    if workers > 1:
        print(
            "WARNING: concurrent cg engine evaluation is fast but not authoritative; "
            "for hard gates use --workers 1 --fresh-workers and trace losses with --start-game.",
            flush=True,
        )
    ex_kwargs = {
        "max_workers": workers,
        "initializer": _init_worker,
        "initargs": (args.policy, deck, False, 48, 4.0, int(args.max_turns), ""),
    }
    if args.fresh_workers:
        ex_kwargs["max_tasks_per_child"] = 1
    with ProcessPoolExecutor(**ex_kwargs) as ex:
        futs = {ex.submit(_worker_play_one, t): t for t in tasks}
        for done, fut in enumerate(as_completed(futs), 1):
            game, seed = futs[fut]
            win, timeout = fut.result()
            wins += int(win)
            if not int(win):
                losses.append({"game": int(game), "seed": int(seed), "timeout": int(timeout)})
                print(
                    f"  loss game={game} seed={seed} timeout={int(timeout)} "
                    f"done={done}/{args.games} wins={wins} wr={wins/done:.3f}",
                    flush=True,
                )
            if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
                print(
                    f"  {done}/{args.games} wins={wins} losses={len(losses)} "
                    f"wr={wins/done:.3f} {_fmt_eta(done, int(args.games), t0)}",
                    flush=True,
                )

    losses.sort(key=lambda r: r["game"])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "seed", "timeout"])
        w.writeheader()
        w.writerows(losses)
    print(
        f"V15_RANDOM_LOSSES wins={wins} games={args.games} losses={len(losses)} out={out}",
        flush=True,
    )


if __name__ == "__main__":
    os.chdir(_REPO)
    main()
