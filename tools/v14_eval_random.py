#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.seq.torch_policy import TorchSequencePolicy

_POLICY = None
_DECK: list[int] | None = None
_DEVICE = "cpu"
_MAX_TURNS = 700


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def legal_random(sel: dict) -> list[int]:
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0))
    mx = int(sel.get("maxCount", 0))
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []


def play_one(policy: TorchSequencePolicy, deck: list[int], game_index: int, seed: int, max_turns: int) -> tuple[int, int]:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    our_side = game_index % 2
    policy.reset_history()
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return 0, 1
    try:
        for _ in range(max_turns):
            cur = obs.get("current", {})
            res = cur.get("result", -1)
            if res != -1:
                return (1 if res == our_side else 0), 0
            sel = obs.get("select")
            if sel is None:
                return 0, 1
            you = int(cur.get("yourIndex", 0))
            if you == our_side:
                try:
                    act = policy.select(obs, greedy=True, update_history=True)
                except Exception:
                    act = legal_random(sel)
            else:
                act = legal_random(sel)
            obs = battle_select(act)
            if obs is None:
                return 0, 1
        return 0, 1
    finally:
        battle_finish()


def init_worker(policy_path: str, deck: list[int], device: str, max_turns: int) -> None:
    global _POLICY, _DECK, _DEVICE, _MAX_TURNS
    _DEVICE = device
    _DECK = deck
    _MAX_TURNS = max_turns
    _POLICY = TorchSequencePolicy.load(policy_path, device=device)


def worker(args):
    game_index, seed = args
    return play_one(_POLICY, _DECK, game_index, seed, _MAX_TURNS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--deck", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args()

    deck = load_deck(args.deck)
    wins = 0
    errors = 0
    t0 = time.time()
    workers = max(1, min(args.workers, args.games))
    print(f"Policy: {args.checkpoint}")
    print(f"Deck: {args.deck} ({len(deck)} cards)")
    print(f"Testing v14 sequence policy vs legal random: games={args.games} workers={workers} device={args.device}")
    if workers == 1:
        policy = TorchSequencePolicy.load(args.checkpoint, device=args.device)
        for g in range(args.games):
            win, err = play_one(policy, deck, g, args.seed + g, args.max_turns)
            wins += win
            errors += err
            done = g + 1
            if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
                _print_progress(done, args.games, wins, t0)
    else:
        tasks = [(g, args.seed + g) for g in range(args.games)]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(args.checkpoint, deck, args.device, args.max_turns)) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for done, fut in enumerate(as_completed(futs), 1):
                win, err = fut.result()
                wins += int(win)
                errors += int(err)
                if args.progress_every and (done == 1 or done % args.progress_every == 0 or done == args.games):
                    _print_progress(done, args.games, wins, t0)
    print(f"\nWin rate vs Random: {wins / args.games * 100:.1f}% ({wins}/{args.games})")
    if errors:
        print(f"Timeout/error games: {errors}/{args.games}")
    print(f"Time: {time.time() - t0:.0f}s")


def _print_progress(done: int, games: int, wins: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = (games - done) / max(rate, 1e-9)
    print(f"  {done}/{games} wins={wins} wr={wins/done:.3f} {rate:.2f} games/s eta={eta:.0f}s", flush=True)


if __name__ == "__main__":
    main()
