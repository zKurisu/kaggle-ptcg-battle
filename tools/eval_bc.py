#!/usr/bin/env python3
"""Evaluate a BC-trained policy.npz against random agents."""
import sys, os, random, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptcg_rl.numpy_policy import NumpyPolicy

_WORKER_POLICY = None
_WORKER_DECK = None
_WORKER_USE_MCTS = False
_WORKER_SIMS = 48
_WORKER_TIME_BUDGET = 4.0

def load_deck(path): 
    with open(path) as f: return [int(l.strip()) for l in f if l.strip()]

def _legal_random(sel: dict) -> list[int]:
    opts = sel.get('option', [])
    mn = int(sel.get('minCount', 0))
    mc = int(sel.get('maxCount', 0))
    if not opts or mc <= 0:
        return []
    hi = min(mc, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []

def _policy_action(policy, obs, deck, use_mcts=False, sims=48, time_budget=4.0):
    sel = obs.get('select') or {}
    n = len(sel.get('option', []))
    mn = int(sel.get('minCount', 0))
    mc = int(sel.get('maxCount', 0))
    if n == 0 or mc <= 0:
        return []
    try:
        if use_mcts:
            act = policy.select_mcts(obs, deck, sims=sims, time_budget=time_budget)
        else:
            act = policy.select(obs, greedy=True)
    except Exception:
        act = []
    act = [a for a in act if 0 <= a < n]
    act = list(dict.fromkeys(act))
    if mn <= len(act) <= mc:
        return act[:mc]
    return _legal_random(sel)

def _play_one_game(policy, deck, game_index, use_mcts=False, sims=48, time_budget=4.0, seed=0):
    from cg.game import battle_start, battle_select, battle_finish

    random.seed(seed)
    our_side = 0 if game_index % 2 == 0 else 1
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return 0

    try:
        while True:
            sel = obs.get('select'); cur = obs.get('current',{}); res = cur.get('result',-1)
            if res != -1:
                return 1 if res == our_side else 0
            if sel is None:
                return 0
            you = cur.get('yourIndex',0)
            if you == our_side:
                act = _policy_action(policy, obs, deck, use_mcts, sims, time_budget)
            else:
                act = _legal_random(sel)
            obs = battle_select(act)
    finally:
        battle_finish()


def _init_worker(policy_path, deck, use_mcts, sims, time_budget):
    global _WORKER_POLICY, _WORKER_DECK, _WORKER_USE_MCTS, _WORKER_SIMS, _WORKER_TIME_BUDGET
    _WORKER_POLICY = NumpyPolicy.load(policy_path)
    _WORKER_DECK = deck
    _WORKER_USE_MCTS = use_mcts
    _WORKER_SIMS = sims
    _WORKER_TIME_BUDGET = time_budget


def _worker_play_one(args):
    game_index, seed = args
    return _play_one_game(
        _WORKER_POLICY, _WORKER_DECK, game_index,
        _WORKER_USE_MCTS, _WORKER_SIMS, _WORKER_TIME_BUDGET, seed,
    )


def _print_progress(done, games, wins, t0):
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = max(games - done, 0) / max(rate, 1e-9)
    print(
        f"  {done}/{games} games wins={wins} wr={wins/done:.3f} "
        f"{rate:.2f} games/s eta={eta:.0f}s",
        flush=True,
    )


def eval_vs_random(policy, deck, policy_path, games=20, use_mcts=False, sims=48,
                   time_budget=4.0, progress_every=10, workers=1, seed=1):
    """Win rate against random agent."""
    wins = 0
    t0 = time.time()
    workers = max(1, min(int(workers), games))
    if workers == 1:
        for g in range(games):
            wins += _play_one_game(policy, deck, g, use_mcts, sims, time_budget, seed + g)
            done = g + 1
            if progress_every and (done == 1 or done % progress_every == 0 or done == games):
                _print_progress(done, games, wins, t0)
        return wins / games

    tasks = [(g, seed + g) for g in range(games)]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(policy_path, deck, use_mcts, sims, time_budget),
    ) as ex:
        futs = [ex.submit(_worker_play_one, t) for t in tasks]
        for done, fut in enumerate(as_completed(futs), 1):
            wins += int(fut.result())
            if progress_every and (done == 1 or done % progress_every == 0 or done == games):
                _print_progress(done, games, wins, t0)
    return wins / games

def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="path to policy.npz")
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--mcts", action="store_true",
                   help="evaluate with the same policy.select_mcts fallback used by main.py")
    p.add_argument("--mcts-sims", type=int, default=48)
    p.add_argument("--time-budget", type=float, default=4.0)
    p.add_argument("--progress-every", type=int, default=10,
                   help="print progress every N games; 0 disables progress")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel game worker processes; each worker loads the policy once")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    policy = NumpyPolicy.load(args.policy)
    deck = load_deck(args.deck)
    
    print(f"Policy: {args.policy}")
    print(f"Deck: {args.deck} ({len(deck)} cards)")
    mode = f"MCTS sims={args.mcts_sims} budget={args.time_budget}s" if args.mcts else "greedy"
    print(f"Testing {args.games} games vs legal random ({mode})...")
    
    t0 = time.time()
    wr = eval_vs_random(
        policy, deck, args.policy, args.games, args.mcts, args.mcts_sims,
        args.time_budget, args.progress_every, args.workers, args.seed,
    )
    elapsed = time.time() - t0
    
    print(f"\nWin rate vs Random: {wr*100:.1f}% ({int(wr*args.games)}/{args.games})")
    print(f"Time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")
    
    # Interpretation
    if wr < 0.5:   print("⚠️  Worse than random — model broken or undertrained")
    elif wr < 0.7: print("📊 Better than random but weak — needs more training")
    elif wr < 0.9: print("📈 Decent — BC is learning")
    else:          print("🔥 Strong — ready for Kaggle submission")

if __name__ == "__main__": main()
