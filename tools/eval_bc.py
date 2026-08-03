#!/usr/bin/env python3
"""Evaluate a BC-trained policy.npz against random agents."""
import sys, os, random, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptcg_rl.numpy_policy import NumpyPolicy
from ptcg_rl.deck_registry import registry_deck_for_policy
from ptcg_rl.rule_overlay import apply_rule_overlay

_WORKER_POLICY = None
_WORKER_DECK = None
_WORKER_RULES = ""
_WORKER_USE_MCTS = False
_WORKER_SIMS = 48
_WORKER_TIME_BUDGET = 4.0
_WORKER_MAX_TURNS = 700
_LAST_TIMEOUTS = 0

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

def _policy_action(policy, obs, deck, use_mcts=False, sims=48, time_budget=4.0, rules=""):
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
    if rules:
        try:
            act = apply_rule_overlay(obs, act, deck, mode=rules).action
        except Exception:
            pass
    act = [a for a in act if 0 <= a < n]
    act = list(dict.fromkeys(act))
    if mn <= len(act) <= mc:
        return act[:mc]
    return _legal_random(sel)

def _play_one_game(policy, deck, game_index, use_mcts=False, sims=48,
                   time_budget=4.0, seed=0, max_turns=700, rules=""):
    from cg.game import battle_start, battle_select, battle_finish

    random.seed(seed)
    our_side = 0 if game_index % 2 == 0 else 1
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return 0, 1

    try:
        for _ in range(max_turns):
            sel = obs.get('select'); cur = obs.get('current',{}); res = cur.get('result',-1)
            if res != -1:
                return (1 if res == our_side else 0), 0
            if sel is None:
                return 0, 1
            you = cur.get('yourIndex',0)
            if you == our_side:
                act = _policy_action(policy, obs, deck, use_mcts, sims, time_budget, rules)
            else:
                act = _legal_random(sel)
            obs = battle_select(act)
            if obs is None:
                return 0, 1
        return 0, 1
    finally:
        battle_finish()


def _init_worker(policy_path, deck, use_mcts, sims, time_budget, max_turns, rules):
    global _WORKER_POLICY, _WORKER_DECK, _WORKER_USE_MCTS, _WORKER_SIMS, _WORKER_TIME_BUDGET, _WORKER_MAX_TURNS, _WORKER_RULES
    _WORKER_POLICY = NumpyPolicy.load(policy_path)
    _WORKER_DECK = deck
    _WORKER_USE_MCTS = use_mcts
    _WORKER_SIMS = sims
    _WORKER_TIME_BUDGET = time_budget
    _WORKER_MAX_TURNS = max_turns
    _WORKER_RULES = rules


def _worker_play_one(args):
    game_index, seed = args
    return _play_one_game(
        _WORKER_POLICY, _WORKER_DECK, game_index,
        _WORKER_USE_MCTS, _WORKER_SIMS, _WORKER_TIME_BUDGET, seed, _WORKER_MAX_TURNS, _WORKER_RULES,
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
                   time_budget=4.0, progress_every=10, workers=1, seed=1, max_turns=700, rules=""):
    """Win rate against random agent."""
    global _LAST_TIMEOUTS
    wins = 0
    timeouts = 0
    t0 = time.time()
    workers = max(1, min(int(workers), games))
    if workers == 1:
        for g in range(games):
            win, timeout = _play_one_game(policy, deck, g, use_mcts, sims, time_budget, seed + g, max_turns, rules)
            wins += win
            timeouts += timeout
            done = g + 1
            if progress_every and (done == 1 or done % progress_every == 0 or done == games):
                _print_progress(done, games, wins, t0)
        _LAST_TIMEOUTS = timeouts
        return wins / games

    tasks = [(g, seed + g) for g in range(games)]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(policy_path, deck, use_mcts, sims, time_budget, max_turns, rules),
    ) as ex:
        futs = [ex.submit(_worker_play_one, t) for t in tasks]
        for done, fut in enumerate(as_completed(futs), 1):
            win, timeout = fut.result()
            wins += int(win)
            timeouts += int(timeout)
            if progress_every and (done == 1 or done % progress_every == 0 or done == games):
                _print_progress(done, games, wins, t0)
    _LAST_TIMEOUTS = timeouts
    return wins / games

def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="path to policy.npz")
    p.add_argument("--deck", default="")
    p.add_argument("--registry", default="",
                   help="CSV mapping policy_path to deck_path; used when --deck is omitted")
    p.add_argument("--auto-deck", action="store_true",
                   help="resolve --deck from --registry; implied when --registry is passed without --deck")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--mcts", action="store_true",
                   help="evaluate with the same policy.select_mcts fallback used by main.py")
    p.add_argument("--mcts-sims", type=int, default=48)
    p.add_argument("--time-budget", type=float, default=4.0)
    p.add_argument("--progress-every", type=int, default=10,
                   help="print progress every N games; 0 disables progress")
    p.add_argument("--max-turns", type=int, default=700,
                   help="maximum engine decisions per game before counting it as a loss/error")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel game worker processes; each worker loads the policy once")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rules", choices=["", "conservative", "aggressive"], default="",
                   help="experimental BC+rule overlay for local evaluation only")
    args = p.parse_args()

    if not args.deck and args.registry:
        args.deck = registry_deck_for_policy(args.registry, args.policy) or ""
    if not args.deck:
        args.deck = "deck.csv"
        if args.auto_deck:
            raise FileNotFoundError(f"no registry deck found for policy: {args.policy}")

    policy = NumpyPolicy.load(args.policy)
    deck = load_deck(args.deck)
    
    print(f"Policy: {args.policy}")
    print(f"Deck: {args.deck} ({len(deck)} cards)")
    mode = f"MCTS sims={args.mcts_sims} budget={args.time_budget}s" if args.mcts else "greedy"
    if args.rules:
        mode += f"+rules:{args.rules}"
    print(f"Testing {args.games} games vs legal random ({mode})...")
    
    t0 = time.time()
    wr = eval_vs_random(
        policy, deck, args.policy, args.games, args.mcts, args.mcts_sims,
        args.time_budget, args.progress_every, args.workers, args.seed, args.max_turns, args.rules,
    )
    elapsed = time.time() - t0
    
    print(f"\nWin rate vs Random: {wr*100:.1f}% ({int(wr*args.games)}/{args.games})")
    if _LAST_TIMEOUTS:
        print(f"Timeout/error games: {_LAST_TIMEOUTS}/{args.games} (max_turns={args.max_turns})")
    print(f"Time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")
    
    # Interpretation
    if wr < 0.5:   print("⚠️  Worse than random — model broken or undertrained")
    elif wr < 0.7: print("📊 Better than random but weak — needs more training")
    elif wr < 0.9: print("📈 Decent — BC is learning")
    else:          print("🔥 Strong — ready for Kaggle submission")

if __name__ == "__main__": main()
