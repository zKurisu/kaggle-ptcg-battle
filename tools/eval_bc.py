#!/usr/bin/env python3
"""Evaluate a BC-trained policy.npz against random agents."""
import sys, os, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptcg_rl.numpy_policy import NumpyPolicy

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

def eval_vs_random(policy, deck, games=20, use_mcts=False, sims=48, time_budget=4.0, progress_every=10):
    """Win rate against random agent."""
    from cg.game import battle_start, battle_select, battle_finish

    wins = 0
    t0 = time.time()
    for g in range(games):
        if g % 2 == 0:
            obs, sd = battle_start(deck, deck)
            our_side = 0
        else:
            obs, sd = battle_start(deck, deck)
            our_side = 1
        if obs is None: continue
        
        while True:
            sel = obs.get('select'); cur = obs.get('current',{}); res = cur.get('result',-1)
            if res != -1:
                if res == our_side: wins += 1
                break
            if sel is None: break
            you = cur.get('yourIndex',0)
            if you == our_side:
                act = _policy_action(policy, obs, deck, use_mcts, sims, time_budget)
            else:
                act = _legal_random(sel)
            obs = battle_select(act)
        battle_finish()
        done = g + 1
        if progress_every and (done == 1 or done % progress_every == 0 or done == games):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = max(games - done, 0) / max(rate, 1e-9)
            print(
                f"  {done}/{games} games wins={wins} wr={wins/done:.3f} "
                f"{rate:.2f} games/s eta={eta:.0f}s",
                flush=True,
            )
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
    args = p.parse_args()

    policy = NumpyPolicy.load(args.policy)
    deck = load_deck(args.deck)
    
    print(f"Policy: {args.policy}")
    print(f"Deck: {args.deck} ({len(deck)} cards)")
    mode = f"MCTS sims={args.mcts_sims} budget={args.time_budget}s" if args.mcts else "greedy"
    print(f"Testing {args.games} games vs legal random ({mode})...")
    
    t0 = time.time()
    wr = eval_vs_random(policy, deck, args.games, args.mcts, args.mcts_sims,
                        args.time_budget, args.progress_every)
    elapsed = time.time() - t0
    
    print(f"\nWin rate vs Random: {wr*100:.1f}% ({int(wr*args.games)}/{args.games})")
    print(f"Time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")
    
    # Interpretation
    if wr < 0.5:   print("⚠️  Worse than random — model broken or undertrained")
    elif wr < 0.7: print("📊 Better than random but weak — needs more training")
    elif wr < 0.9: print("📈 Decent — BC is learning")
    else:          print("🔥 Strong — ready for Kaggle submission")

if __name__ == "__main__": main()
