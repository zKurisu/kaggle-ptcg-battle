#!/usr/bin/env python3
"""Evaluate a BC-trained policy.npz against random or rule-based agents."""
import sys, os, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptcg_rl.numpy_policy import NumpyPolicy
from cg.game import battle_start, battle_select, battle_finish

def load_deck(path): 
    with open(path) as f: return [int(l.strip()) for l in f if l.strip()]

def eval_vs_random(policy, deck, games=20):
    """Win rate against random agent."""
    wins = 0
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
                if sel.get('option'):
                    act = policy.select(obs, greedy=True)
                else:
                    act = []
            else:
                opts = sel.get('option',[]); mc = sel.get('maxCount',0)
                act = random.sample(range(len(opts)), min(mc, len(opts))) if opts and mc>0 else []
            obs = battle_select(act)
        battle_finish()
    return wins / games

def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="path to policy.npz")
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--games", type=int, default=20)
    args = p.parse_args()

    policy = NumpyPolicy.load(args.policy)
    deck = load_deck(args.deck)
    
    print(f"Policy: {args.policy}")
    print(f"Deck: {args.deck} ({len(deck)} cards)")
    print(f"Testing {args.games} games vs random...")
    
    t0 = time.time()
    wr = eval_vs_random(policy, deck, args.games)
    elapsed = time.time() - t0
    
    print(f"\nWin rate vs Random: {wr*100:.1f}% ({int(wr*args.games)}/{args.games})")
    print(f"Time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")
    
    # Interpretation
    if wr < 0.5:   print("⚠️  Worse than random — model broken or undertrained")
    elif wr < 0.7: print("📊 Better than random but weak — needs more training")
    elif wr < 0.9: print("📈 Decent — BC is learning")
    else:          print("🔥 Strong — ready for Kaggle submission")

if __name__ == "__main__": main()
