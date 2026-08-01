"""Kaggle submission — numpy-only, no torch."""
import os, sys, random

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = "/kaggle_simulations/agent"
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ptcg_rl.numpy_policy import NumpyPolicy

# Load deck
DECK_PATH = os.path.join(HERE, "deck.csv")
with open(DECK_PATH) as f:
    MY_DECK = [int(l.strip()) for l in f if l.strip()]

# Load policy
POLICY_PATH = os.path.join(HERE, "policy.npz")
policy = NumpyPolicy.load(POLICY_PATH)


def _safe_random(obs_dict: dict) -> list[int]:
    """Always-valid random fallback."""
    sel = obs_dict.get("select", {})
    opts = sel.get("option", [])
    mc = sel.get("maxCount", 0)
    mn = sel.get("minCount", 0)
    if not opts or mc <= 0:
        return []
    n = len(opts)
    k = max(mn, min(mc, n))
    if k <= 0:
        return []
    return random.sample(range(n), k)


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return list(MY_DECK)

    sel = obs_dict.get("select", {})
    opts = sel.get("option", [])
    mc = sel.get("maxCount", 0)
    mn = sel.get("minCount", 0)
    n = len(opts)

    if n == 0 or mc <= 0:
        return []

    # Try numpy policy
    try:
        picks = policy.select(obs_dict, greedy=True)
        # Validate
        picks = [p for p in picks if 0 <= p < n]
        picks = list(dict.fromkeys(picks))  # dedup
        if len(picks) >= mn and len(picks) <= mc:
            return picks[:mc]
    except Exception:
        pass

    return _safe_random(obs_dict)


if __name__ == "__main__":
    from cg.game import battle_start, battle_select, battle_finish
    deck = list(MY_DECK)
    obs, sd = battle_start(deck, deck)
    assert obs is not None, f"start failed: {sd.errorType}"
    turn = 0
    while True:
        sel = obs.get("select")
        cur = obs.get("current", {})
        res = cur.get("result", -1)
        if res != -1:
            print(f"Game over: result={res}, turns={turn}")
            break
        if sel is None: break
        obs = battle_select(agent(obs)); turn += 1
        if turn > 500: break
    battle_finish()
