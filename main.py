"""Kaggle submission — numpy-only, MCTS search, no torch."""
import os, sys, random

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = "/kaggle_simulations/agent"
for p in [HERE, os.path.dirname(HERE)]:  # repo root + workspace root
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg_rl.numpy_policy import NumpyPolicy
from ptcg_rl.resource_planner import ResourcePlanner
from ptcg_rl.rule_overlay import apply_rule_overlay

with open(os.path.join(HERE, "deck.csv")) as f:
    MY_DECK = [int(l.strip()) for l in f if l.strip()]

policy = NumpyPolicy.load(os.path.join(HERE, "policy.npz"))
planner = ResourcePlanner(MY_DECK)


def _load_rule_mode() -> str:
    mode = os.environ.get("PTCG_RULE_MODE", "").strip()
    cfg = os.path.join(HERE, "rules.txt")
    if not mode and os.path.exists(cfg):
        try:
            with open(cfg) as f:
                mode = f.read().strip()
        except Exception:
            mode = ""
    return mode


RULE_MODE = _load_rule_mode()

# BC checkpoints only train the policy head. Keep MCTS off until RL/PPO trains
# a meaningful value head for leaf evaluation.
USE_MCTS = False
MCTS_SIMS = 48
MCTS_TIME_BUDGET = 4.0


def _safe_random(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select", {})
    opts = sel.get("option", [])
    mc, mn = sel.get("maxCount", 0), sel.get("minCount", 0)
    if not opts or mc <= 0: return []
    k = max(mn, min(mc, len(opts)))
    return random.sample(range(len(opts)), k) if k > 0 else []


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        if hasattr(policy, "reset_history"):
            policy.reset_history()
        planner.reset(MY_DECK)
        return list(MY_DECK)

    sel = obs_dict.get("select", {})
    opts = sel.get("option", [])
    mc, mn, n = sel.get("maxCount", 0), sel.get("minCount", 0), len(opts)
    if n == 0 or mc <= 0: return []

    # MCTS search
    if USE_MCTS:
        try:
            picks = policy.select_mcts(obs_dict, MY_DECK,
                                       sims=MCTS_SIMS, time_budget=MCTS_TIME_BUDGET)
            if RULE_MODE == "resource_plan":
                picks = planner.decide(obs_dict, picks, MY_DECK).action
            elif RULE_MODE:
                picks = apply_rule_overlay(obs_dict, picks, MY_DECK, mode=RULE_MODE).action
            picks = [p for p in picks if 0 <= p < n]
            picks = list(dict.fromkeys(picks))
            if mn <= len(picks) <= mc: return picks[:mc]
        except Exception: pass

    # Greedy fallback
    try:
        picks = policy.select(obs_dict, greedy=True, temperature=1.2)
        if RULE_MODE == "resource_plan":
            picks = planner.decide(obs_dict, picks, MY_DECK).action
        elif RULE_MODE:
            picks = apply_rule_overlay(obs_dict, picks, MY_DECK, mode=RULE_MODE).action
        picks = [p for p in picks if 0 <= p < n]
        picks = list(dict.fromkeys(picks))
        if mn <= len(picks) <= mc: return picks[:mc]
    except Exception: pass

    return _safe_random(obs_dict)


if __name__ == "__main__":
    from cg.game import battle_start, battle_select, battle_finish
    deck = list(MY_DECK)
    obs, sd = battle_start(deck, deck)
    assert obs is not None, f"start: {sd.errorType}"
    turn = 0
    while True:
        sel = obs.get("select")
        cur = obs.get("current", {})
        if cur.get("result", -1) != -1:
            print(f"Game: result={cur['result']}, turns={turn}"); break
        if sel is None: break
        obs = battle_select(agent(obs)); turn += 1
        if turn > 500: break
    battle_finish()
