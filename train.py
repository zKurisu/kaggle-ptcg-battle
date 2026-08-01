#!/usr/bin/env python3
"""PTCG RL Training — clean pipeline. Usage: python train_ptcg.py"""

import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# cg engine is in workspace root, not in this repo
_WORKSPACE = os.path.dirname(_HERE)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import torch
from ptcg_rl.trainer import PPOTrainer, export_numpy


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(l.strip()) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mcts", action="store_true", help="Use MCTS during self-play")
    parser.add_argument("--mcts-sims", type=int, default=32, help="MCTS simulations per decision")
    args = parser.parse_args()

    deck = read_deck(args.deck)
    assert len(deck) == 60

    # Opponent pool: 26 real archetypes + mirror self-play
    opponent_decks = [deck]  # always include mirror
    pool_dir = os.path.join(_HERE, "decks")
    if os.path.isdir(pool_dir):
        for f in sorted(os.listdir(pool_dir)):
            if f.endswith(".csv"):
                opponent_decks.append(read_deck(os.path.join(pool_dir, f)))
    print(f"Opponent pool: {len(opponent_decks)} decks (1 mirror + {len(opponent_decks)-1} archetypes)")

    trainer = PPOTrainer(
        deck=deck, opponent_decks=opponent_decks,
        lr=args.lr, device=args.device,
        use_mcts=args.mcts, mcts_sims=args.mcts_sims,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Resumed from iter {ckpt['iter']}")

    trainer.train(
        iterations=args.iterations,
        games_per_iter=args.games,
        save_every=args.save_every,
    )

    # Export
    export_numpy(trainer.model, "policy.npz")

    # Smoke test numpy policy
    from ptcg_rl.numpy_policy import NumpyPolicy
    p = NumpyPolicy.load("policy.npz")
    from cg.game import battle_start, battle_finish
    obs, sd = battle_start(deck, deck)
    if obs and obs.get("select"):
        p.select(obs, greedy=True)
    battle_finish()
    print("Numpy policy smoke test: OK")


if __name__ == "__main__":
    main()
