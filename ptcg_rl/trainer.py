"""
PPO trainer — single-process, correct, fast.

Self-play games run sequentially (C engine is single-threaded anyway).
PPO update runs on GPU. Opponent pool for diversity.
"""

from __future__ import annotations

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from .encoder import FastEncoder, EncodedDecision
from .model import PolicyValueNet


# ── GAE ─────────────────────────────────────────────────────────────────

def compute_gae(samples: list[EncodedDecision], gamma: float = 0.99,
                lam: float = 0.95) -> None:
    """Compute GAE advantages. Modifies samples in-place."""
    gae = 0.0
    for i in range(len(samples) - 1, -1, -1):
        next_val = samples[i + 1].value if i + 1 < len(samples) else 0.0
        delta = samples[i].reward + gamma * next_val - samples[i].value
        gae = delta + gamma * lam * gae
        samples[i].adv = gae
        samples[i].ret = gae + samples[i].value


# ── Self-play game ──────────────────────────────────────────────────────

def play_game(model: PolicyValueNet, deck: list[int],
              opp_deck: list[int], encoder: FastEncoder,
              temperature: float = 1.0) -> list[EncodedDecision]:
    """Play one game. Returns list of decisions from our perspective (player 0)."""
    from cg.game import battle_start, battle_select, battle_finish, _get_battle_data

    obs, sd = battle_start(deck, opp_deck)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={sd.errorType}")

    decisions = []
    model.eval()
    dev = next(model.parameters()).device

    while True:
        sel = obs.get("select")
        cur = obs.get("current", {})
        result = cur.get("result", -1)
        if result != -1:
            break
        if sel is None:
            break

        opts = sel.get("option", [])
        mc = sel.get("maxCount", 0)
        if not opts or mc == 0:
            obs = battle_select([])
            continue

        try:
            d = encoder.encode(obs)
        except ValueError:
            obs = battle_select([])
            continue

        # Act with model
        with torch.no_grad():
            picks, lp, val = model.act(
                d.board_cards, d.hand_cards, d.state_feats,
                d.opt_type, d.opt_card, d.opt_card2, d.opt_attack, d.opt_feats,
                d.min_count, d.max_count,
                greedy=(temperature < 0.1 or random.random() > temperature),
            )
        d.action = picks
        d.logprob = lp
        d.value = val
        decisions.append(d)
        obs = battle_select(picks)

        if len(decisions) > 500:
            break

    battle_finish()

    # Reward: +1 win, -1 loss, 0 draw (from player 0 perspective)
    final = obs.get("current", {}).get("result", -1)
    if decisions:
        decisions[-1].reward = 1.0 if final == 0 else (-1.0 if final == 1 else 0.0)

    return decisions


# ── PPO Trainer ─────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(self, deck: list[int], opponent_decks: list[list[int]] | None = None,
                 lr: float = 3e-4, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 gamma: float = 0.99, lam: float = 0.95,
                 epochs: int = 4, minibatch: int = 128,
                 device: str = 'cuda'):
        self.deck = deck
        self.opp_decks = opponent_decks or [deck]
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch = minibatch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = PolicyValueNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.encoder = FastEncoder()
        self.metrics = []

        print(f"Model: {self.model.count_params():,} params on {self.device}")

    def collect(self, n_games: int, temperature: float = 1.0) -> list[EncodedDecision]:
        """Play n_games, return all decisions."""
        all_d = []
        for _ in range(n_games):
            opp = random.choice(self.opp_decks)
            d = play_game(self.model, self.deck, opp, self.encoder, temperature)
            all_d.extend(d)
        return all_d

    def update(self, samples: list[EncodedDecision]) -> dict:
        """One full PPO update over all samples."""
        self.model.train()
        dev = self.device

        advs = np.array([s.adv for s in samples], dtype=np.float32)
        adv_mean, adv_std = advs.mean(), advs.std() + 1e-8

        stats = {"pol_loss": 0.0, "val_loss": 0.0, "entropy": 0.0,
                 "kl": 0.0, "clip": 0.0, "n": 0}
        idx = np.arange(len(samples))

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.minibatch):
                mb_idx = idx[start:start + self.minibatch]
                if len(mb_idx) < 2:
                    continue
                mb = [samples[i] for i in mb_idx]

                old_lp = torch.tensor([s.logprob for s in mb], device=dev)
                adv = (torch.tensor([s.adv for s in mb], device=dev) - adv_mean) / adv_std
                ret = torch.tensor([s.ret for s in mb], device=dev)

                new_lp, ent, val = self.model.evaluate_actions(mb)

                ratio = (new_lp - old_lp).exp()
                pol_loss = -torch.min(ratio * adv,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv).mean()
                val_loss = (val - ret).pow(2).mean()
                loss = pol_loss + self.value_coef * val_loss - self.entropy_coef * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    stats["kl"] += ((ratio - 1) - (new_lp - old_lp)).mean().item()
                    stats["clip"] += (ratio - 1).abs().gt(self.clip_eps).float().mean().item()
                stats["pol_loss"] += float(pol_loss)
                stats["val_loss"] += float(val_loss)
                stats["entropy"] += float(ent.mean())
                stats["n"] += 1

        n = max(stats["n"], 1)
        return {"pol_loss": stats["pol_loss"] / n, "val_loss": stats["val_loss"] / n,
                "entropy": stats["entropy"] / n, "kl": stats["kl"] / n,
                "clip": stats["clip"] / n}

    def train(self, iterations: int, games_per_iter: int = 32,
              checkpoint_dir: str = "checkpoints", save_every: int = 50):
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        for it in range(1, iterations + 1):
            t0 = time.time()

            # Collect
            samples = self.collect(games_per_iter)
            t_coll = time.time() - t0

            if not samples:
                print(f"iter {it}: no samples collected, skipping")
                continue

            # GAE
            compute_gae(samples, self.gamma, self.lam)

            # PPO update
            stats = self.update(samples)
            t_upd = time.time() - t0 - t_coll

            stats.update({"iter": it, "steps": len(samples), "games": games_per_iter,
                          "t_collect": t_coll, "t_update": t_upd,
                          "t_total": time.time() - t0})
            self.metrics.append(stats)

            print(f"iter {it:>4} | steps {len(samples):>5} | "
                  f"pol={stats['pol_loss']:.4f} val={stats['val_loss']:.4f} "
                  f"ent={stats['entropy']:.3f} kl={stats['kl']:.4f} "
                  f"t={stats['t_total']:.1f}s")

            # Save
            if it % save_every == 0 or it == iterations:
                path = os.path.join(checkpoint_dir, f"ppo_iter{it:04d}.pt")
                torch.save({"model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "iter": it, "metrics": self.metrics}, path)
                print(f"  → {path}")

        return self.metrics


def export_numpy(model: PolicyValueNet, path: str):
    """Export weights for Kaggle submission."""
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **sd)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"Exported: {path} ({mb:.1f} MB)")
