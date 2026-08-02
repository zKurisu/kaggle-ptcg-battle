#!/usr/bin/env python3
"""
BC (Behavior Cloning) trainer — learn from high-score Kaggle replays.

Usage:
    python tools/bc_trainer.py --archetype "Marnie Grimmsnarl" \
        --score-bands "1200+" "1100-1199" \
        --epochs 10 --device cuda:0

Output:
    checkpoints/bc_policy.npz — numpy weights for Kaggle submission
"""

import sys, os, glob, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

_HERE = Path(__file__).resolve().parent; _REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))

from ptcg_rl.model import PolicyValueNet
from ptcg_rl.encoder import STATE_FEAT_DIM, OPT_FEAT_DIM, MAX_HAND, BOARD_SLOTS
from ptcg_rl.trainer import export_numpy


class BCTrainer:
    def __init__(self, corpus_dir: str, archetype: str,
                 score_bands: list[str], device: str = "cuda",
                 lr: float = 1e-4, dropout: float = 0.1):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet().to(self.device)
        self.model.train()
        # Enable dropout for BC (fixed corpus → overfitting risk)
        # Note: current model has no dropout layers; adding would need model change

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.corpus_dir = corpus_dir
        self.archetype = archetype
        self.score_bands = score_bands
        print(f"BC Trainer: {archetype} [{', '.join(score_bands)}]", flush=True)

    def _iter_batches(self, corpus_dir, archetype, bands, batch_size):
        """Generator: yield batches directly from .npz arrays."""
        arch_dir = os.path.join(corpus_dir, archetype.replace(' ', '_'))
        for band in bands:
            band_dir = os.path.join(arch_dir, band.replace(' ', '_'))
            for npz_path in sorted(glob.glob(os.path.join(band_dir, "*.npz"))):
                data = np.load(npz_path, allow_pickle=True)
                n = len(data['board'])
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    mb_idx = idx[start:start + batch_size]
                    if len(mb_idx) < 2: continue
                    # Build batch directly from npz indices
                    mb = []
                    for i in mb_idx:
                        mb.append({
                            'board': data['board'][i], 'hand': data['hand'][i],
                            'feats': data['feats'][i], 'ot': data['ot'][i],
                            'oc': data['oc'][i], 'oc2': data['oc2'][i],
                            'oa': data['oa'][i],
                            'of': data['of_arr'][i],
                            'action': data['action'][i],
                            'min_c': int(data['min_c'][i]), 'max_c': int(data['max_c'][i]),
                        })
                    yield mb

    def _collate(self, mb):
        """Pad variable-length tensors into a batch."""
        B = len(mb)
        n_max = max(len(s['ot']) for s in mb)

        board = torch.zeros(B, BOARD_SLOTS, dtype=torch.long, device=self.device)
        hand = torch.zeros(B, MAX_HAND, dtype=torch.long, device=self.device)
        feats = torch.zeros(B, STATE_FEAT_DIM, device=self.device)
        ot = torch.zeros(B, n_max, dtype=torch.long, device=self.device)
        oc = torch.zeros(B, n_max, dtype=torch.long, device=self.device)
        oc2 = torch.zeros(B, n_max, dtype=torch.long, device=self.device)
        oa = torch.zeros(B, n_max, dtype=torch.long, device=self.device)
        of_arr = torch.zeros(B, n_max, OPT_FEAT_DIM, device=self.device)
        actions = []  # list of tensors (variable picks per sample)
        min_c = []; max_c = []

        for i, s in enumerate(mb):
            n = len(s['ot'])
            board[i, :] = torch.from_numpy(np.asarray(s['board'], dtype=np.int64))
            hand[i, :len(s['hand'])] = torch.from_numpy(np.asarray(s['hand'], dtype=np.int64))
            feats[i, :] = torch.from_numpy(np.asarray(s['feats'], dtype=np.float32))
            ot[i, :n] = torch.from_numpy(np.asarray(s['ot'], dtype=np.int64))
            oc[i, :n] = torch.from_numpy(np.asarray(s['oc'], dtype=np.int64))
            oc2[i, :n] = torch.from_numpy(np.asarray(s['oc2'], dtype=np.int64))
            oa[i, :n] = torch.from_numpy(np.asarray(s['oa'], dtype=np.int64))
            of_arr[i, :n] = torch.from_numpy(np.asarray(s['of'], dtype=np.float32))
            actions.append(torch.tensor(s['action'].astype(np.int64), device=self.device))
            min_c.append(s['min_c']); max_c.append(s['max_c'])

        return {
            'board': board, 'hand': hand, 'feats': feats,
            'ot': ot, 'oc': oc, 'oc2': oc2, 'oa': oa, 'of': of_arr,
            'actions': actions, 'min_c': min_c, 'max_c': max_c, 'n_max': n_max,
        }

    def train(self, epochs: int = 10, batch_size: int = 128, save_path: str = "checkpoints/bc_policy.npz"):
        # Count total batches for progress
        n_total = 0
        for _ in self._iter_batches(self.corpus_dir, self.archetype, self.score_bands, batch_size):
            n_total += 1
        print(f"  Total batches: {n_total}", flush=True)

        for epoch in range(epochs):
            total_loss = 0.0; total_pol = 0.0; total_val = 0.0; steps = 0
            t0 = time.time()
            print(f"  Epoch {epoch+1}/{epochs} [", end="", flush=True)

            for mb in self._iter_batches(self.corpus_dir, self.archetype,
                                         self.score_bands, batch_size):
                if steps % 50 == 0:
                    print(f"{steps}", end="", flush=True)
                elif steps % 10 == 0:
                    print(".", end="", flush=True)
                batch = self._collate(mb)

                # Forward: state → h, options → opts
                h = self.model.encode_state(batch['board'], batch['hand'], batch['feats'])
                opts = self.model.encode_options(batch['ot'], batch['oc'], batch['oc2'],
                                                  batch['oa'], batch['of'])
                value = self.model.value(h)

                # Compute policy loss: sequential cross-entropy (VECTORIZED)
                B = len(mb)
                picked_sum = torch.zeros(B, 128, device=self.device)  # OPT_ENC
                avail = torch.ones(B, batch['n_max'] + 1, dtype=torch.bool, device=self.device)

                max_picks = max(len(a) for a in batch['actions'])
                # Build target matrix: [B, max_picks+1], -1 = no target
                targets = torch.full((B, max_picks + 1), -1, dtype=torch.long, device=self.device)
                for i in range(B):
                    acts = batch['actions'][i]
                    for k, a in enumerate(acts):
                        if 0 <= int(a) < batch['n_max']:
                            targets[i, k] = int(a)
                    if len(acts) > 0:  # STOP at end
                        targets[i, len(acts)] = batch['n_max']

                pol_loss = torch.tensor(0.0, device=self.device)
                active = torch.ones(B, dtype=torch.bool, device=self.device)

                for k in range(max_picks + 1):
                    if not active.any(): break
                    stop_ok = torch.tensor([k >= batch['min_c'][i] for i in range(B)], device=self.device)
                    mask = avail.clone()
                    mask[:, batch['n_max']] = stop_ok

                    logits = self.model.option_logits(h, opts, picked_sum, mask)
                    logp = F.log_softmax(logits, dim=-1)  # [B, N+1]

                    # Gather logprobs at target indices (vectorized)
                    step_targets = targets[:, k]  # [B]
                    valid = (step_targets >= 0) & active
                    if valid.any():
                        safe_tgt = step_targets.clamp(min=0)
                        step_lp = logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)  # [B]
                        pol_loss = pol_loss - (step_lp * valid.float()).sum()

                        # Update picked_sum only for real options (not STOP)
                        real_pick = valid & (step_targets < batch['n_max'])
                        if real_pick.any():
                            idx = safe_tgt[real_pick]
                            chosen = opts[real_pick].gather(1, idx.view(-1, 1, 1).expand(-1, 1, 128)).squeeze(1)
                            picked_sum[real_pick] += chosen
                            for ri, rv in enumerate(real_pick):
                                if rv:
                                    avail[ri, idx[ri]] = False

                pol_loss = pol_loss / B

                # Value loss: predict game outcome (approximate as +1 for all)
                val_target = torch.ones(B, device=self.device)  # Winners-only filtering later
                val_loss = (value - val_target).pow(2).mean()

                loss = pol_loss + 0.5 * val_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item(); total_pol += pol_loss.item()
                total_val += val_loss.item(); steps += 1

            elapsed = time.time() - t0
            print(f"] loss={total_loss/steps:.4f} pol={total_pol/steps:.4f} val={total_val/steps:.4f} t={elapsed:.0f}s", flush=True)

        # Save
        export_numpy(self.model, save_path)
        print(f"\nSaved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded")
    p.add_argument("--archetype", default="Marnie Grimmsnarl")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save", default="checkpoints/bc_policy.npz")
    args = p.parse_args()

    trainer = BCTrainer(args.corpus, args.archetype, args.score_bands,
                        device=args.device, lr=args.lr)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save)


if __name__ == "__main__":
    main()
