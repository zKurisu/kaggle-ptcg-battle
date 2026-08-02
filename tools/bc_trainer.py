#!/usr/bin/env python3
"""BC trainer — cosine LR, train/val split, best-checkpoint saving."""
import sys, os, glob, time, argparse, math, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

_HERE = Path(__file__).resolve().parent; _REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))
from ptcg_rl.model import PolicyValueNet

class BCTrainer:
    def __init__(self, corpus_dir, archetype, score_bands, device="cuda", lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet().to(self.device); self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.base_lr = lr
        print(f"BC: {archetype} [{', '.join(score_bands)}] {self.device} lr={lr}", flush=True)

        t0 = time.time()
        self.npz_data = []
        ad = os.path.join(corpus_dir, archetype.replace(' ', '_'))
        n_files = sum(1 for band in score_bands
                      for _ in glob.glob(os.path.join(ad, band.replace(' ', '_'), "*.npz")))
        fi = 0; total = 0
        print(f"  Loading {n_files} files:", flush=True)
        for band in score_bands:
            for npz_path in sorted(glob.glob(os.path.join(ad, band.replace(' ', '_'), "*.npz"))):
                t1 = time.time(); d = np.load(npz_path, allow_pickle=True)
                n = len(d['board']); self.npz_data.append(d); total += n; fi += 1
                print(f"    [{fi}/{n_files}] {os.path.basename(npz_path)}: {n} decs, {time.time()-t1:.1f}s", flush=True)
        print(f"  {total} decisions, {time.time()-t0:.0f}s", flush=True)

    def _compute_loss(self, mb_idx):
        B = len(mb_idx)
        N = max(len(self.npz_data[di]['ot'][si]) for di, si in mb_idx)
        board = torch.zeros(B, 12, dtype=torch.long, device=self.device)
        hand = torch.zeros(B, 25, dtype=torch.long, device=self.device)
        feats = torch.zeros(B, 32, device=self.device)
        ot = torch.zeros(B, N, dtype=torch.long, device=self.device)
        oc = torch.zeros(B, N, dtype=torch.long, device=self.device)
        oc2 = torch.zeros(B, N, dtype=torch.long, device=self.device)
        oa = torch.zeros(B, N, dtype=torch.long, device=self.device)
        of_arr = torch.zeros(B, N, 16, device=self.device)
        acts = []; mn = []; mx = []
        for bi, (di, si) in enumerate(mb_idx):
            d = self.npz_data[di]; nn = len(d['ot'][si])
            board[bi] = torch.from_numpy(np.asarray(d['board'][si], dtype=np.int64))
            hand[bi,:len(d['hand'][si])] = torch.from_numpy(np.asarray(d['hand'][si], dtype=np.int64))
            feats[bi] = torch.from_numpy(np.asarray(d['feats'][si], dtype=np.float32))
            ot[bi,:nn] = torch.from_numpy(np.asarray(d['ot'][si], dtype=np.int64))
            oc[bi,:nn] = torch.from_numpy(np.asarray(d['oc'][si], dtype=np.int64))
            oc2[bi,:nn] = torch.from_numpy(np.asarray(d['oc2'][si], dtype=np.int64))
            oa[bi,:nn] = torch.from_numpy(np.asarray(d['oa'][si], dtype=np.int64))
            of_arr[bi,:nn] = torch.from_numpy(np.asarray(d['of_arr'][si], dtype=np.float32))
            acts.append(torch.tensor(d['action'][si].astype(np.int64), device=self.device))
            mn.append(int(d['min_c'][si])); mx.append(int(d['max_c'][si]))

        h = self.model.encode_state(board, hand, feats)
        opts = self.model.encode_options(ot, oc, oc2, oa, of_arr)

        ps = torch.zeros(B, 128, device=self.device)
        avail = torch.ones(B, N + 1, dtype=torch.bool, device=self.device)
        mp = max(len(a) for a in acts)
        tgt = torch.full((B, mp + 1), -1, dtype=torch.long, device=self.device)
        for i in range(B):
            for k, a in enumerate(acts[i]):
                if 0 <= int(a) < N: tgt[i, k] = int(a)
            if len(acts[i]) > 0 and len(acts[i]) >= mn[i]:
                tgt[i, len(acts[i])] = N

        pol = torch.tensor(0.0, device=self.device)
        weights = torch.ones(B, device=self.device)
        for k in range(mp + 1):
            sok = torch.tensor([k >= mn[i] for i in range(B)], device=self.device)
            mask = avail.clone(); mask[:, N] = sok
            logits = self.model.option_logits(h, opts, ps, mask)
            logp = F.log_softmax(logits, dim=-1)
            tk = tgt[:, k]; vld = (tk >= 0)
            if vld.any():
                idx = tk.clamp(min=0).unsqueeze(1)
                lp = logp.gather(1, idx).squeeze(1)
                # Apply inverse-frequency weights
                for i in range(B):
                    if vld[i]:
                        ti = int(tk[i])
                        weights[i] = self.action_weights.get(ti, 1.0)
                pol = pol - (lp * vld.float() * weights).sum()
                for i in range(B):
                    if vld[i] and tk[i] < N:
                        ps[i] = ps[i] + opts[i, int(tk[i])]
                        avail[i, int(tk[i])] = False
        return pol / B

    def train(self, epochs=10, batch_size=128, save_path="checkpoints/bc_policy.npz"):
        # Build flat index list — FILTER empty actions (no learning signal)
        def _has_action(di, si):
            return len(self.npz_data[di]['action'][si]) > 0
        npz_groups = [(di, [si for si in range(len(d['board'])) if _has_action(di, si)])
                      for di, d in enumerate(self.npz_data)]
        npz_groups = [(di, idxs) for di, idxs in npz_groups if idxs]  # remove empty groups
        np.random.shuffle(npz_groups)
        split = max(1, int(len(npz_groups) * 0.9))
        train_idx = [(di, si) for di, indices in npz_groups[:split] for si in indices]
        val_idx   = [(di, si) for di, indices in npz_groups[split:] for si in indices]
        n_train = len(train_idx); train_batches = (n_train + batch_size - 1) // batch_size
        n_val = len(val_idx); val_batches = (n_val + batch_size - 1) // batch_size
        print(f"  Train: {n_train} ({train_batches} batches) | Val: {n_val} ({val_batches} batches) [empty filtered]", flush=True)

        # Inverse-frequency weights for action balancing
        action_counts = {}
        for di, d in enumerate(self.npz_data):
            for si in range(len(d['board'])):
                acts = d['action'][si]
                for a in acts:
                    a_int = int(a)
                    action_counts[a_int] = action_counts.get(a_int, 0) + 1
        total = sum(action_counts.values())
        self.action_weights = {a: total / max(c, 1) for a, c in action_counts.items()}
        print(f"  Action weights: {len(self.action_weights)} unique actions (range {min(self.action_weights.values()):.1f}-{max(self.action_weights.values()):.1f})", flush=True)

        # Cosine LR over all training steps
        total_steps = epochs * train_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.base_lr * 0.01)
        best_val = float('inf')

        for ep in range(epochs):
            # --- Train ---
            self.model.train()
            np.random.shuffle(train_idx)
            tl = 0.0; st = 0; t0 = time.time()
            for start in range(0, n_train, batch_size):
                mb_idx = train_idx[start:start + batch_size]
                if len(mb_idx) < 2: continue
                loss = self._compute_loss(mb_idx)
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step(); scheduler.step()
                tl += loss.item(); st += 1
                e=time.time()-t0; pct=st/train_batches*100
                print(f"  {pct:3.0f}% {st}/{train_batches} loss={tl/st:.4f} lr={scheduler.get_last_lr()[0]:.2e} {e:.0f}s", flush=True)
            train_loss = tl / max(st, 1)
            e_train = time.time() - t0

            # --- Val ---
            self.model.eval()
            vl = 0.0; vs = 0
            with torch.no_grad():
                for start in range(0, n_val, batch_size):
                    mb_idx = val_idx[start:start + batch_size]
                    if len(mb_idx) < 2: continue
                    vl += self._compute_loss(mb_idx).item(); vs += 1
            val_loss = vl / max(vs, 1)

            overfit = "⚠️ OVERFIT" if (val_loss > train_loss * 1.2) else ""
            print(f"  epoch {ep+1}/{epochs} train={train_loss:.4f} val={val_loss:.4f} {e_train:.0f}s {overfit}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                state = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
                np.savez_compressed(save_path, **state)
                print(f"  → best val={val_loss:.4f} saved", flush=True)

        print(f"Best: {best_val:.4f} → {save_path} ({os.path.getsize(save_path)/1024**2:.1f}MB)")


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
    BCTrainer(args.corpus, args.archetype, args.score_bands, device=args.device, lr=args.lr
             ).train(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save)

if __name__ == "__main__": main()
