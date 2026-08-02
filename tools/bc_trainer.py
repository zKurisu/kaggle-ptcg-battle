#!/usr/bin/env python3
"""BC trainer — efficient npz-backed collation, cosine LR, checkpoints."""
import sys, os, glob, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

_HERE = Path(__file__).resolve().parent; _REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))
from ptcg_rl.model import PolicyValueNet

class BCTrainer:
    def __init__(self, corpus_dir, archetype, score_bands, device="cuda", lr=1e-4,
                 width=2.0, include_empty=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.include_empty = include_empty
        self.model = PolicyValueNet(width=width).to(self.device); self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        p = sum(p.numel() for p in self.model.parameters())
        print(f"BC: {archetype} [{', '.join(score_bands)}] {self.device} lr={lr} width={width} params={p/1e6:.1f}M", flush=True)

        def keep_sample(d, i):
            action = np.asarray(d['action'][i], dtype=np.int64)
            n_opt = len(d['ot'][i])
            mn = int(d['min_c'][i])
            mx = int(d['max_c'][i])
            if len(action) == 0:
                return include_empty and mn == 0
            if len(action) < mn or len(action) > mx:
                return False
            if len(set(action.tolist())) != len(action):
                return False
            return bool(((action >= 0) & (action < n_opt)).all())

        t0 = time.time()
        ad = os.path.join(corpus_dir, archetype.replace(' ', '_'))
        n_files = sum(1 for b in score_bands for _ in glob.glob(os.path.join(ad, b.replace(' ', '_'), "*.npz")))
        fi = 0; total = 0; kept = 0; skipped_empty = 0; skipped_bad = 0
        self.npz_data = []
        self.groups = []
        print(f"  Loading {n_files} files:", flush=True)
        for band in score_bands:
            for npz_path in sorted(glob.glob(os.path.join(ad, band.replace(' ', '_'), "*.npz"))):
                t1 = time.time()
                with np.load(npz_path, allow_pickle=True) as z:
                    d = {k: z[k] for k in z.files}
                n = len(d['board'])
                di = len(self.npz_data)
                actions = d['action']
                idxs = []
                bad = empty = 0
                for i in range(n):
                    if keep_sample(d, i):
                        idxs.append(i)
                    elif len(actions[i]) == 0:
                        empty += 1
                    else:
                        bad += 1
                self.npz_data.append(d)
                self.groups.append([(di, i) for i in idxs])
                total += n; fi += 1
                kept += len(idxs)
                skipped_empty += empty
                skipped_bad += bad
                suffix = f", skipped {empty} empty, {bad} bad" if empty or bad else ""
                print(f"    [{fi}/{n_files}] {os.path.basename(npz_path)}: {n} decs{suffix}, {time.time()-t1:.1f}s", flush=True)
        if kept == 0:
            raise RuntimeError(f"No BC samples found under {ad} for bands {score_bands}")
        print(f"  {kept}/{total} decisions kept, skipped {skipped_empty} empty, {skipped_bad} bad, {time.time()-t0:.0f}s", flush=True)

    def _collate(self, indices):
        B = len(indices)
        n_opt = [len(self.npz_data[di]['ot'][si]) for di, si in indices]
        N = max(n_opt)
        board_np = np.empty((B, 12), dtype=np.int64)
        hand_np = np.zeros((B, 25), dtype=np.int64)
        feats_np = np.empty((B, 32), dtype=np.float32)
        ot_np = np.zeros((B, N), dtype=np.int64)
        oc_np = np.zeros((B, N), dtype=np.int64)
        oc2_np = np.zeros((B, N), dtype=np.int64)
        oa_np = np.zeros((B, N), dtype=np.int64)
        of_np = np.zeros((B, N, 16), dtype=np.float32)
        acts = []
        mn_np = np.empty(B, dtype=np.int64)
        mx_np = np.empty(B, dtype=np.int64)

        for bi, (di, si) in enumerate(indices):
            d = self.npz_data[di]
            nn = n_opt[bi]
            board_np[bi] = np.asarray(d['board'][si], dtype=np.int64)
            h = np.asarray(d['hand'][si], dtype=np.int64)
            hand_np[bi, :min(len(h), 25)] = h[:25]
            feats_np[bi] = np.asarray(d['feats'][si], dtype=np.float32)
            ot_np[bi, :nn] = np.asarray(d['ot'][si], dtype=np.int64)
            oc_np[bi, :nn] = np.asarray(d['oc'][si], dtype=np.int64)
            oc2_np[bi, :nn] = np.asarray(d['oc2'][si], dtype=np.int64)
            oa_np[bi, :nn] = np.asarray(d['oa'][si], dtype=np.int64)
            of_np[bi, :nn] = np.asarray(d['of_arr'][si], dtype=np.float32)
            acts.append(np.asarray(d['action'][si], dtype=np.int64))
            mn_np[bi] = int(d['min_c'][si])
            mx_np[bi] = int(d['max_c'][si])

        mp = max(len(a) for a in acts)
        tgt_np = np.full((B, mp + 1), -1, dtype=np.int64)
        for bi, (a, nn) in enumerate(zip(acts, n_opt)):
            valid = a[(a >= 0) & (a < nn)]
            tgt_np[bi, :len(valid)] = valid
            if len(a) < mx_np[bi] and len(a) >= mn_np[bi] and (len(a) > 0 or self.include_empty):
                tgt_np[bi, len(a)] = N

        to_dev = self.device
        return (
            torch.as_tensor(board_np, device=to_dev),
            torch.as_tensor(hand_np, device=to_dev),
            torch.as_tensor(feats_np, device=to_dev),
            torch.as_tensor(ot_np, device=to_dev),
            torch.as_tensor(oc_np, device=to_dev),
            torch.as_tensor(oc2_np, device=to_dev),
            torch.as_tensor(oa_np, device=to_dev),
            torch.as_tensor(of_np, device=to_dev),
            torch.as_tensor(tgt_np, device=to_dev),
            torch.as_tensor(mn_np, device=to_dev),
            torch.as_tensor(n_opt, dtype=torch.long, device=to_dev),
            N,
        )

    def _compute_loss(self, indices):
        (board, hand, feats, ot, oc, oc2, oa, of_arr,
         tgt, mn, opt_len, N) = self._collate(indices)
        B = board.shape[0]

        h = self.model.encode_state(board, hand, feats)
        opts = self.model.encode_options(ot, oc, oc2, oa, of_arr)

        total_loss = torch.tensor(0.0, device=self.device)
        total_valid = torch.tensor(0.0, device=self.device)
        ps = torch.zeros(B, self.model._oe, device=self.device)
        opt_mask = torch.arange(N, device=self.device).unsqueeze(0) < opt_len.unsqueeze(1)
        avail = torch.cat([
            opt_mask,
            torch.ones(B, 1, dtype=torch.bool, device=self.device),
        ], dim=1)

        for k in range(tgt.shape[1]):
            sok = k >= mn
            mask = avail.clone(); mask[:, N] = sok
            logits = self.model.option_logits(h, opts, ps, mask)
            logp = F.log_softmax(logits, dim=-1)
            tk = tgt[:, k]; vld = (tk >= 0)
            if vld.any():
                idx = tk.clamp(min=0).unsqueeze(1)
                lp = logp.gather(1, idx).squeeze(1)
                vf = vld.float()
                total_loss = total_loss - (lp * vf).sum()
                total_valid = total_valid + vf.sum()
                with torch.no_grad():
                    pick = vld & (tk < N)
                    if pick.any():
                        rows = torch.nonzero(pick, as_tuple=True)[0]
                        cols = tk[pick]
                        ps[rows] += opts[rows, cols]
                        avail[rows, cols] = False
        return total_loss / total_valid.clamp(min=1.0)

    def train(self, epochs=30, batch_size=128, save_path="checkpoints/bc_policy.npz", checkpoint_every=5):
        groups = [g for g in self.groups if g]
        np.random.shuffle(groups)
        split = max(1, int(len(groups) * 0.9))
        if len(groups) > 1:
            split = min(split, len(groups) - 1)
        train_idx = [item for g in groups[:split] for item in g]
        val_idx = [item for g in groups[split:] for item in g]
        if not val_idx:
            n_val = max(1, int(len(train_idx) * 0.1))
            val_idx = train_idx[-n_val:]
            train_idx = train_idx[:-n_val]
        train_batches = (len(train_idx) + batch_size - 1) // batch_size
        val_batches = (len(val_idx) + batch_size - 1) // batch_size
        empty_note = "" if self.include_empty else " [empty filtered]"
        print(f"  Train: {len(train_idx)} ({train_batches} batches) | Val: {len(val_idx)} ({val_batches} batches){empty_note}", flush=True)

        total_steps = epochs * train_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01)
        best_val = float('inf')

        for ep in range(epochs):
            self.model.train()
            np.random.shuffle(train_idx)
            tl = 0.0; st = 0; t0 = time.time()
            for start in range(0, len(train_idx), batch_size):
                idxs = train_idx[start:start + batch_size]
                if len(idxs) < 2: continue
                loss = self._compute_loss(idxs)
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step(); scheduler.step()
                tl += loss.item(); st += 1
                e = time.time() - t0; pct = st / train_batches * 100
                print(f"  {pct:3.0f}% {st}/{train_batches} loss={tl/st:.4f} lr={scheduler.get_last_lr()[0]:.2e} {e:.0f}s", flush=True)
            train_loss = tl / max(st, 1); e_train = time.time() - t0

            self.model.eval()
            vl = 0.0; vs = 0
            with torch.no_grad():
                for start in range(0, len(val_idx), batch_size):
                    idxs = val_idx[start:start + batch_size]
                    if len(idxs) < 2: continue
                    vl += self._compute_loss(idxs).item(); vs += 1
            val_loss = vl / max(vs, 1)
            overfit = " ⚠️OVERFIT" if val_loss > train_loss * 1.2 else ""
            print(f"  epoch {ep+1}/{epochs} train={train_loss:.4f} val={val_loss:.4f} {e_train:.0f}s{overfit}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                sd = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
                np.savez_compressed(save_path, **sd)
                print(f"  → best val={val_loss:.4f} saved", flush=True)
            if checkpoint_every and (ep + 1) % checkpoint_every == 0:
                ckpt = save_path.replace('.npz', f'_ep{ep+1:03d}.npz')
                sd = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
                np.savez_compressed(ckpt, **sd)
                print(f"  → checkpoint {ckpt}", flush=True)

        print(f"Best: {best_val:.4f} → {save_path} ({os.path.getsize(save_path)/1024**2:.1f}MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded")
    p.add_argument("--archetype", default="Marnie Grimmsnarl")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--include-empty", action="store_true",
                   help="also train empty human actions as STOP when legal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save", default="checkpoints/bc_policy.npz")
    args = p.parse_args()
    BCTrainer(args.corpus, args.archetype, args.score_bands, device=args.device,
              lr=args.lr, width=args.width, include_empty=args.include_empty
             ).train(epochs=args.epochs, batch_size=args.batch_size,
                     save_path=args.save, checkpoint_every=args.checkpoint_every)

if __name__ == "__main__": main()
