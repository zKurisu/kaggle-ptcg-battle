#!/usr/bin/env python3
"""BC trainer — preloads data into RAM for fast iteration."""
import sys, os, glob, time, argparse, numpy as np
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent; _REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))
from ptcg_rl.model import PolicyValueNet

class BCTrainer:
    def __init__(self, corpus_dir, archetype, score_bands, device="cuda", lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet().to(self.device); self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"BC: {archetype} [{', '.join(score_bands)}] {self.device}", flush=True)

        # Pre-load npz files (keep as npz, don't extract dicts)
        t0 = time.time()
        self.npz_data = []
        ad = os.path.join(corpus_dir, archetype.replace(' ', '_'))
        n_files = sum(1 for band in score_bands for _ in glob.glob(os.path.join(ad, band.replace(' ', '_'), "*.npz")))
        fi = 0; total = 0
        print(f"  Loading {n_files} files:", flush=True)
        for band in score_bands:
            for npz_path in sorted(glob.glob(os.path.join(ad, band.replace(' ', '_'), "*.npz"))):
                t1 = time.time()
                d = np.load(npz_path, allow_pickle=True)
                n = len(d['board'])
                self.npz_data.append(d)
                total += n; fi += 1
                print(f"    [{fi}/{n_files}] {os.path.basename(npz_path)}: {n} decs, load={time.time()-t1:.1f}s", flush=True)
        self.n_samples = total
        print(f"  {total} decisions, {time.time()-t0:.0f}s", flush=True)

    def train(self, epochs=10, batch_size=128, save_path="checkpoints/bc_policy.npz"):
        # Flatten: list of (npz_idx, sample_idx) for all samples
        all_idx = []
        for di, d in enumerate(self.npz_data):
            for i in range(len(d['board'])):
                all_idx.append((di, i))
        n = len(all_idx); per_epoch = (n + batch_size - 1) // batch_size
        print(f"  {per_epoch} batches/epoch ({n} samples)", flush=True)

        for ep in range(epochs):
            np.random.shuffle(all_idx)
            tl = tp = tv = 0.0; st = 0; t0 = time.time()

            for start in range(0, n, batch_size):
                mb_idx = all_idx[start:start + batch_size]
                if len(mb_idx) < 2: continue

                # Build batch directly from npz indices (no dicts!)
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
                    hand[bi, :len(d['hand'][si])] = torch.from_numpy(np.asarray(d['hand'][si], dtype=np.int64))
                    feats[bi] = torch.from_numpy(np.asarray(d['feats'][si], dtype=np.float32))
                    ot[bi, :nn] = torch.from_numpy(np.asarray(d['ot'][si], dtype=np.int64))
                    oc[bi, :nn] = torch.from_numpy(np.asarray(d['oc'][si], dtype=np.int64))
                    oc2[bi, :nn] = torch.from_numpy(np.asarray(d['oc2'][si], dtype=np.int64))
                    oa[bi, :nn] = torch.from_numpy(np.asarray(d['oa'][si], dtype=np.int64))
                    of_arr[bi, :nn] = torch.from_numpy(np.asarray(d['of_arr'][si], dtype=np.float32))
                    acts.append(torch.tensor(d['action'][si].astype(np.int64), device=self.device))
                    mn.append(int(d['min_c'][si])); mx.append(int(d['max_c'][si]))

                # Forward (keep on GPU, no CPU round-trip)
                h = self.model.encode_state(board, hand, feats)
                opts = self.model.encode_options(ot, oc, oc2, oa, of_arr)
                value = self.model.value(h)

                # Sequential policy loss — all on GPU
                ps = torch.zeros(B, 128, device=self.device)
                avail = torch.ones(B, N + 1, dtype=torch.bool, device=self.device)
                mp = max(len(a) for a in acts)
                tgt = torch.full((B, mp + 1), -1, dtype=torch.long, device=self.device)
                for i in range(B):
                    for k, a in enumerate(acts[i]):
                        a_int = int(a)
                        if 0 <= a_int < N: tgt[i, k] = a_int
                    # Only set STOP target if human's action count ≥ min_count
                    # (otherwise the data is inconsistent — skip)
                    if len(acts[i]) > 0 and len(acts[i]) >= mn[i]:
                        tgt[i, len(acts[i])] = N

                pol_sum = 0.0
                for k in range(mp + 1):
                    sok = torch.tensor([k >= mn[i] for i in range(B)], device=self.device)
                    mask = avail.clone(); mask[:, N] = sok
                    logits = self.model.option_logits(h, opts, ps, mask)
                    logp = F.log_softmax(logits, dim=-1)
                    tk = tgt[:, k]; vld = (tk >= 0)
                    if vld.any():
                        idx = tk.clamp(min=0).unsqueeze(1)
                        lp = logp.gather(1, idx).squeeze(1)
                        pol_sum += float(-(lp * vld.float()).sum().item())
                        for i in range(B):
                            if vld[i] and tk[i] < N:
                                ti = int(tk[i])
                                ps[i] = ps[i].detach() + opts[i, ti].detach()
                                avail[i, ti] = False
                pol = pol_sum / B

                vloss = (value - torch.ones(B, device=self.device)).pow(2).mean()
                loss = torch.tensor(pol, device=self.device) + 0.5 * vloss
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                tl += loss.item(); tp += pol; tv += vloss.item(); st += 1
                dbg = f" | pol={pol:.1f}"
                e=time.time()-t0;pct=st/per_epoch*100
                print(f"  {pct:3.0f}% {st}/{per_epoch} loss={tl/st:.1f} pol={tp/st:.1f} val={tv/st:.3f} {e:.0f}s{dbg}", flush=True)

            e=time.time()-t0
            print(f"  100% {st}/{per_epoch} loss={tl/st:.3f} pol={tp/st:.3f} val={tv/st:.3f} {e:.0f}s", flush=True)

        state={k:v.cpu().numpy() for k,v in self.model.state_dict().items()}
        np.savez_compressed(save_path,**state)
        print(f"Saved: {save_path} ({os.path.getsize(save_path)/1024**2:.1f}MB)")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",default="data/bc_corpus_banded")
    p.add_argument("--archetype",default="Marnie Grimmsnarl")
    p.add_argument("--score-bands",nargs="+",default=["1200+","1100-1199"])
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--save",default="checkpoints/bc_policy.npz")
    args=p.parse_args()
    BCTrainer(args.corpus,args.archetype,args.score_bands,device=args.device,lr=args.lr
             ).train(epochs=args.epochs,batch_size=args.batch_size,save_path=args.save)

if __name__=="__main__": main()
