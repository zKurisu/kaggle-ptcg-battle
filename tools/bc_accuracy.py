#!/usr/bin/env python3
"""Offline imitation accuracy for BC checkpoints on extracted .npz corpora."""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

NEG_INF = -1e9


def _relu(x):
    return np.maximum(x, 0.0)


def _linear(w, b, x):
    return x @ w.T + b


def _pool(w, ids):
    e = w["card_emb.weight"][ids]
    mask = (ids > 0).astype(np.float32)[:, None]
    return (e * mask).sum(axis=0) / (mask.sum() + 1e-8)


def _predict(w, sample, include_stop=True):
    board = np.asarray(sample["board"], dtype=np.int64)
    hand = np.asarray(sample["hand"], dtype=np.int64)
    feats = np.asarray(sample["feats"], dtype=np.float32)
    ot = np.asarray(sample["ot"], dtype=np.int64)
    oc = np.asarray(sample["oc"], dtype=np.int64)
    oc2 = np.asarray(sample["oc2"], dtype=np.int64)
    oa = np.asarray(sample["oa"], dtype=np.int64)
    of_arr = np.asarray(sample["of_arr"], dtype=np.float32)
    mn = int(sample["min_c"])
    mx = int(sample["max_c"])
    n = len(ot)

    my = _pool(w, board[:6])
    opp = _pool(w, board[6:])
    hnd = _pool(w, hand)
    x = np.concatenate([my, opp, hnd, feats])
    h = _relu(_linear(w["state_fc1.weight"], w["state_fc1.bias"], x))
    h = _relu(_linear(w["state_fc2.weight"], w["state_fc2.bias"], h))

    opt_x = np.concatenate([
        w["card_emb.weight"][oc],
        w["card_emb.weight"][oc2],
        w["attack_emb.weight"][oa],
        w["opt_type_emb.weight"][ot],
        of_arr,
    ], axis=-1)
    opts = _relu(_linear(w["opt_fc.weight"], w["opt_fc.bias"], opt_x))

    oe = w["stop_vec"].shape[0]
    hd = w["state_fc2.bias"].shape[0]
    picks = []
    picked_sum = np.zeros(oe, dtype=np.float32)
    avail = np.ones(n + 1, dtype=bool)
    while len(picks) < mx:
        avail[n] = include_stop and len(picks) >= mn
        rows = np.concatenate([opts, w["stop_vec"][None, :]], axis=0)
        hx = np.broadcast_to(h, (n + 1, hd))
        px = np.broadcast_to(picked_sum, (n + 1, oe))
        score_x = np.concatenate([hx, rows, px], axis=-1)
        logits = _linear(
            w["score_fc2.weight"],
            w["score_fc2.bias"],
            _relu(_linear(w["score_fc1.weight"], w["score_fc1.bias"], score_x)),
        ).reshape(-1)
        logits = np.where(avail, logits, NEG_INF)
        idx = int(np.argmax(logits))
        if idx >= n:
            break
        picks.append(idx)
        picked_sum += opts[idx]
        avail[idx] = False
    return picks


def _iter_samples(paths, include_empty):
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        for i in range(len(data["board"])):
            act = np.asarray(data["action"][i], dtype=np.int64).tolist()
            if not include_empty and len(act) == 0:
                continue
            yield {
                "board": data["board"][i],
                "hand": data["hand"][i],
                "feats": data["feats"][i],
                "ot": data["ot"][i],
                "oc": data["oc"][i],
                "oc2": data["oc2"][i],
                "oa": data["oa"][i],
                "of_arr": data["of_arr"][i],
                "action": act,
                "min_c": int(data["min_c"][i]),
                "max_c": int(data["max_c"][i]),
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--corpus", default="data/bc_corpus_banded")
    p.add_argument("--archetype", default="Marnie Grimmsnarl")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every Nth sample after filtering")
    args = p.parse_args()

    arch_dir = os.path.join(args.corpus, args.archetype.replace(" ", "_"))
    paths = []
    for band in args.score_bands:
        paths.extend(sorted(glob.glob(os.path.join(arch_dir, band.replace(" ", "_"), "*.npz"))))
    if not paths:
        raise FileNotFoundError(f"No corpus files found under {arch_dir}")

    with np.load(args.policy) as z:
        w = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}

    n = seen = exact = first = pred_empty = true_empty = 0
    len_match = 0
    t0 = time.time()
    for s in _iter_samples(paths, args.include_empty):
        seen += 1
        if args.stride > 1 and (seen - 1) % args.stride != 0:
            continue
        pred = _predict(w, s)
        true = [int(a) for a in s["action"] if 0 <= int(a) < len(s["ot"])]
        n += 1
        exact += int(pred == true)
        len_match += int(len(pred) == len(true))
        first += int(bool(pred) and bool(true) and pred[0] == true[0])
        pred_empty += int(len(pred) == 0)
        true_empty += int(len(true) == 0)
        if args.progress_every and n % args.progress_every == 0:
            dt = time.time() - t0
            rate = n / max(dt, 1e-9)
            remaining = max(args.max_samples - n, 0) / max(rate, 1e-9)
            print(
                f"  {n}/{args.max_samples} "
                f"exact={exact/n:.3f} first={first/max(n-true_empty,1):.3f} "
                f"pred_empty={pred_empty/n:.3f} {rate:.1f}/s eta={remaining:.0f}s",
                flush=True,
            )
        if n >= args.max_samples:
            break

    print(f"Policy: {args.policy}")
    print(f"Samples: {n} from {len(paths)} files")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Exact action seq: {exact / max(n, 1):.3f}")
    print(f"First action:     {first / max(n - true_empty, 1):.3f} over non-empty labels")
    print(f"Length match:     {len_match / max(n, 1):.3f}")
    print(f"True empty:       {true_empty / max(n, 1):.3f}")
    print(f"Pred empty:       {pred_empty / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
