#!/usr/bin/env python3
"""Offline imitation accuracy for BC checkpoints on extracted .npz corpora."""
import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

NEG_INF = -1e9

CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH",
    4: "TO_ACTIVE", 5: "TO_BENCH", 7: "TO_HAND", 8: "DISCARD",
    13: "DAMAGE_COUNTER", 15: "DAMAGE", 16: "REMOVE_DAMAGE_COUNTER",
    21: "ATTACH_FROM", 22: "ATTACH_TO", 29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY", 34: "SKILL_ORDER", 35: "ATTACK", 37: "EVOLVE",
    38: "DRAW_COUNT", 39: "DAMAGE_COUNTER_COUNT", 40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
}

OPT_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
    5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH",
    9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
    13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}


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

    ec = w["card_emb.weight"].shape[1]
    if w["state_fc1.weight"].shape[1] == 5 * ec + len(feats):
        emb = w["card_emb.weight"]
        my_active = emb[board[0]]
        my_bench = _pool(w, board[1:6])
        opp_active = emb[board[6]]
        opp_bench = _pool(w, board[7:])
        hnd = _pool(w, hand)
        x = np.concatenate([my_active, my_bench, opp_active, opp_bench, hnd, feats])
    else:
        my = _pool(w, board[:6])
        opp = _pool(w, board[6:])
        hnd = _pool(w, hand)
        x = np.concatenate([my, opp, hnd, feats])
    h = _relu(_linear(w["state_fc1.weight"], w["state_fc1.bias"], x))
    h = _relu(_linear(w["state_fc2.weight"], w["state_fc2.bias"], h))

    parts = [
        w["card_emb.weight"][oc],
        w["card_emb.weight"][oc2],
        w["attack_emb.weight"][oa],
        w["opt_type_emb.weight"][ot],
    ]
    if "context_emb.weight" in w:
        ctx = np.rint(of_arr[:, 3] * 64.0).astype(np.int64).clip(0, 64)
        sel_type = np.rint(of_arr[:, 4] * 16.0).astype(np.int64).clip(0, 16)
        area = np.rint(of_arr[:, 7] * 16.0).astype(np.int64).clip(0, 16)
        idx = np.rint(of_arr[:, 8] * 64.0).astype(np.int64).clip(0, 64)
        inplay_area = np.rint(of_arr[:, 9] * 16.0).astype(np.int64).clip(0, 16)
        inplay_idx = np.rint(of_arr[:, 10] * 10.0).astype(np.int64).clip(0, 16)
        parts.extend([
            w["context_emb.weight"][ctx],
            w["select_type_emb.weight"][sel_type],
            w["area_emb.weight"][area],
            w["index_emb.weight"][idx],
            w["inplay_area_emb.weight"][inplay_area],
            w["inplay_index_emb.weight"][inplay_idx],
        ])
    parts.append(of_arr)
    opt_x = np.concatenate(parts, axis=-1)
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


def _label_status(data, i, include_empty):
    action = np.asarray(data["action"][i], dtype=np.int64)
    n_opt = len(data["ot"][i])
    mn = int(data["min_c"][i])
    mx = int(data["max_c"][i])
    if len(action) == 0:
        return "keep" if include_empty and mn == 0 else "empty"
    if len(action) < mn or len(action) > mx:
        return "bad"
    if len(set(action.tolist())) != len(action):
        return "bad"
    if not ((action >= 0) & (action < n_opt)).all():
        return "bad"
    return "keep"


def _iter_samples(paths, include_empty):
    stats = {"raw": 0, "empty": 0, "bad": 0, "kept": 0}
    _iter_samples.stats = stats
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        for i in range(len(data["board"])):
            stats["raw"] += 1
            status = _label_status(data, i, include_empty)
            if status != "keep":
                stats[status] += 1
                continue
            stats["kept"] += 1
            act = np.asarray(data["action"][i], dtype=np.int64).tolist()
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
    by_ctx = defaultdict(lambda: [0, 0, 0])
    by_opt = defaultdict(lambda: [0, 0, 0])
    by_nopt = defaultdict(lambda: [0, 0, 0])
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
        first_ok = int(bool(pred) and bool(true) and pred[0] == true[0])
        exact_ok = int(pred == true)
        first += first_ok
        pred_empty += int(len(pred) == 0)
        true_empty += int(len(true) == 0)
        ctx = int(round(float(np.asarray(s["feats"], dtype=np.float32)[17]) * 64.0))
        true0 = true[0] if true else -1
        opt0 = int(np.asarray(s["ot"], dtype=np.int64)[true0]) if true0 >= 0 else -1
        nopt = len(s["ot"])
        nopt_bucket = "1" if nopt == 1 else "2" if nopt == 2 else "3-5" if nopt <= 5 else "6-10" if nopt <= 10 else "11+"
        for table, key in ((by_ctx, ctx), (by_opt, opt0), (by_nopt, nopt_bucket)):
            table[key][0] += 1
            table[key][1] += first_ok
            table[key][2] += exact_ok
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
    stats = getattr(_iter_samples, "stats", {})
    if stats:
        print(
            f"Corpus labels: raw={stats['raw']} kept={stats['kept']} "
            f"skipped_empty={stats['empty']} skipped_bad={stats['bad']}"
        )
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Exact action seq: {exact / max(n, 1):.3f}")
    print(f"First action:     {first / max(n - true_empty, 1):.3f} over non-empty labels")
    print(f"Length match:     {len_match / max(n, 1):.3f}")
    print(f"True empty:       {true_empty / max(n, 1):.3f}")
    print(f"Pred empty:       {pred_empty / max(n, 1):.3f}")
    print("\nBy context:")
    for key, (cnt, fst, ex) in sorted(by_ctx.items(), key=lambda kv: kv[1][0], reverse=True)[:20]:
        print(f"  {key:2d} {CONTEXT_NAMES.get(key, '?'):<18} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f}")
    print("\nBy true first option type:")
    for key, (cnt, fst, ex) in sorted(by_opt.items(), key=lambda kv: kv[1][0], reverse=True)[:20]:
        print(f"  {key:2d} {OPT_NAMES.get(key, '?'):<18} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f}")
    print("\nBy option count:")
    for key, (cnt, fst, ex) in sorted(by_nopt.items(), key=lambda kv: str(kv[0])):
        print(f"  {key:<4} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f}")


if __name__ == "__main__":
    main()
