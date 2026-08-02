#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, greedy_decode
from ptcg_rl.encoder import STATE_FEAT_DIM
from ptcg_rl.model import PolicyValueNet

CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
    5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK",
    10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE", 13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE", 16: "REMOVE_DAMAGE_COUNTER", 17: "HEAL",
    18: "EVOLVES_FROM", 19: "EVOLVES_TO", 20: "DEVOLVE", 21: "ATTACH_FROM",
    22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK", 25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD", 28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD", 30: "DISCARD_ENERGY", 31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY", 33: "SWITCH_ENERGY", 34: "SKILL_ORDER", 35: "ATTACK",
    36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT", 39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT", 41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
    44: "FIRST_EFFECT", 45: "MORE_DEVOLVE", 46: "COIN_HEAD", 47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}
OPT_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD",
    6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}


def _bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    parser.add_argument("--archetype", default="Marnie Grimmsnarl")
    parser.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--include-empty", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    corpus = BCCorpus(paths, include_empty=args.include_empty)
    indices = corpus.all_indices()
    if args.stride > 1:
        indices = indices[:: args.stride]
    indices = indices[: args.max_samples]

    with np.load(args.policy) as z:
        option_context = "context_emb.weight" in z.files
        ec = z["card_emb.weight"].shape[1]
        slot_state = z["state_fc1.weight"].shape[1] == 5 * ec + STATE_FEAT_DIM
        model = PolicyValueNet(
            width=args.width,
            option_context=option_context,
            slot_state=slot_state,
        ).to(device)
        state = {k: torch.as_tensor(z[k], device=device) for k in z.files}
    model.load_state_dict(state)
    model.eval()

    n = exact = first = pred_empty = true_empty = len_match = 0
    by_ctx = defaultdict(lambda: [0, 0, 0])
    by_opt = defaultdict(lambda: [0, 0, 0])
    by_nopt = defaultdict(lambda: [0, 0, 0])
    start = time.time()
    next_progress = args.progress_every if args.progress_every else None

    for batch_start in range(0, len(indices), args.batch_size):
        batch = corpus.collate(indices[batch_start : batch_start + args.batch_size], device)
        preds = greedy_decode(model, batch)
        for pred, true, ctx, opt, nopt in zip(
            preds, batch.actions, batch.contexts, batch.true_first_types, batch.n_options
        ):
            n += 1
            exact_ok = int(pred == true)
            first_ok = int(bool(pred) and bool(true) and pred[0] == true[0])
            exact += exact_ok
            first += first_ok
            len_match += int(len(pred) == len(true))
            pred_empty += int(not pred)
            true_empty += int(not true)
            for table, key in ((by_ctx, ctx), (by_opt, opt), (by_nopt, _bucket(nopt))):
                table[key][0] += 1
                table[key][1] += first_ok
                table[key][2] += exact_ok
        if next_progress is not None and (n >= next_progress or n == len(indices)):
            rate = n / max(time.time() - start, 1e-9)
            eta = max(len(indices) - n, 0) / max(rate, 1e-9)
            print(
                f"  {n}/{len(indices)} exact={exact/n:.3f} "
                f"first={first/max(n-true_empty,1):.3f} {rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            while next_progress is not None and next_progress <= n:
                next_progress += args.progress_every

    print(f"Policy: {args.policy}")
    print(f"Samples: {n} from {len(paths)} files")
    print(f"Corpus labels: {corpus.stats}")
    print(f"Elapsed: {time.time() - start:.1f}s")
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
