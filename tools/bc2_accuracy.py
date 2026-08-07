#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from ptcg_rl.model import (
    build_policy_model,
    checkpoint_arch,
    checkpoint_feature_dims,
    checkpoint_hierarchical_plan,
    checkpoint_history_k,
    checkpoint_plan_dim,
    checkpoint_width,
)

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
SET_CONTEXTS = {
    5,   # TO_BENCH
    7,   # TO_HAND
    8,   # DISCARD
    9,   # TO_DECK
    10,  # TO_DECK_BOTTOM
    11,  # TO_PRIZE
    13,  # DAMAGE_COUNTER
    14,  # DAMAGE_COUNTER_ANY
    16,  # REMOVE_DAMAGE_COUNTER
    22,  # ATTACH_TO
    34,  # SKILL_ORDER
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


@torch.no_grad()
def first_action_topk(model, batch, ks: tuple[int, ...]) -> list[list[int]]:
    h = model.encode_state(batch.board, batch.hand, batch.feats, batch.history)
    opts = model.encode_options(batch.opt_type, batch.opt_card, batch.opt_card2, batch.opt_attack, batch.opt_feats)
    max_options = batch.max_options
    device = batch.board.device
    opt_mask = torch.arange(max_options, device=device).unsqueeze(0) < batch.opt_len.unsqueeze(1)
    stop_ok = batch.min_count <= 0
    mask = torch.cat([opt_mask, stop_ok.unsqueeze(1)], dim=1)
    picked_sum = torch.zeros(batch.board.shape[0], model._oe, device=device)
    logits = model.option_logits(h, opts, picked_sum, mask)
    out: list[list[int]] = []
    kmax = max(ks)
    choices = torch.topk(logits, k=min(kmax, logits.shape[1]), dim=-1).indices.detach().cpu().numpy()
    for row, nopt in zip(choices, batch.n_options):
        out.append([int(x) for x in row if int(x) < nopt])
    return out


def _add(table, key, first_ok: int, exact_ok: int, top2_ok: int, top3_ok: int) -> None:
    row = table[key]
    row[0] += 1
    row[1] += first_ok
    row[2] += exact_ok
    row[3] += top2_ok
    row[4] += top3_ok


def _add_set(table, key, pred: list[int], true: list[int]) -> None:
    ps = set(pred)
    ts = set(true)
    inter = len(ps & ts)
    prec = inter / max(len(ps), 1)
    rec = inter / max(len(ts), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    row = table[key]
    row[0] += 1
    row[1] += int(ps == ts)
    row[2] += prec
    row[3] += rec
    row[4] += f1


ACC_FIELDS = [
    "table", "key", "label", "n",
    "first", "exact", "top2", "top3",
    "set_exact", "precision", "recall", "f1",
]


def _ensure_header(path: str) -> tuple[Path, bool]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists() and out.stat().st_size > 0
    return out, exists


def _write_table(path: str, name: str, table: dict, labels: dict | None = None) -> None:
    if not path:
        return
    out, exists = _ensure_header(path)
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACC_FIELDS)
        if not exists:
            w.writeheader()
        for key, vals in sorted(table.items(), key=lambda kv: kv[1][0], reverse=True):
            cnt, fst, ex, t2, t3 = vals
            label = labels.get(key, "") if labels else ""
            w.writerow({
                "table": name, "key": key, "label": label, "n": cnt,
                "first": fst / cnt, "exact": ex / cnt, "top2": t2 / cnt, "top3": t3 / cnt,
                "set_exact": "", "precision": "", "recall": "", "f1": "",
            })


def _write_set_table(path: str, name: str, table: dict, labels: dict | None = None) -> None:
    if not path:
        return
    out, exists = _ensure_header(path)
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACC_FIELDS)
        if not exists:
            w.writeheader()
        for key, vals in sorted(table.items(), key=lambda kv: kv[1][0], reverse=True):
            cnt, ex, prec, rec, f1 = vals
            label = labels.get(key, "") if labels else ""
            w.writerow({
                "table": name, "key": key, "label": label, "n": cnt,
                "first": "", "exact": "", "top2": "", "top3": "",
                "set_exact": ex / cnt, "precision": prec / cnt, "recall": rec / cnt, "f1": f1 / cnt,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    parser.add_argument("--archetype", default="Marnie Grimmsnarl")
    parser.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    parser.add_argument("--deck-sig", action="append", default=[],
                        help="filter to one or more deck signatures; repeatable. Requires freshly extracted corpus metadata.")
    parser.add_argument("--team-name", action="append", default=[],
                        help="filter to one or more exact team names; repeatable")
    parser.add_argument("--opponent-deck-sig", action="append", default=[],
                        help="filter to decisions from games against one or more opponent deck signatures")
    parser.add_argument("--opponent-archetype", action="append", default=[],
                        help="filter to decisions from games against one or more opponent archetypes")
    parser.add_argument("--opponent-team-name", action="append", default=[],
                        help="filter to decisions from games against one or more exact opponent team names")
    parser.add_argument("--width", type=float, default=0.0,
                        help="model width; 0 infers from checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--load-progress-every", type=int, default=200000,
                        help="print corpus indexing progress every N raw decisions; 0 disables")
    parser.add_argument("--winner-only", action="store_true",
                        help="evaluate only labels from games this player won; requires outcome metadata")
    parser.add_argument("--out-csv", default="", help="append grouped accuracy tables to CSV")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with np.load(args.policy) as z:
        arch = checkpoint_arch(z.files)
        state_feat_dim, opt_feat_dim, option_context, slot_state = checkpoint_feature_dims(z)
        width = float(args.width) if args.width > 0 else checkpoint_width(z)
        plan_dim = checkpoint_plan_dim(z)
        hierarchical_plan = checkpoint_hierarchical_plan(z)
        history_k = checkpoint_history_k(z)
        model = build_policy_model(
            arch,
            width=width,
            option_context=option_context,
            slot_state=slot_state,
            state_feat_dim=state_feat_dim,
            opt_feat_dim=opt_feat_dim,
            plan_dim=plan_dim,
            hierarchical_plan=hierarchical_plan,
            history_k=history_k,
        ).to(device)
        state = {k: torch.as_tensor(z[k], device=device) for k in z.files}
    current = model.state_dict()
    state = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(state, strict=False)
    model.eval()

    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    corpus = BCCorpus(
        paths,
        include_empty=args.include_empty,
        state_feat_dim=state_feat_dim,
        opt_feat_dim=opt_feat_dim,
        deck_sigs=args.deck_sig,
        team_names=args.team_name,
        opponent_deck_sigs=args.opponent_deck_sig,
        opponent_archetypes=args.opponent_archetype,
        opponent_team_names=args.opponent_team_name,
        winner_only=args.winner_only,
        history_k=history_k,
        load_progress_every=args.load_progress_every,
    )
    indices = corpus.all_indices()
    if args.stride > 1:
        indices = indices[:: args.stride]
    indices = indices[: args.max_samples]

    n = exact = first = top2 = top3 = pred_empty = true_empty = len_match = 0
    by_ctx = defaultdict(lambda: [0, 0, 0, 0, 0])
    by_opt = defaultdict(lambda: [0, 0, 0, 0, 0])
    by_nopt = defaultdict(lambda: [0, 0, 0, 0, 0])
    set_ctx = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])
    set_all = [0, 0, 0.0, 0.0, 0.0]
    start = time.time()
    next_progress = args.progress_every if args.progress_every else None

    for batch_start in range(0, len(indices), args.batch_size):
        batch = corpus.collate(indices[batch_start : batch_start + args.batch_size], device)
        preds = greedy_decode(model, batch)
        topk = first_action_topk(model, batch, (2, 3))
        for bi, (pred, true, ctx, opt, nopt) in enumerate(zip(
            preds, batch.actions, batch.contexts, batch.true_first_types, batch.n_options
        )):
            tk = topk[bi]
            n += 1
            exact_ok = int(pred == true)
            first_ok = int(bool(pred) and bool(true) and pred[0] == true[0])
            true_first = true[0] if true else -1
            top2_ok = int(true_first >= 0 and true_first in tk[:2])
            top3_ok = int(true_first >= 0 and true_first in tk)
            exact += exact_ok
            first += first_ok
            top2 += top2_ok
            top3 += top3_ok
            len_match += int(len(pred) == len(true))
            pred_empty += int(not pred)
            true_empty += int(not true)
            for table, key in ((by_ctx, ctx), (by_opt, opt), (by_nopt, _bucket(nopt))):
                _add(table, key, first_ok, exact_ok, top2_ok, top3_ok)
            if len(true) > 1 or ctx in SET_CONTEXTS:
                _add_set(set_ctx, ctx, pred, true)
                _add_set({"all": set_all}, "all", pred, true)
        if next_progress is not None and (n >= next_progress or n == len(indices)):
            rate = n / max(time.time() - start, 1e-9)
            eta = max(len(indices) - n, 0) / max(rate, 1e-9)
            print(
                f"  {n}/{len(indices)} exact={exact/n:.3f} "
                f"first={first/max(n-true_empty,1):.3f} "
                f"top3={top3/max(n-true_empty,1):.3f} {rate:.1f}/s eta={eta:.0f}s",
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
    print(f"Top-2 first:      {top2 / max(n - true_empty, 1):.3f}")
    print(f"Top-3 first:      {top3 / max(n - true_empty, 1):.3f}")
    print(f"Length match:     {len_match / max(n, 1):.3f}")
    print(f"True empty:       {true_empty / max(n, 1):.3f}")
    print(f"Pred empty:       {pred_empty / max(n, 1):.3f}")
    print("\nBy context:")
    for key, (cnt, fst, ex, t2, t3) in sorted(by_ctx.items(), key=lambda kv: kv[1][0], reverse=True)[:20]:
        print(f"  {key:2d} {CONTEXT_NAMES.get(key, '?'):<18} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f} top3={t3/cnt:.3f}")
    print("\nBy true first option type:")
    for key, (cnt, fst, ex, t2, t3) in sorted(by_opt.items(), key=lambda kv: kv[1][0], reverse=True)[:20]:
        print(f"  {key:2d} {OPT_NAMES.get(key, '?'):<18} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f} top3={t3/cnt:.3f}")
    print("\nBy option count:")
    for key, (cnt, fst, ex, t2, t3) in sorted(by_nopt.items(), key=lambda kv: str(kv[0])):
        print(f"  {key:<4} n={cnt:5d} first={fst/cnt:.3f} exact={ex/cnt:.3f} top3={t3/cnt:.3f}")
    if set_all[0]:
        cnt, ex, prec, rec, f1 = set_all
        print("\nSet-style multi/select contexts:")
        print(f"  ALL                  n={cnt:5d} set_exact={ex/cnt:.3f} precision={prec/cnt:.3f} recall={rec/cnt:.3f} f1={f1/cnt:.3f}")
        for key, (cnt, ex, prec, rec, f1) in sorted(set_ctx.items(), key=lambda kv: kv[1][0], reverse=True)[:20]:
            print(
                f"  {key:2d} {CONTEXT_NAMES.get(key, '?'):<18} "
                f"n={cnt:5d} set_exact={ex/cnt:.3f} precision={prec/cnt:.3f} recall={rec/cnt:.3f} f1={f1/cnt:.3f}"
            )
    if args.out_csv:
        _write_table(args.out_csv, "context", by_ctx, CONTEXT_NAMES)
        _write_table(args.out_csv, "option_type", by_opt, OPT_NAMES)
        _write_table(args.out_csv, "option_count", by_nopt)
        if set_all[0]:
            _write_set_table(args.out_csv, "set_context", set_ctx, CONTEXT_NAMES)
            _write_set_table(args.out_csv, "set_all", {"all": set_all})
        print(f"\nWrote grouped tables to {args.out_csv}")


if __name__ == "__main__":
    main()
