#!/usr/bin/env python3
"""Detailed BC2 error report for weak deck specialists."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, greedy_decode
from ptcg_rl.deck_plans import get_plan, tag_cards
from ptcg_rl.model import (
    build_policy_model,
    checkpoint_arch,
    checkpoint_feature_dims,
    checkpoint_width,
)
from tools.bc2_accuracy import CONTEXT_NAMES, OPT_NAMES, SET_CONTEXTS, first_action_topk


STOP_TYPE = -2


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


def _load_model(path: str, width: float, device: torch.device):
    with np.load(path) as z:
        arch = checkpoint_arch(z.files)
        state_feat_dim, opt_feat_dim, option_context, slot_state = checkpoint_feature_dims(z)
        model_width = float(width) if width > 0 else checkpoint_width(z)
        model = build_policy_model(
            arch,
            width=model_width,
            option_context=option_context,
            slot_state=slot_state,
            state_feat_dim=state_feat_dim,
            opt_feat_dim=opt_feat_dim,
        ).to(device)
        state = {k: torch.as_tensor(z[k], device=device) for k in z.files}
    current = model.state_dict()
    state = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, state_feat_dim, opt_feat_dim


def _opt_type(batch, row: int, action: list[int]) -> int:
    if not action:
        return STOP_TYPE
    idx = action[0]
    if idx < 0 or idx >= batch.n_options[row]:
        return STOP_TYPE
    return int(batch.opt_type[row, idx].detach().cpu().item())


def _seq_types(batch, row: int, action: list[int]) -> list[int]:
    out = []
    for idx in action:
        if 0 <= idx < batch.n_options[row]:
            out.append(int(batch.opt_type[row, idx].detach().cpu().item()))
    return out


def _seq_cards(batch, row: int, action: list[int]) -> list[int]:
    out = []
    for idx in action:
        if 0 <= idx < batch.n_options[row]:
            out.append(int(batch.opt_card[row, idx].detach().cpu().item()))
    return out


def _add_rate(table: dict, key, first_ok: int, exact_ok: int, top3_ok: int,
              true_len: int, pred_len: int, true_type: int, pred_type: int) -> None:
    row = table[key]
    row["n"] += 1
    row["first"] += first_ok
    row["exact"] += exact_ok
    row["top3"] += top3_ok
    row["true_len"] += true_len
    row["pred_len"] += pred_len
    row["under_len"] += int(pred_len < true_len)
    row["over_len"] += int(pred_len > true_len)
    row["pred_empty"] += int(pred_len == 0)
    row["early_end"] += int(pred_type == 14 and true_type != 14)
    row["miss_end"] += int(true_type == 14 and pred_type != 14)
    row["early_attack"] += int(pred_type == 13 and true_type != 13)
    row["miss_attack"] += int(true_type == 13 and pred_type != 13)


def _add_set(table: dict, key, pred: list[int], true: list[int]) -> None:
    ps = set(pred)
    ts = set(true)
    inter = len(ps & ts)
    precision = inter / max(len(ps), 1)
    recall = inter / max(len(ts), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    row = table[key]
    row["n"] += 1
    row["set_exact"] += int(ps == ts)
    row["precision"] += precision
    row["recall"] += recall
    row["f1"] += f1


def _row_id(corpus: BCCorpus, idx: tuple[int, int]) -> dict[str, str]:
    di, si = idx
    data = corpus.npz_data[di]
    def get(name: str, default: str = "") -> str:
        if name not in data:
            return default
        try:
            return str(data[name][si])
        except Exception:
            return default
    return {
        "episode_id": get("episode_id"),
        "deck_sig": get("deck_sig"),
        "opponent_deck_sig": get("opponent_deck_sig"),
        "opponent_archetype": get("opponent_archetype"),
        "opponent_team_name": get("opponent_team_name"),
        "player_index": get("player_index"),
        "won": get("won"),
        "draw": get("draw"),
    }


def _write_summary(path: Path, tables: list[tuple[str, dict, dict | None]]) -> None:
    fields = [
        "table", "key", "label", "n", "first", "exact", "top3",
        "avg_true_len", "avg_pred_len", "under_len", "over_len", "pred_empty",
        "early_end", "miss_end", "early_attack", "miss_attack",
        "set_exact", "precision", "recall", "f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for table_name, table, labels in tables:
            for key, row in sorted(table.items(), key=lambda item: item[1]["n"], reverse=True):
                n = max(row["n"], 1)
                label = labels.get(key, "") if labels else ""
                out = {
                    "table": table_name,
                    "key": key,
                    "label": label,
                    "n": row["n"],
                    "first": row.get("first", 0) / n if "first" in row else "",
                    "exact": row.get("exact", 0) / n if "exact" in row else "",
                    "top3": row.get("top3", 0) / n if "top3" in row else "",
                    "avg_true_len": row.get("true_len", 0) / n if "true_len" in row else "",
                    "avg_pred_len": row.get("pred_len", 0) / n if "pred_len" in row else "",
                    "under_len": row.get("under_len", 0) / n if "under_len" in row else "",
                    "over_len": row.get("over_len", 0) / n if "over_len" in row else "",
                    "pred_empty": row.get("pred_empty", 0) / n if "pred_empty" in row else "",
                    "early_end": row.get("early_end", 0) / n if "early_end" in row else "",
                    "miss_end": row.get("miss_end", 0) / n if "miss_end" in row else "",
                    "early_attack": row.get("early_attack", 0) / n if "early_attack" in row else "",
                    "miss_attack": row.get("miss_attack", 0) / n if "miss_attack" in row else "",
                    "set_exact": row.get("set_exact", 0) / n if "set_exact" in row else "",
                    "precision": row.get("precision", 0.0) / n if "precision" in row else "",
                    "recall": row.get("recall", 0.0) / n if "recall" in row else "",
                    "f1": row.get("f1", 0.0) / n if "f1" in row else "",
                }
                w.writerow(out)


def _write_examples(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--corpus", default="data/bc_corpus_banded_v8")
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[],
                   help="filter to decisions from games against one or more opponent deck signatures")
    p.add_argument("--opponent-archetype", action="append", default=[],
                   help="filter to decisions from games against one or more opponent archetypes")
    p.add_argument("--opponent-team-name", action="append", default=[],
                   help="filter to decisions from games against one or more exact opponent team names")
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--width", type=float, default=0.0,
                   help="model width; 0 infers from checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--load-progress-every", type=int, default=200000)
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--out-prefix", default="")
    args = p.parse_args()

    plan = get_plan(args.archetype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, state_feat_dim, opt_feat_dim = _load_model(args.policy, args.width, device)
    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    corpus = BCCorpus(
        paths,
        state_feat_dim=state_feat_dim,
        opt_feat_dim=opt_feat_dim,
        deck_sigs=args.deck_sig,
        team_names=args.team_name,
        opponent_deck_sigs=args.opponent_deck_sig,
        opponent_archetypes=args.opponent_archetype,
        opponent_team_names=args.opponent_team_name,
        winner_only=args.winner_only,
        load_progress_every=args.load_progress_every,
    )
    indices = corpus.all_indices()
    if args.stride > 1:
        indices = indices[:: args.stride]
    indices = indices[: args.max_samples]

    rate_default = lambda: defaultdict(int)
    by_context = defaultdict(rate_default)
    by_true_type = defaultdict(rate_default)
    by_option_count = defaultdict(rate_default)
    by_len_pair = defaultdict(rate_default)
    set_context = defaultdict(rate_default)
    confusion = Counter()
    examples: list[dict] = []
    n = exact = first = top3 = 0
    t0 = time.time()
    next_progress = args.progress_every if args.progress_every else None

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start: batch_start + args.batch_size]
        batch = corpus.collate(batch_indices, device)
        preds = greedy_decode(model, batch)
        topk = first_action_topk(model, batch, (3,))
        for bi, (idx, pred, true, ctx, true_type, nopt) in enumerate(
            zip(batch_indices, preds, batch.actions, batch.contexts, batch.true_first_types, batch.n_options)
        ):
            pred_type = _opt_type(batch, bi, pred)
            true_type = int(true_type) if true else STOP_TYPE
            true_first = true[0] if true else -1
            first_ok = int(bool(pred) and bool(true) and pred[0] == true[0])
            exact_ok = int(pred == true)
            top3_ok = int(true_first >= 0 and true_first in topk[bi])
            n += 1
            first += first_ok
            exact += exact_ok
            top3 += top3_ok
            for table, key in (
                (by_context, ctx),
                (by_true_type, true_type),
                (by_option_count, _bucket(nopt)),
                (by_len_pair, f"{len(true)}->{len(pred)}"),
            ):
                _add_rate(table, key, first_ok, exact_ok, top3_ok, len(true), len(pred), true_type, pred_type)
            if len(true) > 1 or ctx in SET_CONTEXTS:
                _add_set(set_context, ctx, pred, true)
            confusion[(true_type, pred_type)] += 1
            if not exact_ok and len(examples) < args.max_examples:
                meta = _row_id(corpus, idx)
                true_cards = _seq_cards(batch, bi, true)
                pred_cards = _seq_cards(batch, bi, pred)
                examples.append({
                    **meta,
                    "context": ctx,
                    "context_name": CONTEXT_NAMES.get(ctx, "?"),
                    "n_options": nopt,
                    "min_count": int(batch.min_count[bi].detach().cpu().item()),
                    "max_count": int(batch.max_count[bi].detach().cpu().item()),
                    "true_action": " ".join(map(str, true)),
                    "pred_action": " ".join(map(str, pred)),
                    "true_types": " ".join(OPT_NAMES.get(x, str(x)) for x in _seq_types(batch, bi, true)),
                    "pred_types": " ".join(OPT_NAMES.get(x, str(x)) for x in _seq_types(batch, bi, pred)),
                    "true_cards": " ".join(map(str, true_cards)),
                    "pred_cards": " ".join(map(str, pred_cards)),
                    "true_plan_tags": " ".join(tag_cards(plan, true_cards)),
                    "pred_plan_tags": " ".join(tag_cards(plan, pred_cards)),
                    "top3_first": " ".join(map(str, topk[bi])),
                })
        if next_progress is not None and (n >= next_progress or n == len(indices)):
            rate = n / max(time.time() - t0, 1e-9)
            eta = max(len(indices) - n, 0) / max(rate, 1e-9)
            print(
                f"  {n}/{len(indices)} exact={exact/n:.3f} first={first/n:.3f} "
                f"top3={top3/n:.3f} {rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            while next_progress is not None and next_progress <= n:
                next_progress += args.progress_every

    prefix = Path(args.out_prefix) if args.out_prefix else Path("logs") / f"bc2_failure_{Path(args.policy).stem}"
    summary_csv = prefix.with_suffix(".summary.csv")
    examples_csv = prefix.with_suffix(".examples.csv")
    confusion_csv = prefix.with_suffix(".confusion.csv")
    _write_summary(
        summary_csv,
        [
            ("context", by_context, CONTEXT_NAMES),
            ("true_type", by_true_type, OPT_NAMES | {STOP_TYPE: "STOP"}),
            ("option_count", by_option_count, None),
            ("len_pair", by_len_pair, None),
            ("set_context", set_context, CONTEXT_NAMES),
        ],
    )
    _write_examples(examples_csv, examples)
    with confusion_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_type", "true_label", "pred_type", "pred_label", "n"])
        for (tt, pt), cnt in confusion.most_common():
            w.writerow([tt, OPT_NAMES.get(tt, "STOP"), pt, OPT_NAMES.get(pt, "STOP"), cnt])

    print(f"Policy: {args.policy}")
    if plan:
        print(f"Deck plan: {plan.archetype}")
    print(f"Samples: {n} from {len(paths)} files")
    print(f"Corpus labels: {corpus.stats}")
    print(f"Exact={exact/max(n,1):.3f} First={first/max(n,1):.3f} Top3={top3/max(n,1):.3f}")
    print("\nWorst contexts by exact accuracy:")
    ctx_rows = []
    for ctx, row in by_context.items():
        if row["n"] >= 50:
            ctx_rows.append((row["exact"] / row["n"], ctx, row))
    for acc, ctx, row in sorted(ctx_rows)[:12]:
        print(
            f"  {ctx:2d} {CONTEXT_NAMES.get(ctx, '?'):<18} n={row['n']:5d} "
            f"exact={acc:.3f} first={row['first']/row['n']:.3f} "
            f"early_end={row['early_end']/row['n']:.3f} miss_attack={row['miss_attack']/row['n']:.3f}",
            flush=True,
        )
    if set_context:
        print("\nWorst set-style contexts by F1:")
        set_rows = []
        for ctx, row in set_context.items():
            if row["n"] >= 20:
                set_rows.append((row["f1"] / row["n"], ctx, row))
        for f1, ctx, row in sorted(set_rows)[:12]:
            print(
                f"  {ctx:2d} {CONTEXT_NAMES.get(ctx, '?'):<18} n={row['n']:5d} "
                f"set_exact={row['set_exact']/row['n']:.3f} f1={f1:.3f} "
                f"precision={row['precision']/row['n']:.3f} recall={row['recall']/row['n']:.3f}",
                flush=True,
            )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {examples_csv}")
    print(f"Wrote {confusion_csv}")


if __name__ == "__main__":
    main()
