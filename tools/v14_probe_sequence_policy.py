#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.seq.data import SequenceCorpus, discover_sequence_npz
from ptcg_rl.seq.model import SequencePolicyNet, sequence_accuracy, sequence_policy_loss, SequenceLossConfig
from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.seq.constants import LEDGER_FEAT_DIM, FUTURE_PLAN_DIM


def _split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _eval_batch(model, batch, loss_cfg):
    with torch.no_grad():
        outputs = model(batch)
        loss, parts = sequence_policy_loss(outputs, batch, loss_cfg)
        acc = sequence_accuracy(outputs, batch)
    return loss.item(), parts, acc


def _prefix_mask(batch, prefix_len: int):
    clone = batch.to(batch.board.device)
    clone.step_mask = clone.step_mask.clone()
    clone.step_mask[:, prefix_len:] = 0
    clone.target_first = clone.target_first.clone()
    clone.target_first[:, prefix_len:] = -1
    clone.target_order = clone.target_order.clone()
    clone.target_order[:, prefix_len:] = -1
    clone.target_multi = clone.target_multi.clone()
    clone.target_multi[:, prefix_len:] = 0
    clone.target_type = clone.target_type.clone()
    clone.target_type[:, prefix_len:] = 0
    clone.target_context = clone.target_context.clone()
    clone.target_context[:, prefix_len:] = 0
    clone.future_plan = clone.future_plan.clone()
    clone.future_plan[:, prefix_len:] = 0
    clone.outcome = clone.outcome.clone()
    clone.outcome[:, prefix_len:] = 0
    return clone


def _last_only(batch):
    clone = batch.to(batch.board.device)
    clone.step_mask = torch.zeros_like(clone.step_mask)
    clone.step_mask[:, -1] = batch.step_mask[:, -1]
    clone.target_first = clone.target_first.clone()
    clone.target_first[:, :-1] = -1
    clone.target_order = clone.target_order.clone()
    clone.target_order[:, :-1] = -1
    clone.target_multi = clone.target_multi.clone()
    clone.target_multi[:, :-1] = 0
    clone.target_type = clone.target_type.clone()
    clone.target_type[:, :-1] = 0
    clone.target_context = clone.target_context.clone()
    clone.target_context[:, :-1] = 0
    clone.future_plan = clone.future_plan.clone()
    clone.future_plan[:, :-1] = 0
    clone.outcome = clone.outcome.clone()
    clone.outcome[:, :-1] = 0
    return clone


def _zero_prefix_inputs(batch, *, zero_current_ledger: bool = False):
    clone = batch.to(batch.board.device)
    for name in (
        "board",
        "hand",
        "feats",
        "state_token_feats",
        "ledger_feats",
        "prev_type",
        "prev_card",
        "prev_card2",
        "prev_attack",
        "prev_context",
        "prev_select_type",
        "prev_count",
        "opt_type",
        "opt_card",
        "opt_card2",
        "opt_attack",
        "opt_feats",
        "option_mask",
    ):
        value = getattr(clone, name).clone()
        value[:, :-1] = 0
        setattr(clone, name, value)
    if zero_current_ledger:
        clone.ledger_feats = clone.ledger_feats.clone()
        clone.ledger_feats[:, -1] = 0
        clone.prev_type = clone.prev_type.clone()
        clone.prev_card = clone.prev_card.clone()
        clone.prev_card2 = clone.prev_card2.clone()
        clone.prev_attack = clone.prev_attack.clone()
        clone.prev_context = clone.prev_context.clone()
        clone.prev_select_type = clone.prev_select_type.clone()
        clone.prev_count = clone.prev_count.clone()
        clone.prev_type[:, -1] = 0
        clone.prev_card[:, -1] = 0
        clone.prev_card2[:, -1] = 0
        clone.prev_attack[:, -1] = 0
        clone.prev_context[:, -1] = 0
        clone.prev_select_type[:, -1] = 0
        clone.prev_count[:, -1] = 0
    return _last_only(clone)


def _shuffle_prefix_inputs(batch):
    clone = batch.to(batch.board.device)
    seq = clone.board.shape[1]
    if seq <= 2:
        return _last_only(clone)
    perm = torch.randperm(seq - 1, device=clone.board.device)
    for name in (
        "board",
        "hand",
        "feats",
        "state_token_feats",
        "ledger_feats",
        "prev_type",
        "prev_card",
        "prev_card2",
        "prev_attack",
        "prev_context",
        "prev_select_type",
        "prev_count",
        "opt_type",
        "opt_card",
        "opt_card2",
        "opt_attack",
        "opt_feats",
        "option_mask",
    ):
        value = getattr(clone, name).clone()
        value[:, :-1] = value[:, perm]
        setattr(clone, name, value)
    return _last_only(clone)


def _zero_ledger_prev(batch):
    clone = batch.to(batch.board.device)
    clone.ledger_feats = torch.zeros_like(clone.ledger_feats)
    clone.prev_type = torch.zeros_like(clone.prev_type)
    clone.prev_card = torch.zeros_like(clone.prev_card)
    clone.prev_card2 = torch.zeros_like(clone.prev_card2)
    clone.prev_attack = torch.zeros_like(clone.prev_attack)
    clone.prev_context = torch.zeros_like(clone.prev_context)
    clone.prev_select_type = torch.zeros_like(clone.prev_select_type)
    clone.prev_count = torch.zeros_like(clone.prev_count)
    return _last_only(clone)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--prefixes", type=str, default="4,8,16,24,32")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    paths = discover_sequence_npz(args.corpus, args.archetype, _split_csv(args.score_bands))
    corpus = SequenceCorpus(
        paths,
        seq_len=args.seq_len,
        deck_sigs=_split_csv(args.deck_sig),
        team_names=_split_csv(args.team_name),
        winner_only=args.winner_only,
    )
    ids = list(range(len(corpus.samples)))
    random.shuffle(ids)
    ids = ids[: min(args.samples, len(ids))]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = SequencePolicyNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(args.device).eval()
    loss_cfg = SequenceLossConfig()

    batch = corpus.collate(ids).to(torch.device(args.device))
    base_loss, base_parts, base_acc = _eval_batch(model, batch, loss_cfg)
    print("base", json.dumps({"loss": base_loss, **base_parts, **base_acc}, ensure_ascii=False, sort_keys=True))

    last = _last_only(batch)
    loss, parts, acc = _eval_batch(model, last, loss_cfg)
    print("last_only", json.dumps({"loss": loss, **parts, **acc}, ensure_ascii=False, sort_keys=True))

    for name, variant in (
        ("last_zero_prefix_keep_ledger", _zero_prefix_inputs(batch, zero_current_ledger=False)),
        ("last_zero_prefix_zero_ledger", _zero_prefix_inputs(batch, zero_current_ledger=True)),
        ("last_shuffle_prefix", _shuffle_prefix_inputs(batch)),
        ("last_zero_ledger_prev", _zero_ledger_prev(batch)),
    ):
        loss, parts, acc = _eval_batch(model, variant, loss_cfg)
        print(name, json.dumps({"loss": loss, **parts, **acc}, ensure_ascii=False, sort_keys=True))

    seq = batch.board.shape[1]
    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    for prefix in prefixes:
        prefix = max(1, min(prefix, seq))
        masked = _prefix_mask(batch, prefix)
        loss, parts, acc = _eval_batch(model, masked, loss_cfg)
        print(
            "prefix",
            json.dumps(
                {"keep": prefix, "loss": loss, **parts, **acc},
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    shuffled = batch.to(torch.device(args.device))
    perm = torch.randperm(seq, device=shuffled.board.device)
    shuffled.board = shuffled.board[:, perm]
    shuffled.hand = shuffled.hand[:, perm]
    shuffled.feats = shuffled.feats[:, perm]
    shuffled.state_token_feats = shuffled.state_token_feats[:, perm]
    shuffled.ledger_feats = shuffled.ledger_feats[:, perm]
    shuffled.prev_type = shuffled.prev_type[:, perm]
    shuffled.prev_card = shuffled.prev_card[:, perm]
    shuffled.prev_card2 = shuffled.prev_card2[:, perm]
    shuffled.prev_attack = shuffled.prev_attack[:, perm]
    shuffled.prev_context = shuffled.prev_context[:, perm]
    shuffled.prev_select_type = shuffled.prev_select_type[:, perm]
    shuffled.prev_count = shuffled.prev_count[:, perm]
    shuffled.opt_type = shuffled.opt_type[:, perm]
    shuffled.opt_card = shuffled.opt_card[:, perm]
    shuffled.opt_card2 = shuffled.opt_card2[:, perm]
    shuffled.opt_attack = shuffled.opt_attack[:, perm]
    shuffled.opt_feats = shuffled.opt_feats[:, perm]
    shuffled.option_mask = shuffled.option_mask[:, perm]
    shuffled.target_first = shuffled.target_first[:, perm]
    shuffled.target_order = shuffled.target_order[:, perm]
    shuffled.target_multi = shuffled.target_multi[:, perm]
    shuffled.target_type = shuffled.target_type[:, perm]
    shuffled.target_context = shuffled.target_context[:, perm]
    shuffled.min_count = shuffled.min_count[:, perm]
    shuffled.max_count = shuffled.max_count[:, perm]
    shuffled.step_mask = shuffled.step_mask[:, perm]
    shuffled.sample_weight = shuffled.sample_weight[:, perm]
    shuffled.future_plan = shuffled.future_plan[:, perm]
    shuffled.outcome = shuffled.outcome[:, perm]
    loss, parts, acc = _eval_batch(model, shuffled, loss_cfg)
    print("shuffle", json.dumps({"loss": loss, **parts, **acc}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
