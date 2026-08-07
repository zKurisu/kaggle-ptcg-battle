from __future__ import annotations

import torch

from .data import BCBatch


@torch.no_grad()
def greedy_decode(model, batch: BCBatch) -> list[list[int]]:
    h = model.encode_state(batch.board, batch.hand, batch.feats, batch.history)
    opts = model.encode_options(batch.opt_type, batch.opt_card, batch.opt_card2, batch.opt_attack, batch.opt_feats)
    bsz = batch.board.shape[0]
    max_options = batch.max_options
    device = batch.board.device

    opt_mask = torch.arange(max_options, device=device).unsqueeze(0) < batch.opt_len.unsqueeze(1)
    avail = torch.cat([opt_mask, torch.ones(bsz, 1, dtype=torch.bool, device=device)], dim=1)
    picked_sum = torch.zeros(bsz, model._oe, device=device)
    active = torch.ones(bsz, dtype=torch.bool, device=device)
    picks: list[list[int]] = [[] for _ in range(bsz)]

    for step in range(int(batch.max_count.max().item())):
        if not bool(active.any()):
            break
        stop_ok = (step >= batch.min_count) & active
        mask = avail.clone()
        mask[:, :max_options] &= active.unsqueeze(1)
        mask[:, max_options] = stop_ok
        logits = model.option_logits(h, opts, picked_sum, mask)
        choice = logits.argmax(dim=-1)
        for row, idx_t in enumerate(choice):
            if not bool(active[row]):
                continue
            idx = int(idx_t.item())
            if idx >= max_options:
                active[row] = False
                continue
            picks[row].append(idx)
            picked_sum[row] += opts[row, idx]
            avail[row, idx] = False
            if len(picks[row]) >= int(batch.max_count[row].item()):
                active[row] = False
    return picks
