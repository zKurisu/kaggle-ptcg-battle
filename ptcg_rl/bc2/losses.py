from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import BCBatch


def sequence_nll(model, batch: BCBatch, *, first_action_weight: float = 1.0) -> torch.Tensor:
    """Autoregressive sequence NLL with padded options masked out."""
    h = model.encode_state(batch.board, batch.hand, batch.feats)
    opts = model.encode_options(batch.opt_type, batch.opt_card, batch.opt_card2, batch.opt_attack, batch.opt_feats)
    bsz = batch.board.shape[0]
    max_options = batch.max_options
    device = batch.board.device

    opt_mask = torch.arange(max_options, device=device).unsqueeze(0) < batch.opt_len.unsqueeze(1)
    avail = torch.cat([opt_mask, torch.ones(bsz, 1, dtype=torch.bool, device=device)], dim=1)
    picked_sum = torch.zeros(bsz, model._oe, device=device)
    total = torch.tensor(0.0, device=device)
    weight_total = torch.tensor(0.0, device=device)

    for step in range(batch.targets.shape[1]):
        stop_ok = step >= batch.min_count
        mask = avail.clone()
        mask[:, max_options] = stop_ok
        logits = model.option_logits(h, opts, picked_sum, mask)
        logp = F.log_softmax(logits, dim=-1)
        target = batch.targets[:, step]
        valid = target >= 0
        if valid.any():
            row_weight = batch.sample_weight * (float(first_action_weight) if step == 0 else 1.0)
            selected = logp.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
            weights = row_weight * valid.float()
            total = total - (selected * weights).sum()
            weight_total = weight_total + weights.sum()
            with torch.no_grad():
                picked = valid & (target < max_options)
                if picked.any():
                    rows = torch.nonzero(picked, as_tuple=True)[0]
                    cols = target[picked]
                    picked_sum[rows] += opts[rows, cols]
                    avail[rows, cols] = False
    return total / weight_total.clamp(min=1.0)
