from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import BCBatch


def _set_aux_loss(
    model,
    h: torch.Tensor,
    opts: torch.Tensor,
    batch: BCBatch,
    *,
    min_count: int,
    negative_weight: float,
    plan_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Order-free auxiliary loss for decisions that select multiple options."""
    bsz = batch.board.shape[0]
    max_options = batch.max_options
    device = batch.board.device
    opt_mask = torch.arange(max_options, device=device).unsqueeze(0) < batch.opt_len.unsqueeze(1)

    target = torch.zeros(bsz, max_options, dtype=opts.dtype, device=device)
    row_keep = torch.zeros(bsz, dtype=torch.bool, device=device)
    for row, action in enumerate(batch.actions):
        cols = [int(a) for a in action if 0 <= int(a) < batch.n_options[row]]
        if len(cols) >= min_count:
            target[row, cols] = 1.0
            row_keep[row] = True
    if not bool(row_keep.any()):
        return torch.tensor(0.0, device=device)

    picked_sum = torch.zeros(bsz, model._oe, device=device)
    mask = torch.cat([opt_mask, torch.zeros(bsz, 1, dtype=torch.bool, device=device)], dim=1)
    option_logits = model.option_logits(h, opts, picked_sum, mask, plan_override=plan_override)[:, :max_options]
    elem = F.binary_cross_entropy_with_logits(option_logits, target, reduction="none")

    pos_mask = target * opt_mask.float()
    neg_mask = (1.0 - target) * opt_mask.float()
    pos_loss = (elem * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
    neg_loss = (elem * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1.0)
    row_loss = pos_loss + float(negative_weight) * neg_loss
    row_weight = batch.sample_weight * row_keep.float()
    return (row_loss * row_weight).sum() / row_weight.sum().clamp(min=1.0)


def sequence_nll(model, batch: BCBatch, *, first_action_weight: float = 1.0,
                 value_weight: float = 0.0,
                 plan_weight: float = 0.0,
                 step_plan_weight: float = 0.0,
                 plan_teacher_forcing: float = 0.0,
                 set_loss_weight: float = 0.0,
                 set_loss_min_count: int = 2,
                 set_loss_negative_weight: float = 0.25) -> torch.Tensor:
    """Autoregressive sequence NLL with padded options masked out."""
    h = model.encode_state(batch.board, batch.hand, batch.feats, batch.history)
    opts = model.encode_options(batch.opt_type, batch.opt_card, batch.opt_card2, batch.opt_attack, batch.opt_feats)
    bsz = batch.board.shape[0]
    max_options = batch.max_options
    device = batch.board.device

    opt_mask = torch.arange(max_options, device=device).unsqueeze(0) < batch.opt_len.unsqueeze(1)
    avail = torch.cat([opt_mask, torch.ones(bsz, 1, dtype=torch.bool, device=device)], dim=1)
    picked_sum = torch.zeros(bsz, model._oe, device=device)
    total = torch.tensor(0.0, device=device)
    weight_total = torch.tensor(0.0, device=device)
    plan_override = None
    if (
        float(plan_teacher_forcing) > 0
        and getattr(model, "hierarchical_plan", False)
        and batch.plan_step_mask.numel() > 0
        and bool((batch.plan_step_mask > 0).any())
    ):
        pred_plan = torch.sigmoid(model.plan_logits(h))
        target_plan = torch.where(batch.plan_step_mask > 0, batch.plan_step_target, pred_plan)
        if float(plan_teacher_forcing) >= 1.0:
            plan_override = target_plan
        else:
            use_teacher = (
                torch.rand(bsz, 1, dtype=pred_plan.dtype, device=device)
                < float(plan_teacher_forcing)
            ).to(dtype=pred_plan.dtype)
            plan_override = use_teacher * target_plan + (1.0 - use_teacher) * pred_plan

    for step in range(batch.targets.shape[1]):
        stop_ok = step >= batch.min_count
        mask = avail.clone()
        mask[:, max_options] = stop_ok
        logits = model.option_logits(h, opts, picked_sum, mask, plan_override=plan_override)
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
    policy_loss = total / weight_total.clamp(min=1.0)
    loss = policy_loss
    if set_loss_weight > 0:
        loss = loss + float(set_loss_weight) * _set_aux_loss(
            model,
            h,
            opts,
            batch,
            min_count=int(set_loss_min_count),
            negative_weight=float(set_loss_negative_weight),
            plan_override=plan_override,
        )
    if value_weight <= 0:
        value_loss = None
    else:
        value_mask = batch.outcome_mask > 0
        value_loss = None
        if bool(value_mask.any()):
            pred = model.value(h)
            value_loss = ((pred - batch.outcome_value).pow(2) * batch.sample_weight * batch.outcome_mask).sum()
            value_loss = value_loss / (batch.sample_weight * batch.outcome_mask).sum().clamp(min=1.0)
    if value_loss is not None:
        loss = loss + float(value_weight) * value_loss
    if plan_weight > 0 and batch.trajectory_mask.numel() > 0 and bool((batch.trajectory_mask > 0).any()):
        logits = model.plan_logits(h)
        elem = F.binary_cross_entropy_with_logits(
            logits,
            batch.trajectory_target,
            reduction="none",
        )
        plan_weight_rows = batch.sample_weight.unsqueeze(1) * batch.trajectory_mask
        plan_loss = (elem * plan_weight_rows).sum() / plan_weight_rows.sum().clamp(min=1.0)
        loss = loss + float(plan_weight) * plan_loss
    if step_plan_weight > 0 and batch.plan_step_mask.numel() > 0 and bool((batch.plan_step_mask > 0).any()):
        logits = model.plan_logits(h)
        elem = F.binary_cross_entropy_with_logits(
            logits,
            batch.plan_step_target,
            reduction="none",
        )
        plan_weight_rows = batch.sample_weight.unsqueeze(1) * batch.plan_step_mask
        plan_loss = (elem * plan_weight_rows).sum() / plan_weight_rows.sum().clamp(min=1.0)
        loss = loss + float(step_plan_weight) * plan_loss
    return loss
