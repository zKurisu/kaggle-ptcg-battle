from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import BCBatch


def _plan_logits(model, h: torch.Tensor) -> torch.Tensor | None:
    if not getattr(model, "plan_dim", 0):
        return None
    return model.plan_logits(h)


def _plan_parts(batch: BCBatch, logits: torch.Tensor) -> list[tuple[str, torch.Tensor, torch.Tensor, slice]]:
    """Return trajectory-level and step-level plan targets aligned to logits.

    The plan head can now be a concatenation of coarse game strategy targets
    followed by per-decision step-plan labels. Keeping this alignment in one
    helper avoids silently training only one half of the plan vector.
    """
    parts: list[tuple[str, torch.Tensor, torch.Tensor, slice]] = []
    offset = 0
    dim = int(logits.shape[1])
    if batch.trajectory_target.numel() > 0:
        n = min(int(batch.trajectory_target.shape[1]), max(0, dim - offset))
        if n > 0:
            parts.append((
                "trajectory",
                batch.trajectory_target[:, :n],
                batch.trajectory_mask[:, :n],
                slice(offset, offset + n),
            ))
        offset += int(batch.trajectory_target.shape[1])
    if batch.plan_step_target.numel() > 0:
        n = min(int(batch.plan_step_target.shape[1]), max(0, dim - offset))
        if n > 0:
            parts.append((
                "step",
                batch.plan_step_target[:, :n],
                batch.plan_step_mask[:, :n],
                slice(offset, offset + n),
            ))
    return parts


def _combined_plan_override(
    model,
    h: torch.Tensor,
    batch: BCBatch,
    *,
    teacher_forcing: float,
) -> torch.Tensor | None:
    if (
        float(teacher_forcing) <= 0
        or not getattr(model, "hierarchical_plan", False)
        or not getattr(model, "plan_dim", 0)
    ):
        return None
    logits = _plan_logits(model, h)
    if logits is None:
        return None
    pred_plan = torch.sigmoid(logits)
    target = pred_plan.detach().clone()
    mask = torch.zeros_like(pred_plan)
    for _, part_target, part_mask, sl in _plan_parts(batch, logits):
        target[:, sl] = torch.where(part_mask > 0, part_target, target[:, sl])
        mask[:, sl] = torch.maximum(mask[:, sl], part_mask)
    if not bool((mask > 0).any()):
        return None
    if float(teacher_forcing) >= 1.0:
        return target
    use_teacher = (
        torch.rand(pred_plan.shape[0], 1, dtype=pred_plan.dtype, device=pred_plan.device)
        < float(teacher_forcing)
    ).to(dtype=pred_plan.dtype)
    return use_teacher * target + (1.0 - use_teacher) * pred_plan


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


def sequence_loss_parts(model, batch: BCBatch, *, first_action_weight: float = 1.0,
                        raw_policy_loss_weight: float = 0.0,
                        value_weight: float = 0.0,
                        plan_weight: float = 0.0,
                        step_plan_weight: float = 0.0,
                        plan_teacher_forcing: float = 0.0,
                        set_loss_weight: float = 0.0,
                        set_loss_min_count: int = 2,
                        set_loss_negative_weight: float = 0.25) -> dict[str, torch.Tensor]:
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
    raw_total = torch.tensor(0.0, device=device)
    raw_count = torch.tensor(0.0, device=device)
    first_total = torch.tensor(0.0, device=device)
    first_weight_total = torch.tensor(0.0, device=device)
    first_raw_total = torch.tensor(0.0, device=device)
    first_count = torch.tensor(0.0, device=device)
    first_correct = torch.tensor(0.0, device=device)
    plan_override = _combined_plan_override(
        model,
        h,
        batch,
        teacher_forcing=plan_teacher_forcing,
    )

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
            nll = -selected
            total = total + (nll * weights).sum()
            weight_total = weight_total + weights.sum()
            raw_total = raw_total + (nll * valid.float()).sum()
            raw_count = raw_count + valid.float().sum()
            if step == 0:
                first_total = first_total + (nll * weights).sum()
                first_weight_total = first_weight_total + weights.sum()
                first_raw_total = first_raw_total + (nll * valid.float()).sum()
                first_count = first_count + valid.float().sum()
                pred = logits.argmax(dim=-1)
                first_correct = first_correct + ((pred == target) & valid).float().sum()
            with torch.no_grad():
                picked = valid & (target < max_options)
                if picked.any():
                    rows = torch.nonzero(picked, as_tuple=True)[0]
                    cols = target[picked]
                    picked_sum[rows] += opts[rows, cols]
                    avail[rows, cols] = False
    policy_loss = total / weight_total.clamp(min=1.0)
    policy_raw_loss = raw_total / raw_count.clamp(min=1.0)
    first_action_loss = first_total / first_weight_total.clamp(min=1.0)
    first_action_raw_loss = first_raw_total / first_count.clamp(min=1.0)
    first_action_acc = first_correct / first_count.clamp(min=1.0)
    loss = policy_loss
    if raw_policy_loss_weight > 0:
        loss = loss + float(raw_policy_loss_weight) * policy_raw_loss
    set_loss = torch.tensor(0.0, device=device)
    if set_loss_weight > 0:
        set_loss = _set_aux_loss(
            model,
            h,
            opts,
            batch,
            min_count=int(set_loss_min_count),
            negative_weight=float(set_loss_negative_weight),
            plan_override=plan_override,
        )
        loss = loss + float(set_loss_weight) * set_loss
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
    value_loss_out = value_loss if value_loss is not None else torch.tensor(0.0, device=device)
    logits = _plan_logits(model, h) if (plan_weight > 0 or step_plan_weight > 0) else None
    trajectory_loss = torch.tensor(0.0, device=device)
    trajectory_count = torch.tensor(0.0, device=device)
    step_plan_loss = torch.tensor(0.0, device=device)
    step_plan_count = torch.tensor(0.0, device=device)
    if logits is not None:
        parts = _plan_parts(batch, logits)
        if plan_weight > 0:
            traj = next((part for part in parts if part[0] == "trajectory"), None)
            if traj is not None:
                _, target, mask_part, sl = traj
            else:
                target = mask_part = None
                sl = slice(0, 0)
            if target is not None and bool((mask_part > 0).any()):
                elem = F.binary_cross_entropy_with_logits(logits[:, sl], target, reduction="none")
                plan_weight_rows = batch.sample_weight.unsqueeze(1) * mask_part
                trajectory_count = plan_weight_rows.sum()
                trajectory_loss = (elem * plan_weight_rows).sum() / trajectory_count.clamp(min=1.0)
                loss = loss + float(plan_weight) * trajectory_loss
        if step_plan_weight > 0:
            step = next((part for part in parts if part[0] == "step"), None)
            if step is not None:
                _, target, mask_part, sl = step
            else:
                target = mask_part = None
                sl = slice(0, 0)
            if target is not None and bool((mask_part > 0).any()):
                elem = F.binary_cross_entropy_with_logits(logits[:, sl], target, reduction="none")
                plan_weight_rows = batch.sample_weight.unsqueeze(1) * mask_part
                step_plan_count = plan_weight_rows.sum()
                step_plan_loss = (elem * plan_weight_rows).sum() / step_plan_count.clamp(min=1.0)
                loss = loss + float(step_plan_weight) * step_plan_loss
    return {
        "loss": loss,
        "policy": policy_loss,
        "policy_raw": policy_raw_loss,
        "first_action": first_action_loss,
        "first_action_raw": first_action_raw_loss,
        "first_action_acc": first_action_acc,
        "set": set_loss,
        "value": value_loss_out,
        "trajectory": trajectory_loss,
        "step_plan": step_plan_loss,
        "rows": torch.tensor(float(bsz), device=device),
        "policy_weight": weight_total.detach(),
        "policy_raw_count": raw_count.detach(),
        "first_weight": first_weight_total.detach(),
        "first_count": first_count.detach(),
        "trajectory_count": trajectory_count.detach(),
        "step_plan_count": step_plan_count.detach(),
    }


def sequence_nll(model, batch: BCBatch, *, first_action_weight: float = 1.0,
                 raw_policy_loss_weight: float = 0.0,
                 value_weight: float = 0.0,
                 plan_weight: float = 0.0,
                 step_plan_weight: float = 0.0,
                 plan_teacher_forcing: float = 0.0,
                 set_loss_weight: float = 0.0,
                 set_loss_min_count: int = 2,
                 set_loss_negative_weight: float = 0.25) -> torch.Tensor:
    parts = sequence_loss_parts(
        model,
        batch,
        first_action_weight=first_action_weight,
        raw_policy_loss_weight=raw_policy_loss_weight,
        value_weight=value_weight,
        plan_weight=plan_weight,
        step_plan_weight=step_plan_weight,
        plan_teacher_forcing=plan_teacher_forcing,
        set_loss_weight=set_loss_weight,
        set_loss_min_count=set_loss_min_count,
        set_loss_negative_weight=set_loss_negative_weight,
    )
    return parts["loss"]
