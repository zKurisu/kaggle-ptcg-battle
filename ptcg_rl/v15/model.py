from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_rl.encoder import (
    BOARD_SLOTS,
    MAX_HAND,
    N_ATTACKS,
    N_CARDS,
    N_OPT_TYPES,
    OPT_FEAT_DIM,
    STATE_FEAT_DIM,
    STATE_TOKEN_FEAT_DIM,
)
from ptcg_rl.v15.constants import (
    DEFAULT_PLAN_STEPS,
    KNOWN_OPP_CARDS,
    N_ACTION_TYPES,
    N_PLAN_MODES,
    TYPE_ABILITY,
    TYPE_ATTACH,
    TYPE_ATTACK,
    TYPE_END,
    TYPE_EVOLVE,
    TYPE_PLAY,
    TYPE_RETREAT,
)
from ptcg_rl.v15.data import V15Batch

NEG_INF = -1e4


@dataclass
class V15LossConfig:
    action_weight: float = 1.0
    multi_weight: float = 0.20
    type_weight: float = 0.12
    history_type_weight: float = 0.10
    known_type_weight: float = 0.06
    context_weight: float = 0.05
    plan_type_weight: float = 0.55
    plan_card_weight: float = 0.08
    plan_attack_weight: float = 0.08
    plan_context_weight: float = 0.04
    continue_weight: float = 0.20
    mode_weight: float = 0.15
    outcome_weight: float = 0.05


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = mask.float()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1.0)


def _safe_card(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, N_CARDS + 1)


def _safe_attack(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, N_ATTACKS)


class V15PlanPolicyNet(nn.Module):
    """Turn-block policy with explicit event history and plan-conditioned scoring."""

    def __init__(
        self,
        *,
        width: int = 384,
        layers: int = 3,
        heads: int = 6,
        dropout: float = 0.10,
        history_k: int = 48,
        plan_steps: int = DEFAULT_PLAN_STEPS,
        state_feat_dim: int = STATE_FEAT_DIM,
        opt_feat_dim: int = OPT_FEAT_DIM,
        state_token_feat_dim: int = STATE_TOKEN_FEAT_DIM,
        type_prior_scale: float = 1.50,
        history_type_prior_scale: float = 0.35,
        known_type_prior_scale: float = 0.25,
    ):
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads")
        self.width = int(width)
        self.history_k = int(history_k)
        self.plan_steps = int(plan_steps)
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.state_token_feat_dim = int(state_token_feat_dim)
        self.type_prior_scale = float(type_prior_scale)
        self.history_type_prior_scale = float(history_type_prior_scale)
        self.known_type_prior_scale = float(known_type_prior_scale)

        card_dim = max(32, width // 4)
        attack_dim = max(24, width // 10)
        type_dim = max(24, width // 10)
        ctx_dim = max(16, width // 12)

        self.card_emb = nn.Embedding(N_CARDS + 2, card_dim, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, attack_dim, padding_idx=0)
        self.type_emb = nn.Embedding(N_OPT_TYPES + 2, type_dim, padding_idx=0)
        self.action_type_emb = nn.Embedding(N_ACTION_TYPES + 2, type_dim, padding_idx=0)
        self.event_type_emb = nn.Embedding(192, type_dim, padding_idx=0)
        self.source_emb = nn.Embedding(4, ctx_dim, padding_idx=0)
        self.owner_emb = nn.Embedding(4, ctx_dim, padding_idx=0)
        self.context_emb = nn.Embedding(128, ctx_dim, padding_idx=0)
        self.select_type_emb = nn.Embedding(32, ctx_dim, padding_idx=0)
        self.area_emb = nn.Embedding(64, ctx_dim, padding_idx=0)
        self.slot_emb = nn.Embedding(BOARD_SLOTS + MAX_HAND, ctx_dim)
        self.event_pos_emb = nn.Embedding(max(history_k, 1), width)

        token_in = card_dim + ctx_dim + self.state_token_feat_dim
        self.state_token_fc = nn.Sequential(nn.Linear(token_in, width), nn.LayerNorm(width), nn.GELU())
        state_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(state_layer, num_layers=1)
        self.state_feat_fc = nn.Sequential(
            nn.Linear(self.state_feat_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        event_scalar_dim = 4
        event_in = type_dim + ctx_dim * 5 + card_dim * 2 + attack_dim + event_scalar_dim
        self.event_fc = nn.Sequential(nn.Linear(event_in, width), nn.LayerNorm(width), nn.GELU())
        event_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=max(1, int(layers)))

        known_in = card_dim + 2
        self.known_fc = nn.Sequential(nn.Linear(known_in, width), nn.LayerNorm(width), nn.GELU())

        opt_in = type_dim + card_dim * 2 + attack_dim + self.opt_feat_dim
        self.option_fc = nn.Sequential(
            nn.Linear(opt_in, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        fused_in = width * 4
        self.fuse_fc = nn.Sequential(
            nn.Linear(fused_in, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.plan_latent_fc = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_context_fc = nn.Sequential(
            nn.Linear(width * 5, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_score = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.multi_score = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

        self.type_head = nn.Linear(width, N_ACTION_TYPES)
        self.history_type_head = nn.Linear(width, N_ACTION_TYPES)
        self.known_type_head = nn.Linear(width, N_ACTION_TYPES)
        self.context_head = nn.Linear(width, 128)
        self.mode_head = nn.Linear(width, N_PLAN_MODES)
        self.continue_head = nn.Linear(width, 1)
        self.outcome_head = nn.Linear(width, 1)
        self.plan_step_emb = nn.Embedding(self.plan_steps, width)
        self.plan_query = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())
        self.plan_type_head = nn.Linear(width, N_ACTION_TYPES)
        self.plan_card_head = nn.Linear(width, N_CARDS + 2)
        self.plan_card2_head = nn.Linear(width, N_CARDS + 2)
        self.plan_attack_head = nn.Linear(width, N_ATTACKS + 1)
        self.plan_context_head = nn.Linear(width, 128)

    def _state_encode(self, batch: V15Batch) -> torch.Tensor:
        cards = torch.cat([batch.board, batch.hand], dim=1).clamp(0, N_CARDS + 1)
        bsz, n_tok = cards.shape
        slots = torch.arange(n_tok, device=cards.device).unsqueeze(0).expand(bsz, -1)
        token_feats = batch.state_token_feats
        if token_feats.shape[-1] != self.state_token_feat_dim:
            if token_feats.shape[-1] > self.state_token_feat_dim:
                token_feats = token_feats[..., : self.state_token_feat_dim]
            else:
                pad = torch.zeros(*token_feats.shape[:-1], self.state_token_feat_dim - token_feats.shape[-1], device=token_feats.device)
                token_feats = torch.cat([token_feats, pad], dim=-1)
        x = torch.cat([self.card_emb(cards), self.slot_emb(slots), token_feats.float()], dim=-1)
        x = self.state_token_fc(x)
        x = self.state_encoder(x)
        mask = (cards > 0).float()
        pooled = _masked_mean(x, mask)
        return pooled

    def _event_encode(self, batch: V15Batch, *, ablate: str | None = None) -> torch.Tensor:
        et = batch.event_type.clamp(0, 191)
        source = batch.event_source.clamp(0, 3)
        owner = batch.event_owner.clamp(0, 3)
        card = _safe_card(batch.event_card)
        card2 = _safe_card(batch.event_card2)
        attack = _safe_attack(batch.event_attack)
        ctx = batch.event_context.clamp(0, 127)
        sel = batch.event_select_type.clamp(0, 31)
        fa = batch.event_from_area.clamp(0, 63)
        ta = batch.event_to_area.clamp(0, 63)
        scalar = torch.stack(
            [
                batch.event_value.float(),
                batch.event_turn_delta.float(),
                batch.event_step_delta.float(),
                batch.event_same_turn.float(),
            ],
            dim=-1,
        )
        mask = batch.event_mask.float()
        if ablate == "no_history":
            mask = torch.zeros_like(mask)
        if ablate == "reverse_history" and et.shape[1] > 1:
            order = torch.arange(et.shape[1] - 1, -1, -1, device=et.device)
            et = et.index_select(1, order)
            source = source.index_select(1, order)
            owner = owner.index_select(1, order)
            card = card.index_select(1, order)
            card2 = card2.index_select(1, order)
            attack = attack.index_select(1, order)
            ctx = ctx.index_select(1, order)
            sel = sel.index_select(1, order)
            fa = fa.index_select(1, order)
            ta = ta.index_select(1, order)
            scalar = scalar.index_select(1, order)
            mask = mask.index_select(1, order)
        x = torch.cat(
            [
                self.event_type_emb(et),
                self.source_emb(source),
                self.owner_emb(owner),
                self.card_emb(card),
                self.card_emb(card2),
                self.attack_emb(attack),
                self.context_emb(ctx),
                self.select_type_emb(sel),
                self.area_emb(fa) + self.area_emb(ta),
                scalar,
            ],
            dim=-1,
        )
        x = self.event_fc(x)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        x = x + self.event_pos_emb(pos.clamp(max=self.event_pos_emb.num_embeddings - 1))
        pad = mask <= 0
        empty_rows = pad.all(dim=1)
        if bool(empty_rows.any()) and pad.shape[1] > 0:
            pad = pad.clone()
            x = x.clone()
            pad[empty_rows, 0] = False
            x[empty_rows, 0] = 0.0
        if bool((~pad).any()):
            x = self.event_encoder(x, src_key_padding_mask=pad)
        pos_weight = torch.linspace(0.2, 1.0, steps=max(x.shape[1], 1), device=x.device, dtype=x.dtype).unsqueeze(0)
        weights = mask.to(dtype=x.dtype) * pos_weight
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        idx_weight = mask * torch.arange(1, x.shape[1] + 1, device=x.device, dtype=mask.dtype).unsqueeze(0)
        tail_idx = idx_weight.argmax(dim=1).clamp(0, x.shape[1] - 1)
        tail = x[torch.arange(x.shape[0], device=x.device), tail_idx]
        tail = tail * (mask.sum(dim=1, keepdim=True) > 0).float()
        return pooled + tail

    def _known_encode(self, batch: V15Batch, *, ablate: str | None = None) -> torch.Tensor:
        mask = batch.known_mask.float()
        if ablate == "no_known":
            mask = torch.zeros_like(mask)
        cards = _safe_card(batch.known_cards)
        scalar = torch.stack([batch.known_counts.float(), batch.known_age.float()], dim=-1)
        x = self.known_fc(torch.cat([self.card_emb(cards), scalar], dim=-1))
        return _masked_mean(x, mask)

    def _option_encode(self, batch: V15Batch) -> torch.Tensor:
        return self.option_fc(
            torch.cat(
                [
                    self.type_emb(batch.opt_type.clamp(0, N_OPT_TYPES + 1)),
                    self.card_emb(_safe_card(batch.opt_card)),
                    self.card_emb(_safe_card(batch.opt_card2)),
                    self.attack_emb(_safe_attack(batch.opt_attack)),
                    batch.opt_feats.float(),
                ],
                dim=-1,
            )
        )

    def forward(self, batch: V15Batch, *, ablate: str | None = None) -> dict[str, torch.Tensor]:
        state_pool = self._state_encode(batch)
        hist_pool = self._event_encode(batch, ablate=ablate)
        known_pool = self._known_encode(batch, ablate=ablate)
        feat_pool = self.state_feat_fc(batch.feats.float())
        fused = self.fuse_fc(torch.cat([state_pool, hist_pool, known_pool, feat_pool], dim=-1))
        plan_latent = self.plan_latent_fc(fused)
        scorer_plan = torch.zeros_like(plan_latent) if ablate == "no_plan" else plan_latent
        action_ctx = self.action_context_fc(torch.cat([state_pool, hist_pool, known_pool, feat_pool, scorer_plan], dim=-1))
        opt = self._option_encode(batch)
        ctx = action_ctx.unsqueeze(1).expand_as(opt)
        plan_exp = scorer_plan.unsqueeze(1).expand_as(opt)
        score_in = torch.cat([opt, ctx, opt * ctx, opt * plan_exp], dim=-1)
        action_logits = self.action_score(score_in).squeeze(-1).masked_fill(batch.option_mask <= 0, NEG_INF)
        multi_logits = self.multi_score(score_in).squeeze(-1).masked_fill(batch.option_mask <= 0, NEG_INF)
        steps = torch.arange(self.plan_steps, device=plan_latent.device).unsqueeze(0).expand(plan_latent.shape[0], -1)
        step_emb = self.plan_step_emb(steps)
        plan_tokens = self.plan_query(torch.cat([plan_latent.unsqueeze(1).expand(-1, self.plan_steps, -1), step_emb], dim=-1))
        type_logits = self.type_head(plan_latent)
        history_type_logits = self.history_type_head(hist_pool)
        known_type_logits = self.known_type_head(known_pool)
        if self.type_prior_scale:
            prior_latent = torch.zeros_like(plan_latent) if ablate == "no_plan" else plan_latent
            prior_logits = self.type_head(prior_latent)
            type_logp = torch.log_softmax(prior_logits, dim=-1)
            gathered = type_logp.gather(1, batch.opt_type.clamp(0, N_ACTION_TYPES - 1))
            action_logits = action_logits + self.type_prior_scale * gathered.masked_fill(batch.option_mask <= 0, 0.0)
        if self.history_type_prior_scale:
            hist_logits = self.history_type_head(torch.zeros_like(hist_pool) if ablate == "no_history" else hist_pool)
            hist_logp = torch.log_softmax(hist_logits, dim=-1)
            hist_gathered = hist_logp.gather(1, batch.opt_type.clamp(0, N_ACTION_TYPES - 1))
            action_logits = action_logits + self.history_type_prior_scale * hist_gathered.masked_fill(batch.option_mask <= 0, 0.0)
        if self.known_type_prior_scale:
            known_logits_for_prior = self.known_type_head(torch.zeros_like(known_pool) if ablate == "no_known" else known_pool)
            known_logp = torch.log_softmax(known_logits_for_prior, dim=-1)
            known_gathered = known_logp.gather(1, batch.opt_type.clamp(0, N_ACTION_TYPES - 1))
            action_logits = action_logits + self.known_type_prior_scale * known_gathered.masked_fill(batch.option_mask <= 0, 0.0)
        return {
            "action_logits": action_logits,
            "multi_logits": multi_logits,
            "type_logits": type_logits,
            "history_type_logits": history_type_logits,
            "known_type_logits": known_type_logits,
            "context_logits": self.context_head(plan_latent),
            "mode_logits": self.mode_head(plan_latent),
            "continue_logits": self.continue_head(plan_latent).squeeze(-1),
            "outcome_logits": self.outcome_head(plan_latent).squeeze(-1),
            "plan_latent": plan_latent,
            "plan_type_logits": self.plan_type_head(plan_tokens),
            "plan_card_logits": self.plan_card_head(plan_tokens),
            "plan_card2_logits": self.plan_card2_head(plan_tokens),
            "plan_attack_logits": self.plan_attack_head(plan_tokens),
            "plan_context_logits": self.plan_context_head(plan_tokens),
        }


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def _masked_plan_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1).clamp(0, logits.shape[-1] - 1), reduction="none")
    w = (mask.reshape(-1).float() * weight.unsqueeze(1).expand_as(mask).reshape(-1).float())
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


def v15_policy_loss(outputs: dict[str, torch.Tensor], batch: V15Batch, cfg: V15LossConfig) -> tuple[torch.Tensor, dict[str, float]]:
    weight = batch.sample_weight.float()
    valid = batch.target_first >= 0
    if bool(valid.any()):
        action_ce = F.cross_entropy(outputs["action_logits"][valid], batch.target_first[valid], reduction="none")
        action_loss = _weighted_mean(action_ce, weight[valid])
    else:
        action_loss = outputs["action_logits"].sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(outputs["multi_logits"], batch.target_multi, reduction="none")
    bce = (bce * batch.option_mask.float()).sum(dim=1) / batch.option_mask.sum(dim=1).clamp_min(1.0)
    multi_loss = _weighted_mean(bce, weight)
    type_loss = _weighted_mean(F.cross_entropy(outputs["type_logits"], batch.target_type.clamp(0, N_ACTION_TYPES - 1), reduction="none"), weight)
    history_type_loss = _weighted_mean(F.cross_entropy(outputs["history_type_logits"], batch.target_type.clamp(0, N_ACTION_TYPES - 1), reduction="none"), weight)
    known_type_loss = _weighted_mean(F.cross_entropy(outputs["known_type_logits"], batch.target_type.clamp(0, N_ACTION_TYPES - 1), reduction="none"), weight)
    context_loss = _weighted_mean(F.cross_entropy(outputs["context_logits"], batch.target_context.clamp(0, 127), reduction="none"), weight)
    plan_mask = batch.plan_mask.float()
    plan_type_loss = _masked_plan_ce(outputs["plan_type_logits"], batch.plan_type, plan_mask, weight)
    plan_card_loss = _masked_plan_ce(outputs["plan_card_logits"], _safe_card(batch.plan_card), plan_mask * (batch.plan_card > 0).float(), weight)
    plan_attack_loss = _masked_plan_ce(outputs["plan_attack_logits"], _safe_attack(batch.plan_attack), plan_mask * (batch.plan_attack > 0).float(), weight)
    plan_context_loss = _masked_plan_ce(outputs["plan_context_logits"], batch.plan_context.clamp(0, 127), plan_mask, weight)
    continue_loss = _weighted_mean(F.binary_cross_entropy_with_logits(outputs["continue_logits"], batch.turn_continue.float(), reduction="none"), weight)
    mode_loss = _weighted_mean(F.cross_entropy(outputs["mode_logits"], batch.plan_mode.clamp(0, N_PLAN_MODES - 1), reduction="none"), weight)
    outcome_loss = _weighted_mean(F.binary_cross_entropy_with_logits(outputs["outcome_logits"], batch.won.float(), reduction="none"), weight)
    total = (
        cfg.action_weight * action_loss
        + cfg.multi_weight * multi_loss
        + cfg.type_weight * type_loss
        + cfg.history_type_weight * history_type_loss
        + cfg.known_type_weight * known_type_loss
        + cfg.context_weight * context_loss
        + cfg.plan_type_weight * plan_type_loss
        + cfg.plan_card_weight * plan_card_loss
        + cfg.plan_attack_weight * plan_attack_loss
        + cfg.plan_context_weight * plan_context_loss
        + cfg.continue_weight * continue_loss
        + cfg.mode_weight * mode_loss
        + cfg.outcome_weight * outcome_loss
    )
    parts = {
        "loss": float(total.detach().item()),
        "action": float(action_loss.detach().item()),
        "multi": float(multi_loss.detach().item()),
        "type": float(type_loss.detach().item()),
        "hist_type": float(history_type_loss.detach().item()),
        "known_type": float(known_type_loss.detach().item()),
        "ctx": float(context_loss.detach().item()),
        "plan_type": float(plan_type_loss.detach().item()),
        "plan_card": float(plan_card_loss.detach().item()),
        "plan_attack": float(plan_attack_loss.detach().item()),
        "plan_ctx": float(plan_context_loss.detach().item()),
        "continue": float(continue_loss.detach().item()),
        "mode": float(mode_loss.detach().item()),
        "outcome": float(outcome_loss.detach().item()),
    }
    return total, parts


@torch.no_grad()
def v15_accuracy(outputs: dict[str, torch.Tensor], batch: V15Batch) -> dict[str, float]:
    out: dict[str, float] = {}
    pred = outputs["action_logits"].argmax(dim=-1)
    valid = batch.target_first >= 0
    if bool(valid.any()):
        out["top1"] = float((pred[valid] == batch.target_first[valid]).float().mean().item())
        top3 = outputs["action_logits"].topk(k=min(3, outputs["action_logits"].shape[-1]), dim=-1).indices
        out["top3"] = float((top3[valid] == batch.target_first[valid].unsqueeze(-1)).any(dim=-1).float().mean().item())
    else:
        out["top1"] = 0.0
        out["top3"] = 0.0
    target_type = batch.target_type.clamp(0, N_ACTION_TYPES - 1)
    out["type_acc"] = float((outputs["type_logits"].argmax(dim=-1) == target_type).float().mean().item())
    out["history_type_acc"] = float((outputs["history_type_logits"].argmax(dim=-1) == target_type).float().mean().item())
    out["known_type_acc"] = float((outputs["known_type_logits"].argmax(dim=-1) == target_type).float().mean().item())
    out["continue_acc"] = float(((torch.sigmoid(outputs["continue_logits"]) >= 0.5).float() == batch.turn_continue).float().mean().item())
    cont_target = batch.turn_continue > 0.5
    cont_pred = torch.sigmoid(outputs["continue_logits"]) >= 0.5
    tp = (cont_target & cont_pred).sum().float()
    fp = ((~cont_target) & cont_pred).sum().float()
    fn = (cont_target & (~cont_pred)).sum().float()
    out["continue_f1"] = float((2 * tp / (2 * tp + fp + fn).clamp_min(1.0)).item())
    pm = batch.plan_mask > 0
    if bool(pm.any()):
        out["plan_type_acc"] = float((outputs["plan_type_logits"].argmax(dim=-1)[pm] == batch.plan_type[pm]).float().mean().item())
        card_mask = pm & (batch.plan_card > 0)
        out["plan_card_acc"] = float((outputs["plan_card_logits"].argmax(dim=-1)[card_mask] == _safe_card(batch.plan_card)[card_mask]).float().mean().item()) if bool(card_mask.any()) else 0.0
        atk_mask = pm & (batch.plan_attack > 0)
        out["plan_attack_acc"] = float((outputs["plan_attack_logits"].argmax(dim=-1)[atk_mask] == _safe_attack(batch.plan_attack)[atk_mask]).float().mean().item()) if bool(atk_mask.any()) else 0.0
    else:
        out["plan_type_acc"] = 0.0
        out["plan_card_acc"] = 0.0
        out["plan_attack_acc"] = 0.0
    multi_pred = (torch.sigmoid(outputs["multi_logits"]) >= 0.5) & (batch.option_mask > 0)
    multi_tgt = batch.target_multi > 0.5
    tp = (multi_pred & multi_tgt).sum().float()
    fp = (multi_pred & (~multi_tgt) & (batch.option_mask > 0)).sum().float()
    fn = ((~multi_pred) & multi_tgt).sum().float()
    out["multi_f1"] = float((2 * tp / (2 * tp + fp + fn).clamp_min(1.0)).item())
    for typ, name in (
        (TYPE_PLAY, "play"),
        (TYPE_ATTACH, "attach"),
        (TYPE_EVOLVE, "evolve"),
        (TYPE_ABILITY, "ability"),
        (TYPE_RETREAT, "retreat"),
        (TYPE_ATTACK, "attack"),
        (TYPE_END, "end"),
    ):
        mask = valid & (batch.target_type == typ)
        out[f"{name}_top1"] = float((pred[mask] == batch.target_first[mask]).float().mean().item()) if bool(mask.any()) else 0.0
        out[f"{name}_n"] = float(mask.sum().item())
    return out


@torch.no_grad()
def compare_logits(name: str, full: dict[str, torch.Tensor], ablated: dict[str, torch.Tensor], batch: V15Batch) -> dict[str, float]:
    valid = batch.target_first >= 0
    if not bool(valid.any()):
        return {f"{name}_delta": 0.0, f"{name}_agree": 0.0, f"{name}_kl": 0.0, f"{name}_top1": 0.0}
    f = full["action_logits"][valid]
    a = ablated["action_logits"][valid]
    fp = f.argmax(dim=-1)
    ap = a.argmax(dim=-1)
    f_lp = torch.log_softmax(f, dim=-1)
    a_lp = torch.log_softmax(a, dim=-1)
    p = f_lp.exp()
    kl = (p * (f_lp - a_lp)).sum(dim=-1).mean()
    acc_full = (fp == batch.target_first[valid]).float().mean()
    acc_ab = (ap == batch.target_first[valid]).float().mean()
    return {
        f"{name}_top1": float(acc_ab.item()),
        f"{name}_delta": float((acc_ab - acc_full).item()),
        f"{name}_agree": float((fp == ap).float().mean().item()),
        f"{name}_kl": float(kl.item()),
    }
