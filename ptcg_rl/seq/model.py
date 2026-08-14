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
from ptcg_rl.seq.constants import DAMAGE_COUNTER_ANY_CONTEXT, FUTURE_PLAN_DIM, LEDGER_FEAT_DIM, MAX_SELECT_COUNT, N_ACTION_TYPES
from ptcg_rl.seq.data import SequenceBatch

# Keep this fp16-safe. ``masked_fill`` runs under AMP during training, and
# values like -1e9 cannot be represented in float16.
NEG_INF = -1e4


@dataclass
class SequenceLossConfig:
    action_weight: float = 1.0
    multi_weight: float = 0.15
    count_weight: float = 0.20
    plan_weight: float = 0.35
    outcome_weight: float = 0.10
    type_weight: float = 0.10
    multi_target_weight: float = 1.0
    damage_counter_weight: float = 1.0


class SequencePolicyNet(nn.Module):
    """Causal sequence policy for game-window imitation.

    This model is intentionally separate from ``PolicyValueNet``.  The old BC
    stack optimizes independent decision points; this network encodes a prefix
    of decisions and trains auxiliary heads on future behavior so long-game
    resource planning is part of the primary objective.
    """

    def __init__(
        self,
        *,
        width: int = 384,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.10,
        state_feat_dim: int = STATE_FEAT_DIM,
        opt_feat_dim: int = OPT_FEAT_DIM,
        state_token_feat_dim: int = STATE_TOKEN_FEAT_DIM,
        ledger_feat_dim: int = LEDGER_FEAT_DIM,
        future_plan_dim: int = FUTURE_PLAN_DIM,
        max_seq_len: int = 64,
    ):
        super().__init__()
        width = int(width)
        if width % int(heads) != 0:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.layers = int(layers)
        self.heads = int(heads)
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.state_token_feat_dim = int(state_token_feat_dim)
        self.ledger_feat_dim = int(ledger_feat_dim)
        self.future_plan_dim = int(future_plan_dim)
        self.max_seq_len = int(max_seq_len)

        card_dim = width // 4
        attack_dim = max(16, width // 12)
        type_dim = max(16, width // 12)
        ctx_dim = max(16, width // 16)

        self.card_emb = nn.Embedding(N_CARDS + 2, card_dim, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, attack_dim, padding_idx=0)
        self.type_emb = nn.Embedding(N_OPT_TYPES + 2, type_dim, padding_idx=0)
        self.prev_type_emb = nn.Embedding(N_ACTION_TYPES + 2, type_dim, padding_idx=0)
        self.context_emb = nn.Embedding(66, ctx_dim, padding_idx=0)
        self.select_type_emb = nn.Embedding(18, ctx_dim, padding_idx=0)
        self.slot_emb = nn.Embedding(BOARD_SLOTS + MAX_HAND, ctx_dim)
        self.seq_pos_emb = nn.Embedding(max_seq_len, width)

        self.state_feat_fc = nn.Sequential(
            nn.Linear(self.state_feat_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.state_token_feat_fc = nn.Sequential(
            nn.Linear(self.state_token_feat_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width // 2),
        ) if self.state_token_feat_dim > 0 else None
        token_in = card_dim + ctx_dim + (width // 2 if self.state_token_feat_dim > 0 else 0)
        self.state_token_fc = nn.Sequential(
            nn.Linear(token_in, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
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

        opt_in = card_dim * 2 + attack_dim + type_dim + self.opt_feat_dim
        self.option_fc = nn.Sequential(
            nn.Linear(opt_in, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        prev_in = type_dim + card_dim * 2 + attack_dim + ctx_dim * 2 + 1
        self.prev_action_fc = nn.Sequential(
            nn.Linear(prev_in, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.ledger_fc = nn.Sequential(
            nn.Linear(self.ledger_feat_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

        seq_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(seq_layer, num_layers=self.layers)
        self.sequence_norm = nn.LayerNorm(width)

        self.option_query = nn.Linear(width, width)
        self.option_key = nn.Linear(width, width)
        self.order_pos_emb = nn.Embedding(MAX_SELECT_COUNT, width)
        self.action_score = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.plan_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.future_plan_dim),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_OPT_TYPES + 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, MAX_SELECT_COUNT + 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.card_emb.weight, std=0.02)
        nn.init.normal_(self.attack_emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.prev_type_emb.weight, std=0.02)

    def config(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "state_feat_dim": self.state_feat_dim,
            "opt_feat_dim": self.opt_feat_dim,
            "state_token_feat_dim": self.state_token_feat_dim,
            "ledger_feat_dim": self.ledger_feat_dim,
            "future_plan_dim": self.future_plan_dim,
            "max_seq_len": self.max_seq_len,
        }

    def forward(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        board = batch.board.long().clamp(0, N_CARDS + 1)
        hand = batch.hand.long().clamp(0, N_CARDS + 1)
        bsz, seq_len = board.shape[:2]
        flat = bsz * seq_len

        state_tokens, state_mask = self._state_tokens(batch, board, hand)
        state_ctx = self.state_encoder(
            state_tokens.reshape(flat, BOARD_SLOTS + MAX_HAND, self.width),
            src_key_padding_mask=state_mask.reshape(flat, BOARD_SLOTS + MAX_HAND),
        )
        valid = (~state_mask.reshape(flat, BOARD_SLOTS + MAX_HAND)).float().unsqueeze(-1)
        state_pool = (state_ctx * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        state_pool = state_pool.reshape(bsz, seq_len, self.width)

        state_feat = self.state_feat_fc(_fit_last_dim(batch.feats.float(), self.state_feat_dim))
        opt_emb = self._option_embeddings(batch)
        opt_mask = batch.option_mask.float()
        opt_summary = (opt_emb * opt_mask.unsqueeze(-1)).sum(dim=2) / opt_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        prev = self._prev_action_embedding(batch)
        ledger = self.ledger_fc(_fit_last_dim(batch.ledger_feats.float(), self.ledger_feat_dim))

        step_mask = batch.step_mask.float()
        decision = self.decision_fc(torch.cat([state_pool + state_feat, opt_summary, prev, ledger], dim=-1))
        pos = torch.arange(seq_len, device=decision.device).clamp(max=self.max_seq_len - 1)
        decision = decision + self.seq_pos_emb(pos).unsqueeze(0)
        decision = decision * step_mask.unsqueeze(-1)

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=decision.device, dtype=torch.bool),
            diagonal=1,
        )
        seq = self.sequence_encoder(decision, mask=causal)
        seq = self.sequence_norm(seq) * step_mask.unsqueeze(-1)

        q_base = self.option_query(seq)
        q = q_base.unsqueeze(2)
        k = self.option_key(opt_emb)
        scores = self.action_score(torch.cat([q.expand_as(k), k, q.expand_as(k) * k], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(opt_mask <= 0, NEG_INF)
        order_queries = q_base.unsqueeze(2) + self.order_pos_emb(
            torch.arange(MAX_SELECT_COUNT, device=seq.device)
        ).view(1, 1, MAX_SELECT_COUNT, self.width)
        # Do not reuse ``action_score`` here. Expanding [B,T,K,N,W] and
        # concatenating three copies is the largest activation in the model and
        # OOMs under population training. The dot-product scorer keeps only
        # [B,T,K,N] order logits while still conditioning on selection position.
        order_scores = torch.einsum("btkd,btnd->btkn", order_queries, k)
        order_scores = order_scores * (float(self.width) ** -0.5)
        order_scores = order_scores + scores.unsqueeze(2)
        order_scores = order_scores.masked_fill(opt_mask.unsqueeze(2) <= 0, NEG_INF)
        return {
            "action_logits": scores,
            "order_logits": order_scores,
            "plan_logits": self.plan_head(seq),
            "outcome_logits": self.outcome_head(seq).squeeze(-1),
            "type_logits": self.type_head(seq),
            "count_logits": self.count_head(seq),
        }

    def _state_tokens(
        self,
        batch: SequenceBatch,
        board: torch.Tensor,
        hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.cat([board, hand], dim=-1)
        bsz, seq_len, n_tokens = ids.shape
        flat_ids = ids.reshape(bsz * seq_len, n_tokens)
        card = self.card_emb(flat_ids)
        pos = torch.arange(n_tokens, device=ids.device).unsqueeze(0).expand(bsz * seq_len, -1)
        parts = [card, self.slot_emb(pos)]
        if self.state_token_feat_fc is not None:
            stf = _fit_last_dim(batch.state_token_feats.float(), self.state_token_feat_dim)
            parts.append(self.state_token_feat_fc(stf.reshape(bsz * seq_len, n_tokens, self.state_token_feat_dim)))
        token = self.state_token_fc(torch.cat(parts, dim=-1))
        mask = flat_ids <= 0
        # Empty hands are common; keep one padding token unmasked so transformer
        # never receives an all-masked row.
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask[all_masked, 0] = False
        return token.reshape(bsz, seq_len, n_tokens, self.width), mask.reshape(bsz, seq_len, n_tokens)

    def _option_embeddings(self, batch: SequenceBatch) -> torch.Tensor:
        opt_type = batch.opt_type.long().clamp(0, N_OPT_TYPES)
        card = batch.opt_card.long().clamp(0, N_CARDS + 1)
        card2 = batch.opt_card2.long().clamp(0, N_CARDS + 1)
        attack = batch.opt_attack.long().clamp(0, N_ATTACKS)
        feats = _fit_last_dim(batch.opt_feats.float(), self.opt_feat_dim)
        return self.option_fc(torch.cat([
            self.card_emb(card),
            self.card_emb(card2),
            self.attack_emb(attack),
            self.type_emb(opt_type + 1),
            feats,
        ], dim=-1))

    def _prev_action_embedding(self, batch: SequenceBatch) -> torch.Tensor:
        return self.prev_action_fc(torch.cat([
            self.prev_type_emb(batch.prev_type.long().clamp(0, N_ACTION_TYPES) + 1),
            self.card_emb(batch.prev_card.long().clamp(0, N_CARDS + 1)),
            self.card_emb(batch.prev_card2.long().clamp(0, N_CARDS + 1)),
            self.attack_emb(batch.prev_attack.long().clamp(0, N_ATTACKS)),
            self.context_emb(batch.prev_context.long().clamp(0, 65)),
            self.select_type_emb(batch.prev_select_type.long().clamp(0, 17)),
            batch.prev_count.float().unsqueeze(-1),
        ], dim=-1))


def sequence_policy_loss(
    outputs: dict[str, torch.Tensor],
    batch: SequenceBatch,
    cfg: SequenceLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    step_mask = batch.step_mask.float()
    weights = batch.sample_weight.float() * step_mask
    target_count = batch.target_multi.float().sum(dim=-1).long().clamp(0, MAX_SELECT_COUNT)
    multi_target = (target_count > 1) & (step_mask > 0)
    damage_counter = (batch.target_context.long() == DAMAGE_COUNTER_ANY_CONTEXT) & (step_mask > 0)
    multi_boost = torch.where(
        multi_target,
        torch.full_like(weights, max(float(cfg.multi_target_weight), 1.0)),
        torch.ones_like(weights),
    )
    damage_boost = torch.where(
        damage_counter,
        torch.full_like(weights, max(float(cfg.damage_counter_weight), 1.0)),
        torch.ones_like(weights),
    )
    decision_weights = weights * torch.maximum(multi_boost, damage_boost)
    valid_action = (batch.target_first >= 0) & (step_mask > 0)
    valid_weight_sum = weights[valid_action].sum().clamp(min=1.0) if bool(valid_action.any()) else weights.sum().clamp(min=1.0)
    decision_weight_sum = decision_weights[valid_action].sum().clamp(min=1.0) if bool(valid_action.any()) else decision_weights.sum().clamp(min=1.0)
    if bool(valid_action.any()):
        action_loss = F.cross_entropy(
            outputs["action_logits"][valid_action],
            batch.target_first.long()[valid_action],
            reduction="none",
        )
        valid_decision_weights = decision_weights[valid_action]
        raw_action_loss = action_loss
        action_loss = (action_loss * valid_decision_weights).sum() / decision_weight_sum
    else:
        raw_action_loss = torch.zeros(0, device=outputs["action_logits"].device)
        valid_decision_weights = torch.zeros(0, device=outputs["action_logits"].device)
        action_loss = outputs["action_logits"].sum() * 0.0

    order_logits = outputs.get("order_logits")
    if order_logits is not None:
        valid_order = (batch.target_order >= 0) & (step_mask.unsqueeze(-1) > 0)
        if bool(valid_order.any()):
            order_loss_raw = F.cross_entropy(
                order_logits[valid_order],
                batch.target_order.long()[valid_order],
                reduction="none",
            )
            order_weights = decision_weights.unsqueeze(-1).expand_as(batch.target_order.float())[valid_order]
            order_loss = (order_loss_raw * order_weights).sum() / order_weights.sum().clamp(min=1.0)
        else:
            order_loss = outputs["action_logits"].sum() * 0.0
    else:
        order_loss = outputs["action_logits"].sum() * 0.0

    opt_mask = batch.option_mask.float() * step_mask.unsqueeze(-1)
    multi_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["action_logits"].clamp(min=-30.0, max=30.0),
        batch.target_multi.float(),
        reduction="none",
    )
    multi_weighted_mask = opt_mask * decision_weights.unsqueeze(-1)
    multi_loss = (multi_loss_raw * multi_weighted_mask).sum() / multi_weighted_mask.sum().clamp(min=1.0)

    count_loss_raw = F.cross_entropy(
        outputs["count_logits"].reshape(-1, outputs["count_logits"].shape[-1]),
        target_count.reshape(-1),
        reduction="none",
    ).reshape_as(step_mask)
    count_loss = (count_loss_raw * decision_weights).sum() / decision_weights.sum().clamp(min=1.0)

    plan_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["plan_logits"],
        batch.future_plan.float().clamp(0.0, 1.0),
        reduction="none",
    )
    plan_loss = (plan_loss_raw * weights.unsqueeze(-1)).sum() / (weights.sum().clamp(min=1.0) * batch.future_plan.shape[-1])

    outcome_loss_raw = F.binary_cross_entropy_with_logits(outputs["outcome_logits"], batch.outcome.float(), reduction="none")
    outcome_loss = (outcome_loss_raw * weights).sum() / weights.sum().clamp(min=1.0)

    type_loss_raw = F.cross_entropy(
        outputs["type_logits"].reshape(-1, outputs["type_logits"].shape[-1]),
        batch.target_type.long().reshape(-1).clamp(0, outputs["type_logits"].shape[-1] - 1),
        reduction="none",
    ).reshape_as(step_mask)
    type_loss = (type_loss_raw * decision_weights).sum() / decision_weights.sum().clamp(min=1.0)

    loss = (
        cfg.action_weight * action_loss
        + 0.35 * cfg.action_weight * order_loss
        + cfg.multi_weight * multi_loss
        + cfg.count_weight * count_loss
        + cfg.plan_weight * plan_loss
        + cfg.outcome_weight * outcome_loss
        + cfg.type_weight * type_loss
    )

    valid_dca = damage_counter[valid_action] if bool(valid_action.any()) else torch.zeros(0, dtype=torch.bool, device=weights.device)
    valid_multi = multi_target[valid_action] if bool(valid_action.any()) else torch.zeros(0, dtype=torch.bool, device=weights.device)

    def weighted_valid_loss(mask: torch.Tensor) -> float:
        if raw_action_loss.numel() == 0 or not bool(mask.any()):
            return 0.0
        ww = valid_decision_weights[mask]
        return float(((raw_action_loss[mask] * ww).sum() / ww.sum().clamp(min=1.0)).detach().cpu())

    dca_rows = damage_counter.float().sum().clamp(min=1.0)
    dca_focus = batch.dca_group_focus_frac.float()
    dca_unique = batch.dca_group_unique_slots.float()
    dca_spread = damage_counter & (batch.dca_group_unique_slots.long() > 1)
    dca_pos = batch.dca_pos.float()
    dca_prior_unique = batch.dca_prior_unique_slots.float()
    dca_prior_same = batch.dca_prior_same_slot.float()
    parts = {
        "loss": float(loss.detach().cpu()),
        "action": float(action_loss.detach().cpu()),
        "order": float(order_loss.detach().cpu()),
        "multi": float(multi_loss.detach().cpu()),
        "count": float(count_loss.detach().cpu()),
        "plan": float(plan_loss.detach().cpu()),
        "outcome": float(outcome_loss.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "multi_target_rate": float((multi_target.float().sum() / step_mask.sum().clamp(min=1.0)).detach().cpu()),
        "damage_counter_rate": float((damage_counter.float().sum() / step_mask.sum().clamp(min=1.0)).detach().cpu()),
        "weight_boost": float((decision_weight_sum / valid_weight_sum).detach().cpu()),
        "dca_weight_share": float((decision_weights[damage_counter].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach().cpu()) if bool(valid_action.any()) else 0.0,
        "multi_weight_share": float((decision_weights[multi_target].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach().cpu()) if bool(valid_action.any()) else 0.0,
        "dca_action": weighted_valid_loss(valid_dca),
        "non_dca_action": weighted_valid_loss(~valid_dca) if raw_action_loss.numel() else 0.0,
        "multi2_action": weighted_valid_loss(valid_multi),
        "dca_focus_mean": float(((dca_focus * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_spread_rate": float((dca_spread.float().sum() / dca_rows).detach().cpu()),
        "dca_unique_mean": float(((dca_unique * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_pos_mean": float(((dca_pos.clamp(min=0) * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_prior_unique_mean": float(((dca_prior_unique * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_prior_same_mean": float(((dca_prior_same * damage_counter.float()).sum() / dca_rows).detach().cpu()),
    }
    return loss, parts


@torch.no_grad()
def sequence_accuracy(outputs: dict[str, torch.Tensor], batch: SequenceBatch) -> dict[str, float]:
    valid = (batch.target_first >= 0) & (batch.step_mask > 0)
    if not bool(valid.any()):
        return {
            "top1": 0.0,
            "type_acc": 0.0,
            "count_acc": 0.0,
            "n": 0.0,
            "multi2_n": 0.0,
            "multi2_rate": 0.0,
            "capable2_rate": 0.0,
            "dca_n": 0.0,
            "dca_rate": 0.0,
            "dca_top1": 0.0,
            "dca_count_acc": 0.0,
            "dca_count_mae": 0.0,
            "dca_pred_k": 0.0,
            "dca_target_k": 0.0,
            "dca_focus_mean": 0.0,
            "dca_spread_rate": 0.0,
            "dca_spread_top1": 0.0,
            "dca_focus_top1": 0.0,
            "dca_first_top1": 0.0,
            "dca_late_top1": 0.0,
            "dca_prior_same_top1": 0.0,
            "dca_prior_unique_mean": 0.0,
        }
    pred = outputs["action_logits"].argmax(dim=-1)
    top1 = (pred[valid] == batch.target_first[valid]).float().mean().item()
    type_pred = outputs["type_logits"].argmax(dim=-1)
    type_acc = (type_pred[valid] == batch.target_type[valid]).float().mean().item()
    count_target = batch.target_multi.float().sum(dim=-1).long().clamp(0, MAX_SELECT_COUNT)
    count_pred = outputs["count_logits"].argmax(dim=-1)
    count_acc = (count_pred[valid] == count_target[valid]).float().mean().item()
    count_mae = (count_pred[valid].float() - count_target[valid].float()).abs().mean().item()
    multi2 = valid & (count_target > 1)
    dca = valid & (batch.target_context.long() == DAMAGE_COUNTER_ANY_CONTEXT)
    dca_spread = dca & (batch.dca_group_unique_slots.long() > 1)
    dca_focus = dca & (batch.dca_group_unique_slots.long() <= 1)
    dca_first = dca & (batch.dca_pos.long() == 0)
    dca_late = dca & (batch.dca_pos.long() > 0)
    dca_prior_same = dca & (batch.dca_prior_same_slot.long() > 0)
    valid_n = float(valid.sum().item())
    multi2_n = float(multi2.sum().item())
    dca_n = float(dca.sum().item())
    capable2_n = float((valid & (batch.max_count.long() > 1)).sum().item())

    opt_mask = batch.option_mask.float()
    logits = outputs["action_logits"].masked_fill(opt_mask <= 0, NEG_INF)
    max_k = min(MAX_SELECT_COUNT, logits.shape[-1])
    multi2_precision = multi2_recall = multi2_f1 = 0.0
    multi2_top1 = multi2_count_acc = multi2_count_mae = 0.0
    dca_precision = dca_recall = dca_f1 = 0.0
    dca_top1 = dca_count_acc = dca_count_mae = 0.0
    dca_pred_k = dca_target_k = 0.0
    dca_spread_top1 = dca_focus_top1 = 0.0
    dca_first_top1 = dca_late_top1 = dca_prior_same_top1 = 0.0
    if max_k > 0:
        top_idx = logits.topk(k=max_k, dim=-1).indices
        pred_k = torch.minimum(
            torch.maximum(count_pred, batch.min_count.long()),
            torch.minimum(batch.max_count.long(), opt_mask.sum(dim=-1).long()).clamp(max=MAX_SELECT_COUNT),
        ).clamp(min=0, max=max_k)
        rank = torch.arange(max_k, device=logits.device).view(1, 1, max_k)
        pred_multi = torch.zeros_like(batch.target_multi.float())
        pred_multi.scatter_(-1, top_idx, (rank < pred_k.unsqueeze(-1)).float())
        pred_multi = pred_multi * opt_mask
        target_multi = batch.target_multi.float() * opt_mask
        tp = (pred_multi * target_multi).sum(dim=-1)
        pp = pred_multi.sum(dim=-1)
        gp = target_multi.sum(dim=-1)
        precision = (tp[valid] / pp[valid].clamp(min=1.0)).mean().item()
        recall = (tp[valid] / gp[valid].clamp(min=1.0)).mean().item()
        f1 = (2.0 * tp[valid] / (pp[valid] + gp[valid]).clamp(min=1.0)).mean().item()
        if bool(multi2.any()):
            multi2_precision = (tp[multi2] / pp[multi2].clamp(min=1.0)).mean().item()
            multi2_recall = (tp[multi2] / gp[multi2].clamp(min=1.0)).mean().item()
            multi2_f1 = (2.0 * tp[multi2] / (pp[multi2] + gp[multi2]).clamp(min=1.0)).mean().item()
            multi2_top1 = (pred[multi2] == batch.target_first[multi2]).float().mean().item()
            multi2_count_acc = (count_pred[multi2] == count_target[multi2]).float().mean().item()
            multi2_count_mae = (count_pred[multi2].float() - count_target[multi2].float()).abs().mean().item()
        if bool(dca.any()):
            dca_precision = (tp[dca] / pp[dca].clamp(min=1.0)).mean().item()
            dca_recall = (tp[dca] / gp[dca].clamp(min=1.0)).mean().item()
            dca_f1 = (2.0 * tp[dca] / (pp[dca] + gp[dca]).clamp(min=1.0)).mean().item()
            dca_top1 = (pred[dca] == batch.target_first[dca]).float().mean().item()
            dca_count_acc = (count_pred[dca] == count_target[dca]).float().mean().item()
            dca_count_mae = (count_pred[dca].float() - count_target[dca].float()).abs().mean().item()
            dca_pred_k = count_pred[dca].float().mean().item()
            dca_target_k = count_target[dca].float().mean().item()
            if bool(dca_spread.any()):
                dca_spread_top1 = (pred[dca_spread] == batch.target_first[dca_spread]).float().mean().item()
            if bool(dca_focus.any()):
                dca_focus_top1 = (pred[dca_focus] == batch.target_first[dca_focus]).float().mean().item()
            if bool(dca_first.any()):
                dca_first_top1 = (pred[dca_first] == batch.target_first[dca_first]).float().mean().item()
            if bool(dca_late.any()):
                dca_late_top1 = (pred[dca_late] == batch.target_first[dca_late]).float().mean().item()
            if bool(dca_prior_same.any()):
                dca_prior_same_top1 = (pred[dca_prior_same] == batch.target_first[dca_prior_same]).float().mean().item()
    else:
        precision = recall = f1 = 0.0

    order_logits = outputs.get("order_logits")
    order_acc = 0.0
    order_n = 0.0
    multi2_order_acc = 0.0
    multi2_order_n = 0.0
    if order_logits is not None:
        order_valid = (batch.target_order >= 0) & (batch.step_mask.unsqueeze(-1) > 0)
        if bool(order_valid.any()):
            order_pred = order_logits.argmax(dim=-1)
            order_acc = (order_pred[order_valid] == batch.target_order.long()[order_valid]).float().mean().item()
            order_n = float(order_valid.sum().item())
            order_valid_multi2 = order_valid & (count_target.unsqueeze(-1) > 1)
            if bool(order_valid_multi2.any()):
                multi2_order_acc = (order_pred[order_valid_multi2] == batch.target_order.long()[order_valid_multi2]).float().mean().item()
                multi2_order_n = float(order_valid_multi2.sum().item())

    outcome_pred = (torch.sigmoid(outputs["outcome_logits"]) >= 0.5).float()
    step_valid = batch.step_mask > 0
    outcome_acc = (outcome_pred[step_valid] == batch.outcome.float()[step_valid]).float().mean().item() if bool(step_valid.any()) else 0.0
    dca_rows = dca.float().sum().clamp(min=1.0)
    dca_focus_mean = ((batch.dca_group_focus_frac.float() * dca.float()).sum() / dca_rows).item()
    dca_spread_rate = (dca_spread.float().sum() / dca_rows).item()
    dca_prior_unique_mean = ((batch.dca_prior_unique_slots.float() * dca.float()).sum() / dca_rows).item()

    return {
        "top1": float(top1),
        "type_acc": float(type_acc),
        "count_acc": float(count_acc),
        "count_mae": float(count_mae),
        "target_k": float(count_target[valid].float().mean().item()),
        "pred_k": float(count_pred[valid].float().mean().item()),
        "set_precision": float(precision),
        "set_recall": float(recall),
        "set_f1": float(f1),
        "order_acc": float(order_acc),
        "order_n": float(order_n),
        "outcome_acc": float(outcome_acc),
        "n": valid_n,
        "multi2_n": multi2_n,
        "multi2_rate": float(multi2_n / max(valid_n, 1.0)),
        "capable2_rate": float(capable2_n / max(valid_n, 1.0)),
        "multi2_top1": float(multi2_top1),
        "multi2_count_acc": float(multi2_count_acc),
        "multi2_count_mae": float(multi2_count_mae),
        "multi2_precision": float(multi2_precision),
        "multi2_recall": float(multi2_recall),
        "multi2_f1": float(multi2_f1),
        "multi2_order_acc": float(multi2_order_acc),
        "multi2_order_n": float(multi2_order_n),
        "dca_n": dca_n,
        "dca_rate": float(dca_n / max(valid_n, 1.0)),
        "dca_top1": float(dca_top1),
        "dca_count_acc": float(dca_count_acc),
        "dca_count_mae": float(dca_count_mae),
        "dca_pred_k": float(dca_pred_k),
        "dca_target_k": float(dca_target_k),
        "dca_precision": float(dca_precision),
        "dca_recall": float(dca_recall),
        "dca_f1": float(dca_f1),
        "dca_focus_mean": float(dca_focus_mean),
        "dca_spread_rate": float(dca_spread_rate),
        "dca_spread_top1": float(dca_spread_top1),
        "dca_focus_top1": float(dca_focus_top1),
        "dca_first_top1": float(dca_first_top1),
        "dca_late_top1": float(dca_late_top1),
        "dca_prior_same_top1": float(dca_prior_same_top1),
        "dca_prior_unique_mean": float(dca_prior_unique_mean),
    }


def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)
