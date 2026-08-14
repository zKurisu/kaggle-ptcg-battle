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
from ptcg_rl.seq.constants import FUTURE_PLAN_DIM, LEDGER_FEAT_DIM, N_ACTION_TYPES
from ptcg_rl.seq.data import SequenceBatch

NEG_INF = -1e9


@dataclass
class SequenceLossConfig:
    action_weight: float = 1.0
    multi_weight: float = 0.15
    plan_weight: float = 0.35
    outcome_weight: float = 0.10
    type_weight: float = 0.10


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

        q = self.option_query(seq).unsqueeze(2)
        k = self.option_key(opt_emb)
        scores = self.action_score(torch.cat([q.expand_as(k), k, q.expand_as(k) * k], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(opt_mask <= 0, NEG_INF)
        return {
            "action_logits": scores,
            "plan_logits": self.plan_head(seq),
            "outcome_logits": self.outcome_head(seq).squeeze(-1),
            "type_logits": self.type_head(seq),
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
    valid_action = (batch.target_first >= 0) & (step_mask > 0)
    if bool(valid_action.any()):
        action_loss = F.cross_entropy(
            outputs["action_logits"][valid_action],
            batch.target_first.long()[valid_action],
            reduction="none",
        )
        action_loss = (action_loss * batch.sample_weight.float()[valid_action]).sum() / weights[valid_action].sum().clamp(min=1.0)
    else:
        action_loss = outputs["action_logits"].sum() * 0.0

    opt_mask = batch.option_mask.float() * step_mask.unsqueeze(-1)
    multi_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["action_logits"].clamp(min=-30.0, max=30.0),
        batch.target_multi.float(),
        reduction="none",
    )
    multi_loss = (multi_loss_raw * opt_mask * batch.sample_weight.float().unsqueeze(-1)).sum() / opt_mask.sum().clamp(min=1.0)

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
    type_loss = (type_loss_raw * weights).sum() / weights.sum().clamp(min=1.0)

    loss = (
        cfg.action_weight * action_loss
        + cfg.multi_weight * multi_loss
        + cfg.plan_weight * plan_loss
        + cfg.outcome_weight * outcome_loss
        + cfg.type_weight * type_loss
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "action": float(action_loss.detach().cpu()),
        "multi": float(multi_loss.detach().cpu()),
        "plan": float(plan_loss.detach().cpu()),
        "outcome": float(outcome_loss.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
    }
    return loss, parts


@torch.no_grad()
def sequence_accuracy(outputs: dict[str, torch.Tensor], batch: SequenceBatch) -> dict[str, float]:
    valid = (batch.target_first >= 0) & (batch.step_mask > 0)
    if not bool(valid.any()):
        return {"top1": 0.0, "type_acc": 0.0, "n": 0.0}
    pred = outputs["action_logits"].argmax(dim=-1)
    top1 = (pred[valid] == batch.target_first[valid]).float().mean().item()
    type_pred = outputs["type_logits"].argmax(dim=-1)
    type_acc = (type_pred[valid] == batch.target_type[valid]).float().mean().item()
    return {"top1": float(top1), "type_acc": float(type_acc), "n": float(valid.sum().item())}


def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)
