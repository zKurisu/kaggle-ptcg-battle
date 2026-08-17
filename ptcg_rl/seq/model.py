from __future__ import annotations

import math
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
from ptcg_rl.seq.constants import DAMAGE_COUNTER_ANY_CONTEXT, FUTURE_PLAN_DIM, KNOWN_OPP_CARDS, LEDGER_FEAT_DIM, MAX_SELECT_COUNT, N_ACTION_TYPES, TURN_PLAN_STEPS
from ptcg_rl.seq.constants import TYPE_ABILITY, TYPE_ATTACH, TYPE_ATTACK, TYPE_END, TYPE_EVOLVE, TYPE_PLAY, TYPE_RETREAT
from ptcg_rl.seq.data import SequenceBatch

# Keep this fp16-safe. ``masked_fill`` runs under AMP during training, and
# values like -1e9 cannot be represented in float16.
NEG_INF = -1e4

_MONITORED_TYPES = (
    (TYPE_PLAY, "play"),
    (TYPE_ATTACH, "attach"),
    (TYPE_EVOLVE, "evolve"),
    (TYPE_ABILITY, "ability"),
    (TYPE_RETREAT, "retreat"),
    (TYPE_ATTACK, "attack"),
    (TYPE_END, "end"),
)

_OPPORTUNITY_TYPES = (
    TYPE_PLAY,
    TYPE_ATTACH,
    TYPE_EVOLVE,
    TYPE_ABILITY,
    TYPE_RETREAT,
    TYPE_ATTACK,
    TYPE_END,
)
_KEY_OPPORTUNITY_TYPES = (TYPE_ATTACH, TYPE_EVOLVE, TYPE_ABILITY, TYPE_ATTACK)


@dataclass
class SequenceLossConfig:
    action_weight: float = 1.0
    current_action_weight: float = 1.0
    prefix_action_weight: float = 0.10
    order_weight: float = 0.15
    multi_weight: float = 0.15
    count_weight: float = 0.20
    plan_weight: float = 0.35
    next_type_weight: float = 0.25
    dca_plan_weight: float = 0.25
    known_action_weight: float = 0.0
    turn_plan_weight: float = 0.0
    turn_terminal_weight: float = 0.0
    turn_next_plan_weight: float = 0.0
    turn_next_type_weight: float = 1.0
    turn_next_card_weight: float = 0.25
    turn_next_attack_weight: float = 0.25
    turn_next_context_weight: float = 0.10
    turn_seq_plan_weight: float = 0.0
    turn_seq_type_weight: float = 1.0
    turn_seq_card_weight: float = 0.15
    turn_seq_attack_weight: float = 0.15
    turn_seq_context_weight: float = 0.05
    opportunity_type_weight: float = 0.0
    opportunity_margin_weight: float = 0.0
    opportunity_margin: float = 0.25
    current_rank_margin_weight: float = 0.0
    current_rank_margin: float = 0.25
    current_rank_margin_min_options: int = 2
    current_complexity_weight: float = 0.0
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
        next_type_horizon: int = 4,
        history_condition_scale: float = 0.0,
        plan_condition_scale: float = 0.0,
        next_type_condition_scale: float = 0.0,
        dca_condition_scale: float = 0.0,
        known_condition_scale: float = 0.0,
        known_logit_scale: float = 0.0,
        turn_condition_scale: float = 0.0,
        turn_next_condition_scale: float = 0.0,
        turn_seq_condition_scale: float = 0.0,
        type_prior_scale: float = 0.0,
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
        self.next_type_horizon = max(1, int(next_type_horizon))
        self.history_condition_scale = float(history_condition_scale)
        self.plan_condition_scale = float(plan_condition_scale)
        self.next_type_condition_scale = float(next_type_condition_scale)
        self.dca_condition_scale = float(dca_condition_scale)
        self.known_condition_scale = float(known_condition_scale)
        self.known_logit_scale = float(known_logit_scale)
        self.turn_condition_scale = float(turn_condition_scale)
        self.turn_next_condition_scale = float(turn_next_condition_scale)
        self.turn_seq_condition_scale = float(turn_seq_condition_scale)
        self.type_prior_scale = float(type_prior_scale)

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
        self.known_opp_fc = nn.Sequential(
            nn.Linear(card_dim + 1, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
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
        self.history_query = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.plan_query = nn.Sequential(
            nn.Linear(self.future_plan_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.next_type_query = nn.Sequential(
            nn.Linear(N_OPT_TYPES + 1, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.dca_query = nn.Sequential(
            nn.Linear(MAX_SELECT_COUNT + 3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.known_query = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.turn_query = nn.Sequential(
            nn.Linear(N_ACTION_TYPES + 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.turn_next_query = nn.Sequential(
            nn.Linear(type_dim + card_dim * 2 + attack_dim + ctx_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.turn_seq_step_emb = nn.Embedding(TURN_PLAN_STEPS, width)
        self.turn_seq_token_fc = nn.Sequential(
            nn.Linear(type_dim + card_dim + attack_dim + ctx_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.turn_seq_query = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_query_norm = nn.LayerNorm(width)
        self.order_pos_emb = nn.Embedding(MAX_SELECT_COUNT, width)
        self.action_score = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.known_action_score = nn.Sequential(
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
        self.next_type_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.next_type_horizon * (N_OPT_TYPES + 1)),
        )
        self.dca_spread_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.dca_unique_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, MAX_SELECT_COUNT + 1),
        )
        self.dca_focus_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.turn_continue_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.turn_remaining_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, MAX_SELECT_COUNT + 1),
        )
        self.turn_future_type_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_ACTION_TYPES),
        )
        self.turn_next_type_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_ACTION_TYPES + 1),
        )
        self.turn_next_card_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_CARDS + 2),
        )
        self.turn_next_card2_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_CARDS + 2),
        )
        self.turn_next_attack_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, N_ATTACKS + 1),
        )
        self.turn_next_context_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 66),
        )
        self.turn_seq_type_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, TURN_PLAN_STEPS * (N_ACTION_TYPES + 1)),
        )
        self.turn_seq_card_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, TURN_PLAN_STEPS * (N_CARDS + 2)),
        )
        self.turn_seq_attack_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, TURN_PLAN_STEPS * (N_ATTACKS + 1)),
        )
        self.turn_seq_context_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, TURN_PLAN_STEPS * 66),
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
            "next_type_horizon": self.next_type_horizon,
            "history_condition_scale": self.history_condition_scale,
            "plan_condition_scale": self.plan_condition_scale,
            "next_type_condition_scale": self.next_type_condition_scale,
            "dca_condition_scale": self.dca_condition_scale,
            "known_condition_scale": self.known_condition_scale,
            "known_logit_scale": self.known_logit_scale,
            "turn_condition_scale": self.turn_condition_scale,
            "turn_next_condition_scale": self.turn_next_condition_scale,
            "turn_seq_condition_scale": self.turn_seq_condition_scale,
            "type_prior_scale": self.type_prior_scale,
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
        known_ctx = self._known_opp_context(batch)
        if self.known_condition_scale != 0.0:
            ledger = ledger + known_ctx * self.known_condition_scale

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

        plan_logits = self.plan_head(seq)
        next_type_logits = self.next_type_head(seq).view(
            bsz,
            seq_len,
            self.next_type_horizon,
            N_OPT_TYPES + 1,
        )
        dca_spread_logits = self.dca_spread_head(seq).squeeze(-1)
        dca_unique_logits = self.dca_unique_head(seq)
        dca_focus_logits = self.dca_focus_head(seq).squeeze(-1)
        turn_continue_logits = self.turn_continue_head(seq).squeeze(-1)
        turn_remaining_logits = self.turn_remaining_head(seq)
        turn_future_type_logits = self.turn_future_type_head(seq)
        turn_next_type_logits = self.turn_next_type_head(seq)
        turn_next_card_logits = self.turn_next_card_head(seq)
        turn_next_card2_logits = self.turn_next_card2_head(seq)
        turn_next_attack_logits = self.turn_next_attack_head(seq)
        turn_next_context_logits = self.turn_next_context_head(seq)
        turn_seq_type_logits = self.turn_seq_type_head(seq).view(
            bsz,
            seq_len,
            TURN_PLAN_STEPS,
            N_ACTION_TYPES + 1,
        )
        turn_seq_card_logits = self.turn_seq_card_head(seq).view(
            bsz,
            seq_len,
            TURN_PLAN_STEPS,
            N_CARDS + 2,
        )
        turn_seq_attack_logits = self.turn_seq_attack_head(seq).view(
            bsz,
            seq_len,
            TURN_PLAN_STEPS,
            N_ATTACKS + 1,
        )
        turn_seq_context_logits = self.turn_seq_context_head(seq).view(
            bsz,
            seq_len,
            TURN_PLAN_STEPS,
            66,
        )
        type_logits = self.type_head(seq)
        count_logits = self.count_head(seq)

        prefix_count = torch.cumsum(step_mask, dim=1) - step_mask
        prefix_sum = torch.cumsum(seq * step_mask.unsqueeze(-1), dim=1) - seq * step_mask.unsqueeze(-1)
        prefix_summary = prefix_sum / prefix_count.clamp(min=1.0).unsqueeze(-1)
        prefix_summary = prefix_summary * (prefix_count > 0).float().unsqueeze(-1)
        prev_seq = torch.zeros_like(seq)
        if seq_len > 1:
            prev_seq[:, 1:] = seq[:, :-1]
        hist_ctx = self.history_query(torch.cat([
            seq,
            prefix_summary,
            seq - prefix_summary,
            seq * prefix_summary + prev_seq,
        ], dim=-1))
        plan_ctx = self.plan_query(torch.sigmoid(plan_logits))
        next_ctx = self.next_type_query(torch.softmax(next_type_logits[:, :, 0, :], dim=-1))
        dca_ctx = self.dca_query(torch.cat([
            torch.sigmoid(dca_spread_logits).unsqueeze(-1),
            torch.softmax(dca_unique_logits, dim=-1),
            torch.sigmoid(dca_focus_logits).unsqueeze(-1),
        ], dim=-1))
        known_ctx_q = self.known_query(known_ctx)
        remain_bins = torch.arange(MAX_SELECT_COUNT + 1, device=seq.device, dtype=seq.dtype)
        remain_bins = remain_bins / float(max(MAX_SELECT_COUNT, 1))
        turn_remaining_mean = (
            torch.softmax(turn_remaining_logits, dim=-1) * remain_bins.view(1, 1, -1)
        ).sum(dim=-1, keepdim=True)
        turn_ctx = self.turn_query(torch.cat([
            torch.sigmoid(turn_future_type_logits),
            torch.sigmoid(turn_continue_logits).unsqueeze(-1),
            turn_remaining_mean,
        ], dim=-1))
        next_type_prob = torch.softmax(turn_next_type_logits.float(), dim=-1).to(seq.dtype)
        next_card_prob = torch.softmax(turn_next_card_logits.float(), dim=-1).to(seq.dtype)
        next_card2_prob = torch.softmax(turn_next_card2_logits.float(), dim=-1).to(seq.dtype)
        next_attack_prob = torch.softmax(turn_next_attack_logits.float(), dim=-1).to(seq.dtype)
        next_context_prob = torch.softmax(turn_next_context_logits.float(), dim=-1).to(seq.dtype)
        next_type_emb = torch.matmul(next_type_prob, self.prev_type_emb.weight[1:N_ACTION_TYPES + 2])
        next_card_emb = torch.matmul(next_card_prob, self.card_emb.weight[:N_CARDS + 2])
        next_card2_emb = torch.matmul(next_card2_prob, self.card_emb.weight[:N_CARDS + 2])
        next_attack_emb = torch.matmul(next_attack_prob, self.attack_emb.weight[:N_ATTACKS + 1])
        next_context_emb = torch.matmul(next_context_prob, self.context_emb.weight[:66])
        turn_next_ctx = self.turn_next_query(torch.cat([
            next_type_emb,
            next_card_emb,
            next_card2_emb,
            next_attack_emb,
            next_context_emb,
        ], dim=-1))
        seq_type_prob = torch.softmax(turn_seq_type_logits.float(), dim=-1).to(seq.dtype)
        seq_card_prob = torch.softmax(turn_seq_card_logits.float(), dim=-1).to(seq.dtype)
        seq_attack_prob = torch.softmax(turn_seq_attack_logits.float(), dim=-1).to(seq.dtype)
        seq_context_prob = torch.softmax(turn_seq_context_logits.float(), dim=-1).to(seq.dtype)
        seq_type_emb = torch.matmul(seq_type_prob, self.prev_type_emb.weight[1:N_ACTION_TYPES + 2])
        seq_card_emb = torch.matmul(seq_card_prob, self.card_emb.weight[:N_CARDS + 2])
        seq_attack_emb = torch.matmul(seq_attack_prob, self.attack_emb.weight[:N_ATTACKS + 1])
        seq_context_emb = torch.matmul(seq_context_prob, self.context_emb.weight[:66])
        step_ids = torch.arange(TURN_PLAN_STEPS, device=seq.device)
        seq_tokens = self.turn_seq_token_fc(torch.cat([
            seq_type_emb,
            seq_card_emb,
            seq_attack_emb,
            seq_context_emb,
        ], dim=-1))
        seq_tokens = seq_tokens + self.turn_seq_step_emb(step_ids).view(1, 1, TURN_PLAN_STEPS, self.width)
        seq_step_weight = (1.0 - seq_type_prob[..., N_ACTION_TYPES]).clamp(0.0, 1.0).unsqueeze(-1)
        turn_seq_ctx = (seq_tokens * seq_step_weight).sum(dim=2) / seq_step_weight.sum(dim=2).clamp(min=1e-3)
        turn_seq_ctx = self.turn_seq_query(turn_seq_ctx)

        q_raw = self.option_query(seq)
        hist_applied = hist_ctx * self.history_condition_scale
        plan_applied = plan_ctx * self.plan_condition_scale
        next_applied = next_ctx * self.next_type_condition_scale
        dca_applied = dca_ctx * self.dca_condition_scale
        known_applied = known_ctx_q * self.known_condition_scale
        turn_applied = turn_ctx * self.turn_condition_scale
        turn_next_applied = turn_next_ctx * self.turn_next_condition_scale
        turn_seq_applied = turn_seq_ctx * self.turn_seq_condition_scale
        q_base = self.action_query_norm(
            q_raw
            + hist_applied
            + plan_applied
            + next_applied
            + dca_applied
            + known_applied
            + turn_applied
            + turn_next_applied
            + turn_seq_applied
        )
        q = q_base.unsqueeze(2)
        k = self.option_key(opt_emb)
        scores = self.action_score(torch.cat([q.expand_as(k), k, q.expand_as(k) * k], dim=-1)).squeeze(-1)
        known_q = known_ctx_q.unsqueeze(2)
        known_scores = self.known_action_score(torch.cat([
            known_q.expand_as(k),
            k,
            known_q.expand_as(k) * k,
        ], dim=-1)).squeeze(-1)
        if self.known_logit_scale != 0.0:
            scores = scores + known_scores * self.known_logit_scale
        if self.type_prior_scale != 0.0:
            opt_type = batch.opt_type.long().clamp(0, N_OPT_TYPES)
            type_prior = F.log_softmax(type_logits.float(), dim=-1).gather(-1, opt_type)
            scores = scores + type_prior.to(scores.dtype) * self.type_prior_scale
        scores = scores.masked_fill(opt_mask <= 0, NEG_INF)
        known_scores = known_scores.masked_fill(opt_mask <= 0, NEG_INF)
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
            "known_action_logits": known_scores,
            "order_logits": order_scores,
            "plan_logits": plan_logits,
            "next_type_logits": next_type_logits,
            "dca_spread_logits": dca_spread_logits,
            "dca_unique_logits": dca_unique_logits,
            "dca_focus_logits": dca_focus_logits,
            "turn_continue_logits": turn_continue_logits,
            "turn_remaining_logits": turn_remaining_logits,
            "turn_future_type_logits": turn_future_type_logits,
            "turn_next_type_logits": turn_next_type_logits,
            "turn_next_card_logits": turn_next_card_logits,
            "turn_next_card2_logits": turn_next_card2_logits,
            "turn_next_attack_logits": turn_next_attack_logits,
            "turn_next_context_logits": turn_next_context_logits,
            "turn_seq_type_logits": turn_seq_type_logits,
            "turn_seq_card_logits": turn_seq_card_logits,
            "turn_seq_attack_logits": turn_seq_attack_logits,
            "turn_seq_context_logits": turn_seq_context_logits,
            "outcome_logits": self.outcome_head(seq).squeeze(-1),
            "type_logits": type_logits,
            "count_logits": count_logits,
            "base_query_norm": q_raw.detach().float().norm(dim=-1),
            "conditioned_query_norm": q_base.detach().float().norm(dim=-1),
            "history_query_norm": hist_applied.detach().float().norm(dim=-1),
            "plan_query_norm": plan_applied.detach().float().norm(dim=-1),
            "next_type_query_norm": next_applied.detach().float().norm(dim=-1),
            "dca_query_norm": dca_applied.detach().float().norm(dim=-1),
            "known_query_norm": known_applied.detach().float().norm(dim=-1),
            "turn_query_norm": turn_applied.detach().float().norm(dim=-1),
            "turn_next_query_norm": turn_next_applied.detach().float().norm(dim=-1),
            "turn_seq_query_norm": turn_seq_applied.detach().float().norm(dim=-1),
            "prefix_summary_norm": prefix_summary.detach().float().norm(dim=-1),
            "known_opp_card_slots": batch.known_opp_mask.detach().float().sum(dim=-1),
            "type_prior_abs": (
                type_prior.detach().float().abs().mean(dim=-1)
                if self.type_prior_scale != 0.0
                else torch.zeros_like(step_mask, dtype=torch.float32)
            ),
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

    def _known_opp_context(self, batch: SequenceBatch) -> torch.Tensor:
        cards = batch.known_opp_cards.long().clamp(0, N_CARDS + 1)
        counts = batch.known_opp_counts.float().clamp(0.0, 1.0).unsqueeze(-1)
        mask = batch.known_opp_mask.float().clamp(0.0, 1.0).unsqueeze(-1)
        tok = self.known_opp_fc(torch.cat([self.card_emb(cards), counts], dim=-1))
        denom = mask.sum(dim=2).clamp(min=1.0)
        return (tok * mask).sum(dim=2) / denom

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
    known_present = (batch.known_opp_mask.float().sum(dim=-1) > 0) & (step_mask > 0)
    option_count = batch.option_mask.float().sum(dim=-1).clamp(min=1.0)
    if float(cfg.current_complexity_weight) != 0.0:
        max_opt = max(int(batch.option_mask.shape[-1]), 1)
        current_complexity = 1.0 + float(cfg.current_complexity_weight) * (
            torch.log1p(option_count) / math.log1p(float(max_opt))
        )
    else:
        current_complexity = torch.ones_like(option_count)
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
    current_step = torch.zeros_like(valid_action, dtype=torch.bool)
    if current_step.shape[1] > 0:
        current_step[:, -1] = True
    valid_weight_sum = weights[valid_action].sum().clamp(min=1.0) if bool(valid_action.any()) else weights.sum().clamp(min=1.0)
    decision_weight_sum = decision_weights[valid_action].sum().clamp(min=1.0) if bool(valid_action.any()) else decision_weights.sum().clamp(min=1.0)
    if bool(valid_action.any()):
        raw_action_loss = F.cross_entropy(
            outputs["action_logits"][valid_action].float().clamp(min=-50.0, max=50.0),
            batch.target_first.long()[valid_action],
            reduction="none",
        )
        valid_decision_weights = decision_weights[valid_action]
        valid_current = current_step[valid_action]
        valid_prefix = ~valid_current
        if bool(valid_current.any()):
            current_action_loss = (
                raw_action_loss[valid_current]
                * valid_decision_weights[valid_current]
                * current_complexity[valid_action][valid_current]
            ).sum() / (valid_decision_weights[valid_current] * current_complexity[valid_action][valid_current]).sum().clamp(min=1.0)
        else:
            current_action_loss = outputs["action_logits"].sum() * 0.0
        if bool(valid_prefix.any()) and float(cfg.prefix_action_weight) > 0.0:
            prefix_action_loss = (
                raw_action_loss[valid_prefix] * valid_decision_weights[valid_prefix]
            ).sum() / valid_decision_weights[valid_prefix].sum().clamp(min=1.0)
        else:
            prefix_action_loss = outputs["action_logits"].sum() * 0.0
        action_loss = (
            float(cfg.current_action_weight) * current_action_loss
            + float(cfg.prefix_action_weight) * prefix_action_loss
        )
    else:
        raw_action_loss = torch.zeros(0, device=outputs["action_logits"].device)
        valid_decision_weights = torch.zeros(0, device=outputs["action_logits"].device)
        current_action_loss = outputs["action_logits"].sum() * 0.0
        prefix_action_loss = outputs["action_logits"].sum() * 0.0
        action_loss = outputs["action_logits"].sum() * 0.0

    known_action_mask = valid_action & known_present
    if bool(known_action_mask.any()) and "known_action_logits" in outputs:
        known_action_raw = F.cross_entropy(
            outputs["known_action_logits"][known_action_mask].float().clamp(min=-50.0, max=50.0),
            batch.target_first.long()[known_action_mask],
            reduction="none",
        )
        known_w = decision_weights[known_action_mask]
        known_action_loss = (known_action_raw * known_w).sum() / known_w.sum().clamp(min=1.0)
    else:
        known_action_loss = outputs["action_logits"].sum() * 0.0

    order_logits = outputs.get("order_logits")
    if order_logits is not None and float(cfg.order_weight) > 0.0:
        valid_order = (
            (batch.target_order >= 0)
            & (step_mask.unsqueeze(-1) > 0)
            & multi_target.unsqueeze(-1)
        )
        if bool(valid_order.any()):
            order_loss_raw = F.cross_entropy(
                order_logits[valid_order].float().clamp(min=-50.0, max=50.0),
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

    next_type_logits = outputs.get("next_type_logits")
    next_type_loss = outputs["action_logits"].sum() * 0.0
    next_type_weight_sum = torch.tensor(0.0, device=weights.device)
    if next_type_logits is not None and float(cfg.next_type_weight) > 0.0:
        horizon = int(next_type_logits.shape[2])
        seq_len = int(step_mask.shape[1])
        losses: list[torch.Tensor] = []
        loss_weights: list[torch.Tensor] = []
        for offset in range(1, min(horizon, seq_len - 1) + 1):
            src = slice(0, seq_len - offset)
            tgt = slice(offset, seq_len)
            valid_next = (
                (step_mask[:, src] > 0)
                & (step_mask[:, tgt] > 0)
                & (batch.target_first[:, tgt] >= 0)
            )
            if not bool(valid_next.any()):
                continue
            ce = F.cross_entropy(
                next_type_logits[:, src, offset - 1, :].float().reshape(-1, next_type_logits.shape[-1]).clamp(min=-50.0, max=50.0),
                batch.target_type[:, tgt].long().reshape(-1).clamp(0, next_type_logits.shape[-1] - 1),
                reduction="none",
            ).reshape_as(step_mask[:, src])
            losses.append(ce[valid_next])
            loss_weights.append(weights[:, src][valid_next])
        if losses:
            next_raw = torch.cat(losses)
            next_w = torch.cat(loss_weights)
            next_type_weight_sum = next_w.sum()
            next_type_loss = (next_raw * next_w).sum() / next_w.sum().clamp(min=1.0)

    dca_plan_loss = outputs["action_logits"].sum() * 0.0
    dca_spread_pos_weight_value = torch.tensor(1.0, device=weights.device)
    dca_plan_mask = damage_counter & valid_action
    if bool(dca_plan_mask.any()) and float(cfg.dca_plan_weight) > 0.0:
        dca_w = decision_weights[dca_plan_mask]
        spread_target = (batch.dca_group_unique_slots.long()[dca_plan_mask] > 1).float()
        spread_pos = spread_target.sum()
        spread_neg = (1.0 - spread_target).sum()
        if bool(spread_pos > 0):
            dca_spread_pos_weight_value = (spread_neg / spread_pos.clamp(min=1.0)).clamp(min=1.0, max=8.0)
        spread_loss = F.binary_cross_entropy_with_logits(
            outputs["dca_spread_logits"][dca_plan_mask].float().clamp(min=-30.0, max=30.0),
            spread_target,
            pos_weight=dca_spread_pos_weight_value,
            reduction="none",
        )
        unique_target = batch.dca_group_unique_slots.long()[dca_plan_mask].clamp(0, MAX_SELECT_COUNT)
        unique_loss = F.cross_entropy(
            outputs["dca_unique_logits"][dca_plan_mask].float().clamp(min=-50.0, max=50.0),
            unique_target,
            reduction="none",
        )
        focus_target = batch.dca_group_focus_frac.float()[dca_plan_mask].clamp(0.0, 1.0)
        focus_loss = F.mse_loss(
            torch.sigmoid(outputs["dca_focus_logits"][dca_plan_mask].float()),
            focus_target,
            reduction="none",
        )
        dca_plan_loss = (
            (spread_loss + 0.50 * unique_loss + 0.50 * focus_loss) * dca_w
        ).sum() / dca_w.sum().clamp(min=1.0)

    turn_plan_loss = outputs["action_logits"].sum() * 0.0
    turn_continue_loss = outputs["action_logits"].sum() * 0.0
    turn_remaining_loss = outputs["action_logits"].sum() * 0.0
    turn_future_type_loss = outputs["action_logits"].sum() * 0.0
    turn_terminal_loss = outputs["action_logits"].sum() * 0.0
    if (
        float(cfg.turn_plan_weight) > 0.0
        and "turn_continue_logits" in outputs
        and "turn_remaining_logits" in outputs
        and "turn_future_type_logits" in outputs
    ):
        turn_target = batch.turn_continue.float().clamp(0.0, 1.0)
        turn_continue_raw = F.binary_cross_entropy_with_logits(
            outputs["turn_continue_logits"].float().clamp(min=-30.0, max=30.0),
            turn_target,
            reduction="none",
        )
        turn_continue_loss = (turn_continue_raw * weights).sum() / weights.sum().clamp(min=1.0)
        turn_remaining_raw = F.cross_entropy(
            outputs["turn_remaining_logits"].reshape(-1, outputs["turn_remaining_logits"].shape[-1]).float().clamp(min=-50.0, max=50.0),
            batch.turn_remaining.long().reshape(-1).clamp(0, MAX_SELECT_COUNT),
            reduction="none",
        ).reshape_as(step_mask)
        turn_remaining_loss = (turn_remaining_raw * weights).sum() / weights.sum().clamp(min=1.0)
        turn_future_raw = F.binary_cross_entropy_with_logits(
            outputs["turn_future_type_logits"].float().clamp(min=-30.0, max=30.0),
            batch.turn_future_types.float().clamp(0.0, 1.0),
            reduction="none",
        )
        turn_future_type_loss = (
            turn_future_raw * weights.unsqueeze(-1)
        ).sum() / (weights.sum().clamp(min=1.0) * max(batch.turn_future_types.shape[-1], 1))
        turn_plan_loss = turn_continue_loss + 0.25 * turn_remaining_loss + 0.50 * turn_future_type_loss

    if float(cfg.turn_terminal_weight) > 0.0:
        opt_mask_for_terminal = batch.option_mask.float() * step_mask.unsqueeze(-1)
        terminal_options = (
            ((batch.opt_type.long() == TYPE_END) | (batch.opt_type.long() == TYPE_ATTACK))
            & (opt_mask_for_terminal > 0)
        )
        nonterminal_options = (
            ((batch.opt_type.long() != TYPE_END) & (batch.opt_type.long() != TYPE_ATTACK))
            & (opt_mask_for_terminal > 0)
        )
        terminal_valid = valid_action & terminal_options.any(dim=-1) & nonterminal_options.any(dim=-1)
        if bool(terminal_valid.any()):
            probs_terminal = torch.softmax(outputs["action_logits"].float(), dim=-1)
            terminal_prob = (probs_terminal * terminal_options.float()).sum(dim=-1).clamp(1e-5, 1.0 - 1e-5)
            terminal_target = (
                (batch.target_type.long() == TYPE_END) | (batch.target_type.long() == TYPE_ATTACK)
            ).float()
            terminal_raw = F.binary_cross_entropy(
                terminal_prob[terminal_valid],
                terminal_target[terminal_valid],
                reduction="none",
            )
            cont_boost = 1.0 + 2.0 * batch.turn_continue.float().clamp(0.0, 1.0)[terminal_valid]
            terminal_w = decision_weights[terminal_valid] * cont_boost
            turn_terminal_loss = (terminal_raw * terminal_w).sum() / terminal_w.sum().clamp(min=1.0)

    turn_next_plan_loss = outputs["action_logits"].sum() * 0.0
    turn_next_type_loss = outputs["action_logits"].sum() * 0.0
    turn_next_card_loss = outputs["action_logits"].sum() * 0.0
    turn_next_card2_loss = outputs["action_logits"].sum() * 0.0
    turn_next_attack_loss = outputs["action_logits"].sum() * 0.0
    turn_next_context_loss = outputs["action_logits"].sum() * 0.0
    turn_next_rows = torch.tensor(0.0, device=weights.device)
    turn_next_card_rows = torch.tensor(0.0, device=weights.device)
    turn_next_attack_rows = torch.tensor(0.0, device=weights.device)
    if (
        float(cfg.turn_next_plan_weight) > 0.0
        and "turn_next_type_logits" in outputs
        and "turn_next_card_logits" in outputs
        and "turn_next_attack_logits" in outputs
    ):
        turn_next_exists = batch.turn_next_exists.float().clamp(0.0, 1.0)
        next_step_valid = step_mask > 0
        next_type_target = torch.where(
            turn_next_exists > 0.5,
            batch.turn_next_type.long().clamp(0, N_ACTION_TYPES),
            torch.full_like(batch.turn_next_type.long(), N_ACTION_TYPES),
        )
        next_type_raw = F.cross_entropy(
            outputs["turn_next_type_logits"].reshape(-1, outputs["turn_next_type_logits"].shape[-1]).float().clamp(min=-50.0, max=50.0),
            next_type_target.reshape(-1),
            reduction="none",
        ).reshape_as(step_mask)
        next_type_w = weights * torch.where(
            turn_next_exists > 0.5,
            torch.ones_like(weights),
            torch.full_like(weights, 0.25),
        )
        turn_next_type_loss = (
            next_type_raw[next_step_valid] * next_type_w[next_step_valid]
        ).sum() / next_type_w[next_step_valid].sum().clamp(min=1.0)
        turn_next_rows = (next_step_valid & (turn_next_exists > 0.5)).float().sum()

        next_card_valid = next_step_valid & (turn_next_exists > 0.5) & (batch.turn_next_card.long() > 0)
        if bool(next_card_valid.any()):
            card_raw = F.cross_entropy(
                outputs["turn_next_card_logits"][next_card_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_next_card.long()[next_card_valid].clamp(0, N_CARDS + 1),
                reduction="none",
            )
            card_w = weights[next_card_valid]
            turn_next_card_loss = (card_raw * card_w).sum() / card_w.sum().clamp(min=1.0)
            turn_next_card_rows = next_card_valid.float().sum()

        next_card2_valid = next_step_valid & (turn_next_exists > 0.5) & (batch.turn_next_card2.long() > 0)
        if bool(next_card2_valid.any()):
            card2_raw = F.cross_entropy(
                outputs["turn_next_card2_logits"][next_card2_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_next_card2.long()[next_card2_valid].clamp(0, N_CARDS + 1),
                reduction="none",
            )
            card2_w = weights[next_card2_valid]
            turn_next_card2_loss = (card2_raw * card2_w).sum() / card2_w.sum().clamp(min=1.0)

        next_attack_valid = next_step_valid & (turn_next_exists > 0.5) & (batch.turn_next_attack.long() > 0)
        if bool(next_attack_valid.any()):
            attack_raw = F.cross_entropy(
                outputs["turn_next_attack_logits"][next_attack_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_next_attack.long()[next_attack_valid].clamp(0, N_ATTACKS),
                reduction="none",
            )
            attack_w = weights[next_attack_valid]
            turn_next_attack_loss = (attack_raw * attack_w).sum() / attack_w.sum().clamp(min=1.0)
            turn_next_attack_rows = next_attack_valid.float().sum()

        next_context_valid = next_step_valid & (turn_next_exists > 0.5) & (batch.turn_next_context.long() > 0)
        if bool(next_context_valid.any()):
            context_raw = F.cross_entropy(
                outputs["turn_next_context_logits"][next_context_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_next_context.long()[next_context_valid].clamp(0, 65),
                reduction="none",
            )
            context_w = weights[next_context_valid]
            turn_next_context_loss = (context_raw * context_w).sum() / context_w.sum().clamp(min=1.0)

        turn_next_plan_loss = (
            float(cfg.turn_next_type_weight) * turn_next_type_loss
            + float(cfg.turn_next_card_weight) * (turn_next_card_loss + 0.50 * turn_next_card2_loss)
            + float(cfg.turn_next_attack_weight) * turn_next_attack_loss
            + float(cfg.turn_next_context_weight) * turn_next_context_loss
        )

    turn_seq_plan_loss = outputs["action_logits"].sum() * 0.0
    turn_seq_type_loss = outputs["action_logits"].sum() * 0.0
    turn_seq_card_loss = outputs["action_logits"].sum() * 0.0
    turn_seq_attack_loss = outputs["action_logits"].sum() * 0.0
    turn_seq_context_loss = outputs["action_logits"].sum() * 0.0
    turn_seq_slots = torch.tensor(0.0, device=weights.device)
    if (
        float(cfg.turn_seq_plan_weight) > 0.0
        and "turn_seq_type_logits" in outputs
        and "turn_seq_card_logits" in outputs
    ):
        seq_mask = batch.turn_plan_mask.float().clamp(0.0, 1.0)
        seq_step_valid = (step_mask > 0).unsqueeze(-1).expand_as(seq_mask)
        seq_type_target = torch.where(
            seq_mask > 0.5,
            batch.turn_plan_types.long().clamp(0, N_ACTION_TYPES),
            torch.full_like(batch.turn_plan_types.long(), N_ACTION_TYPES),
        )
        seq_type_raw = F.cross_entropy(
            outputs["turn_seq_type_logits"].reshape(-1, outputs["turn_seq_type_logits"].shape[-1]).float().clamp(min=-50.0, max=50.0),
            seq_type_target.reshape(-1),
            reduction="none",
        ).reshape_as(seq_mask)
        seq_w = weights.unsqueeze(-1) * torch.where(
            seq_mask > 0.5,
            torch.ones_like(seq_mask),
            torch.full_like(seq_mask, 0.25),
        )
        turn_seq_type_loss = (
            seq_type_raw[seq_step_valid] * seq_w[seq_step_valid]
        ).sum() / seq_w[seq_step_valid].sum().clamp(min=1.0)
        turn_seq_slots = (seq_step_valid & (seq_mask > 0.5)).float().sum()

        seq_card_valid = seq_step_valid & (seq_mask > 0.5) & (batch.turn_plan_cards.long() > 0)
        if bool(seq_card_valid.any()):
            card_raw = F.cross_entropy(
                outputs["turn_seq_card_logits"][seq_card_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_plan_cards.long()[seq_card_valid].clamp(0, N_CARDS + 1),
                reduction="none",
            )
            card_w = weights.unsqueeze(-1).expand_as(seq_mask)[seq_card_valid]
            turn_seq_card_loss = (card_raw * card_w).sum() / card_w.sum().clamp(min=1.0)

        seq_attack_valid = seq_step_valid & (seq_mask > 0.5) & (batch.turn_plan_attacks.long() > 0)
        if bool(seq_attack_valid.any()):
            attack_raw = F.cross_entropy(
                outputs["turn_seq_attack_logits"][seq_attack_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_plan_attacks.long()[seq_attack_valid].clamp(0, N_ATTACKS),
                reduction="none",
            )
            attack_w = weights.unsqueeze(-1).expand_as(seq_mask)[seq_attack_valid]
            turn_seq_attack_loss = (attack_raw * attack_w).sum() / attack_w.sum().clamp(min=1.0)

        seq_context_valid = seq_step_valid & (seq_mask > 0.5) & (batch.turn_plan_contexts.long() > 0)
        if bool(seq_context_valid.any()):
            context_raw = F.cross_entropy(
                outputs["turn_seq_context_logits"][seq_context_valid].float().clamp(min=-50.0, max=50.0),
                batch.turn_plan_contexts.long()[seq_context_valid].clamp(0, 65),
                reduction="none",
            )
            context_w = weights.unsqueeze(-1).expand_as(seq_mask)[seq_context_valid]
            turn_seq_context_loss = (context_raw * context_w).sum() / context_w.sum().clamp(min=1.0)

        turn_seq_plan_loss = (
            float(cfg.turn_seq_type_weight) * turn_seq_type_loss
            + float(cfg.turn_seq_card_weight) * turn_seq_card_loss
            + float(cfg.turn_seq_attack_weight) * turn_seq_attack_loss
            + float(cfg.turn_seq_context_weight) * turn_seq_context_loss
        )

    opportunity_type_loss = outputs["action_logits"].sum() * 0.0
    opportunity_type_rows = torch.tensor(0.0, device=weights.device)
    opportunity_key_share = torch.tensor(0.0, device=weights.device)
    if float(cfg.opportunity_type_weight) > 0.0:
        opt_valid = batch.option_mask.float() > 0
        action_probs = torch.softmax(
            outputs["action_logits"].float().masked_fill(~opt_valid, NEG_INF),
            dim=-1,
        )
        opt_type = batch.opt_type.long()
        type_mass = torch.stack([
            (action_probs * ((opt_type == typ) & opt_valid).float()).sum(dim=-1)
            for typ in _OPPORTUNITY_TYPES
        ], dim=-1).clamp(1e-5, 1.0)
        target_slot = torch.full_like(batch.target_type.long(), -1)
        for j, typ in enumerate(_OPPORTUNITY_TYPES):
            target_slot = torch.where(batch.target_type.long() == typ, torch.full_like(target_slot, j), target_slot)
        legal_type_count = torch.stack([
            ((opt_type == typ) & opt_valid).any(dim=-1)
            for typ in _OPPORTUNITY_TYPES
        ], dim=-1).sum(dim=-1)
        opportunity_valid = (
            valid_action
            & current_step
            & (target_slot >= 0)
            & (legal_type_count > 1)
            & (option_count > 1)
        )
        if bool(opportunity_valid.any()):
            target_mass = type_mass.gather(-1, target_slot.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            opportunity_raw = -torch.log(target_mass)
            key_target = torch.zeros_like(opportunity_raw, dtype=torch.bool)
            for typ in _KEY_OPPORTUNITY_TYPES:
                key_target |= batch.target_type.long() == typ
            key_boost = torch.where(
                key_target,
                torch.full_like(decision_weights, 2.0),
                torch.ones_like(decision_weights),
            )
            opportunity_w = decision_weights * current_complexity * key_boost
            opportunity_type_loss = (
                opportunity_raw[opportunity_valid] * opportunity_w[opportunity_valid]
            ).sum() / opportunity_w[opportunity_valid].sum().clamp(min=1.0)
            opportunity_type_rows = opportunity_valid.float().sum()
            opportunity_key_share = (
                key_target[opportunity_valid].float().sum() / opportunity_type_rows.clamp(min=1.0)
            )

    opportunity_margin_loss = outputs["action_logits"].sum() * 0.0
    opportunity_margin_rows = torch.tensor(0.0, device=weights.device)
    opportunity_margin_violation = torch.tensor(0.0, device=weights.device)
    if float(cfg.opportunity_margin_weight) > 0.0:
        opt_valid = batch.option_mask.float() > 0
        action_logits = outputs["action_logits"].float().masked_fill(~opt_valid, NEG_INF)
        target_idx = batch.target_first.long().clamp(min=0, max=action_logits.shape[-1] - 1)
        target_logit = action_logits.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
        target_type = batch.target_type.long()
        key_target = torch.zeros_like(valid_action, dtype=torch.bool)
        for typ in _KEY_OPPORTUNITY_TYPES:
            key_target |= target_type == typ
        other_type_mask = opt_valid & (batch.opt_type.long() != target_type.unsqueeze(-1))
        has_other_type = other_type_mask.any(dim=-1)
        best_other = action_logits.masked_fill(~other_type_mask, NEG_INF).max(dim=-1).values
        margin_valid = (
            valid_action
            & current_step
            & key_target
            & has_other_type
            & (option_count > 1)
        )
        if bool(margin_valid.any()):
            margin_raw = F.softplus(best_other - target_logit + float(cfg.opportunity_margin))
            key_boost = torch.full_like(decision_weights, 2.0)
            margin_w = decision_weights * current_complexity * key_boost
            opportunity_margin_loss = (
                margin_raw[margin_valid] * margin_w[margin_valid]
            ).sum() / margin_w[margin_valid].sum().clamp(min=1.0)
            opportunity_margin_rows = margin_valid.float().sum()
            opportunity_margin_violation = (
                (target_logit[margin_valid] < best_other[margin_valid] + float(cfg.opportunity_margin))
                .float()
                .mean()
            )

    current_rank_margin_loss = outputs["action_logits"].sum() * 0.0
    current_rank_margin_rows = torch.tensor(0.0, device=weights.device)
    current_rank_margin_violation = torch.tensor(0.0, device=weights.device)
    if float(cfg.current_rank_margin_weight) > 0.0:
        opt_valid = batch.option_mask.float() > 0
        action_logits = outputs["action_logits"].float().masked_fill(~opt_valid, NEG_INF)
        target_idx = batch.target_first.long().clamp(min=0, max=action_logits.shape[-1] - 1)
        target_logit = action_logits.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
        non_target_mask = opt_valid.clone()
        non_target_mask.scatter_(-1, target_idx.unsqueeze(-1), False)
        has_non_target = non_target_mask.any(dim=-1)
        best_non_target = action_logits.masked_fill(~non_target_mask, NEG_INF).max(dim=-1).values
        rank_valid = (
            valid_action
            & current_step
            & has_non_target
            & (option_count >= float(max(2, int(cfg.current_rank_margin_min_options))))
        )
        if bool(rank_valid.any()):
            rank_raw = F.softplus(best_non_target - target_logit + float(cfg.current_rank_margin))
            same_type_count = (
                ((batch.opt_type.long() == batch.target_type.long().unsqueeze(-1)) & opt_valid)
                .sum(dim=-1)
                .float()
            )
            same_type_boost = torch.where(
                same_type_count > 1,
                torch.full_like(decision_weights, 1.5),
                torch.ones_like(decision_weights),
            )
            rank_w = decision_weights * current_complexity * same_type_boost
            current_rank_margin_loss = (
                rank_raw[rank_valid] * rank_w[rank_valid]
            ).sum() / rank_w[rank_valid].sum().clamp(min=1.0)
            current_rank_margin_rows = rank_valid.float().sum()
            current_rank_margin_violation = (
                (target_logit[rank_valid] < best_non_target[rank_valid] + float(cfg.current_rank_margin))
                .float()
                .mean()
            )

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
        + cfg.order_weight * cfg.action_weight * order_loss
        + cfg.multi_weight * multi_loss
        + cfg.count_weight * count_loss
        + cfg.plan_weight * plan_loss
        + cfg.next_type_weight * next_type_loss
        + cfg.dca_plan_weight * dca_plan_loss
        + cfg.known_action_weight * known_action_loss
        + cfg.turn_plan_weight * turn_plan_loss
        + cfg.turn_terminal_weight * turn_terminal_loss
        + cfg.turn_next_plan_weight * turn_next_plan_loss
        + cfg.turn_seq_plan_weight * turn_seq_plan_loss
        + cfg.opportunity_type_weight * opportunity_type_loss
        + cfg.opportunity_margin_weight * opportunity_margin_loss
        + cfg.current_rank_margin_weight * current_rank_margin_loss
        + cfg.outcome_weight * outcome_loss
        + cfg.type_weight * type_loss
    )

    valid_dca = damage_counter[valid_action] if bool(valid_action.any()) else torch.zeros(0, dtype=torch.bool, device=weights.device)
    valid_multi = multi_target[valid_action] if bool(valid_action.any()) else torch.zeros(0, dtype=torch.bool, device=weights.device)
    valid_current = current_step[valid_action] if bool(valid_action.any()) else torch.zeros(0, dtype=torch.bool, device=weights.device)

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
    raw_current_weight = (decision_weights[current_step & valid_action].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach() if bool(valid_action.any()) else torch.tensor(0.0, device=weights.device)
    action_objective_mass = max(float(cfg.current_action_weight), 0.0) + max(float(cfg.prefix_action_weight), 0.0)
    current_objective_share = max(float(cfg.current_action_weight), 0.0) / max(action_objective_mass, 1e-9)

    def norm_ratio(name: str) -> float:
        value = outputs.get(name)
        base = outputs.get("base_query_norm")
        if value is None or base is None:
            return 0.0
        mask = step_mask > 0
        if not bool(mask.any()):
            return 0.0
        ratio = value.float()[mask].mean() / base.float()[mask].mean().clamp(min=1e-6)
        return float(ratio.detach().cpu())

    def masked_mean(name: str) -> float:
        value = outputs.get(name)
        if value is None:
            return 0.0
        mask = step_mask > 0
        if not bool(mask.any()):
            return 0.0
        return float(value.float()[mask].mean().detach().cpu())

    parts = {
        "loss": float(loss.detach().cpu()),
        "action": float(action_loss.detach().cpu()),
        "action_current_head": float(current_action_loss.detach().cpu()),
        "action_prefix_head": float(prefix_action_loss.detach().cpu()),
        "order": float(order_loss.detach().cpu()),
        "multi": float(multi_loss.detach().cpu()),
        "count": float(count_loss.detach().cpu()),
        "plan": float(plan_loss.detach().cpu()),
        "next_type": float(next_type_loss.detach().cpu()),
        "dca_plan": float(dca_plan_loss.detach().cpu()),
        "known_action": float(known_action_loss.detach().cpu()),
        "turn_plan": float(turn_plan_loss.detach().cpu()),
        "turn_continue": float(turn_continue_loss.detach().cpu()),
        "turn_remaining": float(turn_remaining_loss.detach().cpu()),
        "turn_future_type": float(turn_future_type_loss.detach().cpu()),
        "turn_terminal": float(turn_terminal_loss.detach().cpu()),
        "turn_next_plan": float(turn_next_plan_loss.detach().cpu()),
        "turn_next_type": float(turn_next_type_loss.detach().cpu()),
        "turn_next_card": float(turn_next_card_loss.detach().cpu()),
        "turn_next_card2": float(turn_next_card2_loss.detach().cpu()),
        "turn_next_attack": float(turn_next_attack_loss.detach().cpu()),
        "turn_next_context": float(turn_next_context_loss.detach().cpu()),
        "turn_next_rows": float(turn_next_rows.detach().cpu()),
        "turn_next_card_rows": float(turn_next_card_rows.detach().cpu()),
        "turn_next_attack_rows": float(turn_next_attack_rows.detach().cpu()),
        "turn_seq_plan": float(turn_seq_plan_loss.detach().cpu()),
        "turn_seq_type": float(turn_seq_type_loss.detach().cpu()),
        "turn_seq_card": float(turn_seq_card_loss.detach().cpu()),
        "turn_seq_attack": float(turn_seq_attack_loss.detach().cpu()),
        "turn_seq_context": float(turn_seq_context_loss.detach().cpu()),
        "turn_seq_slots": float(turn_seq_slots.detach().cpu()),
        "opportunity_type": float(opportunity_type_loss.detach().cpu()),
        "opportunity_type_rows": float(opportunity_type_rows.detach().cpu()),
        "opportunity_key_share": float(opportunity_key_share.detach().cpu()),
        "opportunity_margin": float(opportunity_margin_loss.detach().cpu()),
        "opportunity_margin_rows": float(opportunity_margin_rows.detach().cpu()),
        "opportunity_margin_violation": float(opportunity_margin_violation.detach().cpu()),
        "current_rank_margin": float(current_rank_margin_loss.detach().cpu()),
        "current_rank_margin_rows": float(current_rank_margin_rows.detach().cpu()),
        "current_rank_margin_violation": float(current_rank_margin_violation.detach().cpu()),
        "outcome": float(outcome_loss.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "multi_target_rate": float((multi_target.float().sum() / step_mask.sum().clamp(min=1.0)).detach().cpu()),
        "damage_counter_rate": float((damage_counter.float().sum() / step_mask.sum().clamp(min=1.0)).detach().cpu()),
        "known_present_rate": float((known_present.float().sum() / step_mask.sum().clamp(min=1.0)).detach().cpu()),
        "turn_continue_rate": float((batch.turn_continue.float().clamp(0.0, 1.0) * step_mask).sum().detach().cpu() / step_mask.sum().clamp(min=1.0).detach().cpu()),
        "turn_remaining_mean": float((batch.turn_remaining.float().clamp(0.0, float(MAX_SELECT_COUNT)) * step_mask).sum().detach().cpu() / step_mask.sum().clamp(min=1.0).detach().cpu()),
        "current_complexity_mean": float(current_complexity[valid_action].mean().detach().cpu()) if bool(valid_action.any()) else 0.0,
        "current_complexity_max": float(current_complexity[valid_action].max().detach().cpu()) if bool(valid_action.any()) else 0.0,
        "weight_boost": float((decision_weight_sum / valid_weight_sum).detach().cpu()),
        "dca_weight_share": float((decision_weights[damage_counter].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach().cpu()) if bool(valid_action.any()) else 0.0,
        "multi_weight_share": float((decision_weights[multi_target].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach().cpu()) if bool(valid_action.any()) else 0.0,
        "next_type_rows": float(next_type_weight_sum.detach().cpu()),
        "dca_plan_weight_share": float((decision_weights[dca_plan_mask].sum() / decision_weights[valid_action].sum().clamp(min=1.0)).detach().cpu()) if bool(valid_action.any()) else 0.0,
        "dca_spread_pos_weight": float(dca_spread_pos_weight_value.detach().cpu()),
        "current_row_share": float(raw_current_weight.cpu()),
        "current_weight_share": float(current_objective_share),
        "current_action": weighted_valid_loss(valid_current),
        "prefix_action": weighted_valid_loss(~valid_current) if raw_action_loss.numel() else 0.0,
        "dca_action": weighted_valid_loss(valid_dca),
        "non_dca_action": weighted_valid_loss(~valid_dca) if raw_action_loss.numel() else 0.0,
        "multi2_action": weighted_valid_loss(valid_multi),
        "dca_focus_mean": float(((dca_focus * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_spread_rate": float((dca_spread.float().sum() / dca_rows).detach().cpu()),
        "dca_unique_mean": float(((dca_unique * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_pos_mean": float(((dca_pos.clamp(min=0) * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_prior_unique_mean": float(((dca_prior_unique * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "dca_prior_same_mean": float(((dca_prior_same * damage_counter.float()).sum() / dca_rows).detach().cpu()),
        "hist_query_ratio": norm_ratio("history_query_norm"),
        "plan_query_ratio": norm_ratio("plan_query_norm"),
        "next_query_ratio": norm_ratio("next_type_query_norm"),
        "dca_query_ratio": norm_ratio("dca_query_norm"),
        "known_query_ratio": norm_ratio("known_query_norm"),
        "turn_query_ratio": norm_ratio("turn_query_norm"),
        "turn_next_query_ratio": norm_ratio("turn_next_query_norm"),
        "turn_seq_query_ratio": norm_ratio("turn_seq_query_norm"),
        "prefix_summary_ratio": norm_ratio("prefix_summary_norm"),
        "conditioned_query_ratio": norm_ratio("conditioned_query_norm"),
        "known_opp_slots_mean": masked_mean("known_opp_card_slots"),
        "type_prior_abs": masked_mean("type_prior_abs"),
    }
    return loss, parts


@torch.no_grad()
def sequence_accuracy(outputs: dict[str, torch.Tensor], batch: SequenceBatch) -> dict[str, float]:
    valid = (batch.target_first >= 0) & (batch.step_mask > 0)
    step_mask = batch.step_mask.float()
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
            "cur_n": 0.0,
            "cur_top1": 0.0,
            "cur_type_acc": 0.0,
            "cur_set_f1": 0.0,
            "cur_entropy": 0.0,
            "cur_margin": 0.0,
            "cur_target_margin_best": 0.0,
            "cur_rank_violation_025": 0.0,
            "cur_ambig_target_margin_best": 0.0,
            "cur_forced_rate": 0.0,
            "cur_nonforced_top1": 0.0,
            "cur_bigopt_rate": 0.0,
            "cur_bigopt_top1": 0.0,
            "cur_ambig_type_rate": 0.0,
            "cur_ambig_type_top1": 0.0,
            "cur_target_end_rate": 0.0,
            "cur_pred_end_rate": 0.0,
            "cur_pred_end_when_nonend_legal": 0.0,
            "cur_pred_end_when_target_nonend": 0.0,
            "cur_play_legal_rate": 0.0,
            "cur_play_target_if_legal": 0.0,
            "cur_play_pred_if_legal": 0.0,
            "cur_play_miss_if_target": 0.0,
            "cur_attach_legal_rate": 0.0,
            "cur_attach_target_if_legal": 0.0,
            "cur_attach_pred_if_legal": 0.0,
            "cur_attach_miss_if_target": 0.0,
            "cur_attach_target_mass": 0.0,
            "cur_attach_target_margin_other": 0.0,
            "cur_evolve_legal_rate": 0.0,
            "cur_evolve_target_if_legal": 0.0,
            "cur_evolve_pred_if_legal": 0.0,
            "cur_evolve_miss_if_target": 0.0,
            "cur_evolve_target_mass": 0.0,
            "cur_evolve_target_margin_other": 0.0,
            "cur_ability_legal_rate": 0.0,
            "cur_ability_target_if_legal": 0.0,
            "cur_ability_pred_if_legal": 0.0,
            "cur_ability_miss_if_target": 0.0,
            "cur_ability_target_mass": 0.0,
            "cur_ability_target_margin_other": 0.0,
            "cur_attack_legal_rate": 0.0,
            "cur_attack_target_if_legal": 0.0,
            "cur_attack_pred_if_legal": 0.0,
            "cur_attack_miss_if_target": 0.0,
            "cur_attack_target_mass": 0.0,
            "cur_attack_target_margin_other": 0.0,
            "cur_dca_n": 0.0,
            "cur_dca_top1": 0.0,
            "seq_len_mean": 0.0,
            "seq_full_rate": 0.0,
            "history_present_rate": 0.0,
            "ledger_progress": 0.0,
            "prev_nonzero_rate": 0.0,
            "known_opp_rate": 0.0,
            "known_opp_slots_mean": 0.0,
            "known_opp_count_mean": 0.0,
            "known_action_n": 0.0,
            "known_action_top1": 0.0,
            "turn_continue_acc": 0.0,
            "turn_continue_f1": 0.0,
            "turn_continue_target_rate": 0.0,
            "turn_continue_pred_rate": 0.0,
            "turn_remaining_mae": 0.0,
            "turn_future_type_f1": 0.0,
            "turn_future_type_target_rate": 0.0,
            "turn_future_type_pred_rate": 0.0,
            "turn_next_exists_rate": 0.0,
            "turn_next_type_acc": 0.0,
            "turn_next_type_pos_acc": 0.0,
            "turn_next_none_acc": 0.0,
            "turn_next_card_acc": 0.0,
            "turn_next_card_n": 0.0,
            "turn_next_attack_acc": 0.0,
            "turn_next_attack_n": 0.0,
            "turn_next_context_acc": 0.0,
            "turn_next_context_n": 0.0,
            "cur_turn_next_exists_rate": 0.0,
            "cur_turn_next_type_acc": 0.0,
            "cur_turn_next_type_pos_acc": 0.0,
            "cur_turn_next_card_acc": 0.0,
            "cur_turn_next_attack_acc": 0.0,
            "turn_seq_type_acc": 0.0,
            "turn_seq_type_pos_acc": 0.0,
            "turn_seq_none_acc": 0.0,
            "turn_seq_step1_acc": 0.0,
            "turn_seq_step2_acc": 0.0,
            "turn_seq_step3_acc": 0.0,
            "turn_seq_step4_acc": 0.0,
            "turn_seq_card_acc": 0.0,
            "turn_seq_card_n": 0.0,
            "turn_seq_attack_acc": 0.0,
            "turn_seq_attack_n": 0.0,
            "turn_seq_context_acc": 0.0,
            "turn_seq_context_n": 0.0,
            "cur_turn_continue_target_rate": 0.0,
            "cur_turn_continue_pred_rate": 0.0,
            "cur_turn_continue_f1": 0.0,
            "cur_turn_continue_miss": 0.0,
            "cur_terminal_when_continue": 0.0,
            "cur_nonterminal_when_stop": 0.0,
            "cur_terminal_prob_when_continue": 0.0,
            "cur_terminal_prob_when_stop": 0.0,
            "cur_nonterminal_prob_when_stop": 0.0,
            "plan_mae": 0.0,
            "plan_f1": 0.0,
            "plan_pos_rate": 0.0,
            "plan_pred_pos_rate": 0.0,
            "next_type_n": 0.0,
            "next_type_acc": 0.0,
            "next1_acc": 0.0,
            "next2_acc": 0.0,
            "next3_acc": 0.0,
            "next4_acc": 0.0,
            "dca_plan_n": 0.0,
            "dca_plan_spread_acc": 0.0,
            "dca_plan_spread_f1": 0.0,
            "dca_plan_unique_acc": 0.0,
            "dca_plan_unique_mae": 0.0,
            "dca_plan_focus_mae": 0.0,
            "dca_plan_spread_rate": 0.0,
            "dca_plan_pred_spread_rate": 0.0,
            "action_type_acc": 0.0,
            "option_n": 0.0,
            "bigopt_rate": 0.0,
            "bigopt_top1": 0.0,
            "ambig_type_rate": 0.0,
            "ambig_type_top1": 0.0,
            "action_entropy": 0.0,
            "action_margin": 0.0,
            "outcome_pos_rate": 0.0,
            "outcome_pred_pos_rate": 0.0,
            "outcome_brier": 0.0,
        }
    pred = outputs["action_logits"].argmax(dim=-1)
    top1 = (pred[valid] == batch.target_first[valid]).float().mean().item()
    known_action_n = 0.0
    known_action_top1 = 0.0
    if "known_action_logits" in outputs:
        known_mask_for_action = valid & (batch.known_opp_mask.float().sum(dim=-1) > 0)
        known_action_n = float(known_mask_for_action.sum().item())
        if bool(known_mask_for_action.any()):
            known_pred = outputs["known_action_logits"].argmax(dim=-1)
            known_action_top1 = (known_pred[known_mask_for_action] == batch.target_first[known_mask_for_action]).float().mean().item()
    type_pred = outputs["type_logits"].argmax(dim=-1)
    type_acc = (type_pred[valid] == batch.target_type[valid]).float().mean().item()
    count_target = batch.target_multi.float().sum(dim=-1).long().clamp(0, MAX_SELECT_COUNT)
    count_pred = outputs["count_logits"].argmax(dim=-1)
    count_acc = (count_pred[valid] == count_target[valid]).float().mean().item()
    count_mae = (count_pred[valid].float() - count_target[valid].float()).abs().mean().item()
    current_step = torch.zeros_like(valid, dtype=torch.bool)
    if current_step.shape[1] > 0:
        current_step[:, -1] = True
    current_valid = valid & current_step
    multi2 = valid & (count_target > 1)
    dca = valid & (batch.target_context.long() == DAMAGE_COUNTER_ANY_CONTEXT)
    dca_spread = dca & (batch.dca_group_unique_slots.long() > 1)
    dca_focus = dca & (batch.dca_group_unique_slots.long() <= 1)
    dca_first = dca & (batch.dca_pos.long() == 0)
    dca_late = dca & (batch.dca_pos.long() > 0)
    dca_prior_same = dca & (batch.dca_prior_same_slot.long() > 0)
    valid_n = float(valid.sum().item())
    current_n = float(current_valid.sum().item())
    multi2_n = float(multi2.sum().item())
    dca_n = float(dca.sum().item())
    capable2_n = float((valid & (batch.max_count.long() > 1)).sum().item())

    opt_mask = batch.option_mask.float()
    logits = outputs["action_logits"].masked_fill(opt_mask <= 0, NEG_INF)
    opt_n = opt_mask.sum(dim=-1)
    pred_safe = pred.clamp(min=0, max=max(logits.shape[-1] - 1, 0))
    pred_type = batch.opt_type.long().gather(-1, pred_safe.unsqueeze(-1)).squeeze(-1)
    action_type_acc = (pred_type[valid] == batch.target_type.long()[valid]).float().mean().item()
    same_type_count = ((batch.opt_type.long() == batch.target_type.long().unsqueeze(-1)) & (opt_mask > 0)).sum(dim=-1)
    ambig_type = valid & (same_type_count > 1)
    bigopt = valid & (opt_n >= 8)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = (-(probs * log_probs).sum(dim=-1))[valid].mean().item()
    top2 = logits.topk(k=min(2, logits.shape[-1]), dim=-1).values
    nonforced_valid = valid & (opt_n > 1)
    if top2.shape[-1] >= 2 and bool(nonforced_valid.any()):
        margin = (top2[..., 0] - top2[..., 1])[nonforced_valid].mean().item()
    else:
        margin = 0.0
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

    cur_top1 = cur_type_acc = cur_set_f1 = cur_dca_top1 = 0.0
    cur_entropy = cur_margin = 0.0
    cur_target_margin_best = 0.0
    cur_rank_violation_025 = 0.0
    cur_ambig_target_margin_best = 0.0
    cur_forced_rate = cur_nonforced_top1 = 0.0
    cur_bigopt_rate = cur_bigopt_top1 = 0.0
    cur_ambig_type_rate = cur_ambig_type_top1 = 0.0
    cur_target_end_rate = cur_pred_end_rate = 0.0
    cur_pred_end_when_nonend_legal = cur_pred_end_when_target_nonend = 0.0
    current_type_opportunity: dict[str, float] = {}
    cur_nonforced = current_valid & (opt_n > 1)
    cur_bigopt = current_valid & (opt_n >= 8)
    cur_ambig_type = current_valid & (same_type_count > 1)
    cur_nonend_legal = current_valid & (((batch.opt_type.long() != TYPE_END) & (opt_mask > 0)).any(dim=-1))
    if bool(current_valid.any()):
        cur_top1 = (pred[current_valid] == batch.target_first[current_valid]).float().mean().item()
        cur_type_acc = (pred_type[current_valid] == batch.target_type.long()[current_valid]).float().mean().item()
        cur_set_f1 = (2.0 * tp[current_valid] / (pp[current_valid] + gp[current_valid]).clamp(min=1.0)).mean().item() if max_k > 0 else 0.0
        cur_entropy = (-(probs * log_probs).sum(dim=-1))[current_valid].mean().item()
        if top2.shape[-1] >= 2 and bool(cur_nonforced.any()):
            cur_margin = (top2[..., 0] - top2[..., 1])[cur_nonforced].mean().item()
            target_safe = batch.target_first.long().clamp(min=0, max=logits.shape[-1] - 1)
            target_logit = logits.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)
            non_target_mask = opt_mask > 0
            non_target_mask = non_target_mask.clone()
            non_target_mask.scatter_(-1, target_safe.unsqueeze(-1), False)
            target_margin_rows = current_valid & non_target_mask.any(dim=-1)
            if bool(target_margin_rows.any()):
                best_non_target = logits.masked_fill(~non_target_mask, NEG_INF).max(dim=-1).values
                target_margin = target_logit - best_non_target
                cur_target_margin_best = target_margin[target_margin_rows].mean().item()
                cur_rank_violation_025 = (target_margin[target_margin_rows] < 0.25).float().mean().item()
                ambig_margin_rows = cur_ambig_type & non_target_mask.any(dim=-1)
                if bool(ambig_margin_rows.any()):
                    cur_ambig_target_margin_best = target_margin[ambig_margin_rows].mean().item()
        cur_forced_rate = float((current_valid & (opt_n <= 1)).float().sum().item() / max(current_n, 1.0))
        cur_bigopt_rate = float(cur_bigopt.float().sum().item() / max(current_n, 1.0))
        cur_ambig_type_rate = float(cur_ambig_type.float().sum().item() / max(current_n, 1.0))
        cur_target_end_rate = float((batch.target_type.long()[current_valid] == TYPE_END).float().mean().item())
        cur_pred_end_rate = float((pred_type[current_valid] == TYPE_END).float().mean().item())
        if bool(cur_nonforced.any()):
            cur_nonforced_top1 = (pred[cur_nonforced] == batch.target_first[cur_nonforced]).float().mean().item()
        if bool(cur_bigopt.any()):
            cur_bigopt_top1 = (pred[cur_bigopt] == batch.target_first[cur_bigopt]).float().mean().item()
        if bool(cur_ambig_type.any()):
            cur_ambig_type_top1 = (pred[cur_ambig_type] == batch.target_first[cur_ambig_type]).float().mean().item()
        if bool(cur_nonend_legal.any()):
            cur_pred_end_when_nonend_legal = (pred_type[cur_nonend_legal] == TYPE_END).float().mean().item()
        cur_target_nonend = current_valid & (batch.target_type.long() != TYPE_END)
        if bool(cur_target_nonend.any()):
            cur_pred_end_when_target_nonend = (pred_type[cur_target_nonend] == TYPE_END).float().mean().item()
        cur_dca = current_valid & dca
        if bool(cur_dca.any()):
            cur_dca_top1 = (pred[cur_dca] == batch.target_first[cur_dca]).float().mean().item()

    for typ, name in (
        (TYPE_PLAY, "play"),
        (TYPE_ATTACH, "attach"),
        (TYPE_EVOLVE, "evolve"),
        (TYPE_ABILITY, "ability"),
        (TYPE_ATTACK, "attack"),
    ):
        legal = current_valid & (((batch.opt_type.long() == typ) & (opt_mask > 0)).any(dim=-1))
        target = current_valid & (batch.target_type.long() == typ)
        current_type_opportunity[f"cur_{name}_legal_rate"] = float(legal.float().sum().item() / max(current_n, 1.0))
        if bool(legal.any()):
            current_type_opportunity[f"cur_{name}_target_if_legal"] = float((batch.target_type.long()[legal] == typ).float().mean().item())
            current_type_opportunity[f"cur_{name}_pred_if_legal"] = float((pred_type[legal] == typ).float().mean().item())
        else:
            current_type_opportunity[f"cur_{name}_target_if_legal"] = 0.0
            current_type_opportunity[f"cur_{name}_pred_if_legal"] = 0.0
        if bool(target.any()):
            current_type_opportunity[f"cur_{name}_miss_if_target"] = float((pred_type[target] != typ).float().mean().item())
            target_type_mass = (probs * ((batch.opt_type.long() == typ) & (opt_mask > 0)).float()).sum(dim=-1)
            current_type_opportunity[f"cur_{name}_target_mass"] = float(target_type_mass[target].mean().item())
            target_safe = batch.target_first.long().clamp(min=0, max=logits.shape[-1] - 1)
            target_logit = logits.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)
            other_type = (batch.opt_type.long() != typ) & (opt_mask > 0)
            target_with_other = target & other_type.any(dim=-1)
            if bool(target_with_other.any()):
                best_other = logits.masked_fill(~other_type, NEG_INF).max(dim=-1).values
                current_type_opportunity[f"cur_{name}_target_margin_other"] = float(
                    (target_logit[target_with_other] - best_other[target_with_other]).mean().item()
                )
            else:
                current_type_opportunity[f"cur_{name}_target_margin_other"] = 0.0
        else:
            current_type_opportunity[f"cur_{name}_miss_if_target"] = 0.0
            current_type_opportunity[f"cur_{name}_target_mass"] = 0.0
            current_type_opportunity[f"cur_{name}_target_margin_other"] = 0.0

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
    outcome_prob = torch.sigmoid(outputs["outcome_logits"])
    step_valid = batch.step_mask > 0
    outcome_acc = (outcome_pred[step_valid] == batch.outcome.float()[step_valid]).float().mean().item() if bool(step_valid.any()) else 0.0
    outcome_pos_rate = batch.outcome.float()[step_valid].mean().item() if bool(step_valid.any()) else 0.0
    outcome_pred_pos_rate = outcome_pred[step_valid].float().mean().item() if bool(step_valid.any()) else 0.0
    outcome_brier = ((outcome_prob[step_valid] - batch.outcome.float()[step_valid]) ** 2).mean().item() if bool(step_valid.any()) else 0.0
    plan_target = batch.future_plan.float().clamp(0.0, 1.0)
    plan_pred = torch.sigmoid(outputs["plan_logits"])
    plan_mask = step_valid.unsqueeze(-1).expand_as(plan_target)
    plan_mae = (plan_pred[plan_mask] - plan_target[plan_mask]).abs().mean().item() if bool(plan_mask.any()) else 0.0
    plan_target_bin = (plan_target > 0.05) & plan_mask
    plan_pred_bin = (plan_pred > 0.20) & plan_mask
    plan_tp = (plan_target_bin & plan_pred_bin).float().sum()
    plan_pp = plan_pred_bin.float().sum()
    plan_gp = plan_target_bin.float().sum()
    plan_f1 = float((2.0 * plan_tp / (plan_pp + plan_gp).clamp(min=1.0)).item())
    plan_pos_rate = float((plan_gp / plan_mask.float().sum().clamp(min=1.0)).item())
    plan_pred_pos_rate = float((plan_pp / plan_mask.float().sum().clamp(min=1.0)).item())
    next_type_logits = outputs.get("next_type_logits")
    next_type_n = next_type_acc = 0.0
    next_acc_by_offset = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    if next_type_logits is not None:
        horizon = int(next_type_logits.shape[2])
        seq_len = int(step_mask.shape[1])
        total_correct = 0.0
        total_n = 0.0
        for offset in range(1, min(horizon, seq_len - 1) + 1):
            src = slice(0, seq_len - offset)
            tgt = slice(offset, seq_len)
            valid_next = (
                (batch.step_mask[:, src] > 0)
                & (batch.step_mask[:, tgt] > 0)
                & (batch.target_first[:, tgt] >= 0)
            )
            n_next = float(valid_next.sum().item())
            if n_next <= 0:
                continue
            pred_next = next_type_logits[:, src, offset - 1, :].argmax(dim=-1)
            correct = (
                pred_next[valid_next]
                == batch.target_type[:, tgt].long()[valid_next].clamp(0, next_type_logits.shape[-1] - 1)
            ).float().sum().item()
            acc_next = float(correct / max(n_next, 1.0))
            if offset in next_acc_by_offset:
                next_acc_by_offset[offset] = acc_next
            total_correct += correct
            total_n += n_next
        next_type_n = total_n
        next_type_acc = float(total_correct / max(total_n, 1.0))

    dca_plan_n = dca_plan_spread_acc = dca_plan_spread_f1 = 0.0
    dca_plan_unique_acc = dca_plan_unique_mae = dca_plan_focus_mae = 0.0
    dca_plan_spread_rate = dca_plan_pred_spread_rate = 0.0
    if bool(dca.any()) and "dca_spread_logits" in outputs:
        spread_target = (batch.dca_group_unique_slots.long()[dca] > 1).float()
        spread_pred = (torch.sigmoid(outputs["dca_spread_logits"][dca]) >= 0.5).float()
        dca_plan_n = float(spread_target.numel())
        dca_plan_spread_acc = float((spread_pred == spread_target).float().mean().item())
        sp_tp = (spread_pred * spread_target).sum()
        sp_pp = spread_pred.sum()
        sp_gp = spread_target.sum()
        dca_plan_spread_f1 = float((2.0 * sp_tp / (sp_pp + sp_gp).clamp(min=1.0)).item())
        unique_target = batch.dca_group_unique_slots.long()[dca].clamp(0, MAX_SELECT_COUNT)
        unique_pred = outputs["dca_unique_logits"][dca].argmax(dim=-1)
        dca_plan_unique_acc = float((unique_pred == unique_target).float().mean().item())
        dca_plan_unique_mae = float((unique_pred.float() - unique_target.float()).abs().mean().item())
        focus_target = batch.dca_group_focus_frac.float()[dca].clamp(0.0, 1.0)
        focus_pred = torch.sigmoid(outputs["dca_focus_logits"][dca])
        dca_plan_focus_mae = float((focus_pred - focus_target).abs().mean().item())
        dca_plan_spread_rate = float(spread_target.mean().item())
        dca_plan_pred_spread_rate = float(spread_pred.mean().item())
    dca_rows = dca.float().sum().clamp(min=1.0)
    dca_focus_mean = ((batch.dca_group_focus_frac.float() * dca.float()).sum() / dca_rows).item()
    dca_spread_rate = (dca_spread.float().sum() / dca_rows).item()
    dca_prior_unique_mean = ((batch.dca_prior_unique_slots.float() * dca.float()).sum() / dca_rows).item()
    seq_lengths = batch.step_mask.float().sum(dim=-1)
    seq_len_mean = seq_lengths.mean().item()
    seq_full_rate = (seq_lengths >= batch.step_mask.shape[1]).float().mean().item()
    history_present_rate = (seq_lengths > 1).float().mean().item()
    ledger_progress = batch.ledger_feats.float()[step_valid][:, 0].mean().item() if bool(step_valid.any()) and batch.ledger_feats.shape[-1] > 0 else 0.0
    known_mask = batch.known_opp_mask.float().clamp(0.0, 1.0)
    known_counts = batch.known_opp_counts.float().clamp(0.0, 1.0)
    known_slots = known_mask.sum(dim=-1)
    known_step = step_valid & (known_slots > 0)
    known_opp_rate = known_step.float().sum().item() / max(float(step_valid.float().sum().item()), 1.0) if bool(step_valid.any()) else 0.0
    known_opp_slots_mean = known_slots[known_step].mean().item() if bool(known_step.any()) else 0.0
    known_opp_count_mean = (known_counts.sum(dim=-1)[known_step]).mean().item() if bool(known_step.any()) else 0.0
    turn_continue_acc = turn_continue_f1 = 0.0
    turn_continue_target_rate = turn_continue_pred_rate = 0.0
    turn_remaining_mae = 0.0
    turn_future_type_f1 = 0.0
    turn_future_type_target_rate = turn_future_type_pred_rate = 0.0
    turn_next_exists_rate = 0.0
    turn_next_type_acc = turn_next_type_pos_acc = turn_next_none_acc = 0.0
    turn_next_card_acc = turn_next_attack_acc = turn_next_context_acc = 0.0
    turn_next_card_n = turn_next_attack_n = turn_next_context_n = 0.0
    cur_turn_next_exists_rate = 0.0
    cur_turn_next_type_acc = cur_turn_next_type_pos_acc = 0.0
    cur_turn_next_card_acc = cur_turn_next_attack_acc = 0.0
    turn_seq_type_acc = turn_seq_type_pos_acc = turn_seq_none_acc = 0.0
    turn_seq_step_acc = {i: 0.0 for i in range(TURN_PLAN_STEPS)}
    turn_seq_card_acc = turn_seq_attack_acc = turn_seq_context_acc = 0.0
    turn_seq_card_n = turn_seq_attack_n = turn_seq_context_n = 0.0
    cur_turn_continue_target_rate = cur_turn_continue_pred_rate = cur_turn_continue_f1 = 0.0
    cur_turn_continue_miss = cur_terminal_when_continue = cur_nonterminal_when_stop = 0.0
    cur_terminal_prob_when_continue = cur_terminal_prob_when_stop = cur_nonterminal_prob_when_stop = 0.0
    turn_target = batch.turn_continue.float().clamp(0.0, 1.0)
    turn_pred = torch.zeros_like(turn_target)
    if "turn_continue_logits" in outputs and "turn_remaining_logits" in outputs and "turn_future_type_logits" in outputs:
        turn_target = batch.turn_continue.float().clamp(0.0, 1.0)
        turn_pred = (torch.sigmoid(outputs["turn_continue_logits"].float()) >= 0.5).float()
        turn_valid = step_valid
        if bool(turn_valid.any()):
            turn_continue_acc = float((turn_pred[turn_valid] == turn_target[turn_valid]).float().mean().item())
            turn_continue_target_rate = float(turn_target[turn_valid].mean().item())
            turn_continue_pred_rate = float(turn_pred[turn_valid].mean().item())
            tp_turn = (turn_pred[turn_valid] * turn_target[turn_valid]).sum()
            pp_turn = turn_pred[turn_valid].sum()
            gp_turn = turn_target[turn_valid].sum()
            turn_continue_f1 = float((2.0 * tp_turn / (pp_turn + gp_turn).clamp(min=1.0)).item())
            rem_pred = outputs["turn_remaining_logits"].argmax(dim=-1)
            turn_remaining_mae = float(
                (rem_pred[turn_valid].float() - batch.turn_remaining.long()[turn_valid].clamp(0, MAX_SELECT_COUNT).float())
                .abs()
                .mean()
                .item()
            )
            fut_target = batch.turn_future_types.float().clamp(0.0, 1.0)
            fut_pred = (torch.sigmoid(outputs["turn_future_type_logits"].float()) >= 0.20).float()
            fut_mask = turn_valid.unsqueeze(-1).expand_as(fut_target)
            tp_fut = (fut_pred[fut_mask] * fut_target[fut_mask]).sum()
            pp_fut = fut_pred[fut_mask].sum()
            gp_fut = fut_target[fut_mask].sum()
            turn_future_type_f1 = float((2.0 * tp_fut / (pp_fut + gp_fut).clamp(min=1.0)).item())
            turn_future_type_target_rate = float(fut_target[fut_mask].mean().item())
            turn_future_type_pred_rate = float(fut_pred[fut_mask].mean().item())

    if "turn_next_type_logits" in outputs:
        next_valid = step_valid
        next_exists = batch.turn_next_exists.float().clamp(0.0, 1.0) > 0.5
        next_type_target = torch.where(
            next_exists,
            batch.turn_next_type.long().clamp(0, N_ACTION_TYPES),
            torch.full_like(batch.turn_next_type.long(), N_ACTION_TYPES),
        )
        next_type_pred = outputs["turn_next_type_logits"].argmax(dim=-1)
        if bool(next_valid.any()):
            turn_next_exists_rate = float(next_exists[next_valid].float().mean().item())
            turn_next_type_acc = float((next_type_pred[next_valid] == next_type_target[next_valid]).float().mean().item())
            pos = next_valid & next_exists
            none = next_valid & (~next_exists)
            if bool(pos.any()):
                turn_next_type_pos_acc = float((next_type_pred[pos] == next_type_target[pos]).float().mean().item())
            if bool(none.any()):
                turn_next_none_acc = float((next_type_pred[none] == N_ACTION_TYPES).float().mean().item())
        card_valid = next_valid & next_exists & (batch.turn_next_card.long() > 0)
        if bool(card_valid.any()) and "turn_next_card_logits" in outputs:
            turn_next_card_n = float(card_valid.sum().item())
            card_pred = outputs["turn_next_card_logits"].argmax(dim=-1)
            turn_next_card_acc = float(
                (card_pred[card_valid] == batch.turn_next_card.long()[card_valid].clamp(0, N_CARDS + 1)).float().mean().item()
            )
        attack_valid = next_valid & next_exists & (batch.turn_next_attack.long() > 0)
        if bool(attack_valid.any()) and "turn_next_attack_logits" in outputs:
            turn_next_attack_n = float(attack_valid.sum().item())
            attack_pred = outputs["turn_next_attack_logits"].argmax(dim=-1)
            turn_next_attack_acc = float(
                (attack_pred[attack_valid] == batch.turn_next_attack.long()[attack_valid].clamp(0, N_ATTACKS)).float().mean().item()
            )
        context_valid = next_valid & next_exists & (batch.turn_next_context.long() > 0)
        if bool(context_valid.any()) and "turn_next_context_logits" in outputs:
            turn_next_context_n = float(context_valid.sum().item())
            context_pred = outputs["turn_next_context_logits"].argmax(dim=-1)
            turn_next_context_acc = float(
                (context_pred[context_valid] == batch.turn_next_context.long()[context_valid].clamp(0, 65)).float().mean().item()
            )
        if bool(current_valid.any()):
            cur_turn_next_exists_rate = float(next_exists[current_valid].float().mean().item())
            cur_turn_next_type_acc = float(
                (next_type_pred[current_valid] == next_type_target[current_valid]).float().mean().item()
            )
            cur_pos = current_valid & next_exists
            if bool(cur_pos.any()):
                cur_turn_next_type_pos_acc = float((next_type_pred[cur_pos] == next_type_target[cur_pos]).float().mean().item())
                if "turn_next_card_logits" in outputs:
                    cur_card = cur_pos & (batch.turn_next_card.long() > 0)
                    if bool(cur_card.any()):
                        card_pred = outputs["turn_next_card_logits"].argmax(dim=-1)
                        cur_turn_next_card_acc = float(
                            (card_pred[cur_card] == batch.turn_next_card.long()[cur_card].clamp(0, N_CARDS + 1)).float().mean().item()
                        )
                if "turn_next_attack_logits" in outputs:
                    cur_attack = cur_pos & (batch.turn_next_attack.long() > 0)
                    if bool(cur_attack.any()):
                        attack_pred = outputs["turn_next_attack_logits"].argmax(dim=-1)
                        cur_turn_next_attack_acc = float(
                            (attack_pred[cur_attack] == batch.turn_next_attack.long()[cur_attack].clamp(0, N_ATTACKS)).float().mean().item()
                        )

    if "turn_seq_type_logits" in outputs:
        seq_valid = (step_mask > 0).unsqueeze(-1).expand_as(batch.turn_plan_mask)
        seq_exists = batch.turn_plan_mask.float().clamp(0.0, 1.0) > 0.5
        seq_type_target = torch.where(
            seq_exists,
            batch.turn_plan_types.long().clamp(0, N_ACTION_TYPES),
            torch.full_like(batch.turn_plan_types.long(), N_ACTION_TYPES),
        )
        seq_type_pred = outputs["turn_seq_type_logits"].argmax(dim=-1)
        if bool(seq_valid.any()):
            turn_seq_type_acc = float((seq_type_pred[seq_valid] == seq_type_target[seq_valid]).float().mean().item())
            seq_pos = seq_valid & seq_exists
            seq_none = seq_valid & (~seq_exists)
            if bool(seq_pos.any()):
                turn_seq_type_pos_acc = float((seq_type_pred[seq_pos] == seq_type_target[seq_pos]).float().mean().item())
            if bool(seq_none.any()):
                turn_seq_none_acc = float((seq_type_pred[seq_none] == N_ACTION_TYPES).float().mean().item())
            for step in range(min(TURN_PLAN_STEPS, 4)):
                step_mask_i = seq_valid[..., step]
                if bool(step_mask_i.any()):
                    turn_seq_step_acc[step] = float(
                        (seq_type_pred[..., step][step_mask_i] == seq_type_target[..., step][step_mask_i]).float().mean().item()
                    )
        seq_card_valid = seq_valid & seq_exists & (batch.turn_plan_cards.long() > 0)
        if bool(seq_card_valid.any()) and "turn_seq_card_logits" in outputs:
            turn_seq_card_n = float(seq_card_valid.sum().item())
            seq_card_pred = outputs["turn_seq_card_logits"].argmax(dim=-1)
            turn_seq_card_acc = float(
                (seq_card_pred[seq_card_valid] == batch.turn_plan_cards.long()[seq_card_valid].clamp(0, N_CARDS + 1)).float().mean().item()
            )
        seq_attack_valid = seq_valid & seq_exists & (batch.turn_plan_attacks.long() > 0)
        if bool(seq_attack_valid.any()) and "turn_seq_attack_logits" in outputs:
            turn_seq_attack_n = float(seq_attack_valid.sum().item())
            seq_attack_pred = outputs["turn_seq_attack_logits"].argmax(dim=-1)
            turn_seq_attack_acc = float(
                (seq_attack_pred[seq_attack_valid] == batch.turn_plan_attacks.long()[seq_attack_valid].clamp(0, N_ATTACKS)).float().mean().item()
            )
        seq_context_valid = seq_valid & seq_exists & (batch.turn_plan_contexts.long() > 0)
        if bool(seq_context_valid.any()) and "turn_seq_context_logits" in outputs:
            turn_seq_context_n = float(seq_context_valid.sum().item())
            seq_context_pred = outputs["turn_seq_context_logits"].argmax(dim=-1)
            turn_seq_context_acc = float(
                (seq_context_pred[seq_context_valid] == batch.turn_plan_contexts.long()[seq_context_valid].clamp(0, 65)).float().mean().item()
            )

    if bool(current_valid.any()):
        cur_turn_target = turn_target[current_valid]
        cur_turn_pred = turn_pred[current_valid]
        cur_turn_continue_target_rate = float(cur_turn_target.mean().item())
        cur_turn_continue_pred_rate = float(cur_turn_pred.mean().item())
        tp_cur = (cur_turn_pred * cur_turn_target).sum()
        pp_cur = cur_turn_pred.sum()
        gp_cur = cur_turn_target.sum()
        cur_turn_continue_f1 = float((2.0 * tp_cur / (pp_cur + gp_cur).clamp(min=1.0)).item())
        cur_cont = current_valid & (turn_target > 0.5)
        cur_stop = current_valid & (turn_target <= 0.5)
        terminal_options = (
            ((batch.opt_type.long() == TYPE_END) | (batch.opt_type.long() == TYPE_ATTACK))
            & (opt_mask > 0)
        )
        terminal_prob = (probs * terminal_options.float()).sum(dim=-1).clamp(0.0, 1.0)
        if bool(cur_cont.any()):
            cur_turn_continue_miss = float((turn_pred[cur_cont] < 0.5).float().mean().item())
            cur_terminal_when_continue = float(
                ((pred_type[cur_cont] == TYPE_END) | (pred_type[cur_cont] == TYPE_ATTACK)).float().mean().item()
            )
            cur_terminal_prob_when_continue = float(terminal_prob[cur_cont].mean().item())
        if bool(cur_stop.any()):
            cur_nonterminal_when_stop = float(
                ((pred_type[cur_stop] != TYPE_END) & (pred_type[cur_stop] != TYPE_ATTACK)).float().mean().item()
            )
            cur_terminal_prob_when_stop = float(terminal_prob[cur_stop].mean().item())
            cur_nonterminal_prob_when_stop = float((1.0 - terminal_prob[cur_stop]).mean().item())
    prev_nonzero = (
        (batch.prev_type.long() != 0)
        | (batch.prev_card.long() != 0)
        | (batch.prev_card2.long() != 0)
        | (batch.prev_attack.long() != 0)
    )
    prev_nonzero_rate = prev_nonzero[step_valid].float().mean().item() if bool(step_valid.any()) else 0.0

    type_metrics: dict[str, float] = {}
    for typ, name in _MONITORED_TYPES:
        mask = valid & (batch.target_type.long() == typ)
        n = float(mask.sum().item())
        type_metrics[f"{name}_rate"] = float(n / max(valid_n, 1.0))
        if bool(mask.any()):
            type_metrics[f"{name}_top1"] = float((pred[mask] == batch.target_first[mask]).float().mean().item())
            type_metrics[f"{name}_atype"] = float((pred_type[mask] == batch.target_type.long()[mask]).float().mean().item())
        else:
            type_metrics[f"{name}_top1"] = 0.0
            type_metrics[f"{name}_atype"] = 0.0

    metrics = {
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
        "cur_n": current_n,
        "cur_top1": float(cur_top1),
        "cur_type_acc": float(cur_type_acc),
        "cur_set_f1": float(cur_set_f1),
        "cur_entropy": float(cur_entropy),
        "cur_margin": float(cur_margin),
        "cur_target_margin_best": float(cur_target_margin_best),
        "cur_rank_violation_025": float(cur_rank_violation_025),
        "cur_ambig_target_margin_best": float(cur_ambig_target_margin_best),
        "cur_forced_rate": float(cur_forced_rate),
        "cur_nonforced_top1": float(cur_nonforced_top1),
        "cur_bigopt_rate": float(cur_bigopt_rate),
        "cur_bigopt_top1": float(cur_bigopt_top1),
        "cur_ambig_type_rate": float(cur_ambig_type_rate),
        "cur_ambig_type_top1": float(cur_ambig_type_top1),
        "cur_target_end_rate": float(cur_target_end_rate),
        "cur_pred_end_rate": float(cur_pred_end_rate),
        "cur_pred_end_when_nonend_legal": float(cur_pred_end_when_nonend_legal),
        "cur_pred_end_when_target_nonend": float(cur_pred_end_when_target_nonend),
        "cur_dca_n": float((current_valid & dca).float().sum().item()),
        "cur_dca_top1": float(cur_dca_top1),
        "seq_len_mean": float(seq_len_mean),
        "seq_full_rate": float(seq_full_rate),
        "history_present_rate": float(history_present_rate),
        "ledger_progress": float(ledger_progress),
        "prev_nonzero_rate": float(prev_nonzero_rate),
        "known_opp_rate": float(known_opp_rate),
        "known_opp_slots_mean": float(known_opp_slots_mean),
        "known_opp_count_mean": float(known_opp_count_mean),
        "known_action_n": float(known_action_n),
        "known_action_top1": float(known_action_top1),
        "turn_continue_acc": float(turn_continue_acc),
        "turn_continue_f1": float(turn_continue_f1),
        "turn_continue_target_rate": float(turn_continue_target_rate),
        "turn_continue_pred_rate": float(turn_continue_pred_rate),
        "turn_remaining_mae": float(turn_remaining_mae),
        "turn_future_type_f1": float(turn_future_type_f1),
        "turn_future_type_target_rate": float(turn_future_type_target_rate),
        "turn_future_type_pred_rate": float(turn_future_type_pred_rate),
        "turn_next_exists_rate": float(turn_next_exists_rate),
        "turn_next_type_acc": float(turn_next_type_acc),
        "turn_next_type_pos_acc": float(turn_next_type_pos_acc),
        "turn_next_none_acc": float(turn_next_none_acc),
        "turn_next_card_acc": float(turn_next_card_acc),
        "turn_next_card_n": float(turn_next_card_n),
        "turn_next_attack_acc": float(turn_next_attack_acc),
        "turn_next_attack_n": float(turn_next_attack_n),
        "turn_next_context_acc": float(turn_next_context_acc),
        "turn_next_context_n": float(turn_next_context_n),
        "cur_turn_next_exists_rate": float(cur_turn_next_exists_rate),
        "cur_turn_next_type_acc": float(cur_turn_next_type_acc),
        "cur_turn_next_type_pos_acc": float(cur_turn_next_type_pos_acc),
        "cur_turn_next_card_acc": float(cur_turn_next_card_acc),
        "cur_turn_next_attack_acc": float(cur_turn_next_attack_acc),
        "turn_seq_type_acc": float(turn_seq_type_acc),
        "turn_seq_type_pos_acc": float(turn_seq_type_pos_acc),
        "turn_seq_none_acc": float(turn_seq_none_acc),
        "turn_seq_step1_acc": float(turn_seq_step_acc.get(0, 0.0)),
        "turn_seq_step2_acc": float(turn_seq_step_acc.get(1, 0.0)),
        "turn_seq_step3_acc": float(turn_seq_step_acc.get(2, 0.0)),
        "turn_seq_step4_acc": float(turn_seq_step_acc.get(3, 0.0)),
        "turn_seq_card_acc": float(turn_seq_card_acc),
        "turn_seq_card_n": float(turn_seq_card_n),
        "turn_seq_attack_acc": float(turn_seq_attack_acc),
        "turn_seq_attack_n": float(turn_seq_attack_n),
        "turn_seq_context_acc": float(turn_seq_context_acc),
        "turn_seq_context_n": float(turn_seq_context_n),
        "cur_turn_continue_target_rate": float(cur_turn_continue_target_rate),
        "cur_turn_continue_pred_rate": float(cur_turn_continue_pred_rate),
        "cur_turn_continue_f1": float(cur_turn_continue_f1),
        "cur_turn_continue_miss": float(cur_turn_continue_miss),
        "cur_terminal_when_continue": float(cur_terminal_when_continue),
        "cur_nonterminal_when_stop": float(cur_nonterminal_when_stop),
        "cur_terminal_prob_when_continue": float(cur_terminal_prob_when_continue),
        "cur_terminal_prob_when_stop": float(cur_terminal_prob_when_stop),
        "cur_nonterminal_prob_when_stop": float(cur_nonterminal_prob_when_stop),
        "plan_mae": float(plan_mae),
        "plan_f1": float(plan_f1),
        "plan_pos_rate": float(plan_pos_rate),
        "plan_pred_pos_rate": float(plan_pred_pos_rate),
        "next_type_n": float(next_type_n),
        "next_type_acc": float(next_type_acc),
        "next1_acc": float(next_acc_by_offset[1]),
        "next2_acc": float(next_acc_by_offset[2]),
        "next3_acc": float(next_acc_by_offset[3]),
        "next4_acc": float(next_acc_by_offset[4]),
        "dca_plan_n": float(dca_plan_n),
        "dca_plan_spread_acc": float(dca_plan_spread_acc),
        "dca_plan_spread_f1": float(dca_plan_spread_f1),
        "dca_plan_unique_acc": float(dca_plan_unique_acc),
        "dca_plan_unique_mae": float(dca_plan_unique_mae),
        "dca_plan_focus_mae": float(dca_plan_focus_mae),
        "dca_plan_spread_rate": float(dca_plan_spread_rate),
        "dca_plan_pred_spread_rate": float(dca_plan_pred_spread_rate),
        "action_type_acc": float(action_type_acc),
        "option_n": float(opt_n[valid].float().mean().item()),
        "bigopt_rate": float(bigopt.float().sum().item() / max(valid_n, 1.0)),
        "bigopt_top1": float((pred[bigopt] == batch.target_first[bigopt]).float().mean().item()) if bool(bigopt.any()) else 0.0,
        "ambig_type_rate": float(ambig_type.float().sum().item() / max(valid_n, 1.0)),
        "ambig_type_top1": float((pred[ambig_type] == batch.target_first[ambig_type]).float().mean().item()) if bool(ambig_type.any()) else 0.0,
        "action_entropy": float(entropy),
        "action_margin": float(margin),
        "outcome_pos_rate": float(outcome_pos_rate),
        "outcome_pred_pos_rate": float(outcome_pred_pos_rate),
        "outcome_brier": float(outcome_brier),
    }
    metrics.update(type_metrics)
    metrics.update(current_type_opportunity)
    return metrics


def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)
