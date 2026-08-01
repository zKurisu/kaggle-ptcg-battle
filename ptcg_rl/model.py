"""
Pointer-style policy/value network. PyTorch — fast, clean, no pkm.

act() for inference. evaluate_actions() for PPO update (batched re-eval).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import STATE_FEAT_DIM, OPT_FEAT_DIM, N_CARDS, N_ATTACKS, N_OPT_TYPES, BOARD_SLOTS, MAX_HAND

EMB_CARD = 64
EMB_ATTACK = 32
EMB_OPT_TYPE = 16
OPT_ENC = 128
HIDDEN = 256
NEG_INF = -1e9


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(N_CARDS + 2, EMB_CARD, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, EMB_ATTACK, padding_idx=0)
        self.opt_type_emb = nn.Embedding(N_OPT_TYPES + 1, EMB_OPT_TYPE)
        self.stop_vec = nn.Parameter(torch.zeros(OPT_ENC))

        state_in = 2 * EMB_CARD + EMB_CARD + STATE_FEAT_DIM
        self.state_fc1 = nn.Linear(state_in, 512)
        self.state_fc2 = nn.Linear(512, HIDDEN)

        opt_in = EMB_CARD + EMB_CARD + EMB_ATTACK + EMB_OPT_TYPE + OPT_FEAT_DIM
        self.opt_fc = nn.Linear(opt_in, OPT_ENC)

        score_in = HIDDEN + OPT_ENC + OPT_ENC
        self.score_fc1 = nn.Linear(score_in, 128)
        self.score_fc2 = nn.Linear(128, 1)

        self.value_fc1 = nn.Linear(HIDDEN, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    # ── pooling ─────────────────────────────────────────────────────

    def _pool(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.card_emb(ids)
        mask = (ids > 0).float().unsqueeze(-1)
        return (e * mask).sum(-2) / mask.sum(-2).clamp(min=1.0)

    # ── state encoder ───────────────────────────────────────────────

    def encode_state(self, board: torch.Tensor, hand: torch.Tensor,
                     feats: torch.Tensor) -> torch.Tensor:
        my = self._pool(board[..., :6])
        opp = self._pool(board[..., 6:])
        hnd = self._pool(hand)
        x = torch.cat([my, opp, hnd, feats], dim=-1)
        return F.relu(self.state_fc2(F.relu(self.state_fc1(x))))

    # ── option encoder ──────────────────────────────────────────────

    def encode_options(self, opt_type: torch.Tensor, opt_card: torch.Tensor,
                       opt_card2: torch.Tensor, opt_attack: torch.Tensor,
                       opt_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.card_emb(opt_card), self.card_emb(opt_card2),
            self.attack_emb(opt_attack), self.opt_type_emb(opt_type), opt_feats,
        ], dim=-1)
        return F.relu(self.opt_fc(x))

    # ── scoring ─────────────────────────────────────────────────────

    def option_logits(self, h: torch.Tensor, opts: torch.Tensor,
                      picked_sum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = opts.shape
        stop = self.stop_vec.expand(B, 1, OPT_ENC)
        rows = torch.cat([opts, stop], dim=1)
        hx = h.unsqueeze(1).expand(B, N + 1, HIDDEN)
        px = picked_sum.unsqueeze(1).expand(B, N + 1, OPT_ENC)
        x = torch.cat([hx, rows, px], dim=-1)
        logits = self.score_fc2(F.relu(self.score_fc1(x))).squeeze(-1)
        return logits.masked_fill(~mask, NEG_INF)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value_fc2(F.relu(self.value_fc1(h)))).squeeze(-1)

    # ── inference: act on ONE decision ──────────────────────────────

    @torch.no_grad()
    def act(self, board: np.ndarray, hand: np.ndarray, feats: np.ndarray,
            opt_type: np.ndarray, opt_card: np.ndarray, opt_card2: np.ndarray,
            opt_attack: np.ndarray, opt_feats: np.ndarray,
            min_count: int, max_count: int, greedy: bool = False) -> tuple:
        """Single decision → (picks, logprob, value). All inputs are 1-D numpy."""
        dev = next(self.parameters()).device
        b = torch.from_numpy(board).unsqueeze(0).to(dev)
        hd = torch.from_numpy(hand).unsqueeze(0).to(dev)
        ft = torch.from_numpy(feats).unsqueeze(0).to(dev)
        h = self.encode_state(b, hd, ft)
        v = float(self.value(h)[0])

        n = len(opt_type)
        ot = torch.from_numpy(opt_type).unsqueeze(0).to(dev)
        oc = torch.from_numpy(opt_card).unsqueeze(0).to(dev)
        oc2 = torch.from_numpy(opt_card2).unsqueeze(0).to(dev)
        oa = torch.from_numpy(opt_attack).unsqueeze(0).to(dev)
        of = torch.from_numpy(opt_feats).unsqueeze(0).to(dev)
        opts = self.encode_options(ot, oc, oc2, oa, of)

        picks, stopped, lp = [], False, 0.0
        ps = torch.zeros(1, OPT_ENC, device=dev)
        avail = torch.ones(1, n + 1, dtype=torch.bool, device=dev)

        while len(picks) < max_count:
            avail[0, n] = len(picks) >= min_count
            logits = self.option_logits(h, opts, ps, avail)
            logp = F.log_softmax(logits, dim=-1)
            idx = int(logp.argmax(dim=-1)[0]) if greedy else int(torch.multinomial(logp.exp(), 1)[0, 0])
            lp += float(logp[0, idx])
            if idx == n:
                stopped = True; break
            picks.append(idx)
            ps = ps + opts[0, idx]
            avail[0, idx] = False

        return picks, lp, v

    # ── PPO: re-evaluate stored actions ─────────────────────────────

    def evaluate_actions(self, decisions: list) -> tuple:
        """Recompute (new_logprobs, entropies, values) for stored decisions.
        Returns tensors on model's device."""
        if not decisions:
            return (torch.zeros(0), torch.zeros(0), torch.zeros(0))

        dev = next(self.parameters()).device
        B = len(decisions)

        # Batch state
        board = torch.from_numpy(np.stack([d.board_cards for d in decisions])).to(dev)
        hand = torch.from_numpy(np.stack([d.hand_cards for d in decisions])).to(dev)
        feats = torch.from_numpy(np.stack([d.state_feats for d in decisions])).to(dev)
        h = self.encode_state(board, hand, feats)
        values = self.value(h)

        # Batch options (pad to n_max)
        n_max = max(len(d.opt_type) for d in decisions)
        ot = torch.zeros(B, n_max, dtype=torch.long, device=dev)
        oc = torch.zeros(B, n_max, dtype=torch.long, device=dev)
        oc2 = torch.zeros(B, n_max, dtype=torch.long, device=dev)
        oa = torch.zeros(B, n_max, dtype=torch.long, device=dev)
        of = torch.zeros(B, n_max, OPT_FEAT_DIM, device=dev)
        for i, d in enumerate(decisions):
            n = len(d.opt_type)
            ot[i, :n] = torch.from_numpy(d.opt_type).to(dev)
            oc[i, :n] = torch.from_numpy(d.opt_card).to(dev)
            oc2[i, :n] = torch.from_numpy(d.opt_card2).to(dev)
            oa[i, :n] = torch.from_numpy(d.opt_attack).to(dev)
            of[i, :n] = torch.from_numpy(d.opt_feats).to(dev)

        opts = self.encode_options(ot, oc, oc2, oa, of)

        # Sequential re-evaluation
        logprobs = torch.zeros(B, device=dev)
        entropies = torch.zeros(B, device=dev)
        ps = torch.zeros(B, OPT_ENC, device=dev)
        avail = torch.ones(B, n_max + 1, dtype=torch.bool, device=dev)

        # Pad stored actions
        max_k = max(len(d.action) for d in decisions) + 1  # +1 for possible STOP
        for k in range(max_k):
            stop_ok = torch.tensor([k >= d.min_count for d in decisions], device=dev)
            mask = avail.clone()
            mask[:, n_max] = stop_ok

            logits = self.option_logits(h, opts, ps, mask)
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            ent = -(probs * logp).sum(dim=-1)

            for i, d in enumerate(decisions):
                if k < len(d.action):
                    idx = d.action[k]
                    logprobs[i] += logp[i, idx]
                    entropies[i] += ent[i]
                    ps[i] += opts[i, idx]
                    avail[i, idx] = False
                elif k == len(d.action) and hasattr(d, 'stopped') and d.stopped:
                    logprobs[i] += logp[i, n_max]
                    entropies[i] += ent[i]

        entropies /= max_k
        return logprobs, entropies, values

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
