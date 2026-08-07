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

NEG_INF = -1e9
_EC, _EA, _EO, _OE, _HD, _S1, _SC = 64, 32, 16, 128, 256, 512, 128
_CTX, _AREA, _IDX = 16, 8, 8
ARCH_POINTER = "pointer"
ARCH_CROSS_ATTN = "cross_attn"


class PolicyValueNet(nn.Module):
    def __init__(self, width: float = 1.0, option_context: bool = True,
                 slot_state: bool = True, state_feat_dim: int = STATE_FEAT_DIM,
                 opt_feat_dim: int = OPT_FEAT_DIM, plan_dim: int = 0,
                 hierarchical_plan: bool = False, history_k: int = 0):
        """width=1.0→501K, 2.0→4M, 3.0→9M params."""
        super().__init__()
        ec = int(_EC * width); ea = int(_EA * width); eo_t = int(_EO * width)
        oe = int(_OE * width); hd = int(_HD * width)
        s1 = int(_S1 * width); sc = int(_SC * width)
        ctx = int(_CTX * width); area = int(_AREA * width); idx = int(_IDX * width)
        self.option_context = option_context
        self.slot_state = slot_state
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.plan_dim = int(plan_dim)
        self.hierarchical_plan = bool(hierarchical_plan and self.plan_dim > 0)
        self.history_k = max(0, int(history_k))
        self._ec=ec; self._ea=ea; self._eo_t=eo_t; self._oe=oe; self._hd=hd
        self._ctx=ctx; self._area=area; self._idx=idx
        self._plan_cd = sc if self.hierarchical_plan else 0
        self._hist = sc if self.history_k > 0 else 0

        self.card_emb = nn.Embedding(N_CARDS + 2, ec, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, ea, padding_idx=0)
        self.opt_type_emb = nn.Embedding(N_OPT_TYPES + 1, eo_t)
        if option_context:
            self.context_emb = nn.Embedding(65, ctx)
            self.select_type_emb = nn.Embedding(17, ctx)
            self.area_emb = nn.Embedding(17, area)
            self.index_emb = nn.Embedding(65, idx)
            self.inplay_area_emb = nn.Embedding(17, area)
            self.inplay_index_emb = nn.Embedding(17, idx)
        self.stop_vec = nn.Parameter(torch.zeros(oe))

        state_in = (5 * ec if slot_state else 3 * ec) + self.state_feat_dim
        self.state_fc1 = nn.Linear(state_in, s1)
        self.state_fc2 = nn.Linear(s1, hd)
        if self.history_k > 0:
            self.history_type_emb = nn.Embedding(N_OPT_TYPES + 2, eo_t, padding_idx=0)
            self.history_context_emb = nn.Embedding(66, ctx, padding_idx=0)
            self.history_select_type_emb = nn.Embedding(18, ctx, padding_idx=0)
            self.history_pos_emb = nn.Embedding(self.history_k, ctx)
            hist_in = ec + ec + ea + eo_t + ctx + ctx + ctx + 2
            self.history_token_fc = nn.Linear(hist_in, sc)
            self.history_gru = nn.GRU(sc, sc, batch_first=True)
            self.history_out_fc = nn.Linear(hd + sc, hd)
        opt_extra = ctx + ctx + area + idx + area + idx if option_context else 0
        self.opt_fc = nn.Linear(ec + ec + ea + eo_t + opt_extra + self.opt_feat_dim, oe)
        self.score_fc1 = nn.Linear(hd + oe + oe, sc)
        self.score_fc2 = nn.Linear(sc, 1)
        self.value_fc1 = nn.Linear(hd, sc)
        self.value_fc2 = nn.Linear(sc, 1)
        if self.plan_dim > 0:
            self.plan_fc1 = nn.Linear(hd, sc)
            self.plan_fc2 = nn.Linear(sc, self.plan_dim)
            if self.hierarchical_plan:
                self.plan_condition_fc = nn.Linear(self.plan_dim, self._plan_cd)
                self.plan_score_fc1 = nn.Linear(oe + oe + self._plan_cd, sc)
                self.plan_score_fc2 = nn.Linear(sc, 1)
        self._init()
        if self.hierarchical_plan:
            nn.init.zeros_(self.plan_score_fc2.weight)
            nn.init.zeros_(self.plan_score_fc2.bias)

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

    def _encode_history(self, history: dict[str, torch.Tensor] | None, bsz: int,
                        device: torch.device) -> torch.Tensor:
        if self.history_k <= 0:
            return torch.zeros(bsz, 0, device=device)
        if not history or history.get("mask") is None or history["mask"].numel() == 0:
            return torch.zeros(bsz, self._hist, device=device)
        mask = history["mask"].to(device=device, dtype=torch.float32)
        if mask.shape[1] != self.history_k:
            mask = mask[:, -self.history_k:]
        k = mask.shape[1]
        pos = torch.arange(k, device=device).unsqueeze(0).expand(mask.shape[0], -1)
        parts = [
            self.card_emb(history["card"].to(device=device).long()[:, -k:]),
            self.card_emb(history["card2"].to(device=device).long()[:, -k:]),
            self.attack_emb(history["attack"].to(device=device).long()[:, -k:]),
            self.history_type_emb(history["type"].to(device=device).long()[:, -k:].clamp(0, N_OPT_TYPES + 1)),
            self.history_context_emb(history["context"].to(device=device).long()[:, -k:].clamp(0, 65)),
            self.history_select_type_emb(history["select_type"].to(device=device).long()[:, -k:].clamp(0, 17)),
            self.history_pos_emb(pos),
            history["count"].to(device=device, dtype=torch.float32)[:, -k:].unsqueeze(-1),
            mask.unsqueeze(-1),
        ]
        x = torch.cat(parts, dim=-1)
        token = F.relu(self.history_token_fc(x)) * mask.unsqueeze(-1)
        hidden = torch.zeros(1, mask.shape[0], self._hist, device=device, dtype=token.dtype)
        for t in range(k):
            _, next_hidden = self.history_gru(token[:, t:t + 1], hidden)
            valid = mask[:, t].view(1, -1, 1) > 0
            hidden = torch.where(valid, next_hidden, hidden)
        return hidden.squeeze(0)

    def _merge_history(self, h: torch.Tensor, history: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if self.history_k <= 0:
            return h
        hist = self._encode_history(history, h.shape[0], h.device)
        return F.relu(self.history_out_fc(torch.cat([h, hist], dim=-1)))

    def encode_state(self, board: torch.Tensor, hand: torch.Tensor,
                     feats: torch.Tensor, history: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        feats = self._fit_feat_dim(feats, self.state_feat_dim)
        if self.slot_state:
            my_active = self.card_emb(board[..., 0])
            my_bench = self._pool(board[..., 1:6])
            opp_active = self.card_emb(board[..., 6])
            opp_bench = self._pool(board[..., 7:])
            hnd = self._pool(hand)
            x = torch.cat([my_active, my_bench, opp_active, opp_bench, hnd, feats], dim=-1)
            return self._merge_history(F.relu(self.state_fc2(F.relu(self.state_fc1(x)))), history)
        my = self._pool(board[..., :6])
        opp = self._pool(board[..., 6:])
        hnd = self._pool(hand)
        x = torch.cat([my, opp, hnd, feats], dim=-1)
        return self._merge_history(F.relu(self.state_fc2(F.relu(self.state_fc1(x)))), history)

    # ── option encoder ──────────────────────────────────────────────

    def encode_options(self, opt_type: torch.Tensor, opt_card: torch.Tensor,
                       opt_card2: torch.Tensor, opt_attack: torch.Tensor,
                       opt_feats: torch.Tensor) -> torch.Tensor:
        opt_feats = self._fit_feat_dim(opt_feats, self.opt_feat_dim)
        parts = [
            self.card_emb(opt_card), self.card_emb(opt_card2),
            self.attack_emb(opt_attack), self.opt_type_emb(opt_type),
        ]
        if self.option_context:
            ctx = torch.round(opt_feats[..., 3] * 64.0).long().clamp(0, 64)
            sel_type = torch.round(opt_feats[..., 4] * 16.0).long().clamp(0, 16)
            area = torch.round(opt_feats[..., 7] * 16.0).long().clamp(0, 16)
            idx = torch.round(opt_feats[..., 8] * 64.0).long().clamp(0, 64)
            inplay_area = torch.round(opt_feats[..., 9] * 16.0).long().clamp(0, 16)
            inplay_idx = torch.round(opt_feats[..., 10] * 10.0).long().clamp(0, 16)
            parts.extend([
                self.context_emb(ctx), self.select_type_emb(sel_type),
                self.area_emb(area), self.index_emb(idx),
                self.inplay_area_emb(inplay_area), self.inplay_index_emb(inplay_idx),
            ])
        parts.append(opt_feats)
        x = torch.cat(parts, dim=-1)
        return F.relu(self.opt_fc(x))

    @staticmethod
    def _fit_feat_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur == dim:
            return x
        if cur > dim:
            return x[..., :dim]
        return F.pad(x, (0, dim - cur))

    # ── scoring ─────────────────────────────────────────────────────

    def option_logits(self, h: torch.Tensor, opts: torch.Tensor,
                      picked_sum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = opts.shape
        stop = self.stop_vec.expand(B, 1, self._oe)
        rows = torch.cat([opts, stop], dim=1)
        hx = h.unsqueeze(1).expand(B, N + 1, self._hd)
        px = picked_sum.unsqueeze(1).expand(B, N + 1, self._oe)
        x = torch.cat([hx, rows, px], dim=-1)
        logits = self.score_fc2(F.relu(self.score_fc1(x))).squeeze(-1)
        if self.hierarchical_plan:
            plan_prob = torch.sigmoid(self.plan_logits(h))
            plan_ctx = F.relu(self.plan_condition_fc(plan_prob))
            plan_x = torch.cat([
                rows,
                px,
                plan_ctx.unsqueeze(1).expand(B, N + 1, self._plan_cd),
            ], dim=-1)
            logits = logits + self.plan_score_fc2(F.relu(self.plan_score_fc1(plan_x))).squeeze(-1)
        return logits.masked_fill(~mask, NEG_INF)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value_fc2(F.relu(self.value_fc1(h)))).squeeze(-1)

    def plan_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.plan_dim <= 0:
            raise RuntimeError("PolicyValueNet was created without plan_dim")
        return self.plan_fc2(F.relu(self.plan_fc1(h)))

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
        ps = torch.zeros(1, self._oe, device=dev)
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
        of = torch.zeros(B, n_max, self.opt_feat_dim, device=dev)
        for i, d in enumerate(decisions):
            n = len(d.opt_type)
            ot[i, :n] = torch.from_numpy(d.opt_type).to(dev)
            oc[i, :n] = torch.from_numpy(d.opt_card).to(dev)
            oc2[i, :n] = torch.from_numpy(d.opt_card2).to(dev)
            oa[i, :n] = torch.from_numpy(d.opt_attack).to(dev)
            src = torch.from_numpy(d.opt_feats).to(dev)
            of[i, :n, : min(src.shape[-1], self.opt_feat_dim)] = src[:, : self.opt_feat_dim]

        opts = self.encode_options(ot, oc, oc2, oa, of)

        # Sequential re-evaluation
        logprobs = torch.zeros(B, device=dev)
        entropies = torch.zeros(B, device=dev)
        ps = torch.zeros(B, self._oe, device=dev)
        opt_len = torch.tensor([len(d.opt_type) for d in decisions], dtype=torch.long, device=dev)
        opt_mask = torch.arange(n_max, device=dev).unsqueeze(0) < opt_len.unsqueeze(1)
        avail = torch.cat([
            opt_mask,
            torch.ones(B, 1, dtype=torch.bool, device=dev),
        ], dim=1)

        # Pad stored actions
        max_k = max(len(d.action) for d in decisions) + 1  # +1 for possible STOP
        for k in range(max_k):
            stop_ok = torch.tensor([k >= d.min_count for d in decisions], device=dev)
            mask = avail.clone()
            mask[:, :n_max] &= opt_mask
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

    # ── MCTS action (for training-time self-play) ────────────────────

    @torch.no_grad()
    def act_mcts(self, obs_dict: dict, deck: list[int],
                 sims: int = 32) -> tuple[list[int], float, float]:
        """MCTS search using engine search API + this model as leaf evaluator.
        Returns (picks, logprob, value) like act()."""
        import math, random, time
        from .encoder import FastEncoder
        from cg.api import (
            to_observation_class, search_begin, search_step, search_end,
        )

        obs = to_observation_class(obs_dict)
        state = obs.current
        you = state.yourIndex
        my_s, op_s = state.players[you], state.players[1 - you]
        dev = next(self.parameters()).device

        # Predictions for hidden info
        mc_d, pc = my_s.deckCount, len(my_s.prize)
        oc, opc, ohc = op_s.deckCount, len(op_s.prize), op_s.handCount
        pad = (deck * ((max(mc_d, pc, oc, opc, ohc) // len(deck)) + 2)) if deck else [1]

        try:
            ss = search_begin(obs,
                your_deck=pad[:max(1, mc_d)],
                your_prize=pad[:pc] if pc > 0 else [],
                opponent_deck=pad[:max(1, oc)],
                opponent_prize=pad[:opc] if opc > 0 else [],
                opponent_hand=pad[:max(1, ohc)] if ohc > 0 else [1],
                opponent_active=[deck[0]] if (op_s.active and op_s.active[0] is None) else [])
        except Exception:
            return self._greedy_fallback(obs_dict)

        root_sel = ss.observation.select
        if root_sel is None:
            search_end(); return [], 0.0, 0.0
        n = len(root_sel.option)
        mc = root_sel.maxCount

        # Root children
        children = [{'sel': [i], 'sid': None, 'visits': 0, 'total': 0.0,
                      'prior': 0.05 if i == n else 1.0 / n}
                    for i in range(n + 1)]

        if n > 1:
            noise = np.random.dirichlet([0.25] * len(children))
            for i, c in enumerate(children):
                c['prior'] = 0.75 * c['prior'] + 0.25 * noise[i]

        enc = FastEncoder()
        def _eval_obs(o_dict: dict) -> float:
            try:
                d = enc.encode(o_dict)
                b = torch.from_numpy(d.board_cards).unsqueeze(0).to(dev)
                hd = torch.from_numpy(d.hand_cards).unsqueeze(0).to(dev)
                ft = torch.from_numpy(d.state_feats).unsqueeze(0).to(dev)
                return float(self.value(self.encode_state(b, hd, ft))[0])
            except Exception:
                return 0.0

        root_sid = ss.searchId
        t0 = time.time()

        for si in range(sims):
            if time.time() - t0 > 5.0:
                break
            cur_sid = root_sid

            # PUCT selection + expansion
            best_score, best_ci = -1e9, 0
            for ci, c in enumerate(children):
                q = c['total'] / max(1, c['visits'])
                u = 1.25 * c['prior'] * math.sqrt(max(1, si)) / (1 + c['visits'])
                if q + u > best_score:
                    best_score, best_ci = q + u, ci

            chosen = children[best_ci]
            if chosen.get('visits', 0) == 0:
                try:
                    ar = search_step(cur_sid, chosen['sel'])
                    chosen['sid'] = ar.searchId
                    lc = ar.observation.current
                    if lc and lc.result is not None and lc.result != -1:
                        val = 1.0 if lc.result == you else (-1.0 if lc.result != 2 else 0.0)
                    else:
                        val = _eval_obs(ar.observation.__dict__ if hasattr(ar.observation, '__dict__') else {})
                except Exception:
                    val = 0.0

                chosen['visits'] = 1
                chosen['total'] = val
            else:
                chosen['visits'] += 1

        search_end()

        best = max(children, key=lambda c: c.get('visits', 0))
        if best is None or best['sel'] == [n]:
            return [], 0.0, 0.0

        # Get greedy logprob/value for PPO
        d2 = enc.encode(obs_dict)
        _, lp, v = self.act(d2.board_cards, d2.hand_cards, d2.state_feats,
                            d2.opt_type, d2.opt_card, d2.opt_card2,
                            d2.opt_attack, d2.opt_feats,
                            d2.min_count, d2.max_count, greedy=True)

        return best['sel'][:mc], lp, v

    def _greedy_fallback(self, obs_dict: dict):
        from .encoder import FastEncoder
        d = FastEncoder().encode(obs_dict)
        return self.act(d.board_cards, d.hand_cards, d.state_feats,
                        d.opt_type, d.opt_card, d.opt_card2,
                        d.opt_attack, d.opt_feats,
                        d.min_count, d.max_count, greedy=True)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ff1 = nn.Linear(dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(1)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(q.shape[-1], 1))
        scores = scores.masked_fill(~valid, NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        y = self.o(torch.matmul(attn, v))
        x = F.relu(x + y)
        y = self.ff2(F.relu(self.ff1(x)))
        x = F.relu(x + y)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class CrossAttentionPolicyValueNet(nn.Module):
    """Pointer policy with tokenized board/hand state and option-state attention.

    The public interface intentionally matches ``PolicyValueNet`` so BC losses,
    greedy decoding, PPO utilities, and diagnostics can share the same path.
    """

    def __init__(
        self,
        width: float = 1.0,
        option_context: bool = True,
        state_feat_dim: int = STATE_FEAT_DIM,
        opt_feat_dim: int = OPT_FEAT_DIM,
        plan_dim: int = 0,
        hierarchical_plan: bool = False,
        history_k: int = 0,
        state_layers: int = 2,
    ):
        super().__init__()
        ec = int(_EC * width); ea = int(_EA * width); eo_t = int(_EO * width)
        oe = int(_OE * width); hd = int(_HD * width)
        sc = int(_SC * width)
        ctx = int(_CTX * width); area = int(_AREA * width); idx = int(_IDX * width)
        self.option_context = option_context
        self.slot_state = True
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.plan_dim = int(plan_dim)
        self.hierarchical_plan = bool(hierarchical_plan and self.plan_dim > 0)
        self.history_k = max(0, int(history_k))
        self.state_layers_n = int(state_layers)
        self._ec=ec; self._ea=ea; self._eo_t=eo_t; self._oe=oe; self._hd=hd
        self._ctx=ctx; self._area=area; self._idx=idx
        self._plan_cd = sc if self.hierarchical_plan else 0
        self._hist = sc if self.history_k > 0 else 0
        self.register_buffer("arch_code", torch.tensor([1], dtype=torch.int32), persistent=True)

        self.card_emb = nn.Embedding(N_CARDS + 2, ec, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, ea, padding_idx=0)
        self.opt_type_emb = nn.Embedding(N_OPT_TYPES + 1, eo_t)
        if option_context:
            self.context_emb = nn.Embedding(65, ctx)
            self.select_type_emb = nn.Embedding(17, ctx)
            self.area_emb = nn.Embedding(17, area)
            self.index_emb = nn.Embedding(65, idx)
            self.inplay_area_emb = nn.Embedding(17, area)
            self.inplay_index_emb = nn.Embedding(17, idx)

        self.stop_vec = nn.Parameter(torch.zeros(oe))
        self.state_area_emb = nn.Embedding(8, area)
        self.state_index_emb = nn.Embedding(65, idx)
        self.state_token_fc = nn.Linear(ec + area + idx, oe)
        self.feat_token_fc = nn.Linear(self.state_feat_dim, oe)
        ff_dim = max(oe * 2, 64)
        self.state_layers = nn.ModuleList([
            _SelfAttentionBlock(oe, ff_dim) for _ in range(max(1, int(state_layers)))
        ])
        self.state_pool_fc = nn.Linear(oe, 1)
        self.state_out_fc = nn.Linear(oe, hd)
        if self.history_k > 0:
            self.history_type_emb = nn.Embedding(N_OPT_TYPES + 2, eo_t, padding_idx=0)
            self.history_context_emb = nn.Embedding(66, ctx, padding_idx=0)
            self.history_select_type_emb = nn.Embedding(18, ctx, padding_idx=0)
            self.history_pos_emb = nn.Embedding(self.history_k, ctx)
            hist_in = ec + ec + ea + eo_t + ctx + ctx + ctx + 2
            self.history_token_fc = nn.Linear(hist_in, sc)
            self.history_gru = nn.GRU(sc, sc, batch_first=True)
            self.history_out_fc = nn.Linear(hd + sc, hd)

        opt_extra = ctx + ctx + area + idx + area + idx if option_context else 0
        self.opt_fc = nn.Linear(ec + ec + ea + eo_t + opt_extra + self.opt_feat_dim, oe)
        self.cross_q = nn.Linear(oe, oe)
        self.cross_k = nn.Linear(oe, oe)
        self.cross_v = nn.Linear(oe, oe)
        self.cross_out = nn.Linear(oe + oe, oe)

        self.score_fc1 = nn.Linear(hd + oe + oe, sc)
        self.score_fc2 = nn.Linear(sc, 1)
        self.value_fc1 = nn.Linear(hd, sc)
        self.value_fc2 = nn.Linear(sc, 1)
        if self.plan_dim > 0:
            self.plan_fc1 = nn.Linear(hd, sc)
            self.plan_fc2 = nn.Linear(sc, self.plan_dim)
            if self.hierarchical_plan:
                self.plan_condition_fc = nn.Linear(self.plan_dim, self._plan_cd)
                self.plan_score_fc1 = nn.Linear(oe + oe + self._plan_cd, sc)
                self.plan_score_fc2 = nn.Linear(sc, 1)
        self._cached_state_tokens: torch.Tensor | None = None
        self._cached_state_mask: torch.Tensor | None = None
        self._init()
        if self.hierarchical_plan:
            nn.init.zeros_(self.plan_score_fc2.weight)
            nn.init.zeros_(self.plan_score_fc2.bias)

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    @staticmethod
    def _fit_feat_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur == dim:
            return x
        if cur > dim:
            return x[..., :dim]
        return F.pad(x, (0, dim - cur))

    def _state_area_index(self, board: torch.Tensor, hand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = board.shape[0]
        device = board.device
        board_area = torch.tensor([1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4], device=device)
        board_index = torch.tensor([0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4], device=device)
        hand_area = torch.full((hand.shape[1],), 5, dtype=torch.long, device=device)
        hand_index = torch.arange(hand.shape[1], dtype=torch.long, device=device).clamp(max=64)
        area = torch.cat([board_area, hand_area], dim=0).unsqueeze(0).expand(bsz, -1)
        index = torch.cat([board_index, hand_index], dim=0).unsqueeze(0).expand(bsz, -1)
        return area, index

    def _build_state_tokens(
        self,
        board: torch.Tensor,
        hand: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._fit_feat_dim(feats, self.state_feat_dim)
        ids = torch.cat([board, hand], dim=1)
        area, index = self._state_area_index(board, hand)
        card_tokens = F.relu(self.state_token_fc(torch.cat([
            self.card_emb(ids),
            self.state_area_emb(area),
            self.state_index_emb(index),
        ], dim=-1)))
        feat_token = F.relu(self.feat_token_fc(feats)).unsqueeze(1)
        tokens = torch.cat([feat_token, card_tokens], dim=1)
        mask = torch.cat([
            torch.ones(board.shape[0], 1, dtype=torch.bool, device=board.device),
            ids > 0,
        ], dim=1)
        for layer in self.state_layers:
            tokens = layer(tokens, mask)
        return tokens, mask

    def _encode_history(self, history: dict[str, torch.Tensor] | None, bsz: int,
                        device: torch.device) -> torch.Tensor:
        if self.history_k <= 0:
            return torch.zeros(bsz, 0, device=device)
        if not history or history.get("mask") is None or history["mask"].numel() == 0:
            return torch.zeros(bsz, self._hist, device=device)
        mask = history["mask"].to(device=device, dtype=torch.float32)
        if mask.shape[1] != self.history_k:
            mask = mask[:, -self.history_k:]
        k = mask.shape[1]
        pos = torch.arange(k, device=device).unsqueeze(0).expand(mask.shape[0], -1)
        parts = [
            self.card_emb(history["card"].to(device=device).long()[:, -k:]),
            self.card_emb(history["card2"].to(device=device).long()[:, -k:]),
            self.attack_emb(history["attack"].to(device=device).long()[:, -k:]),
            self.history_type_emb(history["type"].to(device=device).long()[:, -k:].clamp(0, N_OPT_TYPES + 1)),
            self.history_context_emb(history["context"].to(device=device).long()[:, -k:].clamp(0, 65)),
            self.history_select_type_emb(history["select_type"].to(device=device).long()[:, -k:].clamp(0, 17)),
            self.history_pos_emb(pos),
            history["count"].to(device=device, dtype=torch.float32)[:, -k:].unsqueeze(-1),
            mask.unsqueeze(-1),
        ]
        x = torch.cat(parts, dim=-1)
        token = F.relu(self.history_token_fc(x)) * mask.unsqueeze(-1)
        hidden = torch.zeros(1, mask.shape[0], self._hist, device=device, dtype=token.dtype)
        for t in range(k):
            _, next_hidden = self.history_gru(token[:, t:t + 1], hidden)
            valid = mask[:, t].view(1, -1, 1) > 0
            hidden = torch.where(valid, next_hidden, hidden)
        return hidden.squeeze(0)

    def _merge_history(self, h: torch.Tensor, history: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if self.history_k <= 0:
            return h
        hist = self._encode_history(history, h.shape[0], h.device)
        return F.relu(self.history_out_fc(torch.cat([h, hist], dim=-1)))

    def encode_state(self, board: torch.Tensor, hand: torch.Tensor,
                     feats: torch.Tensor, history: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        tokens, mask = self._build_state_tokens(board, hand, feats)
        pool_logits = self.state_pool_fc(tokens).squeeze(-1).masked_fill(~mask, NEG_INF)
        pool = torch.softmax(pool_logits, dim=-1).unsqueeze(-1)
        pooled = (tokens * pool).sum(dim=1)
        self._cached_state_tokens = tokens
        self._cached_state_mask = mask
        return self._merge_history(F.relu(self.state_out_fc(pooled)), history)

    def encode_options(self, opt_type: torch.Tensor, opt_card: torch.Tensor,
                       opt_card2: torch.Tensor, opt_attack: torch.Tensor,
                       opt_feats: torch.Tensor) -> torch.Tensor:
        opt_feats = self._fit_feat_dim(opt_feats, self.opt_feat_dim)
        parts = [
            self.card_emb(opt_card), self.card_emb(opt_card2),
            self.attack_emb(opt_attack), self.opt_type_emb(opt_type),
        ]
        if self.option_context:
            ctx = torch.round(opt_feats[..., 3] * 64.0).long().clamp(0, 64)
            sel_type = torch.round(opt_feats[..., 4] * 16.0).long().clamp(0, 16)
            area = torch.round(opt_feats[..., 7] * 16.0).long().clamp(0, 16)
            idx = torch.round(opt_feats[..., 8] * 64.0).long().clamp(0, 64)
            inplay_area = torch.round(opt_feats[..., 9] * 16.0).long().clamp(0, 16)
            inplay_idx = torch.round(opt_feats[..., 10] * 10.0).long().clamp(0, 16)
            parts.extend([
                self.context_emb(ctx), self.select_type_emb(sel_type),
                self.area_emb(area), self.index_emb(idx),
                self.inplay_area_emb(inplay_area), self.inplay_index_emb(inplay_idx),
            ])
        parts.append(opt_feats)
        base = F.relu(self.opt_fc(torch.cat(parts, dim=-1)))

        tokens = self._cached_state_tokens
        token_mask = self._cached_state_mask
        if tokens is None or token_mask is None or tokens.shape[0] != base.shape[0]:
            return base
        q = self.cross_q(base)
        k = self.cross_k(tokens)
        v = self.cross_v(tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(q.shape[-1], 1))
        scores = scores.masked_fill(~token_mask.unsqueeze(1), NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        return F.relu(self.cross_out(torch.cat([base, ctx], dim=-1)))

    def option_logits(self, h: torch.Tensor, opts: torch.Tensor,
                      picked_sum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = opts.shape
        stop = self.stop_vec.expand(B, 1, self._oe)
        rows = torch.cat([opts, stop], dim=1)
        hx = h.unsqueeze(1).expand(B, N + 1, self._hd)
        px = picked_sum.unsqueeze(1).expand(B, N + 1, self._oe)
        x = torch.cat([hx, rows, px], dim=-1)
        logits = self.score_fc2(F.relu(self.score_fc1(x))).squeeze(-1)
        if self.hierarchical_plan:
            plan_prob = torch.sigmoid(self.plan_logits(h))
            plan_ctx = F.relu(self.plan_condition_fc(plan_prob))
            plan_x = torch.cat([
                rows,
                px,
                plan_ctx.unsqueeze(1).expand(B, N + 1, self._plan_cd),
            ], dim=-1)
            logits = logits + self.plan_score_fc2(F.relu(self.plan_score_fc1(plan_x))).squeeze(-1)
        return logits.masked_fill(~mask, NEG_INF)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value_fc2(F.relu(self.value_fc1(h)))).squeeze(-1)

    def plan_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.plan_dim <= 0:
            raise RuntimeError("CrossAttentionPolicyValueNet was created without plan_dim")
        return self.plan_fc2(F.relu(self.plan_fc1(h)))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_policy_model(arch: str = ARCH_POINTER, **kwargs) -> nn.Module:
    arch = (arch or ARCH_POINTER).strip().lower()
    if arch in {ARCH_POINTER, "mlp"}:
        kwargs.pop("state_layers", None)
        return PolicyValueNet(**kwargs)
    if arch in {ARCH_CROSS_ATTN, "cross-attn", "attention"}:
        kwargs.pop("slot_state", None)
        return CrossAttentionPolicyValueNet(**kwargs)
    raise ValueError(f"unknown policy arch: {arch!r}")


def checkpoint_arch(files: list[str] | tuple[str, ...] | set[str]) -> str:
    keys = set(files)
    if "state_token_fc.weight" in keys or "cross_q.weight" in keys:
        return ARCH_CROSS_ATTN
    return ARCH_POINTER


def checkpoint_feature_dims(z) -> tuple[int, int, bool, bool]:
    arch = checkpoint_arch(z.files)
    option_context = "context_emb.weight" in z.files
    ec = z["card_emb.weight"].shape[1]
    if arch == ARCH_CROSS_ATTN:
        state_feat_dim = int(z["feat_token_fc.weight"].shape[1])
        slot_state = True
    else:
        state_in = z["state_fc1.weight"].shape[1]
        slot_feat_dim = state_in - 5 * ec
        legacy_feat_dim = state_in - 3 * ec
        slot_state = 8 <= slot_feat_dim <= 256
        state_feat_dim = slot_feat_dim if slot_state else legacy_feat_dim
    opt_extra = 0
    if option_context:
        opt_extra = (
            z["context_emb.weight"].shape[1]
            + z["select_type_emb.weight"].shape[1]
            + z["area_emb.weight"].shape[1]
            + z["index_emb.weight"].shape[1]
            + z["inplay_area_emb.weight"].shape[1]
            + z["inplay_index_emb.weight"].shape[1]
        )
    opt_feat_dim = z["opt_fc.weight"].shape[1] - (
        2 * ec
        + z["attack_emb.weight"].shape[1]
        + z["opt_type_emb.weight"].shape[1]
        + opt_extra
    )
    return int(state_feat_dim), int(opt_feat_dim), bool(option_context), bool(slot_state)


def checkpoint_width(z) -> float:
    """Infer model width from a checkpoint's embedding dimension."""
    return float(z["card_emb.weight"].shape[1]) / float(_EC)


def checkpoint_plan_dim(z) -> int:
    """Infer auxiliary/hierarchical plan target width from a checkpoint."""
    if "plan_fc2.weight" not in z:
        return 0
    return int(z["plan_fc2.weight"].shape[0])


def checkpoint_hierarchical_plan(z) -> bool:
    """Return whether a checkpoint conditions policy logits on predicted plan."""
    return "plan_condition_fc.weight" in z


def checkpoint_history_k(z) -> int:
    """Infer past-decision history length from a checkpoint."""
    if "history_pos_emb.weight" not in z:
        return 0
    return int(z["history_pos_emb.weight"].shape[0])
