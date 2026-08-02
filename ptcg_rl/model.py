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


class PolicyValueNet(nn.Module):
    def __init__(self, width: float = 1.0):
        """width=1.0→501K, 2.0→4M, 3.0→9M params."""
        super().__init__()
        ec = int(_EC * width); ea = int(_EA * width); eo_t = int(_EO * width)
        oe = int(_OE * width); hd = int(_HD * width)
        s1 = int(_S1 * width); sc = int(_SC * width)
        self._ec=ec; self._ea=ea; self._eo_t=eo_t; self._oe=oe; self._hd=hd

        self.card_emb = nn.Embedding(N_CARDS + 2, ec, padding_idx=0)
        self.attack_emb = nn.Embedding(N_ATTACKS + 1, ea, padding_idx=0)
        self.opt_type_emb = nn.Embedding(N_OPT_TYPES + 1, eo_t)
        self.stop_vec = nn.Parameter(torch.zeros(oe))

        self.state_fc1 = nn.Linear(2 * ec + ec + STATE_FEAT_DIM, s1)
        self.state_fc2 = nn.Linear(s1, hd)
        self.opt_fc = nn.Linear(ec + ec + ea + eo_t + OPT_FEAT_DIM, oe)
        self.score_fc1 = nn.Linear(hd + oe + oe, sc)
        self.score_fc2 = nn.Linear(sc, 1)
        self.value_fc1 = nn.Linear(hd, sc)
        self.value_fc2 = nn.Linear(sc, 1)
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
        stop = self.stop_vec.expand(B, 1, self._oe)
        rows = torch.cat([opts, stop], dim=1)
        hx = h.unsqueeze(1).expand(B, N + 1, self._hd)
        px = picked_sum.unsqueeze(1).expand(B, N + 1, self._oe)
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
        ps = torch.zeros(B, self._oe, device=dev)
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
