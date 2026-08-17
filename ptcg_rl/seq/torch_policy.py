from __future__ import annotations

import numpy as np
import torch
import warnings

from ptcg_rl.encoder import FastEncoder, OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.seq.constants import FUTURE_PLAN_DIM, KNOWN_OPP_CARDS, LEDGER_FEAT_DIM, MAX_SELECT_COUNT, N_ACTION_TYPES, TURN_PLAN_STEPS
from ptcg_rl.seq.data import SequenceBatch
from ptcg_rl.seq.features import SequenceLedger
from ptcg_rl.seq.model import SequencePolicyNet

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*", category=UserWarning)


class TorchSequencePolicy:
    """Live/local inference wrapper for v14 sequence checkpoints."""

    def __init__(
        self,
        model: SequencePolicyNet,
        *,
        device: str = "cpu",
        seq_len: int = 32,
        model_config: dict | None = None,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.seq_len = int(seq_len)
        cfg = model_config or {}
        self.state_feat_dim = int(cfg.get("state_feat_dim", STATE_FEAT_DIM))
        self.opt_feat_dim = int(cfg.get("opt_feat_dim", OPT_FEAT_DIM))
        self.state_token_feat_dim = int(cfg.get("state_token_feat_dim", STATE_TOKEN_FEAT_DIM))
        self.ledger_feat_dim = int(cfg.get("ledger_feat_dim", LEDGER_FEAT_DIM))
        self.future_plan_dim = int(cfg.get("future_plan_dim", FUTURE_PLAN_DIM))
        self.encoder = FastEncoder()
        self.ledger = SequenceLedger()
        self.buffer: list[dict[str, np.ndarray | int | float]] = []

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "TorchSequencePolicy":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        model_config = dict(ckpt["model_config"])
        model = SequencePolicyNet(**model_config)
        # Training-only diagnostic heads may be added over time. They do not
        # participate in live action selection, so older checkpoints should
        # remain usable after such changes.
        model.load_state_dict(ckpt["model_state"], strict=False)
        seq_len = int(model_config.get("max_seq_len", 32))
        return cls(model, device=device, seq_len=seq_len, model_config=model_config)

    def reset_history(self) -> None:
        self.ledger.reset()
        self.buffer.clear()

    def select(
        self,
        obs_dict: dict,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        update_history: bool = True,
        **_: object,
    ) -> list[int]:
        self.ledger.observe_public_logs(obs_dict)
        encoded = self.encoder.encode(obs_dict)
        row = self._row_from_encoded(encoded)
        rows = (self.buffer + [row])[-self.seq_len:]
        batch = self._batch_from_rows(rows).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            logits = out["action_logits"][0, -1].detach().cpu().numpy()
            order_logits = out.get("order_logits")
            order_scores = (
                order_logits[0, -1].detach().cpu().numpy()
                if order_logits is not None else None
            )
            count_logits = out.get("count_logits")
            pred_count = 1
            if count_logits is not None:
                pred_count = int(count_logits[0, -1].detach().cpu().argmax().item())
        n = len(encoded.opt_type)
        sel = obs_dict.get("select") or {}
        mn = int(sel.get("minCount", encoded.min_count))
        mx = int(sel.get("maxCount", encoded.max_count))
        if n == 0 or mx <= 0:
            picks: list[int] = []
        elif greedy:
            k = max(mn, min(mx, pred_count))
            if k <= 0 and mn == 0:
                k = 0
            elif k <= 0:
                k = mn
            picks = []
            if order_scores is not None:
                for pos in range(min(k, order_scores.shape[0])):
                    for idx in np.argsort(-order_scores[pos, :n]):
                        idx = int(idx)
                        if idx not in picks:
                            picks.append(idx)
                            break
            if len(picks) < k:
                for idx in np.argsort(-logits[:n]):
                    idx = int(idx)
                    if idx not in picks:
                        picks.append(idx)
                    if len(picks) >= k:
                        break
        else:
            temp = max(float(temperature), 1e-3)
            scaled = logits[:n] / temp
            probs = np.exp(scaled - np.max(scaled))
            probs = probs / max(float(probs.sum()), 1e-9)
            k = max(mn, min(mx, pred_count))
            picks = list(np.random.choice(np.arange(n), size=k, replace=False, p=probs))
        picks = [int(x) for x in picks if 0 <= int(x) < n]
        picks = list(dict.fromkeys(picks))
        if len(picks) < mn:
            missing = [i for i in range(n) if i not in picks]
            picks.extend(missing[: mn - len(picks)])
        picks = picks[:mx]
        if update_history:
            self.remember_encoded(encoded, picks, row)
        return picks

    def remember_decision(self, obs_dict: dict, picks: list[int]) -> None:
        self.ledger.observe_public_logs(obs_dict)
        encoded = self.encoder.encode(obs_dict)
        self.remember_encoded(encoded, picks, self._row_from_encoded(encoded))

    def select_mcts(
        self,
        obs_dict: dict,
        deck: list[int] | None = None,
        *,
        sims: int = 48,
        time_budget: float = 4.0,
        update_history: bool = True,
        **kwargs: object,
    ) -> list[int]:
        """Compatibility shim for legacy eval/main MCTS switches.

        v14 checkpoints do not train a value head suitable for tree search.
        If an old script accidentally enables MCTS, using the sequence policy is
        safer than raising and silently falling back to random/legal actions.
        """
        return self.select(obs_dict, greedy=True, update_history=update_history, **kwargs)

    def remember_encoded(self, encoded, picks: list[int], row: dict[str, np.ndarray | int | float] | None = None) -> None:
        if row is None:
            row = self._row_from_encoded(encoded)
        self.ledger.update(encoded, picks)
        self.buffer.append(row)
        if len(self.buffer) > self.seq_len - 1:
            del self.buffer[: len(self.buffer) - (self.seq_len - 1)]

    def _row_from_encoded(self, encoded) -> dict[str, np.ndarray | int | float]:
        prev = self.ledger.last_event or {}
        known_cards, known_counts, known_mask = self.ledger.known_opp_arrays()
        return {
            "board": np.asarray(encoded.board_cards, dtype=np.int64),
            "hand": np.asarray(encoded.hand_cards, dtype=np.int64),
            "feats": np.asarray(encoded.state_feats, dtype=np.float32),
            "state_token_feats": np.asarray(encoded.state_token_feats, dtype=np.float32),
            "ledger_feats": self.ledger.features(encoded),
            "known_opp_cards": known_cards,
            "known_opp_counts": known_counts,
            "known_opp_mask": known_mask,
            "prev_type": int(prev.get("type", 0) or 0),
            "prev_card": int(prev.get("card", 0) or 0),
            "prev_card2": int(prev.get("card2", 0) or 0),
            "prev_attack": int(prev.get("attack", 0) or 0),
            "prev_context": int(prev.get("context", 0) or 0),
            "prev_select_type": int(prev.get("select_type", 0) or 0),
            "prev_count": float(prev.get("count", 0.0) or 0.0),
            "opt_type": np.asarray(encoded.opt_type, dtype=np.int64),
            "opt_card": np.asarray(encoded.opt_card, dtype=np.int64),
            "opt_card2": np.asarray(encoded.opt_card2, dtype=np.int64),
            "opt_attack": np.asarray(encoded.opt_attack, dtype=np.int64),
            "opt_feats": np.asarray(encoded.opt_feats, dtype=np.float32),
            "min_count": int(encoded.min_count),
            "max_count": int(encoded.max_count),
        }

    def _batch_from_rows(self, rows: list[dict[str, np.ndarray | int | float]]) -> SequenceBatch:
        seq_len = self.seq_len
        nopt = max(1, max(len(np.asarray(r["opt_type"]).reshape(-1)) for r in rows))
        offset = seq_len - len(rows)
        board = np.zeros((1, seq_len, 12), dtype=np.int64)
        hand = np.zeros((1, seq_len, 25), dtype=np.int64)
        feats = np.zeros((1, seq_len, self.state_feat_dim), dtype=np.float32)
        stf = np.zeros((1, seq_len, 37, self.state_token_feat_dim), dtype=np.float32)
        ledger = np.zeros((1, seq_len, self.ledger_feat_dim), dtype=np.float32)
        known_opp_cards = np.zeros((1, seq_len, KNOWN_OPP_CARDS), dtype=np.int64)
        known_opp_counts = np.zeros((1, seq_len, KNOWN_OPP_CARDS), dtype=np.float32)
        known_opp_mask = np.zeros((1, seq_len, KNOWN_OPP_CARDS), dtype=np.float32)
        prev_type = np.zeros((1, seq_len), dtype=np.int64)
        prev_card = np.zeros((1, seq_len), dtype=np.int64)
        prev_card2 = np.zeros((1, seq_len), dtype=np.int64)
        prev_attack = np.zeros((1, seq_len), dtype=np.int64)
        prev_context = np.zeros((1, seq_len), dtype=np.int64)
        prev_select_type = np.zeros((1, seq_len), dtype=np.int64)
        prev_count = np.zeros((1, seq_len), dtype=np.float32)
        opt_type = np.zeros((1, seq_len, nopt), dtype=np.int64)
        opt_card = np.zeros((1, seq_len, nopt), dtype=np.int64)
        opt_card2 = np.zeros((1, seq_len, nopt), dtype=np.int64)
        opt_attack = np.zeros((1, seq_len, nopt), dtype=np.int64)
        opt_feats = np.zeros((1, seq_len, nopt, self.opt_feat_dim), dtype=np.float32)
        option_mask = np.zeros((1, seq_len, nopt), dtype=np.float32)
        step_mask = np.zeros((1, seq_len), dtype=np.float32)
        for i, row in enumerate(rows, offset):
            board[0, i] = _fit_1d(row["board"], 12, np.int64)
            hand[0, i] = _fit_1d(row["hand"], 25, np.int64)
            feats[0, i] = _fit_1d(row["feats"], self.state_feat_dim, np.float32)
            stf[0, i] = _fit_2d(row["state_token_feats"], 37, self.state_token_feat_dim)
            ledger[0, i] = _fit_1d(row["ledger_feats"], self.ledger_feat_dim, np.float32)
            known_opp_cards[0, i] = _fit_1d(row.get("known_opp_cards", []), KNOWN_OPP_CARDS, np.int64)
            known_opp_counts[0, i] = _fit_1d(row.get("known_opp_counts", []), KNOWN_OPP_CARDS, np.float32)
            known_opp_mask[0, i] = _fit_1d(row.get("known_opp_mask", []), KNOWN_OPP_CARDS, np.float32)
            prev_type[0, i] = int(row["prev_type"])
            prev_card[0, i] = int(row["prev_card"])
            prev_card2[0, i] = int(row["prev_card2"])
            prev_attack[0, i] = int(row["prev_attack"])
            prev_context[0, i] = int(row["prev_context"])
            prev_select_type[0, i] = int(row["prev_select_type"])
            prev_count[0, i] = float(row["prev_count"])
            ot = np.asarray(row["opt_type"], dtype=np.int64).reshape(-1)
            n = len(ot)
            opt_type[0, i, :n] = ot
            opt_card[0, i, :n] = _fit_1d(row["opt_card"], n, np.int64)
            opt_card2[0, i, :n] = _fit_1d(row["opt_card2"], n, np.int64)
            opt_attack[0, i, :n] = _fit_1d(row["opt_attack"], n, np.int64)
            opt_feats[0, i, :n] = _fit_2d(row["opt_feats"], n, self.opt_feat_dim)
            option_mask[0, i, :n] = 1.0
            step_mask[0, i] = 1.0
        dummy_first = np.full((1, seq_len), -1, dtype=np.int64)
        dummy_order = np.full((1, seq_len, MAX_SELECT_COUNT), -1, dtype=np.int64)
        dummy_multi = np.zeros((1, seq_len, nopt), dtype=np.float32)
        dummy_dca = np.zeros((1, seq_len), dtype=np.int64)
        dummy_dca_neg = np.full((1, seq_len), -1, dtype=np.int64)
        dummy_dca_float = np.zeros((1, seq_len), dtype=np.float32)
        dummy_turn_types = np.zeros((1, seq_len, N_ACTION_TYPES), dtype=np.float32)
        return SequenceBatch(
            board=torch.from_numpy(board),
            hand=torch.from_numpy(hand),
            feats=torch.from_numpy(feats),
            state_token_feats=torch.from_numpy(stf),
            ledger_feats=torch.from_numpy(ledger),
            known_opp_cards=torch.from_numpy(known_opp_cards),
            known_opp_counts=torch.from_numpy(known_opp_counts),
            known_opp_mask=torch.from_numpy(known_opp_mask),
            prev_type=torch.from_numpy(prev_type),
            prev_card=torch.from_numpy(prev_card),
            prev_card2=torch.from_numpy(prev_card2),
            prev_attack=torch.from_numpy(prev_attack),
            prev_context=torch.from_numpy(prev_context),
            prev_select_type=torch.from_numpy(prev_select_type),
            prev_count=torch.from_numpy(prev_count),
            opt_type=torch.from_numpy(opt_type),
            opt_card=torch.from_numpy(opt_card),
            opt_card2=torch.from_numpy(opt_card2),
            opt_attack=torch.from_numpy(opt_attack),
            opt_feats=torch.from_numpy(opt_feats),
            option_mask=torch.from_numpy(option_mask),
            target_first=torch.from_numpy(dummy_first),
            target_order=torch.from_numpy(dummy_order),
            target_multi=torch.from_numpy(dummy_multi),
            target_type=torch.zeros((1, seq_len), dtype=torch.int64),
            target_context=torch.zeros((1, seq_len), dtype=torch.int64),
            dca_group_index=torch.from_numpy(dummy_dca_neg),
            dca_pos=torch.from_numpy(dummy_dca_neg),
            dca_len=torch.from_numpy(dummy_dca),
            dca_remaining=torch.from_numpy(dummy_dca),
            dca_prior_same_slot=torch.from_numpy(dummy_dca),
            dca_prior_unique_slots=torch.from_numpy(dummy_dca),
            dca_prior_max_repeat=torch.from_numpy(dummy_dca),
            dca_group_unique_slots=torch.from_numpy(dummy_dca),
            dca_group_focus_frac=torch.from_numpy(dummy_dca_float),
            turn_continue=torch.zeros((1, seq_len), dtype=torch.float32),
            turn_remaining=torch.zeros((1, seq_len), dtype=torch.int64),
            turn_future_types=torch.from_numpy(dummy_turn_types),
            turn_next_exists=torch.zeros((1, seq_len), dtype=torch.float32),
            turn_next_type=torch.full((1, seq_len), N_ACTION_TYPES, dtype=torch.int64),
            turn_next_card=torch.zeros((1, seq_len), dtype=torch.int64),
            turn_next_card2=torch.zeros((1, seq_len), dtype=torch.int64),
            turn_next_attack=torch.zeros((1, seq_len), dtype=torch.int64),
            turn_next_context=torch.zeros((1, seq_len), dtype=torch.int64),
            turn_plan_mask=torch.zeros((1, seq_len, TURN_PLAN_STEPS), dtype=torch.float32),
            turn_plan_types=torch.full((1, seq_len, TURN_PLAN_STEPS), N_ACTION_TYPES, dtype=torch.int64),
            turn_plan_cards=torch.zeros((1, seq_len, TURN_PLAN_STEPS), dtype=torch.int64),
            turn_plan_attacks=torch.zeros((1, seq_len, TURN_PLAN_STEPS), dtype=torch.int64),
            turn_plan_contexts=torch.zeros((1, seq_len, TURN_PLAN_STEPS), dtype=torch.int64),
            min_count=torch.zeros((1, seq_len), dtype=torch.int64),
            max_count=torch.ones((1, seq_len), dtype=torch.int64),
            step_mask=torch.from_numpy(step_mask),
            sample_weight=torch.ones((1, seq_len), dtype=torch.float32),
            future_plan=torch.zeros((1, seq_len, self.future_plan_dim), dtype=torch.float32),
            outcome=torch.zeros((1, seq_len), dtype=torch.float32),
            game_keys=["live"],
            row_refs=[[]],
        )


def _fit_1d(value, dim: int, dtype) -> np.ndarray:
    out = np.zeros(dim, dtype=dtype)
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    n = min(dim, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _fit_2d(value, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        r = min(rows, arr.shape[0])
        c = min(cols, arr.shape[1])
        out[:r, :c] = arr[:r, :c]
    return out
