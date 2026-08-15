from __future__ import annotations

import numpy as np
import torch
import warnings

from ptcg_rl.encoder import FastEncoder, OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.v15.constants import (
    DEFAULT_HISTORY_K,
    DEFAULT_MAX_OPTIONS,
    DEFAULT_PLAN_STEPS,
    KNOWN_OPP_CARDS,
    MAX_SELECT_COUNT,
    N_ACTION_TYPES,
)
from ptcg_rl.v15.data import V15Batch
from ptcg_rl.v15.events import V15Memory, pack_event_history
from ptcg_rl.v15.model import V15PlanPolicyNet
from ptcg_rl.v15.route import normalize_archetype, route_targets

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*", category=UserWarning)


def _fit_1d(value, n: int, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    out = np.zeros(n, dtype=dtype)
    m = min(n, arr.shape[0])
    if m:
        out[:m] = arr[:m]
    return out


def _fit_2d(value, n: int, d: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.zeros((n, d), dtype=np.float32)
    rows = min(n, arr.shape[0])
    cols = min(d, arr.shape[1] if arr.ndim >= 2 else 0)
    if rows and cols:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


class TorchV15Policy:
    """Live/local inference wrapper for v15 turn-block plan checkpoints."""

    def __init__(
        self,
        model: V15PlanPolicyNet,
        *,
        device: str = "cpu",
        model_config: dict | None = None,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        cfg = dict(model_config or {})
        self.state_feat_dim = int(cfg.get("state_feat_dim", STATE_FEAT_DIM))
        self.opt_feat_dim = int(cfg.get("opt_feat_dim", OPT_FEAT_DIM))
        self.state_token_feat_dim = int(cfg.get("state_token_feat_dim", STATE_TOKEN_FEAT_DIM))
        self.history_k = int(cfg.get("history_k", DEFAULT_HISTORY_K))
        self.plan_steps = int(cfg.get("plan_steps", DEFAULT_PLAN_STEPS))
        self.max_options = int(cfg.get("max_options", DEFAULT_MAX_OPTIONS))
        self.archetype = normalize_archetype(str(cfg.get("archetype", "") or cfg.get("_inferred_archetype", "")))
        self.route_prior_scale = float(cfg.get("route_prior_scale", 1.50 if self.archetype in {"dragapult", "alakazam"} else 0.0))
        self.has_trained_count = bool(cfg.get("_has_trained_count", True))
        self.encoder = FastEncoder()
        self.memory = V15Memory()
        self.decision_index = 0
        self._last_ranking: list[dict[str, float | int]] = []

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "TorchV15Policy":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        cfg = dict(ckpt.get("config") or ckpt.get("model_config") or {})
        name = str(path).lower()
        if not cfg.get("archetype"):
            if "dragapult" in name:
                cfg["_inferred_archetype"] = "dragapult"
            elif "alakazam" in name:
                cfg["_inferred_archetype"] = "alakazam"
        model_args = {
            k: cfg[k]
            for k in (
                "width",
                "layers",
                "heads",
                "dropout",
                "history_k",
                "plan_steps",
                "state_feat_dim",
                "opt_feat_dim",
                "state_token_feat_dim",
                "type_prior_scale",
                "history_type_prior_scale",
                "known_type_prior_scale",
            )
            if k in cfg
        }
        model = V15PlanPolicyNet(**model_args)
        state = ckpt.get("model") or ckpt.get("model_state")
        if state is None:
            raise ValueError(f"v15 checkpoint has no model state: {path}")
        cfg["_has_trained_count"] = "count_head.weight" in state
        model.load_state_dict(state, strict=False)
        return cls(model, device=device, model_config=cfg)

    def reset_history(self) -> None:
        self.memory = V15Memory()
        self.decision_index = 0
        self._last_ranking = []

    def select(
        self,
        obs_dict: dict,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        update_history: bool = True,
        **_: object,
    ) -> list[int]:
        self.memory.observe_logs(obs_dict, decision=self.decision_index)
        encoded = self.encoder.encode(obs_dict)
        n = len(encoded.opt_type)
        if n <= 0 or int(encoded.max_count) <= 0:
            return []
        batch = self._batch_from_encoded(obs_dict, encoded).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            action_logits = out["action_logits"][0, :n].detach().cpu().numpy()
            multi_logits = out["multi_logits"][0, :n].detach().cpu().numpy()
            count_logits = out.get("count_logits")
            pred_count = int(count_logits[0].detach().cpu().argmax().item()) if count_logits is not None and self.has_trained_count else -1
        route_multi, route_stage = route_targets(self.archetype, encoded.board_cards, encoded.opt_type, encoded.opt_card, encoded.opt_card2)
        route_multi = np.asarray(route_multi[:n], dtype=np.float32)
        if self.route_prior_scale and route_multi.sum() > 0:
            action_logits = action_logits + self.route_prior_scale * route_multi
        self._last_ranking = self._ranking_from_logits(encoded, action_logits)
        mn = int((obs_dict.get("select") or {}).get("minCount", encoded.min_count))
        mx = int((obs_dict.get("select") or {}).get("maxCount", encoded.max_count))
        force_pick = bool(route_stage and route_multi.sum() > 0)
        picks = self._choose(
            action_logits,
            multi_logits,
            mn=mn,
            mx=mx,
            greedy=greedy,
            temperature=temperature,
            pred_count=pred_count,
            force_pick=force_pick,
            route_multi=route_multi,
        )
        if update_history:
            self.remember_encoded(obs_dict, encoded, picks)
        return picks

    def first_step_ranking(self, obs_dict: dict, *, top_n: int = 0) -> list[dict[str, float | int]]:
        """Return current option ranking without mutating live memory.

        Trace tools call this after policy_action(), so the most accurate answer
        is the cached pre-action ranking from select().  If no cached ranking is
        available, compute one against the current memory snapshot.
        """
        if self._last_ranking:
            return self._last_ranking[:top_n] if top_n else list(self._last_ranking)
        encoded = self.encoder.encode(obs_dict)
        n = len(encoded.opt_type)
        if n <= 0:
            return []
        batch = self._batch_from_encoded(obs_dict, encoded).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)["action_logits"][0, :n].detach().cpu().numpy()
        route_multi, _ = route_targets(self.archetype, encoded.board_cards, encoded.opt_type, encoded.opt_card, encoded.opt_card2)
        route_multi = np.asarray(route_multi[:n], dtype=np.float32)
        if self.route_prior_scale and route_multi.sum() > 0:
            logits = logits + self.route_prior_scale * route_multi
        ranking = self._ranking_from_logits(encoded, logits)
        return ranking[:top_n] if top_n else ranking

    def select_mcts(
        self,
        obs_dict: dict,
        deck: list[int] | None = None,
        *,
        update_history: bool = True,
        **kwargs: object,
    ) -> list[int]:
        return self.select(obs_dict, greedy=True, update_history=update_history, **kwargs)

    def remember_decision(self, obs_dict: dict, picks: list[int]) -> None:
        self.memory.observe_logs(obs_dict, decision=self.decision_index)
        encoded = self.encoder.encode(obs_dict)
        self.remember_encoded(obs_dict, encoded, picks)

    def remember_encoded(self, obs_dict: dict, encoded, picks: list[int]) -> None:
        cur = (obs_dict.get("current") or {})
        turn = int(cur.get("turn", 0) or 0)
        self.memory.add_action(encoded, picks, turn=turn, decision=self.decision_index)
        self.decision_index += 1

    def _choose(
        self,
        action_logits: np.ndarray,
        multi_logits: np.ndarray,
        *,
        mn: int,
        mx: int,
        greedy: bool,
        temperature: float,
        pred_count: int = -1,
        force_pick: bool = False,
        route_multi: np.ndarray | None = None,
    ) -> list[int]:
        n = int(action_logits.shape[0])
        if n <= 0 or mx <= 0:
            return []
        mn = max(0, min(int(mn), n))
        mx = max(mn, min(int(mx), n))
        if not greedy:
            temp = max(float(temperature), 1e-3)
            scaled = action_logits / temp
            probs = np.exp(scaled - np.max(scaled))
            probs = probs / max(float(probs.sum()), 1e-9)
            k = mn if mx > 1 else max(mn, 1)
            k = min(max(k, 0), mx)
            return [int(x) for x in np.random.choice(np.arange(n), size=k, replace=False, p=probs)]
        order = [int(i) for i in np.argsort(-action_logits)]
        if mx <= 1:
            if mn == 0 and not force_pick:
                if pred_count >= 0:
                    return [order[0]] if pred_count >= 1 else []
                top = order[0]
                return [top] if float(multi_logits[top]) >= 0.0 else []
            if force_pick and route_multi is not None and route_multi.sum() > 0:
                for idx in order:
                    if float(route_multi[idx]) > 0:
                        return [idx]
            return [order[0]]
        if pred_count >= 0:
            k = max(mn, min(mx, pred_count))
            if force_pick:
                k = max(k, 1)
            chosen: list[int] = []
            if force_pick and route_multi is not None:
                for i in order:
                    if float(route_multi[i]) > 0:
                        chosen.append(i)
                        break
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= k:
                    break
            return chosen[:mx]
        chosen = []
        if force_pick and route_multi is not None:
            for i in order:
                if float(route_multi[i]) > 0:
                    chosen.append(i)
                    break
        for i in order:
            if i in chosen:
                continue
            if float(multi_logits[i]) >= 0.0:
                chosen.append(i)
            if len(chosen) >= mx:
                break
        if len(chosen) < mn:
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= mn:
                    break
        return chosen[:mx]

    def _ranking_from_logits(self, encoded, logits: np.ndarray) -> list[dict[str, float | int]]:
        n = int(len(encoded.opt_type))
        if n <= 0:
            return []
        raw = np.asarray(logits[:n], dtype=np.float64)
        probs = np.exp(raw - np.max(raw))
        probs = probs / max(float(probs.sum()), 1e-12)
        order = np.argsort(-raw)
        return [
            {
                "index": int(i),
                "logit": float(raw[i]),
                "prob": float(probs[i]),
                "type": int(encoded.opt_type[i]),
                "card": int(encoded.opt_card[i]),
                "card2": int(encoded.opt_card2[i]),
                "attack": int(encoded.opt_attack[i]),
            }
            for i in order
        ]

    def _batch_from_encoded(self, obs_dict: dict, encoded) -> V15Batch:
        cur = obs_dict.get("current") or {}
        turn = int(cur.get("turn", 0) or 0)
        events = pack_event_history(
            self.memory.events,
            k=self.history_k,
            current_turn=turn,
            current_decision=self.decision_index,
        )
        known_cards, known_counts, known_age, known_mask = self.memory.known_arrays(decision=self.decision_index)
        nopt = max(1, len(encoded.opt_type))
        opt_type = _fit_1d(encoded.opt_type, nopt, np.int64)
        opt_card = _fit_1d(encoded.opt_card, nopt, np.int64)
        opt_card2 = _fit_1d(encoded.opt_card2, nopt, np.int64)
        opt_attack = _fit_1d(encoded.opt_attack, nopt, np.int64)
        opt_feats = _fit_2d(encoded.opt_feats, nopt, self.opt_feat_dim)
        option_mask = np.zeros(nopt, dtype=np.float32)
        option_mask[: len(encoded.opt_type)] = 1.0
        route_multi, route_stage = route_targets(
            self.archetype,
            encoded.board_cards,
            encoded.opt_type,
            encoded.opt_card,
            encoded.opt_card2,
            max_options=nopt,
        )
        route_mask = 1.0 if np.asarray(route_multi).sum() > 0 else 0.0
        return V15Batch(
            board=torch.from_numpy(_fit_1d(encoded.board_cards, 12, np.int64)).unsqueeze(0),
            hand=torch.from_numpy(_fit_1d(encoded.hand_cards, 25, np.int64)).unsqueeze(0),
            feats=torch.from_numpy(_fit_1d(encoded.state_feats, self.state_feat_dim, np.float32)).unsqueeze(0),
            state_token_feats=torch.from_numpy(_fit_2d(encoded.state_token_feats, 37, self.state_token_feat_dim)).unsqueeze(0),
            event_type=torch.from_numpy(events["event_type"].astype(np.int64)).unsqueeze(0),
            event_source=torch.from_numpy(events["source"].astype(np.int64)).unsqueeze(0),
            event_owner=torch.from_numpy(events["owner"].astype(np.int64)).unsqueeze(0),
            event_card=torch.from_numpy(events["card"].astype(np.int64)).unsqueeze(0),
            event_card2=torch.from_numpy(events["card2"].astype(np.int64)).unsqueeze(0),
            event_attack=torch.from_numpy(events["attack"].astype(np.int64)).unsqueeze(0),
            event_context=torch.from_numpy(events["context"].astype(np.int64)).unsqueeze(0),
            event_select_type=torch.from_numpy(events["select_type"].astype(np.int64)).unsqueeze(0),
            event_from_area=torch.from_numpy(events["from_area"].astype(np.int64)).unsqueeze(0),
            event_to_area=torch.from_numpy(events["to_area"].astype(np.int64)).unsqueeze(0),
            event_value=torch.from_numpy(events["value"].astype(np.float32)).unsqueeze(0),
            event_turn_delta=torch.from_numpy(events["turn_delta"].astype(np.float32)).unsqueeze(0),
            event_step_delta=torch.from_numpy(events["step_delta"].astype(np.float32)).unsqueeze(0),
            event_same_turn=torch.from_numpy(events["same_turn"].astype(np.float32)).unsqueeze(0),
            event_mask=torch.from_numpy(events["mask"].astype(np.float32)).unsqueeze(0),
            known_cards=torch.from_numpy(known_cards.astype(np.int64)).unsqueeze(0),
            known_counts=torch.from_numpy(known_counts.astype(np.float32)).unsqueeze(0),
            known_age=torch.from_numpy(known_age.astype(np.float32)).unsqueeze(0),
            known_mask=torch.from_numpy(known_mask.astype(np.float32)).unsqueeze(0),
            opt_type=torch.from_numpy(opt_type).unsqueeze(0),
            opt_card=torch.from_numpy(opt_card).unsqueeze(0),
            opt_card2=torch.from_numpy(opt_card2).unsqueeze(0),
            opt_attack=torch.from_numpy(opt_attack).unsqueeze(0),
            opt_feats=torch.from_numpy(opt_feats).unsqueeze(0),
            option_mask=torch.from_numpy(option_mask).unsqueeze(0),
            target_first=torch.full((1,), -1, dtype=torch.long),
            target_multi=torch.zeros((1, nopt), dtype=torch.float32),
            target_order=torch.full((1, MAX_SELECT_COUNT), -1, dtype=torch.long),
            target_type=torch.zeros((1,), dtype=torch.long),
            target_context=torch.zeros((1,), dtype=torch.long),
            plan_mask=torch.zeros((1, self.plan_steps), dtype=torch.float32),
            plan_type=torch.zeros((1, self.plan_steps), dtype=torch.long),
            plan_card=torch.zeros((1, self.plan_steps), dtype=torch.long),
            plan_card2=torch.zeros((1, self.plan_steps), dtype=torch.long),
            plan_attack=torch.zeros((1, self.plan_steps), dtype=torch.long),
            plan_context=torch.zeros((1, self.plan_steps), dtype=torch.long),
            plan_mode=torch.zeros((1,), dtype=torch.long),
            turn_continue=torch.zeros((1,), dtype=torch.float32),
            turn_remaining=torch.zeros((1,), dtype=torch.float32),
            block_pos=torch.zeros((1,), dtype=torch.long),
            block_len=torch.zeros((1,), dtype=torch.float32),
            block_remaining=torch.zeros((1,), dtype=torch.float32),
            block_type_counts=torch.zeros((1, N_ACTION_TYPES), dtype=torch.float32),
            dca_mask=torch.zeros((1,), dtype=torch.float32),
            dca_group_unique=torch.zeros((1,), dtype=torch.float32),
            dca_group_focus=torch.zeros((1,), dtype=torch.float32),
            route_mask=torch.tensor([route_mask], dtype=torch.float32),
            route_stage=torch.tensor([int(route_stage)], dtype=torch.long),
            route_target_multi=torch.from_numpy(np.asarray(route_multi, dtype=np.float32)).unsqueeze(0),
            min_count=torch.tensor([int(encoded.min_count)], dtype=torch.long),
            max_count=torch.tensor([int(encoded.max_count)], dtype=torch.long),
            sample_weight=torch.ones((1,), dtype=torch.float32),
            won=torch.zeros((1,), dtype=torch.float32),
            game_keys=["live"],
        )
