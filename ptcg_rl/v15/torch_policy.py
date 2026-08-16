from __future__ import annotations

import os
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
    TYPE_ABILITY,
    TYPE_ATTACH,
    TYPE_CARD,
    TYPE_END,
    TYPE_ENERGY,
    TYPE_EVOLVE,
    TYPE_PLAY,
    TYPE_RETREAT,
    TYPE_ATTACK,
)
from ptcg_rl.v15.data import V15Batch
from ptcg_rl.v15.events import V15Memory, pack_event_history
from ptcg_rl.v15.model import V15PlanPolicyNet
from ptcg_rl.v15.route import normalize_archetype, route_context_allowed, route_targets

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
        self._last_rule_hits: list[str] = []

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
        sel = obs_dict.get("select") or {}
        select_context = int(sel.get("context", -1) if sel.get("context") is not None else -1)
        route_multi, route_stage = route_targets(
            self.archetype,
            encoded.board_cards,
            encoded.opt_type,
            encoded.opt_card,
            encoded.opt_card2,
            select_context=select_context,
        )
        route_multi = np.asarray(route_multi[:n], dtype=np.float32)
        if self.route_prior_scale and route_multi.sum() > 0:
            action_logits = action_logits + self.route_prior_scale * route_multi
        action_logits = self._apply_tactical_priors(obs_dict, encoded, action_logits)
        self._last_ranking = self._ranking_from_logits(encoded, action_logits)
        mn = int(sel.get("minCount", encoded.min_count))
        mx = int(sel.get("maxCount", encoded.max_count))
        force_pick = bool(route_context_allowed(select_context) and route_stage and route_multi.sum() > 0)
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
        sel = obs_dict.get("select") or {}
        select_context = int(sel.get("context", -1) if sel.get("context") is not None else -1)
        route_multi, _ = route_targets(
            self.archetype,
            encoded.board_cards,
            encoded.opt_type,
            encoded.opt_card,
            encoded.opt_card2,
            select_context=select_context,
        )
        route_multi = np.asarray(route_multi[:n], dtype=np.float32)
        if self.route_prior_scale and route_multi.sum() > 0:
            logits = logits + self.route_prior_scale * route_multi
        logits = self._apply_tactical_priors(obs_dict, encoded, logits)
        ranking = self._ranking_from_logits(encoded, logits)
        return ranking[:top_n] if top_n else ranking

    def last_rule_hits(self) -> list[str]:
        return list(self._last_rule_hits)

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
            # Route targets are already added to action_logits.  Do not hard
            # override a single-choice decision here: tactical priors may have
            # intentionally moved a non-route option, such as ATTACK in a
            # closing game, above a generic route target like RETREAT.
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

    def _apply_tactical_priors(self, obs_dict: dict, encoded, logits: np.ndarray) -> np.ndarray:
        """Hard safety priors for failures that pure one-step BC repeats.

        The first v15 Dragapult random loss showed a clear endgame deck-out:
        already far ahead on prizes, the policy kept choosing Drakloak/Poke Pad
        resource loops until its own deck hit zero.  These priors are deliberately
        narrow and state-based.  They do not replace normal setup; they only
        activate when the game is already in a low-deck / closing phase.
        """
        n = min(int(len(encoded.opt_type)), int(logits.shape[0]))
        self._last_rule_hits = []
        if n <= 0:
            return logits
        experimental_rules = {
            "support_retreat",
            "active_line_attach",
            "backup_before_attack",
            "backup_energy_attach",
            "dragapult_backup_line_energy",
            "dragapult_crispin_readiness",
            "drakloak_ko",
            "drakloak_pressure_attack",
            "alakazam_evolve_pressure",
            "dragapult_pivot_ready",
            "alakazam_fast_stage2",
            "alakazam_active_line_energy",
            "alakazam_attack_pressure",
            "dragapult_vs_alakazam_plan",
            "alakazam_vs_dragapult_plan",
            "dragapult_vs_lopunny_plan",
            "dragapult_vs_crustle_plan",
            "dragapult_blaziken_crustle_plan",
        }
        enabled_rules = {
            x.strip().lower()
            for x in os.environ.get("V15_ENABLE_RULES", "").replace(";", ",").split(",")
            if x.strip()
        }
        disabled_rules = set(experimental_rules)
        if "all" in enabled_rules:
            disabled_rules.clear()
        else:
            disabled_rules.difference_update(enabled_rules)
        disabled_rules.update({
            x.strip().lower()
            for x in os.environ.get("V15_DISABLE_RULES", "").replace(";", ",").split(",")
            if x.strip()
        })

        def rule_enabled(name: str) -> bool:
            return name not in disabled_rules and "all" not in disabled_rules

        def hit(name: str, detail: str = "") -> None:
            self._last_rule_hits.append(f"{name}:{detail}" if detail else name)

        out = np.asarray(logits, dtype=np.float32).copy()
        cur = obs_dict.get("current") or {}
        players = cur.get("players") or []
        you = int(cur.get("yourIndex", 0) or 0)
        if not (0 <= you < len(players)):
            return out
        me = players[you] or {}
        opp = players[1 - you] if 0 <= 1 - you < len(players) else {}
        my_active_obj = (me.get("active") or [None])[0] if me.get("active") else None
        active_hp = 0
        active_max_hp = 0
        if my_active_obj:
            active_hp = int(my_active_obj.get("hp", 0) or 0)
            active_max_hp = int(my_active_obj.get("maxHp", 0) or my_active_obj.get("maxHP", 0) or 0)
        my_deck = int(me.get("deckCount", 60) or 0)
        my_prizes = len(me.get("prize") or [])
        opp_prizes = len((opp or {}).get("prize") or [])
        prize_lead = max(0, opp_prizes - my_prizes)
        prize_deficit = max(0, my_prizes - opp_prizes)
        opt_type = np.asarray(encoded.opt_type[:n], dtype=np.int64)
        opt_card = np.asarray(encoded.opt_card[:n], dtype=np.int64)
        opt_card2 = np.asarray(encoded.opt_card2[:n], dtype=np.int64)
        opt_attack = np.asarray(encoded.opt_attack[:n], dtype=np.int64)
        opt_feats = np.asarray(encoded.opt_feats[:n], dtype=np.float32)
        sel = obs_dict.get("select") or {}
        raw_options = sel.get("option") or sel.get("options") or []
        mn = int(sel.get("minCount", 0) or 0)
        mx = int(sel.get("maxCount", 0) or 0)
        select_context = int(sel.get("context", -1) if sel.get("context") is not None else -1)
        has_attack = bool(np.any(opt_type == TYPE_ATTACK))
        has_attach = bool(np.any(opt_type == TYPE_ATTACH))
        has_retreat = bool(np.any(opt_type == TYPE_RETREAT))
        primary_ids = {121, 245, 345, 381, 431, 648, 678, 743, 849}
        setup_ids = {119, 120, 235, 112, 140, 741, 742, 1071}
        dragapult_line_ids = {119, 120, 121}
        alakazam_line_ids = {741, 742, 743, 245}
        support_ids = {112, 140, 235, 1071}
        active_id = int(encoded.board_cards[0]) if encoded.board_cards.size else 0
        active_is_primary = active_id in primary_ids
        is_dragapult = self.archetype == "dragapult"
        is_alakazam = self.archetype == "alakazam"
        active_is_dragapult_line = active_id in dragapult_line_ids
        active_is_alakazam_line = active_id in alakazam_line_ids
        has_primary_on_bench = False
        has_stage2_on_bench = False
        bench_attack_ready_dragapult = False
        my_board_ids: list[int] = []
        opp_board_ids: list[int] = []
        def energy_ids(pokemon: dict | None) -> set[int]:
            ids: set[int] = set()
            if not pokemon:
                return ids
            for card in pokemon.get("energyCards") or []:
                try:
                    ids.add(int(card.get("id", 0) or 0))
                except Exception:
                    pass
            for eid in pokemon.get("energies") or []:
                try:
                    ids.add(int(eid or 0))
                except Exception:
                    pass
            return ids

        def energy_id_list(pokemon: dict | None) -> list[int]:
            ids: list[int] = []
            if not pokemon:
                return ids
            for card in pokemon.get("energyCards") or []:
                try:
                    eid = int(card.get("id", 0) or 0)
                except Exception:
                    eid = 0
                if eid > 0:
                    ids.append(eid)
            for eid_raw in pokemon.get("energies") or []:
                try:
                    eid = int(eid_raw or 0)
                except Exception:
                    eid = 0
                if eid > 0:
                    ids.append(eid)
            return ids

        def pokemon_id(pokemon: dict | None) -> int:
            if not pokemon:
                return 0
            try:
                return int(pokemon.get("id", 0) or 0)
            except Exception:
                return 0

        def pokemon_hp(pokemon: dict | None) -> tuple[int, int, int]:
            """Return current hp, max hp, and visible damage for an in-play card."""
            if not pokemon:
                return 0, 0, 0
            try:
                hp = int(pokemon.get("hp", 0) or 0)
            except Exception:
                hp = 0
            try:
                max_hp = int(pokemon.get("maxHp", 0) or pokemon.get("maxHP", 0) or 0)
            except Exception:
                max_hp = 0
            try:
                dmg = int(pokemon.get("damage", 0) or pokemon.get("dmg", 0) or 0)
            except Exception:
                dmg = 0
            if dmg <= 0 and hp > 0 and max_hp > 0 and hp < max_hp:
                dmg = max_hp - hp
            return hp, max_hp, dmg

        def option_target_info(i: int) -> tuple[dict | None, int, int, int]:
            if not (0 <= i < len(raw_options)):
                return None, you, -1, -1
            raw = raw_options[i]
            if not isinstance(raw, dict):
                return None, you, -1, -1
            try:
                pid = int(raw.get("playerIndex", you) if raw.get("playerIndex") is not None else you)
            except Exception:
                pid = you
            area = raw.get("inPlayArea", raw.get("area"))
            idx = raw.get("inPlayIndex", raw.get("index", 0))
            try:
                area_i = int(area or 0)
                idx_i = int(idx or 0)
            except Exception:
                return None, pid, -1, -1
            if not (0 <= pid < len(players)):
                return None, pid, area_i, idx_i
            player = players[pid] or {}
            if area_i == 4:
                active = player.get("active") or []
                return (active[idx_i] if 0 <= idx_i < len(active) else None), pid, area_i, idx_i
            if area_i == 5:
                bench = player.get("bench") or []
                return (bench[idx_i] if 0 <= idx_i < len(bench) else None), pid, area_i, idx_i
            return None, pid, area_i, idx_i

        def option_target_pokemon(i: int) -> dict | None:
            target, _, _, _ = option_target_info(i)
            return target

        def option_target_id(i: int) -> int:
            target = option_target_pokemon(i)
            if target:
                return pokemon_id(target)
            return int(opt_card2[i]) if 0 <= i < len(opt_card2) else 0

        phantom_energy_ids = {2, 5}  # Basic R + Basic P for Dragapult ex's Phantom Dive.

        def phantom_ready_energy(pokemon: dict | None) -> bool:
            return phantom_energy_ids.issubset(energy_ids(pokemon))

        def option_makes_phantom_ready(i: int) -> bool:
            if int(opt_type[i]) != TYPE_ATTACH:
                return False
            target = option_target_pokemon(i)
            target_id = option_target_id(i)
            if target_id not in dragapult_line_ids:
                return False
            energies = energy_ids(target)
            card = int(opt_card[i])
            if card > 0:
                energies.add(card)
            return phantom_energy_ids.issubset(energies)

        if encoded.board_cards.size >= 6:
            my_board_ids = [int(x) for x in np.asarray(encoded.board_cards[:6], dtype=np.int64).tolist()]
            bench_ids = my_board_ids[1:6]
            has_primary_on_bench = any(cid in primary_ids for cid in bench_ids)
            has_stage2_on_bench = any(0 <= cid < len(self.encoder.card_stage) and int(self.encoder.card_stage[cid]) >= 2 for cid in bench_ids)
            for poke in me.get("bench") or []:
                if not poke or int(poke.get("id", 0) or 0) != 121:
                    continue
                if phantom_ready_energy(poke):
                    bench_attack_ready_dragapult = True
                    break
        if encoded.board_cards.size >= 12:
            opp_board_ids = [int(x) for x in np.asarray(encoded.board_cards[6:12], dtype=np.int64).tolist()]
        opp_has_alakazam_line = any(cid in {741, 742, 743} for cid in opp_board_ids)
        opp_has_dragapult_line = any(cid in dragapult_line_ids for cid in opp_board_ids)
        opp_has_lopunny_line = any(cid in {174, 848, 849, 860, 861} for cid in opp_board_ids)
        opp_has_crustle_wall_line = any(cid in {344, 345, 756} for cid in opp_board_ids)
        opp_active_id = int(opp_board_ids[0]) if opp_board_ids else 0
        opp_bench_ids = opp_board_ids[1:] if len(opp_board_ids) > 1 else []
        opp_bench_has_dwebble = any(cid == 344 for cid in opp_bench_ids)
        opp_bench_has_crustle = any(cid == 345 for cid in opp_bench_ids)
        opp_bench_has_wall_target = bool(opp_bench_has_dwebble or opp_bench_has_crustle)
        stadium_cards = cur.get("stadium") or []
        stadium_id = 0
        if isinstance(stadium_cards, list) and stadium_cards:
            try:
                stadium_id = int((stadium_cards[0] or {}).get("id", 0) or 0)
            except Exception:
                stadium_id = 0
        battle_cage_active = stadium_id == 1264
        active_energy_ids = energy_ids(my_active_obj)
        active_phantom_energy_ready = bool(active_id in {120, 121} and phantom_ready_energy(my_active_obj))
        bench_phantom_energy_line_ready = any(
            bool(poke)
            and int(poke.get("id", 0) or 0) in {120, 121}
            and phantom_ready_energy(poke)
            for poke in me.get("bench") or []
        )
        dragapult_phantom_energy_line_ready = bool(active_phantom_energy_ready or bench_phantom_energy_line_ready)
        dragapult_stage2_attack_ready_on_board = bool(
            (active_id == 121 and phantom_ready_energy(my_active_obj))
            or any(
                bool(poke)
                and int(poke.get("id", 0) or 0) == 121
                and phantom_ready_energy(poke)
                for poke in me.get("bench") or []
            )
        )
        hand_ids = [int(x) for x in np.asarray(encoded.hand_cards, dtype=np.int64).tolist() if int(x) > 0]
        my_in_play = (me.get("active") or []) + (me.get("bench") or [])
        opp_in_play = ((opp or {}).get("active") or []) + ((opp or {}).get("bench") or [])
        my_munkidori_on_board = any(pokemon_id(poke) == 112 for poke in my_in_play)
        my_munkidori_has_dark = any(pokemon_id(poke) == 112 and 7 in energy_ids(poke) for poke in my_in_play)
        my_munkidori_has_psychic = any(pokemon_id(poke) == 112 and 5 in energy_ids(poke) for poke in my_in_play)
        my_munkidori_needs_dark = bool(my_munkidori_on_board and not my_munkidori_has_dark)
        def munkidori_mind_bend_ready(poke: dict | None) -> bool:
            eids = energy_ids(poke)
            return bool(pokemon_id(poke) == 112 and 5 in eids and len(eids) >= 2)

        bench_munkidori_attack_ready = any(
            munkidori_mind_bend_ready(poke) for poke in (me.get("bench") or [])
        )
        any_munkidori_attack_ready = any(munkidori_mind_bend_ready(poke) for poke in my_in_play)
        my_damaged_any = any(pokemon_hp(poke)[2] > 0 for poke in my_in_play)
        my_damaged_dragapult = any(pokemon_id(poke) == 121 and pokemon_hp(poke)[2] > 0 for poke in my_in_play)
        opp_wall_damage = max(
            [pokemon_hp(poke)[2] for poke in opp_in_play if pokemon_id(poke) in {344, 345}] or [0]
        )
        opp_active_obj = ((opp or {}).get("active") or [None])[0] if (opp or {}).get("active") else None
        opp_active_hp, opp_active_max_hp, opp_active_damage = pokemon_hp(opp_active_obj)
        def dca_protected(poke: dict | None) -> bool:
            eids = energy_ids(poke)
            # Engine check: Mist Energy prevents attack effects such as
            # Phantom Dive's damage counters.  Rock Fighting Energy only
            # protects Fighting Pokemon; the current Crustle/Dwebble ids are
            # Grass-type in this engine, so Rock does not protect them.
            return bool(11 in eids)

        def crustle_attack_ready(poke: dict | None) -> bool:
            if pokemon_id(poke) != 345:
                return False
            eids = energy_id_list(poke)
            # AllAttack()[478] / attackId 479 = Superb Scissors, cost {G}{C}{C}.
            # Basic Grass and Grow Grass provide the required typed energy; any
            # attached energy can cover the colorless part.
            has_grass = any(eid in {1, 18, 10} for eid in eids)
            return bool(has_grass and len(eids) >= 3)

        opp_active_energy_list = energy_id_list(opp_active_obj)
        opp_active_crustle_can_attack = crustle_attack_ready(opp_active_obj)
        opp_active_crustle_energy_count = len(opp_active_energy_list)
        opp_active_crustle_has_grass = any(eid in {1, 18, 10} for eid in opp_active_energy_list)
        opp_active_crustle_has_mist = 11 in opp_active_energy_list

        opp_bench_has_valid_dca_target = any(
            bool(poke)
            and pokemon_hp(poke)[0] > 0
            and not dca_protected(poke)
            for poke in (opp.get("bench") or [])
        )
        my_dragapult_line_count = sum(1 for cid in my_board_ids if cid in dragapult_line_ids)
        my_dreepy_count = sum(1 for cid in my_board_ids if cid == 119)
        my_drakloak_count = sum(1 for cid in my_board_ids if cid == 120)
        my_dragapult_count = sum(1 for cid in my_board_ids if cid == 121)
        bench_line_count = 0
        if encoded.board_cards.size >= 6:
            bench_line_count = sum(
                1
                for c in np.asarray(encoded.board_cards[1:6], dtype=np.int64).tolist()
                if int(c) in dragapult_line_ids
            )
        my_alakazam_line_count = sum(1 for cid in my_board_ids if cid in {741, 742, 743})
        my_abra_count = sum(1 for cid in my_board_ids if cid == 741)
        my_kadabra_count = sum(1 for cid in my_board_ids if cid == 742)
        my_alakazam_count = sum(1 for cid in my_board_ids if cid == 743)
        opp_dreepy_count = sum(1 for cid in opp_board_ids if cid == 119)
        opp_drakloak_count = sum(1 for cid in opp_board_ids if cid == 120)
        opp_dragapult_count = sum(1 for cid in opp_board_ids if cid == 121)
        opp_abra_count = sum(1 for cid in opp_board_ids if cid == 741)
        opp_kadabra_count = sum(1 for cid in opp_board_ids if cid == 742)
        opp_alakazam_count = sum(1 for cid in opp_board_ids if cid == 743)
        opp_hand_count = int(opp.get("handCount", 0) or 0)
        if opp_hand_count <= 0:
            opp_hand_count = len(opp.get("hand") or [])
        opp_alakazam_energy_count = 0
        for poke in ((opp.get("active") or []) + (opp.get("bench") or [])):
            if not poke or int(poke.get("id", 0) or 0) not in {741, 742, 743}:
                continue
            opp_alakazam_energy_count += len(energy_ids(poke))
        has_phantom_dive = bool(np.any((opt_type == TYPE_ATTACK) & (opt_attack == 154)))
        has_jet_headbutt = bool(np.any((opt_type == TYPE_ATTACK) & (opt_attack == 153)))
        has_powerful_hand = bool(np.any((opt_type == TYPE_ATTACK) & (opt_attack == 1072)))
        opp_active_nonwall_phantom_ko = bool(
            opp_active_id not in {344, 345}
            and has_phantom_dive
            and 0 < opp_active_hp <= 200
        )
        low_deck_attack_available = bool(
            select_context == 0
            and my_deck <= 8
            and has_attack
        )
        low_deck_danger_cards = {1080, 1086, 1121, 1152, 1213, 1227, 1231}
        low_deck_danger_abilities = {120, 140}
        has_alakazam_stage2_on_board = any(cid in {743, 245} for cid in my_board_ids)
        has_alakazam_stage2_in_hand = any(cid in {743, 245} for cid in hand_ids)
        rare_candy_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1079)))
        attack_ko_available = bool(
            has_attack
            and opt_feats.ndim == 2
            and opt_feats.shape[1] > 57
            and np.any((opt_type == TYPE_ATTACK) & (opt_feats[:, 57] > 0.0))
        )
        attack_damage_max = 0.0
        if has_attack and opt_feats.ndim == 2 and opt_feats.shape[1] > 55:
            attack_rows = opt_feats[opt_type == TYPE_ATTACK, 55]
            if attack_rows.size:
                attack_damage_max = float(np.max(attack_rows) * 400.0)
        high_damage_attack_available = bool(attack_ko_available or attack_damage_max >= 150.0)
        crustle_single_wall_switch = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_crustle_plan")
            and opp_active_id == 345
            and not any(cid > 0 for cid in opp_bench_ids)
            and select_context in (3, 4)
            and n > 0
            and np.all(opt_type == TYPE_CARD)
        )
        if crustle_single_wall_switch:
            hit("dragapult_vs_crustle_plan", "single_wall_non_ex_switch")
            for i in range(n):
                target, target_pid, _, _ = option_target_info(i)
                if target_pid != you:
                    continue
                tid = pokemon_id(target) or int(opt_card[i])
                tenergy = energy_ids(target)
                mind_bend_ready = munkidori_mind_bend_ready(target)
                safe_munkidori_window = bool(
                    mind_bend_ready
                    and (
                        0 < opp_active_hp <= 60
                        or (
                            not opp_active_crustle_can_attack
                            and (
                                opp_active_crustle_energy_count <= 1
                                or not opp_active_crustle_has_grass
                                or 0 < opp_active_hp <= 120
                            )
                        )
                    )
                )
                if tid == 112:
                    if safe_munkidori_window:
                        out[i] += 260.0
                    elif opp_active_crustle_can_attack:
                        out[i] -= 220.0
                    elif 7 in tenergy and 5 in tenergy:
                        out[i] += 64.0
                    elif 7 in tenergy or 5 in tenergy:
                        out[i] += 24.0
                    else:
                        out[i] -= 42.0
                elif tid == 121:
                    out[i] += 95.0 if opp_active_crustle_can_attack else 22.0
                elif tid == 120:
                    out[i] += 42.0 if not opp_active_crustle_can_attack else -24.0
                elif tid in {235, 140, 1071}:
                    out[i] -= 180.0 if opp_active_crustle_can_attack else 58.0
        crustle_protected_wall_munkidori_switch = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_crustle_plan")
            and opp_active_id == 345
            and not opp_bench_has_valid_dca_target
            and select_context in (3, 4)
            and n > 0
            and np.all(opt_type == TYPE_CARD)
        )
        munkidori_can_ko_active_wall = bool(opp_active_id in {344, 345} and 0 < opp_active_hp <= 60)
        if crustle_protected_wall_munkidori_switch:
            hit("dragapult_vs_crustle_plan", "protected_wall_munkidori_switch")
            psychic_in_hand = 5 in hand_ids
            for i in range(n):
                target, target_pid, _, _ = option_target_info(i)
                if target_pid != you:
                    continue
                tid = pokemon_id(target) or int(opt_card[i])
                safe_munkidori_window = bool(
                    (
                        munkidori_mind_bend_ready(target)
                        or (tid == 112 and 5 in energy_ids(target) and (7 in energy_ids(target) or len(energy_id_list(target)) >= 1))
                    )
                    and (
                        munkidori_can_ko_active_wall
                        or (
                            not opp_active_crustle_can_attack
                            and (
                                opp_active_crustle_energy_count <= 1
                                or not opp_active_crustle_has_grass
                                or 0 < opp_active_hp <= 120
                            )
                        )
                    )
                )
                if munkidori_mind_bend_ready(target):
                    if safe_munkidori_window:
                        out[i] += 360.0
                    elif opp_active_crustle_can_attack:
                        out[i] -= 240.0
                    else:
                        out[i] += 18.0
                elif tid == 112 and 7 in energy_ids(target) and psychic_in_hand:
                    out[i] += 185.0 if safe_munkidori_window else (-85.0 if opp_active_crustle_can_attack else 28.0)
                elif tid == 112 and 5 in energy_ids(target) and len(energy_ids(target)) >= 1:
                    out[i] += 120.0 if safe_munkidori_window else (-70.0 if opp_active_crustle_can_attack else 12.0)
                elif tid == 112:
                    out[i] -= 80.0 if opp_active_crustle_can_attack else 18.0
                elif tid == 121:
                    out[i] += 90.0 if opp_active_crustle_can_attack else 18.0
                elif tid in {140, 1071}:
                    out[i] -= 180.0 if opp_active_crustle_can_attack else 55.0
        stage2_evolve_available = bool(
            np.any((opt_type == TYPE_EVOLVE) & ((opt_card == 121) | (opt_card2 == 121)))
        )
        alakazam_stage2_evolve_available = bool(
            np.any((opt_type == TYPE_EVOLVE) & ((opt_card == 743) | (opt_card2 == 743) | (opt_card == 245) | (opt_card2 == 245)))
        )
        forced_promote = bool(
            active_id <= 0
            and mn == 1
            and mx == 1
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and any(int(c) in dragapult_line_ids for c in opt_card.tolist())
        )
        if forced_promote and self.archetype == "dragapult":
            for i in range(n):
                card = int(opt_card[i])
                if card == 121:
                    out[i] += 16.0
                elif card == 120:
                    out[i] += 10.0
                elif card == 119:
                    out[i] += 6.0
                elif card in support_ids:
                    out[i] -= 8.0
        switch_line_context = bool(
            is_dragapult
            and active_id > 0
            and select_context in (3, 4)
            and mn == 1
            and mx == 1
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and any(int(c) in dragapult_line_ids for c in opt_card.tolist())
        )
        if switch_line_context:
            for i in range(n):
                card = int(opt_card[i])
                if card == 121:
                    out[i] += 16.0
                elif card == 120:
                    out[i] += 12.0
                elif card == 119:
                    out[i] += 5.0
                elif card in support_ids:
                    out[i] -= 12.0
        preserve_line_discard = bool(
            is_dragapult
            and select_context == 8
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and (
                any(int(c) in dragapult_line_ids for c in opt_card.tolist())
                or (
                    opp_has_crustle_wall_line
                    and rule_enabled("dragapult_vs_crustle_plan")
                    and any(int(c) in {5, 7, 112, 1120, 1182, 1198, 1246} for c in opt_card.tolist())
                )
            )
        )
        if preserve_line_discard:
            own_line_count = 0
            own_stage2_count = 0
            if encoded.board_cards.size >= 6:
                own_cards = [int(x) for x in np.asarray(encoded.board_cards[:6], dtype=np.int64).tolist()]
                own_line_count = sum(1 for c in own_cards if c in dragapult_line_ids)
                own_stage2_count = sum(1 for c in own_cards if c == 121)
            line_needed = own_stage2_count <= 0 or own_line_count < 3
            if line_needed or (opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan")):
                for i in range(n):
                    card = int(opt_card[i])
                    crustle_preserve = bool(opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"))
                    if card == 121:
                        out[i] -= 360.0 if crustle_preserve else 18.0
                    elif card == 120:
                        out[i] -= 140.0 if crustle_preserve else 16.0
                    elif card == 119:
                        out[i] -= 85.0 if crustle_preserve else 10.0
                    elif crustle_preserve and card == 7:
                        out[i] -= 150.0
                    elif crustle_preserve and card == 5 and my_munkidori_on_board and not my_munkidori_has_psychic:
                        out[i] -= 135.0
                    elif crustle_preserve and card == 112 and not my_munkidori_on_board:
                        out[i] -= 120.0
                    elif crustle_preserve and card == 1120:
                        out[i] -= 260.0
                    elif crustle_preserve and card == 1182:
                        out[i] -= 90.0
                    elif crustle_preserve and card == 1246 and stadium_id not in {0, 1246}:
                        out[i] -= 80.0
                    elif (
                        card == 1198
                        and own_stage2_count <= 0
                        and (
                            (opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan"))
                            or (opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"))
                            or (opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"))
                        )
                    ):
                        out[i] -= 120.0 if crustle_preserve else 18.0
                        if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                            hit("dragapult_vs_crustle_plan", "preserve_crispin_discard")
                        elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                            hit("dragapult_vs_lopunny_plan", "preserve_crispin_discard")
                        else:
                            hit("dragapult_vs_alakazam_plan", "preserve_crispin_discard")

        dragapult_setup_race_matchup = bool(
            (opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan"))
            or (opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"))
            or (opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"))
        )
        dragapult_vs_alakazam_line_pick = bool(
            is_dragapult
            and dragapult_setup_race_matchup
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and select_context not in (8, 13, 14, 16)
            and my_dragapult_count <= 0
            and my_dragapult_line_count < 4
            and any(int(c) in dragapult_line_ids for c in opt_card.tolist())
        )
        if dragapult_vs_alakazam_line_pick:
            if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                hit("dragapult_vs_crustle_plan", "line_pick")
            elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                hit("dragapult_vs_lopunny_plan", "line_pick")
            else:
                hit("dragapult_vs_alakazam_plan", "line_pick")
            for i in range(n):
                card = int(opt_card[i])
                if my_dragapult_line_count <= 0:
                    if card == 119:
                        out[i] += 28.0
                    elif card == 120:
                        out[i] += 8.0
                    elif card == 121:
                        out[i] -= 6.0
                elif my_drakloak_count <= 0:
                    if card == 120:
                        out[i] += 24.0
                    elif card == 119:
                        out[i] += 10.0
                    elif card == 121 and my_dreepy_count <= 0:
                        out[i] -= 8.0
                else:
                    if card == 121:
                        out[i] += 22.0
                    elif card == 120:
                        out[i] += 8.0
                    elif card == 119:
                        out[i] += 6.0
                if card in support_ids and my_dragapult_line_count <= 0:
                    out[i] -= 12.0

        dragapult_vs_alakazam_opening = bool(
            is_dragapult
            and dragapult_setup_race_matchup
            and select_context == 0
            and my_dragapult_line_count <= 0
        )
        if dragapult_vs_alakazam_opening:
            if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                hit("dragapult_vs_crustle_plan", "open_dreepy_first")
            elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                hit("dragapult_vs_lopunny_plan", "open_dreepy_first")
            else:
                hit("dragapult_vs_alakazam_plan", "open_dreepy_first")
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                if typ == TYPE_PLAY and card == 119:
                    out[i] += 30.0
                elif typ == TYPE_PLAY and card == 1086:
                    out[i] += 26.0
                elif typ == TYPE_PLAY and card == 1121:
                    out[i] += 18.0
                elif typ == TYPE_PLAY and card == 1152:
                    out[i] += 12.0
                elif typ == TYPE_PLAY and card == 1071:
                    # The close-read loss put Meowth before the first Dreepy,
                    # then fell behind the Alakazam setup.  Meowth is still
                    # fine after a line exists, not before it.
                    out[i] -= 18.0
                elif typ == TYPE_ATTACK:
                    out[i] -= 6.0
                elif typ == TYPE_END:
                    out[i] -= 10.0

        dragapult_vs_alakazam_boarding = bool(
            is_dragapult
            and dragapult_setup_race_matchup
            and select_context == 0
            and my_dragapult_line_count > 0
            and my_dragapult_count <= 0
            and my_dragapult_line_count < 3
        )
        if dragapult_vs_alakazam_boarding:
            if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                hit("dragapult_vs_crustle_plan", "build_multiple_lines")
            elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                hit("dragapult_vs_lopunny_plan", "build_multiple_lines")
            else:
                hit("dragapult_vs_alakazam_plan", "build_multiple_lines")
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                if typ == TYPE_PLAY and card == 119:
                    out[i] += 18.0
                elif typ == TYPE_PLAY and card == 1086:
                    out[i] += 18.0
                elif typ == TYPE_EVOLVE and (card == 120 or int(opt_card2[i]) == 120):
                    out[i] += 16.0
                elif typ == TYPE_EVOLVE and (card == 121 or int(opt_card2[i]) == 121):
                    out[i] += 20.0
                elif typ == TYPE_PLAY and card == 1071 and my_dragapult_line_count < 2:
                    out[i] -= 8.0

        dragapult_vs_crustle_early_second_line = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_crustle_plan")
            and opp_has_crustle_wall_line
            and select_context == 0
            and my_dragapult_count <= 0
            and my_dragapult_line_count <= 1
            and bench_line_count <= 0
            and np.any(opt_type == TYPE_PLAY)
        )
        if dragapult_vs_crustle_early_second_line:
            hit("dragapult_vs_crustle_plan", "early_second_line")
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                if typ == TYPE_PLAY and card in {1086, 1121, 1152}:
                    out[i] += 82.0
                elif typ == TYPE_PLAY and card == 119:
                    out[i] += 70.0
                elif typ == TYPE_PLAY and card in {140, 1071}:
                    out[i] -= 46.0
                elif typ == TYPE_END:
                    out[i] -= 60.0

        dragapult_vs_crustle_ability_before_commit = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_crustle_plan")
            and opp_has_crustle_wall_line
            and select_context == 0
            and my_deck > 6
            and np.any((opt_type == TYPE_ABILITY) & (opt_card == 120))
            and not (
                active_id == 112
                and 5 in active_energy_ids
                and len(active_energy_ids) >= 2
                and np.any(opt_type == TYPE_ATTACK)
            )
            and np.any(
                np.isin(opt_type, [TYPE_PLAY, TYPE_EVOLVE, TYPE_ATTACH, TYPE_RETREAT, TYPE_END])
                | ((opt_type == TYPE_ATTACK) & (opt_attack != 154))
            )
        )
        if dragapult_vs_crustle_ability_before_commit:
            hit("dragapult_vs_crustle_plan", "drakloak_ability_before_commit")
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_ABILITY and card == 120:
                    out[i] += 145.0
                elif typ == TYPE_EVOLVE and (card == 121 or card2 == 121):
                    out[i] -= 72.0
                elif typ == TYPE_PLAY and card in {1120, 1246}:
                    out[i] -= 4.0
                elif typ == TYPE_PLAY:
                    out[i] -= 46.0
                elif typ == TYPE_ATTACH:
                    out[i] -= 36.0
                elif typ == TYPE_RETREAT:
                    out[i] -= 30.0
                elif typ == TYPE_ATTACK and int(opt_attack[i]) != 154 and not attack_ko_available:
                    out[i] -= 70.0
                elif typ == TYPE_END:
                    out[i] -= 70.0

        dragapult_vs_alakazam_ability_before_evolve = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_alakazam_plan")
            and select_context == 0
            and opp_has_alakazam_line
            and stage2_evolve_available
            and my_deck > 4
            and np.any((opt_type == TYPE_ABILITY) & (opt_card == 120))
        )
        if dragapult_vs_alakazam_ability_before_evolve:
            hit("dragapult_vs_alakazam_plan", "drakloak_ability_before_stage2")
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_ABILITY and card == 120:
                    out[i] += 80.0
                elif typ == TYPE_EVOLVE and (card == 121 or card2 == 121):
                    out[i] -= 60.0
                elif typ == TYPE_END:
                    out[i] -= 10.0

        dragapult_vs_alakazam_backup_line = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_alakazam_plan")
            and select_context == 0
            and opp_has_alakazam_line
            and active_id == 121
            and bench_line_count <= 0
            and my_prizes >= 2
            and my_deck > 10
            and (
                np.any((opt_type == TYPE_PLAY) & np.isin(opt_card, [119, 1086, 1121, 1152, 1198, 1227, 1231, 1246]))
                or has_attach
            )
        )
        if dragapult_vs_alakazam_backup_line:
            hit("dragapult_vs_alakazam_plan", "backup_line_before_trade")
            active_low = bool(active_hp > 0 and active_max_hp > 0 and active_hp / active_max_hp <= 0.55)
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_own_bench = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 35 and float(opt_feats[i, 35]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_PLAY and card == 119:
                    out[i] += 38.0
                elif typ == TYPE_PLAY and card in {1086, 1121, 1231}:
                    out[i] += 32.0
                elif typ == TYPE_PLAY and card == 1152:
                    out[i] += 22.0
                elif typ == TYPE_PLAY and card in {1227, 1246}:
                    out[i] += 16.0
                elif typ == TYPE_PLAY and card == 1198 and active_low:
                    out[i] += 8.0
                elif typ == TYPE_ATTACH and target_own_bench and target_setup_or_evo:
                    out[i] += 22.0
                elif typ == TYPE_ATTACH and target_own_active and active_low:
                    out[i] -= 10.0
                elif typ == TYPE_ATTACK and my_prizes > 1:
                    out[i] -= 16.0 if active_low else 8.0
                elif typ == TYPE_END:
                    out[i] -= 12.0

        dragapult_line_energy_matchup = bool(
            (opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan"))
            or (opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"))
            or (opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"))
        )
        dragapult_line_energy_plan = bool(
            is_dragapult
            and dragapult_line_energy_matchup
            and select_context == 0
            and has_attach
            and (my_dragapult_line_count > 0 or active_id in dragapult_line_ids)
        )
        if dragapult_line_energy_plan:
            adjusted = False
            crustle_energy_critical = bool(
                opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan")
            )
            for i in range(n):
                typ = int(opt_type[i])
                if typ != TYPE_ATTACH:
                    continue
                card = int(opt_card[i])
                target = option_target_pokemon(i)
                target_id = option_target_id(i)
                target_energy = energy_ids(target)
                missing = phantom_energy_ids.difference(target_energy)
                if target_id in dragapult_line_ids:
                    after = set(target_energy)
                    if card > 0:
                        after.add(card)
                    if card == 7:
                        out[i] -= 125.0 if crustle_energy_critical else 70.0
                        adjusted = True
                    elif not missing:
                        out[i] -= 120.0 if crustle_energy_critical else 54.0
                        adjusted = True
                    elif phantom_energy_ids.issubset(after):
                        out[i] += 190.0 if crustle_energy_critical else 54.0
                        adjusted = True
                    elif card in missing:
                        if crustle_energy_critical:
                            out[i] += 145.0 if target_energy.intersection(phantom_energy_ids) else 96.0
                        else:
                            out[i] += 30.0 if target_energy.intersection(phantom_energy_ids) else 22.0
                        adjusted = True
                    elif card in phantom_energy_ids and card not in missing and missing:
                        out[i] -= 160.0 if crustle_energy_critical else 55.0
                        adjusted = True
                elif card in phantom_energy_ids and my_dragapult_line_count > 0:
                    # Preserve R/P for the attacking line unless the option
                    # explicitly targets that line.
                    out[i] -= 60.0 if crustle_energy_critical else 16.0
                    adjusted = True
                elif card == 7 and target_id == 112:
                    out[i] += 8.0
                    adjusted = True
            if adjusted:
                if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                    hit("dragapult_vs_crustle_plan", "line_energy_rp_not_dark")
                elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                    hit("dragapult_vs_lopunny_plan", "line_energy_rp_not_dark")
                else:
                    hit("dragapult_vs_alakazam_plan", "line_energy_rp_not_dark")

        phantom_ready_attach_available = bool(
            any(option_makes_phantom_ready(i) for i in range(n) if int(opt_type[i]) == TYPE_ATTACH)
        )
        crispin_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1198)))
        dragapult_phantom_energy_matchup = bool(
            (opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan"))
            or (opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"))
            or (opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"))
        )
        dragapult_vs_alakazam_phantom_energy = bool(
            is_dragapult
            and dragapult_phantom_energy_matchup
            and select_context == 0
            and not has_phantom_dive
            and (my_drakloak_count > 0 or my_dragapult_count > 0 or active_id in {120, 121})
            and (has_attach or crispin_available or has_jet_headbutt)
        )
        if dragapult_vs_alakazam_phantom_energy:
            adjusted = False
            readiness_available = bool(crispin_available or phantom_ready_attach_available)
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                target = option_target_pokemon(i)
                target_id = option_target_id(i)
                target_energy = energy_ids(target)
                missing = phantom_energy_ids.difference(target_energy)
                if typ == TYPE_PLAY and card == 1198:
                    out[i] += 36.0
                    adjusted = True
                elif typ == TYPE_ATTACH and target_id in dragapult_line_ids:
                    after = set(target_energy)
                    if card > 0:
                        after.add(card)
                    if not missing:
                        out[i] -= 72.0
                        adjusted = True
                    elif phantom_energy_ids.issubset(after):
                        out[i] += 58.0
                        adjusted = True
                    elif card in missing:
                        out[i] += 26.0
                        adjusted = True
                    elif card in phantom_energy_ids and card not in missing and missing:
                        # Do not strand the attacker on duplicate R or P; the
                        # traced loss repeatedly attached extra R and then only
                        # had Jet Headbutt available.
                        out[i] -= 70.0
                        adjusted = True
                    elif card == 7:
                        out[i] -= 70.0
                        adjusted = True
                elif typ == TYPE_ATTACH and target_id not in dragapult_line_ids:
                    out[i] -= 10.0
                    adjusted = True
                elif typ == TYPE_ATTACK and active_id == 121 and has_jet_headbutt:
                    out[i] += 24.0 if not readiness_available else -45.0
                    adjusted = True
                elif typ == TYPE_PLAY and card == 1182 and opp_has_crustle_wall_line:
                    out[i] -= 145.0
                    adjusted = True
                elif typ == TYPE_PLAY and active_id == 121 and has_jet_headbutt and not readiness_available and card in {119, 140, 235, 1071, 1086}:
                    out[i] -= 14.0
                    adjusted = True
                elif typ == TYPE_PLAY and opp_has_crustle_wall_line and readiness_available and card != 1198:
                    out[i] -= 42.0
                    adjusted = True
                elif typ == TYPE_PLAY and card in {1120, 1197, 1213} and not dragapult_stage2_attack_ready_on_board:
                    out[i] -= 16.0
                    adjusted = True
                elif typ == TYPE_END:
                    out[i] -= 16.0
                    adjusted = True
            if adjusted:
                if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                    hit("dragapult_vs_crustle_plan", "phantom_energy_before_jet")
                elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                    hit("dragapult_vs_lopunny_plan", "phantom_energy_before_jet")
                else:
                    hit("dragapult_vs_alakazam_plan", "phantom_energy_before_jet")

        dragapult_crispin_energy_pick = bool(
            is_dragapult
            and dragapult_phantom_energy_matchup
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and select_context not in (8, 13, 14, 16)
            and any(int(c) in {2, 5, 7} for c in opt_card.tolist())
            and (my_dragapult_line_count > 0 or active_id in dragapult_line_ids)
        )
        if dragapult_crispin_energy_pick:
            line_energy_samples: list[set[int]] = []
            if active_id in dragapult_line_ids:
                line_energy_samples.append(set(active_energy_ids))
            for poke in me.get("bench") or []:
                if not poke:
                    continue
                if int(poke.get("id", 0) or 0) in dragapult_line_ids:
                    line_energy_samples.append(set(energy_ids(poke)))
            if line_energy_samples:
                # Prefer the line closest to Phantom Dive readiness, then take
                # exactly its missing R/P pieces.  This handles Crispin-style
                # deck selections where the raw BC often grabs Basic Dark.
                best_energy = sorted(
                    line_energy_samples,
                    key=lambda es: (len(phantom_energy_ids.difference(es)), -len(es)),
                )[0]
                missing = phantom_energy_ids.difference(best_energy)
            else:
                missing = set(phantom_energy_ids)
            crustle_munkidori_energy_pick = bool(
                opp_has_crustle_wall_line
                and rule_enabled("dragapult_vs_crustle_plan")
                and opp_active_id == 345
                and not opp_bench_has_valid_dca_target
                and my_munkidori_on_board
                and not any_munkidori_attack_ready
                and (dragapult_stage2_attack_ready_on_board or active_id == 121)
            )
            if missing or not dragapult_stage2_attack_ready_on_board or crustle_munkidori_energy_pick:
                adjusted = False
                active_crispin_missing = (
                    phantom_energy_ids.difference(set(active_energy_ids))
                    if active_id in {120, 121} and active_energy_ids
                    else set()
                )
                for i in range(n):
                    card = int(opt_card[i])
                    if len(active_crispin_missing) == 1 and select_context == 7:
                        need = next(iter(active_crispin_missing))
                        if card == need:
                            out[i] -= 360.0
                        elif card in phantom_energy_ids:
                            out[i] += 260.0
                        elif card == 7:
                            out[i] += 36.0
                        adjusted = True
                        continue
                    if len(active_crispin_missing) == 1 and select_context == 22:
                        need = next(iter(active_crispin_missing))
                        if card == need:
                            out[i] += 620.0
                        elif card in phantom_energy_ids:
                            out[i] -= 360.0
                        else:
                            out[i] -= 140.0
                        adjusted = True
                        continue
                    if crustle_munkidori_energy_pick and card == 5 and not my_munkidori_has_psychic:
                        out[i] += 150.0
                        adjusted = True
                    elif crustle_munkidori_energy_pick and card == 7 and my_munkidori_needs_dark:
                        out[i] += 132.0
                        adjusted = True
                    elif crustle_munkidori_energy_pick and card in {2, 5} and not missing:
                        out[i] -= 24.0
                        adjusted = True
                    elif card in missing:
                        out[i] += 88.0
                        adjusted = True
                    elif card in phantom_energy_ids:
                        out[i] += 28.0
                        adjusted = True
                    elif card == 7:
                        out[i] -= 92.0
                        adjusted = True
                    elif card in {1198, 1182, 1246}:
                        out[i] += 8.0
                        adjusted = True
                if adjusted:
                    if opp_has_crustle_wall_line and rule_enabled("dragapult_vs_crustle_plan"):
                        hit("dragapult_vs_crustle_plan", "crispin_pick_missing_rp")
                    elif opp_has_lopunny_line and rule_enabled("dragapult_vs_lopunny_plan"):
                        hit("dragapult_vs_lopunny_plan", "crispin_pick_missing_rp")
                    else:
                        hit("dragapult_vs_alakazam_plan", "crispin_pick_missing_rp")

        dragapult_vs_alakazam_disrupt = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_alakazam_plan")
            and select_context == 0
            and opp_has_alakazam_line
        )
        if dragapult_vs_alakazam_disrupt:
            adjusted = False
            defer_disrupt_until_ready = bool(
                (my_dragapult_count <= 0 and stage2_evolve_available)
                or (active_id == 121 and not has_phantom_dive and (has_jet_headbutt or has_attach))
                or (dragapult_phantom_energy_line_ready and not dragapult_stage2_attack_ready_on_board and stage2_evolve_available)
            )
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if defer_disrupt_until_ready and typ == TYPE_PLAY and card in {1120, 1197, 1213}:
                    out[i] -= 22.0
                    adjusted = True
                elif defer_disrupt_until_ready and typ == TYPE_EVOLVE and (card == 121 or card2 == 121):
                    out[i] += 30.0
                    adjusted = True
                elif defer_disrupt_until_ready and typ == TYPE_PLAY and card == 1198:
                    out[i] += 18.0
                    adjusted = True
                elif not defer_disrupt_until_ready and typ == TYPE_PLAY and card == 1120 and opp_alakazam_energy_count > 0:
                    out[i] += 22.0 if active_id != 121 else 16.0
                    adjusted = True
                elif not defer_disrupt_until_ready and typ == TYPE_PLAY and card in {1213, 1197} and opp_hand_count >= 10:
                    out[i] += 26.0 if card == 1213 else 20.0
                    adjusted = True
                elif not defer_disrupt_until_ready and typ == TYPE_ATTACK and active_id == 121 and opp_hand_count >= 14 and np.any((opt_type == TYPE_PLAY) & (opt_card == 1213)):
                    out[i] -= 6.0
                    adjusted = True
            if adjusted:
                detail = "defer_disrupt_until_ready" if defer_disrupt_until_ready else f"disrupt_hand{opp_hand_count}_energy{opp_alakazam_energy_count}"
                hit("dragapult_vs_alakazam_plan", detail)

        dragapult_vs_alakazam_phantom = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_alakazam_plan")
            and select_context == 0
            and active_id == 121
            and has_phantom_dive
            and opp_has_alakazam_line
        )
        if dragapult_vs_alakazam_phantom:
            hit("dragapult_vs_alakazam_plan", "phantom_over_jet")
            for i in range(n):
                typ = int(opt_type[i])
                attack_id = int(opt_attack[i])
                if typ == TYPE_ATTACK and attack_id == 154:
                    out[i] += 24.0
                elif typ == TYPE_ATTACK and attack_id == 153:
                    out[i] -= 18.0

        dragapult_vs_lopunny_plan = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_lopunny_plan")
            and opp_has_lopunny_line
        )
        if dragapult_vs_lopunny_plan:
            adjusted = False
            jamming_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1246)))
            bench_dragapult_present = any(
                bool(poke) and int(poke.get("id", 0) or 0) == 121
                for poke in me.get("bench") or []
            )
            active_support_ex_stranded = bool(active_id in {140, 1071} and bench_dragapult_present)
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                target_id = option_target_id(i)
                if battle_cage_active:
                    if typ == TYPE_PLAY and card == 1246:
                        out[i] += 80.0
                        adjusted = True
                    elif jamming_available and typ == TYPE_ATTACK and int(opt_attack[i]) == 154:
                        out[i] -= 35.0
                        adjusted = True
                    elif jamming_available and typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE, TYPE_ATTACH, TYPE_END):
                        out[i] -= 8.0
                        adjusted = True
                if select_context not in (8, 13, 14, 16) and typ == TYPE_CARD and card == 1246:
                    out[i] += 56.0 if battle_cage_active else 18.0
                    adjusted = True
                if (
                    select_context == 0
                    and my_prizes == 6
                    and opp_prizes == 6
                    and typ == TYPE_PLAY
                    and card == 140
                    and my_dragapult_line_count > 0
                ):
                    out[i] -= 34.0
                    adjusted = True
                if select_context == 0 and active_support_ex_stranded:
                    if typ == TYPE_RETREAT:
                        out[i] += 70.0
                        adjusted = True
                    elif typ == TYPE_ATTACH and target_id == active_id:
                        out[i] += 45.0
                        adjusted = True
                    elif typ == TYPE_END:
                        out[i] -= 55.0
                        adjusted = True
                    elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE) and not (typ == TYPE_PLAY and card == 1246):
                        out[i] -= 10.0
                        adjusted = True
            if adjusted:
                detail = "battle_cage" if battle_cage_active else "lopunny_froslass"
                if active_support_ex_stranded:
                    detail += "_support_stranded"
                hit("dragapult_vs_lopunny_plan", detail)

        dragapult_vs_crustle_plan = bool(
            is_dragapult
            and rule_enabled("dragapult_vs_crustle_plan")
            and opp_has_crustle_wall_line
        )
        blaziken_route_ids = {31, 272, 324, 325, 326, 1079, 1250, 1256}
        blaziken_route_visible = bool(
            any(cid in blaziken_route_ids for cid in my_board_ids)
            or any(cid in blaziken_route_ids for cid in hand_ids)
            or any(int(c) in blaziken_route_ids for c in opt_card.tolist())
        )
        dragapult_blaziken_crustle_plan = bool(
            is_dragapult
            and rule_enabled("dragapult_blaziken_crustle_plan")
            and blaziken_route_visible
            and (opp_has_crustle_wall_line or (active_id <= 0 and not opp_board_ids))
        )
        if dragapult_blaziken_crustle_plan:
            adjusted = False
            detail_bits: list[str] = []
            torchic_count = sum(1 for cid in my_board_ids if cid == 324)
            combusken_count = sum(1 for cid in my_board_ids if cid == 325)
            blaziken_count = sum(1 for cid in my_board_ids if cid == 326)
            chi_yu_active = active_id == 31
            chi_yu_on_board = any(cid == 31 for cid in my_board_ids)
            route_not_online = blaziken_count <= 0
            can_hit_active_dwebble = bool(chi_yu_active and opp_active_id == 344 and has_attack)
            early_crustle_line = bool(opp_active_id == 344 or opp_bench_has_dwebble)
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                attack_id = int(opt_attack[i])
                target_id = option_target_id(i)
                target = option_target_pokemon(i)
                target_own_active = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0
                )
                target_setup_or_evo = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0
                )
                target_opp_bench = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 70 and float(opt_feats[i, 70]) > 0.0
                )

                # LumenLiquidity-style anti-Crustle route: open/keep Chi-Yu
                # when available, use it to remove Dwebble before the wall is
                # established, and build Torchic -> Blaziken ex as the energy
                # engine.  This is intentionally a separate opt-in rule because
                # cc2e does not contain the route cards.
                if active_id <= 0 and mn == 1 and mx == 1 and typ == TYPE_CARD:
                    if card == 31:
                        out[i] += 130.0
                        adjusted = True
                        detail_bits.append("open_chi_yu")
                    elif card == 119:
                        out[i] += 20.0
                        adjusted = True
                    elif card in {112, 140, 235, 1071}:
                        out[i] -= 28.0
                        adjusted = True

                if select_context == 0:
                    if can_hit_active_dwebble and typ == TYPE_ATTACK:
                        out[i] += 150.0
                        adjusted = True
                        detail_bits.append("chi_yu_ko_dwebble")
                    elif can_hit_active_dwebble and typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE, TYPE_END):
                        out[i] -= 26.0
                        adjusted = True
                    if early_crustle_line and route_not_online:
                        if typ == TYPE_PLAY and card == 324:
                            out[i] += 95.0
                            adjusted = True
                            detail_bits.append("play_torchic")
                        elif typ == TYPE_PLAY and card in {1086, 1121, 1152} and torchic_count <= 0:
                            out[i] += 62.0
                            adjusted = True
                            detail_bits.append("search_torchic")
                        elif typ == TYPE_PLAY and card == 1079 and torchic_count > 0 and any(x == 326 for x in hand_ids):
                            out[i] += 88.0
                            adjusted = True
                            detail_bits.append("rare_candy_blaziken")
                        elif typ == TYPE_EVOLVE and (card == 326 or card2 == 326):
                            out[i] += 110.0
                            adjusted = True
                            detail_bits.append("evolve_blaziken")
                        elif typ == TYPE_EVOLVE and (card == 325 or card2 == 325):
                            out[i] += 54.0
                            adjusted = True
                            detail_bits.append("evolve_combusken")
                        elif typ == TYPE_PLAY and card == 1198 and (chi_yu_active or my_dragapult_line_count > 0):
                            out[i] += 42.0
                            adjusted = True
                        elif typ == TYPE_ATTACH and target_id == 31 and card in {2, 5}:
                            out[i] += 86.0
                            adjusted = True
                            detail_bits.append("fuel_chi_yu")
                        elif typ == TYPE_ATTACH and target_id == 324 and card in {2, 5} and not chi_yu_active:
                            out[i] += 26.0
                            adjusted = True
                        elif typ == TYPE_END:
                            out[i] -= 42.0
                            adjusted = True
                    if typ == TYPE_PLAY and card == 1256 and stadium_id != 1256:
                        out[i] += 42.0
                        adjusted = True
                        detail_bits.append("watchtower")
                    elif typ == TYPE_PLAY and card == 1250 and blaziken_count > 0 and my_dragapult_line_count > 0:
                        out[i] += 26.0
                        adjusted = True
                        detail_bits.append("area_zero")

                if select_context not in (8, 13, 14, 16) and typ == TYPE_CARD:
                    if early_crustle_line and route_not_online:
                        if card == 324:
                            out[i] += 100.0
                            adjusted = True
                            detail_bits.append("pick_torchic")
                        elif card == 326 and torchic_count > 0:
                            out[i] += 92.0
                            adjusted = True
                            detail_bits.append("pick_blaziken")
                        elif card == 1079 and torchic_count > 0:
                            out[i] += 78.0
                            adjusted = True
                            detail_bits.append("pick_rare_candy")
                        elif card == 31 and not chi_yu_on_board and not chi_yu_active:
                            out[i] += 64.0
                            adjusted = True
                            detail_bits.append("pick_chi_yu")
                        elif card in {1256, 1250}:
                            out[i] += 24.0
                            adjusted = True
                    if card in {2, 5} and (chi_yu_active or active_id in dragapult_line_ids):
                        out[i] += 34.0
                        adjusted = True
                    elif card == 7 and route_not_online and early_crustle_line:
                        out[i] -= 48.0
                        adjusted = True

                if select_context == 14 and typ == TYPE_CARD and target_opp_bench:
                    target_hp = int(target.get("hp", 0) or 0) if target else 0
                    if target_id == 344:
                        out[i] += 120.0 if 0 < target_hp <= 60 else 82.0
                        adjusted = True
                        detail_bits.append("dca_dwebble")
                    elif target_id == 345 and opp_bench_has_dwebble:
                        out[i] -= 26.0
                        adjusted = True

            if adjusted:
                detail = "_".join(dict.fromkeys(detail_bits)) or "route"
                hit("dragapult_blaziken_crustle_plan", detail)

        if dragapult_vs_crustle_plan:
            adjusted = False
            dca_redirected = False
            munkidori_route_adjusted = False
            boss_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1182)))
            jamming_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1246)))
            munkidori_ability_available = bool(np.any((opt_type == TYPE_ABILITY) & (opt_card == 112)))
            bench_has_energy_line = any(
                bool(poke)
                and int(poke.get("id", 0) or 0) in {119, 120, 121}
                and bool(energy_ids(poke))
                for poke in me.get("bench") or []
            )
            bench_has_dragapult_line = any(
                bool(poke)
                and int(poke.get("id", 0) or 0) in {119, 120, 121}
                for poke in me.get("bench") or []
            )
            active_support_stranded = bool(
                (
                    active_id in {112, 140, 1071}
                    or (active_id == 235 and dragapult_stage2_attack_ready_on_board)
                )
                and (
                    bench_attack_ready_dragapult
                    or bench_phantom_energy_line_ready
                    or bench_has_energy_line
                    or (
                        opp_has_crustle_wall_line
                        and active_id != 235
                        and bench_has_dragapult_line
                    )
                )
                and (has_attach or has_retreat)
            )
            active_budew_crustle_lock_window = bool(
                select_context == 0
                and active_id == 235
                and opp_has_crustle_wall_line
                and not dragapult_stage2_attack_ready_on_board
            )
            opp_bench_has_nonwall_target = any(cid > 0 and cid not in {344, 345} for cid in opp_bench_ids)
            boss_break_window = bool(
                select_context == 0
                and active_id == 121
                and has_phantom_dive
                and boss_available
                and (opp_bench_has_dwebble or opp_bench_has_nonwall_target)
                and opp_active_id not in {344, 345}
                and not opp_active_nonwall_phantom_ko
            )
            boss_around_wall_window = bool(
                select_context == 0
                and active_id == 121
                and (has_phantom_dive or has_jet_headbutt)
                and boss_available
                and opp_active_id == 345
                and opp_bench_has_nonwall_target
            )
            boss_rotate_wall_window = bool(
                select_context == 0
                and active_id == 121
                and has_phantom_dive
                and boss_available
                and opp_active_id == 345
                and opp_bench_has_wall_target
                and not dca_protected(opp_active_obj)
            )
            crustle_single_wall = bool(
                select_context == 0
                and opp_active_id == 345
                and not any(cid > 0 for cid in opp_bench_ids)
            )
            active_line_under_wall = bool(
                select_context == 0
                and opp_active_id == 345
                and active_id in {120, 121}
                and (has_attack or has_attach or crispin_available or stage2_evolve_available)
            )
            protected_wall_no_dca = bool(
                opp_active_id == 345
                and not opp_bench_has_valid_dca_target
            )
            phantom_into_protected_wall = bool(
                active_line_under_wall
                and active_id == 121
                and has_phantom_dive
                and protected_wall_no_dca
            )
            active_dragapult_ko_risk = bool(
                select_context == 0
                and active_id == 121
                and opp_active_id == 345
                and active_hp > 0
                and active_hp <= 150
                and my_prizes > 1
            )
            bench_replacement_attach_available = any(
                int(opt_type[j]) == TYPE_ATTACH
                and option_target_info(j)[2] == 5
                and option_target_id(j) in dragapult_line_ids
                and (
                    (lambda e, c: bool(phantom_energy_ids.difference(e) and c in phantom_energy_ids.difference(e)))(
                        energy_ids(option_target_pokemon(j)),
                        int(opt_card[j]),
                    )
                )
                for j in range(n)
            )
            munkidori_conversion_window = bool(
                select_context == 0
                and munkidori_ability_available
                and my_damaged_any
                and (opp_has_crustle_wall_line or opp_wall_damage > 0)
            )
            crushing_hammer_available = bool(np.any((opt_type == TYPE_PLAY) & (opt_card == 1120)))
            crustle_single_wall_disrupt_window = bool(
                crustle_single_wall
                and opp_active_crustle_energy_count > 0
                and crushing_hammer_available
            )
            single_wall_munkidori_safe_attack = bool(
                crustle_single_wall
                and active_id == 112
                and (
                    0 < opp_active_hp <= 60
                    or (
                        not opp_active_crustle_can_attack
                        and (
                            opp_active_crustle_energy_count <= 1
                            or not opp_active_crustle_has_grass
                            or 0 < opp_active_hp <= 120
                        )
                    )
                )
            )
            single_wall_munkidori_pivot_window = bool(
                crustle_single_wall
                and active_id != 112
                and bench_munkidori_attack_ready
                and (
                    0 < opp_active_hp <= 60
                    or (
                        not opp_active_crustle_can_attack
                        and (
                            opp_active_crustle_energy_count <= 1
                            or not opp_active_crustle_has_grass
                            or 0 < opp_active_hp <= 120
                        )
                    )
                )
            )
            munkidori_attack_window = bool(
                select_context == 0
                and active_id == 112
                and opp_active_id in {344, 345}
                and (
                    0 < opp_active_hp <= 80
                    or opp_active_damage >= 40
                    or opp_wall_damage >= 40
                    or single_wall_munkidori_safe_attack
                )
                and has_attack
            )
            crustle_force_active_choice = bool(
                select_context in (3, 4)
                and opp_active_id == 345
                and n > 0
                and np.all(opt_type == TYPE_CARD)
            )
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                attack_id = int(opt_attack[i])
                target_id = option_target_id(i)
                target, target_pid, target_area, _ = option_target_info(i)
                target_energy = energy_ids(target)
                target_hp, target_max_hp, target_damage = pokemon_hp(target)
                target_is_opp = bool(target_pid == 1 - you)
                target_is_own = bool(target_pid == you)
                target_is_bench = bool(target_area == 5)
                target_own_active = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0
                )
                if target_is_own and target_area == 4:
                    target_own_active = True
                target_setup_or_evo = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0
                )
                target_opp_bench = bool(
                    opt_feats.ndim == 2 and opt_feats.shape[1] > 70 and float(opt_feats[i, 70]) > 0.0
                )
                target_opp_wall_line = bool(target_is_opp and target_id in {344, 345})
                target_fainted = bool(target_hp <= 0 or (target_max_hp > 0 and target_damage >= target_max_hp))

                if active_budew_crustle_lock_window:
                    if typ == TYPE_RETREAT:
                        out[i] -= 420.0
                        adjusted = True
                    elif typ == TYPE_ATTACK and attack_id == 323:
                        if has_attach or np.any(np.isin(opt_type, [TYPE_PLAY, TYPE_EVOLVE, TYPE_ABILITY])):
                            out[i] += 18.0
                        else:
                            out[i] += 240.0
                        adjusted = True
                    elif typ == TYPE_ATTACH and target_is_own and target_is_bench and target_id in dragapult_line_ids:
                        if card in phantom_energy_ids:
                            out[i] += 165.0
                        else:
                            out[i] += 55.0
                        adjusted = True
                    elif typ == TYPE_ATTACH and target_own_active:
                        out[i] -= 180.0
                        adjusted = True

                if crustle_force_active_choice and target_is_own:
                    target_ready_dragapult = bool(target_id == 121 and phantom_ready_energy(target))
                    target_partial_line = bool(target_id in {119, 120})
                    if target_ready_dragapult:
                        out[i] += 420.0
                        adjusted = True
                    elif target_id == 235:
                        out[i] += 260.0
                        adjusted = True
                    elif target_id == 112:
                        if (
                            munkidori_mind_bend_ready(target)
                            and (
                                0 < opp_active_hp <= 60
                                or not opp_active_crustle_can_attack
                            )
                        ):
                            out[i] += 180.0
                        elif opp_active_crustle_can_attack:
                            out[i] -= 280.0
                        else:
                            out[i] += 35.0
                        adjusted = True
                    elif target_partial_line and opp_active_crustle_can_attack:
                        out[i] -= 520.0
                        adjusted = True
                    elif target_id in {140, 1071} and opp_prizes <= 2:
                        out[i] -= 90.0
                        adjusted = True
                    elif target_id in {140, 1071}:
                        out[i] += 45.0
                        adjusted = True

                if low_deck_attack_available:
                    if typ == TYPE_ATTACK:
                        out[i] += 320.0 if my_deck <= 4 else 190.0
                        adjusted = True
                    elif typ == TYPE_ABILITY and card in low_deck_danger_abilities:
                        out[i] -= 1_000_000.0 if my_deck <= 5 else 520.0
                        adjusted = True
                    elif typ == TYPE_PLAY and card in low_deck_danger_cards:
                        out[i] -= 1_000_000.0 if my_deck <= 4 else 460.0
                        adjusted = True
                    elif typ == TYPE_PLAY and card == 1097 and my_deck <= 4 and has_attack:
                        out[i] -= 180.0
                        adjusted = True
                    elif typ == TYPE_END and my_deck <= 4:
                        out[i] -= 120.0
                        adjusted = True

                if select_context == 21 and typ == TYPE_CARD and target_is_own:
                    active_missing = phantom_energy_ids.difference(active_energy_ids)
                    target_missing = phantom_energy_ids.difference(target_energy)
                    active_nearly_ready = bool(
                        active_id == 121
                        and active_missing
                        and active_energy_ids
                        and not has_phantom_dive
                    )
                    if active_nearly_ready and target_own_active:
                        out[i] += 720.0
                        adjusted = True
                    elif active_nearly_ready and not target_own_active:
                        out[i] -= 260.0
                        adjusted = True
                    elif target_id in dragapult_line_ids and target_missing:
                        out[i] += 150.0 if target_id == 121 else 78.0
                        adjusted = True
                    elif target_id not in dragapult_line_ids and active_id in dragapult_line_ids:
                        out[i] -= 120.0
                        adjusted = True

                if (
                    select_context == 0
                    and opp_active_nonwall_phantom_ko
                    and typ == TYPE_PLAY
                    and card == 1182
                ):
                    out[i] -= 1_000_000.0
                    adjusted = True

                if typ == TYPE_ENERGY and target_is_opp and target_id == 345:
                    if crustle_single_wall:
                        if card in {1, 18, 10} and (opp_active_crustle_can_attack or opp_active_crustle_has_grass):
                            out[i] += 280.0
                            adjusted = True
                        elif card == 11 and (opp_active_crustle_has_mist or any_munkidori_attack_ready):
                            out[i] += 245.0
                            adjusted = True
                        elif card in {14, 20}:
                            out[i] += 88.0
                            adjusted = True
                        else:
                            out[i] += 42.0
                            adjusted = True
                    else:
                        if card == 11:
                            out[i] += 145.0
                            adjusted = True
                        elif card in {18, 20}:
                            out[i] += 72.0
                            adjusted = True
                        elif card == 14:
                            out[i] += 40.0
                            adjusted = True

                if select_context == 0 and typ == TYPE_END and (
                    np.any(opt_type == TYPE_PLAY)
                    or np.any(opt_type == TYPE_EVOLVE)
                    or np.any(opt_type == TYPE_ABILITY)
                    or np.any(opt_type == TYPE_ATTACH)
                ):
                    out[i] -= 96.0
                    adjusted = True

                if (
                    select_context == 0
                    and typ == TYPE_PLAY
                    and card == 1121
                    and len(hand_ids) <= 3
                    and any(c in {121, 1198, 1182, 7} for c in hand_ids)
                ):
                    out[i] -= 185.0
                    adjusted = True

                if (
                    select_context == 0
                    and typ == TYPE_ATTACK
                    and attack_id == 153
                    and not has_phantom_dive
                    and not attack_ko_available
                    and (
                        np.any(opt_type == TYPE_PLAY)
                        or np.any(opt_type == TYPE_EVOLVE)
                        or np.any(opt_type == TYPE_ABILITY)
                        or np.any(opt_type == TYPE_ATTACH)
                        or np.any(opt_type == TYPE_RETREAT)
                    )
                ):
                    out[i] -= 92.0
                    adjusted = True

                if active_support_stranded and select_context == 0:
                    if typ == TYPE_RETREAT:
                        if active_id == 112 and protected_wall_no_dca and (5 in active_energy_ids or 7 in active_energy_ids):
                            out[i] -= 170.0
                        else:
                            out[i] += 96.0
                        adjusted = True
                    elif typ == TYPE_ATTACH and target_own_active:
                        if active_id == 112 and protected_wall_no_dca:
                            current = set(active_energy_ids)
                            if card == 5 and 5 not in current:
                                out[i] += 260.0
                            elif card == 7 and 7 not in current:
                                out[i] += 220.0
                            elif 5 in current and len(current) < 2 and card > 0:
                                out[i] += 190.0
                            else:
                                out[i] += 32.0
                        elif active_id == 112:
                            out[i] -= 90.0
                        else:
                            out[i] += 86.0 if card == 7 else 64.0
                        adjusted = True
                    elif typ == TYPE_ATTACH and target_setup_or_evo:
                        out[i] -= 38.0
                        adjusted = True
                    elif typ == TYPE_END:
                        out[i] -= 85.0
                        adjusted = True
                    elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE) and not (typ == TYPE_PLAY and card in {1182, 1246}):
                        out[i] -= 18.0
                        adjusted = True

                if boss_break_window or boss_around_wall_window or boss_rotate_wall_window:
                    if typ == TYPE_PLAY and card == 1182:
                        if boss_rotate_wall_window:
                            out[i] += 210.0
                        else:
                            out[i] += 86.0 if boss_around_wall_window else 78.0
                        adjusted = True
                    elif typ == TYPE_ATTACK and attack_id == 154:
                        out[i] -= 118.0 if boss_rotate_wall_window else (58.0 if boss_around_wall_window else 48.0)
                        adjusted = True
                    elif typ in (TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE, TYPE_END):
                        out[i] -= 8.0
                        adjusted = True

                if munkidori_conversion_window:
                    if typ == TYPE_ABILITY and card == 112:
                        out[i] += 120.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif typ == TYPE_ATTACK and not attack_ko_available:
                        out[i] -= 24.0
                        adjusted = True
                    elif typ == TYPE_END:
                        out[i] -= 40.0
                        adjusted = True

                if munkidori_attack_window:
                    if typ == TYPE_ATTACK:
                        if single_wall_munkidori_safe_attack:
                            out[i] += 1_000_000.0 if munkidori_mind_bend_ready(my_active_obj) else 220.0
                        else:
                            out[i] += 820.0 if munkidori_mind_bend_ready(my_active_obj) else 135.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE, TYPE_ATTACH, TYPE_RETREAT, TYPE_END):
                        if single_wall_munkidori_safe_attack and munkidori_mind_bend_ready(my_active_obj):
                            out[i] -= 1_000_000.0
                        else:
                            out[i] -= 260.0 if munkidori_mind_bend_ready(my_active_obj) else 20.0
                        adjusted = True

                if select_context == 0 and has_attach and my_munkidori_needs_dark:
                    if typ == TYPE_ATTACH and card == 7 and target_id == 112:
                        bonus = 82.0 if (dragapult_stage2_attack_ready_on_board or my_damaged_any) else 34.0
                        out[i] += bonus
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif typ == TYPE_ATTACH and card == 7 and target_id in dragapult_line_ids:
                        out[i] -= 34.0
                        adjusted = True

                if select_context == 0 and has_attach and protected_wall_no_dca and my_munkidori_on_board:
                    target_es = set(target_energy)
                    target_len = len(energy_id_list(target))
                    if typ == TYPE_ATTACH and target_id == 112:
                        if card == 5 and 5 not in target_es and (7 in target_es or target_len >= 1):
                            out[i] += 330.0
                            adjusted = True
                            munkidori_route_adjusted = True
                        elif card == 5 and 5 not in target_es:
                            out[i] += 210.0
                            adjusted = True
                            munkidori_route_adjusted = True
                        elif card == 7 and 7 not in target_es and 5 in target_es:
                            out[i] += 285.0
                            adjusted = True
                            munkidori_route_adjusted = True
                        elif 5 in target_es and target_len < 2 and card > 0:
                            out[i] += 240.0
                            adjusted = True
                            munkidori_route_adjusted = True
                        elif 7 in target_es and 5 not in target_es and card != 5:
                            out[i] -= 95.0
                            adjusted = True
                    elif typ == TYPE_ATTACH and target_id != 112:
                        if card == 5 and not my_munkidori_has_psychic:
                            out[i] -= 230.0
                            adjusted = True
                        elif card == 7 and my_munkidori_needs_dark:
                            out[i] -= 190.0
                            adjusted = True
                        elif any_munkidori_attack_ready:
                            out[i] -= 24.0
                            adjusted = True
                    elif typ == TYPE_RETREAT and active_id == 112 and (5 in active_energy_ids or 7 in active_energy_ids):
                        out[i] -= 260.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif typ == TYPE_END and active_id == 112 and (5 in active_energy_ids or 7 in active_energy_ids):
                        out[i] -= 160.0
                        adjusted = True
                        munkidori_route_adjusted = True

                if select_context == 0 and jamming_available and stadium_id in {1257, 1264}:
                    if typ == TYPE_PLAY and card == 1246:
                        out[i] += 54.0
                        adjusted = True
                    elif typ == TYPE_ATTACK and active_id != 121:
                        out[i] -= 6.0
                        adjusted = True

                if select_context not in (8, 13, 14, 16) and typ == TYPE_CARD and target_opp_bench:
                    target_has_mist = bool(11 in target_energy)
                    if boss_break_window:
                        if target_id == 344:
                            out[i] += 130.0 if 0 < target_hp <= 80 else 88.0
                        elif target_id == 345:
                            out[i] -= 120.0
                        elif target_id not in {344, 345} and 0 < target_hp <= 200:
                            out[i] += 75.0
                        else:
                            out[i] -= 16.0
                        adjusted = True
                    elif boss_rotate_wall_window and target_id in {344, 345}:
                        out[i] += 155.0
                        if target_damage > 0 or (0 < target_hp <= 80):
                            out[i] += 35.0
                        adjusted = True
                    elif boss_around_wall_window and target_id not in {344, 345}:
                        out[i] += 82.0
                        adjusted = True
                    elif target_id == 344:
                        out[i] += 96.0
                        adjusted = True
                    elif target_id == 345:
                        if target_has_mist:
                            out[i] -= 92.0
                        elif 0 < target_hp <= 70 or target_damage >= 80:
                            out[i] += 72.0
                        else:
                            out[i] -= 18.0
                        adjusted = True
                    elif target_id == 756 and (opp_bench_has_dwebble or opp_bench_has_crustle):
                        out[i] -= 16.0
                        adjusted = True

                if (
                    select_context not in (8, 13, 14, 16)
                    and typ == TYPE_CARD
                    and opp_active_id == 345
                    and opp_bench_has_nonwall_target
                ):
                    if card == 1182:
                        out[i] += 96.0
                        adjusted = True
                    elif card == 1120:
                        out[i] -= 38.0
                        adjusted = True
                    elif card in {1227, 1213} and my_deck <= 12:
                        out[i] -= 18.0
                        adjusted = True

                if select_context == 13 and typ == TYPE_CARD and target_is_own:
                    # Munkidori's transfer route should heal the damaged
                    # Dragapult ex first, then any other damaged attacker.
                    if target_id == 121 and target_damage > 0:
                        out[i] += 118.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif target_damage > 0:
                        out[i] += 42.0
                        adjusted = True
                    else:
                        out[i] -= 25.0
                        adjusted = True

                if select_context == 14 and typ == TYPE_CARD and target_is_opp and target_fainted:
                    out[i] -= 220.0
                    dca_redirected = True
                    adjusted = True

                if select_context == 14 and typ == TYPE_CARD and target_opp_bench and not target_fainted:
                    target_protected = dca_protected(target)
                    if target_protected:
                        out[i] -= 180.0
                        dca_redirected = True
                        adjusted = True
                    elif target_id == 344:
                        out[i] += 90.0 if 0 < target_hp <= 60 else 48.0
                        dca_redirected = True
                        adjusted = True
                    elif target_id == 345:
                        out[i] += 95.0 if 0 < target_hp <= 60 else 52.0
                        dca_redirected = True
                        adjusted = True
                    elif target_id == 756:
                        if 0 < target_hp <= 60:
                            out[i] += 92.0
                        elif 0 < target_hp <= 120:
                            out[i] += 34.0
                        else:
                            out[i] += 8.0
                        dca_redirected = True
                        adjusted = True

                if select_context == 14 and typ == TYPE_CARD and target_is_opp and not target_is_bench and not target_fainted:
                    target_protected = dca_protected(target)
                    if target_protected:
                        out[i] -= 180.0
                        dca_redirected = True
                        adjusted = True
                    elif target_id == 344:
                        out[i] += 110.0 if 0 < target_hp <= 60 else 70.0
                        dca_redirected = True
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif target_id == 345:
                        if 0 < target_hp <= 60 or target_damage >= 70:
                            out[i] += 126.0
                        elif target_damage > 0:
                            out[i] += 88.0
                        else:
                            out[i] += 54.0
                        dca_redirected = True
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif target_id == 756 and not (opp_bench_has_dwebble or opp_bench_has_crustle):
                        out[i] += 20.0
                        dca_redirected = True
                        adjusted = True

                if select_context == 30 and typ == TYPE_ENERGY and target_is_own and target_id == 112:
                    target_e_list = energy_id_list(target)
                    if card == 5:
                        out[i] -= 650.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif card == 7:
                        if target_e_list.count(7) <= 1 or 5 in set(target_e_list):
                            out[i] -= 520.0
                        else:
                            out[i] -= 120.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif 5 in set(target_e_list) and len(target_e_list) >= 2:
                        out[i] += 80.0
                        adjusted = True
                        munkidori_route_adjusted = True

                if select_context == 8 and typ == TYPE_CARD:
                    if card == 1182 and (opp_bench_has_dwebble or opp_bench_has_crustle):
                        out[i] -= 42.0
                        adjusted = True
                    elif card == 1182 and opp_active_id == 345 and opp_bench_has_nonwall_target:
                        out[i] -= 42.0
                        adjusted = True
                    elif card == 1246 and stadium_id not in {0, 1246}:
                        out[i] -= 28.0
                        adjusted = True
                    elif card == 1198 and not dragapult_stage2_attack_ready_on_board:
                        out[i] -= 18.0
                        adjusted = True
                    elif card == 112 and not my_munkidori_on_board:
                        out[i] -= 58.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif card == 7 and my_munkidori_needs_dark:
                        out[i] -= 44.0
                        adjusted = True
                        munkidori_route_adjusted = True

                if (
                    select_context not in (8, 13, 14, 16)
                    and typ == TYPE_CARD
                    and (opp_has_crustle_wall_line or opp_wall_damage > 0)
                ):
                    if (
                        card == 112
                        and not my_munkidori_on_board
                        and my_dragapult_line_count > 0
                        and (
                            dragapult_stage2_attack_ready_on_board
                            or active_id == 121
                            or my_damaged_any
                            or my_prizes <= 3
                        )
                    ):
                        out[i] += 70.0
                        adjusted = True
                        munkidori_route_adjusted = True
                    elif card == 112 and not my_munkidori_on_board and my_dragapult_count <= 0:
                        out[i] -= 80.0
                        adjusted = True
                    elif card == 7 and my_munkidori_needs_dark and (dragapult_stage2_attack_ready_on_board or my_damaged_any):
                        out[i] += 68.0
                        adjusted = True
                        munkidori_route_adjusted = True

                if active_line_under_wall:
                    if active_id == 120:
                        active_can_become_stage2 = bool(stage2_evolve_available)
                        active_stage2_survival_plan = bool(active_can_become_stage2 and not crustle_single_wall)
                        if (
                            opp_active_crustle_can_attack
                            and not active_stage2_survival_plan
                            and typ == TYPE_RETREAT
                        ):
                            out[i] += 360.0
                            adjusted = True
                        elif (
                            opp_active_crustle_can_attack
                            and not active_stage2_survival_plan
                            and typ == TYPE_ATTACK
                        ):
                            out[i] -= 520.0
                            adjusted = True
                        elif crustle_single_wall and typ == TYPE_EVOLVE and (card == 121 or int(opt_card2[i]) == 121):
                            out[i] -= 92.0
                            adjusted = True
                        elif typ == TYPE_EVOLVE and (card == 121 or int(opt_card2[i]) == 121):
                            out[i] += 72.0
                            adjusted = True
                        elif typ == TYPE_ATTACK and not stage2_evolve_available:
                            out[i] += 52.0
                            adjusted = True
                        elif typ == TYPE_RETREAT:
                            out[i] -= 76.0
                            adjusted = True
                        elif typ == TYPE_ATTACH and target_own_active:
                            missing = phantom_energy_ids.difference(active_energy_ids)
                            if card in missing:
                                out[i] += 34.0
                            elif card == 7:
                                out[i] -= 64.0
                            else:
                                out[i] -= 18.0
                            adjusted = True
                        elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_ATTACH, TYPE_END):
                            out[i] -= 22.0
                            adjusted = True
                    elif active_id == 119 and opp_active_crustle_can_attack:
                        if typ == TYPE_RETREAT:
                            out[i] += 300.0
                            adjusted = True
                        elif typ == TYPE_ATTACK:
                            out[i] -= 360.0
                            adjusted = True
                        elif typ == TYPE_ATTACH and target_own_active:
                            out[i] -= 260.0
                            adjusted = True
                        elif typ == TYPE_EVOLVE and not np.any(
                            (opt_type == TYPE_EVOLVE)
                            & ((opt_card == 121) | (opt_card2 == 121))
                        ):
                            out[i] -= 130.0
                            adjusted = True
                    elif active_id == 121 and not has_phantom_dive:
                        missing = phantom_energy_ids.difference(active_energy_ids)
                        after = set(active_energy_ids)
                        if typ == TYPE_PLAY and card == 1198:
                            out[i] += 120.0 if missing else 64.0
                            adjusted = True
                        elif typ == TYPE_PLAY and card == 1182:
                            # Do not Boss a Crustle wall forward before the
                            # Dragapult line can use Phantom Dive.  The traced
                            # loss used Boss with only P attached and lost the
                            # only real pressure window.
                            out[i] -= 150.0
                            adjusted = True
                        elif typ == TYPE_PLAY:
                            out[i] -= 70.0 if missing else 20.0
                            adjusted = True
                        elif typ == TYPE_ATTACH and target_own_active:
                            if card > 0:
                                after.add(card)
                            if phantom_energy_ids.issubset(after):
                                out[i] += 210.0
                            elif card in missing:
                                out[i] += 135.0
                            else:
                                out[i] -= 100.0
                            adjusted = True
                        elif typ == TYPE_ATTACK:
                            out[i] -= 115.0
                            adjusted = True
                        elif typ == TYPE_RETREAT:
                            if active_energy_ids and not bench_attack_ready_dragapult:
                                out[i] -= 1_000_000.0
                            else:
                                out[i] -= 240.0
                            adjusted = True
                        elif typ == TYPE_END:
                            out[i] -= 110.0
                            adjusted = True
                    elif active_id == 121 and has_phantom_dive and not boss_around_wall_window:
                        if crustle_single_wall and typ == TYPE_PLAY and card == 1120:
                            out[i] += 620.0 if crustle_single_wall_disrupt_window else 150.0
                            adjusted = True
                        elif (
                            active_dragapult_ko_risk
                            and typ == TYPE_ATTACH
                            and target_is_own
                            and target_is_bench
                            and target_id in dragapult_line_ids
                            and card in phantom_energy_ids.difference(target_energy)
                        ):
                            out[i] += 220.0
                            adjusted = True
                        elif typ == TYPE_ATTACK and attack_id == 154:
                            # Engine check: Crustle blocks the active ex attack
                            # damage, but Phantom Dive still supplies the bench
                            # damage-counter route unless the target has Mist.
                            # Do not pivot away from a ready Dragapult line.
                            if phantom_into_protected_wall:
                                if crustle_single_wall_disrupt_window or single_wall_munkidori_pivot_window:
                                    out[i] -= 1_000_000.0
                                elif np.any(opt_type != TYPE_ATTACK):
                                    out[i] -= 1_000_000.0
                                else:
                                    out[i] -= 300.0
                            elif active_dragapult_ko_risk and bench_replacement_attach_available:
                                out[i] -= 170.0
                            else:
                                out[i] += 190.0 if opp_bench_has_wall_target else 132.0
                            adjusted = True
                        elif typ == TYPE_ATTACK and attack_id == 153 and (crustle_single_wall or np.any(opt_attack == 154)):
                            if crustle_single_wall_disrupt_window or single_wall_munkidori_pivot_window:
                                out[i] -= 1_000_000.0
                            elif crustle_single_wall:
                                out[i] -= 420.0 if np.any(opt_type != TYPE_ATTACK) else 140.0
                            else:
                                out[i] -= 95.0
                            adjusted = True
                        elif phantom_into_protected_wall and typ == TYPE_ABILITY and card in {112, 120, 140}:
                            out[i] += 118.0 if card == 112 else 58.0
                            adjusted = True
                        elif phantom_into_protected_wall and typ == TYPE_ATTACH and target_id == 112:
                            if card == 5 and 5 not in target_energy:
                                out[i] += 260.0
                            elif card == 7 and 7 not in target_energy:
                                out[i] += 220.0
                            elif 5 in target_energy and len(energy_id_list(target)) < 2 and card > 0:
                                out[i] += 190.0
                            else:
                                out[i] += 42.0
                            adjusted = True
                        elif phantom_into_protected_wall and typ == TYPE_PLAY and card in {1120, 1246, 1182}:
                            out[i] += 620.0 if card == 1120 and crustle_single_wall_disrupt_window else 76.0
                            adjusted = True
                        elif phantom_into_protected_wall and typ == TYPE_RETREAT and bench_munkidori_attack_ready:
                            if munkidori_can_ko_active_wall or single_wall_munkidori_pivot_window:
                                out[i] += 520.0
                            else:
                                out[i] -= 210.0
                            adjusted = True
                        elif crustle_single_wall and typ in (TYPE_ABILITY, TYPE_PLAY) and card != 1182:
                            if card in {1120, 1246}:
                                out[i] += 34.0
                            elif card in {1227, 1231, 1213, 1080, 1152, 1086} and my_deck <= 14:
                                out[i] -= 62.0
                            else:
                                out[i] -= 18.0
                            adjusted = True
                        elif typ == TYPE_RETREAT:
                            out[i] -= 180.0
                            adjusted = True
                        elif typ in (TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE, TYPE_END):
                            out[i] -= 120.0 if typ == TYPE_END else 32.0
                            adjusted = True

            if adjusted:
                detail = "support_pivot" if active_support_stranded else "wall_break"
                if boss_break_window:
                    detail += "_boss"
                if boss_around_wall_window:
                    detail += "_boss_around"
                if boss_rotate_wall_window:
                    detail += "_boss_rotate_wall"
                if active_line_under_wall:
                    detail += "_active_line"
                if active_budew_crustle_lock_window:
                    detail += "_budew_lock"
                if crustle_force_active_choice:
                    detail += "_preserve_line_switch"
                if crustle_single_wall:
                    detail += "_single_wall"
                if dca_redirected:
                    detail += "_dca_redirect"
                if munkidori_route_adjusted:
                    detail += "_munkidori_convert"
                if stadium_id in {1257, 1264}:
                    detail += "_stadium"
                hit("dragapult_vs_crustle_plan", detail)

        alakazam_vs_dragapult_line_pick = bool(
            is_alakazam
            and rule_enabled("alakazam_vs_dragapult_plan")
            and opp_has_dragapult_line
            and n > 0
            and np.all(opt_type == TYPE_CARD)
            and select_context not in (8, 13, 14, 16)
            and my_alakazam_count <= 0
            and any(int(c) in {741, 742, 743, 305, 66} for c in opt_card.tolist())
        )
        if alakazam_vs_dragapult_line_pick:
            hit("alakazam_vs_dragapult_plan", "line_pick")
            for i in range(n):
                card = int(opt_card[i])
                if my_alakazam_line_count <= 0:
                    if card == 741:
                        out[i] += 28.0
                    elif card in {305, 66}:
                        out[i] += 12.0
                    elif card in {742, 743}:
                        out[i] -= 5.0
                elif my_kadabra_count <= 0 and my_alakazam_count <= 0:
                    if card == 742:
                        out[i] += 24.0
                    elif card == 741:
                        out[i] += 12.0
                    elif card == 743 and my_abra_count <= 0:
                        out[i] -= 8.0
                else:
                    if card == 743:
                        out[i] += 24.0
                    elif card == 742:
                        out[i] += 8.0
                if card == 305 and my_alakazam_line_count <= 1:
                    out[i] += 6.0

        alakazam_vs_dragapult_plan = bool(
            is_alakazam
            and rule_enabled("alakazam_vs_dragapult_plan")
            and select_context == 0
            and opp_has_dragapult_line
        )
        if alakazam_vs_dragapult_plan:
            adjusted = False
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                attack_id = int(opt_attack[i])
                if my_alakazam_count <= 0:
                    if typ == TYPE_PLAY and card == 741:
                        out[i] += 24.0
                        adjusted = True
                    elif typ == TYPE_EVOLVE and (card == 742 or card2 == 742):
                        out[i] += 18.0
                        adjusted = True
                    elif typ == TYPE_EVOLVE and (card == 743 or card2 == 743):
                        out[i] += 28.0
                        adjusted = True
                    elif typ == TYPE_PLAY and card == 1079 and has_alakazam_stage2_in_hand:
                        out[i] += 24.0
                        adjusted = True
                    elif typ == TYPE_PLAY and card in {305, 66}:
                        out[i] += 10.0
                        adjusted = True
                    elif typ == TYPE_END:
                        out[i] -= 12.0
                        adjusted = True
                else:
                    if typ == TYPE_PLAY and card == 1182 and (opp_dreepy_count + opp_drakloak_count) > 0:
                        out[i] += 28.0
                        adjusted = True
                    elif typ == TYPE_PLAY and card == 1197 and opp_hand_count >= 4:
                        out[i] += 18.0
                        adjusted = True
                    elif typ == TYPE_ATTACK and attack_id == 1072:
                        out[i] += 24.0
                        adjusted = True
                    elif typ == TYPE_EVOLVE and (card == 742 or card2 == 742 or card == 743 or card2 == 743):
                        out[i] += 8.0
                        adjusted = True
            if adjusted:
                hit("alakazam_vs_dragapult_plan", f"opp_dreepy{opp_dreepy_count}_drak{opp_drakloak_count}_hand{opp_hand_count}")
        prize_danger = bool(active_is_primary and has_attack and opp_prizes <= 2)
        if prize_danger:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    out[i] += 12.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE, TYPE_RETREAT):
                    out[i] -= 4.0
        tempo_attack = bool(
            is_dragapult
            and active_id == 121
            and has_attack
            and high_damage_attack_available
            and (prize_deficit >= 2 or (opp_prizes <= 4 and my_deck <= 18))
        )
        if tempo_attack:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    out[i] += 18.0 if prize_deficit >= 2 else 12.0
                elif typ == TYPE_RETREAT:
                    out[i] -= 10.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE):
                    out[i] -= 6.0 if prize_deficit >= 2 else 4.0
        alakazam_pressure_attack = bool(
            is_dragapult
            and active_id == 121
            and has_attack
            and high_damage_attack_available
            and opp_has_alakazam_line
        )
        if alakazam_pressure_attack:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    bonus = 12.0
                    if opt_feats.ndim == 2 and opt_feats.shape[1] > 57 and float(opt_feats[i, 57]) > 0.0:
                        bonus += 4.0
                    out[i] += bonus
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE):
                    out[i] -= 4.0
                elif typ == TYPE_RETREAT:
                    out[i] -= 8.0
        dragapult_upgrade_attack_attach = bool(
            is_dragapult
            and select_context == 0
            and active_id == 121
            and has_attack
            and has_attach
            and not high_damage_attack_available
            and (opp_has_alakazam_line or opp_prizes <= 4)
        )
        if dragapult_upgrade_attack_attach:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                if typ == TYPE_ATTACH and target_own_active:
                    out[i] += 22.0
                elif typ == TYPE_ATTACK:
                    out[i] -= 14.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_RETREAT):
                    out[i] -= 3.0
        dragapult_crispin_readiness = bool(
            is_dragapult
            and rule_enabled("dragapult_crispin_readiness")
            and select_context == 0
            and opp_has_alakazam_line
            and active_is_dragapult_line
            and np.any((opt_type == TYPE_PLAY) & (opt_card == 1198))
            and not (active_id == 121 and high_damage_attack_available)
            and (2 not in active_energy_ids or 5 not in active_energy_ids or prize_deficit >= 1)
        )
        if dragapult_crispin_readiness:
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                if typ == TYPE_PLAY and card == 1198:
                    out[i] += 28.0
                elif typ == TYPE_ATTACH and target_own_active and active_id in dragapult_line_ids:
                    out[i] += 4.0
                elif typ == TYPE_PLAY and card in {1120, 1182, 1213, 1246}:
                    out[i] -= 8.0
                elif typ == TYPE_ATTACK and active_id == 121:
                    out[i] -= 8.0
                elif typ == TYPE_END:
                    out[i] -= 10.0
        dragapult_damage_counter_cleanup = bool(
            is_dragapult
            and select_context == 14
            and n > 0
            and np.all(opt_type == TYPE_CARD)
        )
        if dragapult_damage_counter_cleanup:
            for i in range(n):
                card = int(opt_card[i])
                target_opp_bench = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 70 and float(opt_feats[i, 70]) > 0.0)
                target_hp_ratio = float(opt_feats[i, 21]) if opt_feats.ndim == 2 and opt_feats.shape[1] > 21 else 0.0
                target_can_ko = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 68 and float(opt_feats[i, 68]) > 0.0)
                target_one_counter_ko = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 67 and float(opt_feats[i, 67]) > 0.0)
                overkill = float(opt_feats[i, 71]) if opt_feats.ndim == 2 and opt_feats.shape[1] > 71 else 0.0
                if card <= 0 or not target_opp_bench:
                    out[i] -= 4.0
                elif target_hp_ratio <= 0.0:
                    # Phantom Dive resolves the six counters as a sequence and
                    # the engine may keep a soon-to-be-discarded bench Pokemon
                    # selectable until the whole attack finishes.  Top
                    # Dragapult traces repeatedly continue selecting Abra past
                    # 0 HP to remove the evolution root at resolution time.
                    if opp_has_alakazam_line and card in {741, 742}:
                        out[i] += 8.0
                        hit("dragapult_vs_alakazam_plan", "dca_finish_line")
                    else:
                        out[i] -= 30.0
                else:
                    if target_can_ko:
                        out[i] += 10.0
                    if target_one_counter_ko:
                        out[i] += 8.0
                    if card in {741, 742, 743}:
                        out[i] += 16.0 if (opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan")) else 5.0
                    elif card in {140, 66}:
                        if opp_has_alakazam_line and rule_enabled("dragapult_vs_alakazam_plan"):
                            out[i] -= 2.0
                        else:
                            out[i] += 2.0
                    out[i] -= 4.0 * overkill
        dragapult_pivot_ready = bool(
            is_dragapult
            and rule_enabled("dragapult_pivot_ready")
            and select_context == 0
            and bench_attack_ready_dragapult
            and active_id != 121
            and (opp_has_alakazam_line or prize_deficit >= 1 or opp_prizes <= 5 or my_deck <= 30)
        )
        if dragapult_pivot_ready:
            active_has_energy = bool(energy_ids(my_active_obj))
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_RETREAT:
                    out[i] += 28.0
                elif typ == TYPE_ATTACH and target_own_active:
                    out[i] += 26.0 if not active_has_energy else 10.0
                elif typ == TYPE_ATTACH and target_setup_or_evo:
                    out[i] -= 8.0
                elif typ == TYPE_END:
                    out[i] -= 12.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_EVOLVE):
                    out[i] -= 4.0
        drakloak_ko_window = bool(
            is_dragapult
            and rule_enabled("drakloak_ko")
            and active_id == 120
            and opp_has_alakazam_line
            and attack_ko_available
            and not stage2_evolve_available
        )
        if drakloak_ko_window:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    bonus = 20.0
                    if opt_feats.ndim == 2 and opt_feats.shape[1] > 57 and float(opt_feats[i, 57]) > 0.0:
                        bonus += 10.0
                    out[i] += bonus
                elif typ == TYPE_EVOLVE and (int(opt_card[i]) == 121 or int(opt_card2[i]) == 121):
                    out[i] += 6.0
                elif typ in (TYPE_ABILITY, TYPE_PLAY, TYPE_ATTACH, TYPE_RETREAT):
                    out[i] -= 8.0
                elif typ == TYPE_EVOLVE:
                    out[i] -= 4.0
        needs_backup_before_attack = bool(
            is_dragapult
            and rule_enabled("backup_before_attack")
            and active_id == 121
            and has_attack
            and opp_has_alakazam_line
            and bench_line_count <= 0
            and my_prizes >= 4
            and opp_prizes >= 4
            and my_deck > 20
            and np.any(opt_type == TYPE_PLAY)
        )
        if needs_backup_before_attack:
            backup_search_ids = {
                119,   # Dreepy
                1086,  # Buddy-Buddy Poffin
                1121,  # Ultra Ball
                1152,  # Poke Pad
                140,   # Fezandipiti ex
            }
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                if typ == TYPE_PLAY and card == 119:
                    out[i] += 28.0
                elif typ == TYPE_PLAY and card == 1086:
                    out[i] += 26.0
                elif typ == TYPE_PLAY and card == 1121:
                    out[i] += 24.0
                elif typ == TYPE_PLAY and card == 1152:
                    out[i] += 18.0
                elif typ == TYPE_PLAY and card == 140:
                    out[i] += 14.0
                elif typ == TYPE_PLAY and card not in backup_search_ids:
                    out[i] -= 4.0
                elif typ == TYPE_ATTACK:
                    out[i] -= 10.0
        tempo_attach = bool(
            is_dragapult
            and has_attach
            and not has_attack
            and (prize_deficit >= 2 or opp_prizes <= 3)
            and active_id in {119, 120, 121}
        )
        if tempo_attach:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_ATTACH:
                    bonus = 7.0
                    if active_id == 121 and target_own_active:
                        bonus += 8.0
                    elif target_own_active or target_setup_or_evo:
                        bonus += 5.0
                    out[i] += bonus
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_RETREAT):
                    out[i] -= 5.0
        setup_attach_to_line = bool(
            is_dragapult
            and has_attach
            and not active_is_dragapult_line
            and encoded.board_cards.size >= 6
            and any(int(c) in dragapult_line_ids for c in np.asarray(encoded.board_cards[1:6], dtype=np.int64).tolist())
        )
        if setup_attach_to_line:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_ATTACH and target_setup_or_evo:
                    out[i] += 8.0
                elif typ == TYPE_ATTACH and target_own_active:
                    out[i] -= 6.0
        active_line_attach = bool(
            is_dragapult
            and rule_enabled("active_line_attach")
            and select_context == 0
            and has_attach
            and not has_attack
            and active_is_dragapult_line
        )
        if active_line_attach:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                if typ == TYPE_ATTACH and target_own_active:
                    bonus = 12.0
                    if active_id == 121:
                        bonus += 8.0
                    if opp_has_alakazam_line or opp_prizes <= 5:
                        bonus += 6.0
                    out[i] += bonus
                elif typ == TYPE_ATTACH:
                    out[i] -= 5.0
                elif typ in (TYPE_ABILITY, TYPE_PLAY):
                    out[i] -= 4.0 if opp_has_alakazam_line else 2.0
                elif typ == TYPE_END:
                    out[i] -= 8.0
        backup_energy_attach = bool(
            is_dragapult
            and rule_enabled("backup_energy_attach")
            and select_context == 0
            and active_id == 121
            and has_attack
            and has_attach
            and bench_line_count > 0
            and (opp_has_alakazam_line or opp_prizes <= 4)
            and (active_hp <= 180 or (active_max_hp > 0 and active_hp / active_max_hp <= 0.55) or opp_prizes <= 3)
        )
        if backup_energy_attach:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_own_bench = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 35 and float(opt_feats[i, 35]) > 0.0)
                target_primary = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 59 and float(opt_feats[i, 59]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_ATTACH and target_own_bench and (target_primary or target_setup_or_evo):
                    out[i] += 18.0
                elif typ == TYPE_ATTACH and target_own_active:
                    out[i] -= 10.0
                elif typ == TYPE_ATTACH:
                    out[i] -= 3.0
        dragapult_backup_line_energy = bool(
            is_dragapult
            and rule_enabled("dragapult_backup_line_energy")
            and select_context == 0
            and active_id == 121
            and has_attack
            and high_damage_attack_available
            and has_attach
            and bench_line_count > 0
            and opp_has_alakazam_line
            and my_prizes >= 2
        )
        if dragapult_backup_line_energy:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_own_bench = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 35 and float(opt_feats[i, 35]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_ATTACH and target_own_bench and target_setup_or_evo:
                    out[i] += 20.0
                elif typ == TYPE_ATTACH and target_own_active:
                    out[i] -= 8.0
                elif typ == TYPE_ATTACK:
                    out[i] -= 3.0
        support_retreat_to_line = bool(
            is_dragapult
            and rule_enabled("support_retreat")
            and select_context == 0
            and has_retreat
            and not active_is_dragapult_line
            and encoded.board_cards.size >= 6
            and any(int(c) in dragapult_line_ids for c in np.asarray(encoded.board_cards[1:6], dtype=np.int64).tolist())
            and (not has_attack or opp_has_alakazam_line or opp_prizes <= 5)
        )
        if support_retreat_to_line:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_RETREAT:
                    out[i] += 14.0 if opp_has_alakazam_line else 10.0
                elif typ == TYPE_END:
                    out[i] -= 8.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY):
                    out[i] -= 3.0
        under_pressure_evolve = bool(
            is_dragapult
            and active_id == 120
            and (
                prize_deficit >= 2
                or opp_prizes <= 3
                or my_deck <= 16
                or (rule_enabled("alakazam_evolve_pressure") and opp_has_alakazam_line)
            )
            and stage2_evolve_available
        )
        if under_pressure_evolve:
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_EVOLVE and (card == 121 or card2 == 121):
                    out[i] += 16.0
                elif typ == TYPE_ABILITY:
                    out[i] -= 8.0
                elif typ in (TYPE_PLAY, TYPE_ATTACH):
                    out[i] -= 4.0
        drakloak_pressure_attack = bool(
            is_dragapult
            and rule_enabled("drakloak_pressure_attack")
            and active_id == 120
            and opp_has_alakazam_line
            and has_attack
            and not stage2_evolve_available
            and (opp_prizes <= 5 or prize_deficit >= 1 or my_deck <= 28)
        )
        if drakloak_pressure_attack:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    bonus = 14.0
                    if opt_feats.ndim == 2 and opt_feats.shape[1] > 57 and float(opt_feats[i, 57]) > 0.0:
                        bonus += 10.0
                    out[i] += bonus
                elif typ in (TYPE_ABILITY, TYPE_PLAY, TYPE_ATTACH, TYPE_RETREAT):
                    out[i] -= 5.0
                elif typ == TYPE_END:
                    out[i] -= 4.0
        empty_retreat = bool(
            is_dragapult
            and select_context == 0
            and has_retreat
            and active_id in {119, 120}
            and not has_attack
            and not has_stage2_on_bench
        )
        if empty_retreat:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_RETREAT:
                    out[i] -= 10.0
                elif typ == TYPE_END:
                    out[i] += 3.0
        primary_idle_retreat = bool(
            is_dragapult
            and select_context == 0
            and has_retreat
            and active_id == 121
            and not has_attack
            and (prize_deficit >= 1 or opp_prizes <= 3)
        )
        if primary_idle_retreat:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_RETREAT:
                    out[i] -= 14.0
                elif typ == TYPE_END:
                    out[i] += 6.0
                elif typ in (TYPE_ATTACH, TYPE_ABILITY):
                    out[i] += 2.0
        alakazam_fast_stage2 = bool(
            is_alakazam
            and rule_enabled("alakazam_fast_stage2")
            and select_context == 0
            and opp_has_dragapult_line
            and not has_alakazam_stage2_on_board
            and (alakazam_stage2_evolve_available or (rare_candy_available and has_alakazam_stage2_in_hand))
        )
        if alakazam_fast_stage2:
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_EVOLVE and (card in {743, 245} or card2 in {743, 245}):
                    out[i] += 26.0
                elif typ == TYPE_PLAY and card == 1079:
                    out[i] += 24.0
                elif typ == TYPE_EVOLVE and (card == 742 or card2 == 742) and rare_candy_available:
                    out[i] -= 10.0
                elif typ == TYPE_END:
                    out[i] -= 14.0
                elif typ == TYPE_PLAY and card in {741, 305, 66, 140}:
                    out[i] -= 5.0
        alakazam_active_line_energy = bool(
            is_alakazam
            and rule_enabled("alakazam_active_line_energy")
            and select_context == 0
            and opp_has_dragapult_line
            and has_attach
            and active_is_alakazam_line
            and not has_attack
        )
        if alakazam_active_line_energy:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                target_own_bench = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 35 and float(opt_feats[i, 35]) > 0.0)
                target_setup_or_evo = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 60 and float(opt_feats[i, 60]) > 0.0)
                if typ == TYPE_ATTACH and target_own_active:
                    out[i] += 18.0
                elif typ == TYPE_ATTACH and target_own_bench and target_setup_or_evo and active_id in {743, 245}:
                    out[i] += 6.0
                elif typ == TYPE_ATTACH:
                    out[i] -= 6.0
                elif typ == TYPE_END:
                    out[i] -= 8.0
        alakazam_attack_pressure = bool(
            is_alakazam
            and rule_enabled("alakazam_attack_pressure")
            and select_context == 0
            and opp_has_dragapult_line
            and has_attack
            and (active_id in {743, 245} or np.any((opt_type == TYPE_ATTACK) & np.isin(opt_attack, [1072, 338, 339])))
        )
        if alakazam_attack_pressure:
            for i in range(n):
                typ = int(opt_type[i])
                attack_id = int(opt_attack[i])
                if typ == TYPE_ATTACK and attack_id in {1072, 338, 339}:
                    out[i] += 18.0
                elif typ == TYPE_ATTACK:
                    out[i] += 4.0
                elif typ in (TYPE_PLAY, TYPE_ABILITY, TYPE_ATTACH, TYPE_EVOLVE):
                    out[i] -= 4.0
                elif typ == TYPE_RETREAT:
                    out[i] -= 6.0
        stuck_active = bool(
            has_primary_on_bench
            and not active_is_primary
            and not has_attack
            and not has_retreat
            and my_deck <= 14
        )
        if stuck_active:
            for i in range(n):
                typ = int(opt_type[i])
                target_own_active = bool(opt_feats.ndim == 2 and opt_feats.shape[1] > 34 and float(opt_feats[i, 34]) > 0.0)
                if typ == TYPE_ATTACH and target_own_active:
                    out[i] += 12.0 if my_deck > 8 else 18.0
                elif typ == TYPE_END:
                    out[i] -= 8.0 if my_deck <= 12 else 4.0
                elif typ == TYPE_ABILITY and my_deck <= 18:
                    out[i] -= 3.0
                elif typ == TYPE_PLAY and my_deck <= 8:
                    out[i] -= 2.0
        attack_pressure = bool(active_is_primary and has_attack and my_deck <= 18)
        if attack_pressure:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    out[i] += 5.0 if my_deck <= 14 else 3.0
                elif typ == TYPE_ABILITY and my_deck <= 16:
                    out[i] -= 3.0
                elif typ == TYPE_PLAY and my_deck <= 12:
                    out[i] -= 2.0
        closing = bool(
            my_deck <= 12
            or (my_deck <= 14 and my_prizes <= 3)
            or (my_deck <= 20 and my_prizes <= 1)
            or (my_deck <= 16 and prize_lead >= 3)
        )
        if not closing:
            return out

        # Avoid spending a won game on more resource churn.  Draw/search actions
        # are not all bad, but when deck is this low they must beat an explicit
        # finish/attack option, not the other way around.
        churn_types = {TYPE_ABILITY, TYPE_PLAY, TYPE_ATTACH}
        for i in range(n):
            typ = int(opt_type[i])
            if typ in churn_types:
                out[i] -= 3.5
            elif typ == TYPE_END and not has_attack and not has_retreat:
                out[i] += 1.5

        if has_attack:
            for i in range(n):
                typ = int(opt_type[i])
                if typ == TYPE_ATTACK:
                    bonus = 6.0
                    if opt_feats.ndim == 2 and opt_feats.shape[1] > 57 and float(opt_feats[i, 57]) > 0.0:
                        bonus += 2.0
                    if my_prizes <= 2:
                        bonus += 2.0
                    out[i] += bonus
                elif typ in (TYPE_EVOLVE, TYPE_RETREAT):
                    out[i] -= 1.0
        elif has_retreat:
            for i in range(n):
                if int(opt_type[i]) == TYPE_RETREAT:
                    out[i] += 5.0
                elif int(opt_type[i]) in churn_types:
                    out[i] -= 2.0
        else:
            # No attack/retreat. Prefer forming or preserving attackers without
            # drawing more. This catches active Stage-1 endgames where evolving
            # is better than another Drakloak ability into deck-out.
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_EVOLVE and (card in primary_ids or card2 in primary_ids):
                    out[i] += 4.0
                elif typ == TYPE_END:
                    out[i] += -4.0 if stuck_active else 2.0
                elif typ == TYPE_ABILITY:
                    out[i] -= 4.0

        if has_stage2_on_bench and not active_is_primary:
            for i in range(n):
                typ = int(opt_type[i])
                card = int(opt_card[i])
                card2 = int(opt_card2[i])
                if typ == TYPE_CARD and card in primary_ids:
                    out[i] += 18.0
                elif typ == TYPE_CARD and card in setup_ids:
                    out[i] -= 8.0
                elif typ == TYPE_EVOLVE and (card in primary_ids or card2 in primary_ids):
                    out[i] += 6.0
                elif typ in (TYPE_ABILITY, TYPE_PLAY) and my_deck <= 15:
                    out[i] -= 1.5
        if has_primary_on_bench and my_deck <= 15 and not has_attack:
            for i in range(n):
                if int(opt_type[i]) == TYPE_CARD and int(opt_card[i]) in primary_ids:
                    out[i] += 4.0

        # In follow-up selection contexts created by a low-deck action, avoid
        # optional picks when the engine allows choosing nothing.
        if int(sel.get("minCount", 0) or 0) == 0 and int(sel.get("maxCount", 0) or 0) > 0:
            if my_deck <= 4 and np.all(opt_type == TYPE_CARD):
                out -= 2.0
        return out

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
            select_context=int((obs_dict.get("select") or {}).get("context", -1) if (obs_dict.get("select") or {}).get("context") is not None else -1),
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
