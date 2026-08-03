"""
Fast observation encoder — raw dict → numpy arrays → torch tensors.

Design:
  - NO pydantic. Raw dict access only. The C engine produces valid data.
  - Pre-compute card metadata once at import time.
  - Vectorize where possible; single-pass over board/hand/options.

Speed target: <0.5ms per decision (pkm's pydantic path: ~2.5ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Constants (from engine spec) ───────────────────────────────────────────
N_CARDS = 1267          # total card IDs in the game
N_ATTACKS = 1556        # total attack IDs
N_OPT_TYPES = 17        # OptionType enum values
MAX_HAND = 25           # practical max hand size
BOARD_SLOTS = 12        # 1 active + 5 bench × 2 players
N_PRIZES = 6
BENCH_MAX = 5
DECK_SIZE = 60
MAX_HP = 400            # practical max HP

# ── Feature dimensions ────────────────────────────────────────────────────
CARD_DIM = 64           # card embedding
STATE_FEAT_DIM = 64     # scalar state features
OPT_FEAT_DIM = 48       # per-option scalar features


@dataclass
class EncodedDecision:
    """One decision point — pure numpy, crosses process boundary cheaply."""
    # State
    board_cards: np.ndarray      # [12] int64 — card IDs per board slot, 0=empty
    hand_cards: np.ndarray       # [MAX_HAND] int64 — card IDs in hand, 0-padded
    state_feats: np.ndarray      # [STATE_FEAT_DIM] float32
    # Options
    opt_type: np.ndarray         # [N] int64 — OptionType per option
    opt_card: np.ndarray         # [N] int64 — primary card ID
    opt_card2: np.ndarray        # [N] int64 — secondary card ID (attach target)
    opt_attack: np.ndarray       # [N] int64 — attack ID if applicable
    opt_feats: np.ndarray        # [N, OPT_FEAT_DIM] float32
    # Selection metadata
    min_count: int
    max_count: int
    # Aux labels (filled by trainer)
    action: list[int] | None = None
    logprob: float = 0.0
    reward: float = 0.0
    value: float = 0.0
    adv: float = 0.0
    ret: float = 0.0


class FastEncoder:
    """Encode an observation dict directly into numpy arrays. No pydantic."""

    def __init__(self):
        # Pre-load card metadata for fast lookup
        from cg.api import all_card_data
        cards = all_card_data()
        # card_id → (hp, retreat_cost, stage, ex, megaEx, weakness, resistance, energyType, attacks)
        self.card_hp = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_stage = np.zeros(N_CARDS + 2, dtype=np.int32)  # 0=basic,1=stage1,2=stage2
        self.card_ex = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_mega = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_weakness = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_resistance = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_energy = np.zeros(N_CARDS + 2, dtype=np.int32)
        self.card_retreat = np.zeros(N_CARDS + 2, dtype=np.int32)
        for c in cards:
            cid = c.cardId
            self.card_hp[cid] = c.hp
            self.card_stage[cid] = 2 if c.stage2 else 1 if c.stage1 else 0
            self.card_ex[cid] = 1 if c.ex else 0
            self.card_mega[cid] = 1 if c.megaEx else 0
            self.card_retreat[cid] = c.retreatCost

    def encode(self, obs_dict: dict) -> EncodedDecision:
        """Fast path: dict → EncodedDecision in one pass."""
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection — use encode_deck() instead")

        cur = obs_dict["current"]
        you = cur["yourIndex"]
        me = cur["players"][you]
        opp = cur["players"][1 - you]

        # ── Board: fixed 12 slots ────────────────────────────────────────
        board = np.zeros(BOARD_SLOTS, dtype=np.int64)
        # My active + bench
        if me.get("active"):
            p = me["active"][0]
            if p: board[0] = p["id"]
        for i, p in enumerate(me.get("bench", [])):
            if p and i < BENCH_MAX: board[1 + i] = p["id"]
        # Opponent active + bench
        if opp.get("active"):
            p = opp["active"][0]
            if p: board[6] = p["id"] if p else 0  # face-down = 0 = unknown
        for i, p in enumerate(opp.get("bench", [])):
            if p and i < BENCH_MAX: board[7 + i] = p["id"]

        # ── Hand ──────────────────────────────────────────────────────────
        hand = np.zeros(MAX_HAND, dtype=np.int64)
        my_hand = me.get("hand") or []
        for i, c in enumerate(my_hand[:MAX_HAND]):
            hand[i] = c["id"]

        options = sel.get("option", [])
        n_opt = len(options)
        opt_type_counts: dict[int, int] = {}
        for o in options:
            ot = int(o.get("type", 0) or 0)
            opt_type_counts[ot] = opt_type_counts.get(ot, 0) + 1

        my_inplay = self._in_play(me)
        opp_inplay = self._in_play(opp)
        my_active = me.get("active", [None])[0] if me.get("active") else None
        opp_active = opp.get("active", [None])[0] if opp.get("active") else None

        # ── State features ────────────────────────────────────────────────
        s = np.zeros(STATE_FEAT_DIM, dtype=np.float32)
        s[0] = cur.get("turn", 0) / 30.0
        s[1] = cur.get("turnActionCount", 0) / 50.0
        s[2] = 1.0 if cur.get("firstPlayer") == you else 0.0
        s[3] = 1.0 if cur.get("supporterPlayed") else 0.0
        s[4] = 1.0 if cur.get("energyAttached") else 0.0
        s[5] = 1.0 if cur.get("retreated") else 0.0
        s[6] = len(me.get("prize", [])) / N_PRIZES
        s[7] = len(opp.get("prize", [])) / N_PRIZES
        s[8] = me.get("deckCount", 60) / DECK_SIZE
        s[9] = opp.get("deckCount", 60) / DECK_SIZE
        s[10] = (len(me.get("hand")) if me.get("hand") else me.get("handCount", 0)) / 25.0
        s[11] = opp.get("handCount", 0) / 25.0
        s[12] = len(me.get("bench", [])) / BENCH_MAX
        s[13] = len(opp.get("bench", [])) / BENCH_MAX
        s[14] = 1.0 if cur.get("stadiumPlayed") else 0.0
        s[15] = len(cur.get("stadium") or []) / 1.0
        s[16] = len(cur.get("looking") or []) / 30.0
        s[17] = sel.get("context", 0) / 64.0
        s[18] = sel.get("type", 0) / 16.0
        s[19] = sel.get("minCount", 0) / 10.0
        s[20] = sel.get("maxCount", 0) / 10.0
        s[21] = sel.get("remainDamageCounter", 0) / 30.0
        s[22] = sel.get("remainEnergyCost", 0) / 10.0
        s[23] = 1.0 if sel.get("contextCard") else 0.0
        s[24] = 1.0 if sel.get("effect") else 0.0
        s[25] = self._damage_ratio(me.get("active", [None])[0] if me.get("active") else None)
        s[26] = self._energy_count(me.get("active", [None])[0] if me.get("active") else None) / 10.0
        s[27] = self._damage_ratio(opp.get("active", [None])[0] if opp.get("active") else None)
        s[28] = self._energy_count(opp.get("active", [None])[0] if opp.get("active") else None) / 10.0
        s[29] = sum(self._damage_ratio(p) for p in me.get("bench", []) if p) / BENCH_MAX
        s[30] = sum(self._damage_ratio(p) for p in opp.get("bench", []) if p) / BENCH_MAX
        s[31] = len(my_hand) / MAX_HAND
        s[32] = self._hp_ratio(my_active)
        s[33] = self._hp_ratio(opp_active)
        s[34] = self._retreat_cost(my_active) / 5.0
        s[35] = self._retreat_cost(opp_active) / 5.0
        s[36] = len((my_active or {}).get("tools") or []) / 4.0
        s[37] = len((opp_active or {}).get("tools") or []) / 4.0
        s[38] = self._stage(my_active) / 2.0
        s[39] = self._stage(opp_active) / 2.0
        s[40] = self._is_ex(my_active)
        s[41] = self._is_ex(opp_active)
        s[42] = self._is_mega(my_active)
        s[43] = self._is_mega(opp_active)
        s[44] = self._special_status(me)
        s[45] = self._special_status(opp)
        s[46] = len(me.get("discard") or []) / DECK_SIZE
        s[47] = len(opp.get("discard") or []) / DECK_SIZE
        # Game-plan / phase features. These expose action availability and
        # board development explicitly instead of forcing the model to infer
        # them only from unordered options and pooled card embeddings.
        s[48] = opt_type_counts.get(13, 0) / 4.0   # ATTACK options
        s[49] = opt_type_counts.get(8, 0) / 8.0    # ATTACH options
        s[50] = opt_type_counts.get(9, 0) / 8.0    # EVOLVE options
        s[51] = opt_type_counts.get(10, 0) / 8.0   # ABILITY options
        s[52] = opt_type_counts.get(7, 0) / 16.0   # PLAY options
        s[53] = opt_type_counts.get(12, 0) / 4.0   # RETREAT options
        s[54] = opt_type_counts.get(14, 0) / 2.0   # END options
        s[55] = n_opt / 32.0
        s[56] = max((self._energy_count(p) for p in my_inplay), default=0) / 10.0
        s[57] = max((self._energy_count(p) for p in opp_inplay), default=0) / 10.0
        s[58] = sum(1 for p in my_inplay if self._is_ex(p)) / 6.0
        s[59] = sum(1 for p in my_inplay if self._is_mega(p)) / 6.0
        s[60] = sum(1 for p in my_inplay if self._stage(p) >= 1) / 6.0
        s[61] = sum(1 for p in my_inplay if self._damage_ratio(p) > 0) / 6.0
        s[62] = sum(1 for p in opp_inplay if self._damage_ratio(p) > 0) / 6.0
        s[63] = max((self._damage_ratio(p) for p in opp_inplay), default=0.0)

        # ── Options ───────────────────────────────────────────────────────
        opt_type = np.zeros(n_opt, dtype=np.int64)
        opt_card = np.zeros(n_opt, dtype=np.int64)
        opt_card2 = np.zeros(n_opt, dtype=np.int64)
        opt_attack = np.zeros(n_opt, dtype=np.int64)
        opt_feats = np.zeros((n_opt, OPT_FEAT_DIM), dtype=np.float32)

        for i, o in enumerate(options):
            ot = int(o.get("type", 0) or 0)
            opt_type[i] = ot
            opt_feats[i, 0] = float(ot) / N_OPT_TYPES
            opt_feats[i, 2] = i / max(n_opt - 1, 1)
            opt_feats[i, 3] = sel.get("context", 0) / 64.0
            opt_feats[i, 4] = sel.get("type", 0) / 16.0
            opt_feats[i, 5] = o.get("number", 0) / 20.0 if o.get("number") is not None else 0.0
            opt_feats[i, 6] = o.get("count", 0) / 20.0 if o.get("count") is not None else 0.0

            # Resolve the card this option acts on
            pid = o.get("playerIndex", you)
            area = o.get("area")
            idx = o.get("index", 0)
            opt_feats[i, 7] = float(area or 0) / 16.0
            opt_feats[i, 8] = float(idx or 0) / 64.0
            opt_feats[i, 9] = float(o.get("inPlayArea") or 0) / 16.0
            opt_feats[i, 10] = float(o.get("inPlayIndex") or 0) / 10.0
            opt_feats[i, 11] = float(o.get("toolIndex") or 0) / 10.0
            opt_feats[i, 12] = float(o.get("energyIndex") or 0) / 10.0
            opt_feats[i, 13] = 1.0 if pid == you else 0.0
            opt_feats[i, 31] = n_opt / 32.0
            same_type_count = opt_type_counts.get(int(ot or 0), 0)
            opt_feats[i, 32] = same_type_count / max(n_opt, 1)
            opt_feats[i, 33] = same_type_count / 16.0
            opt_feats[i, 41] = 1.0 if ot in (7, 8, 9, 10) else 0.0
            opt_feats[i, 42] = 1.0 if ot == 13 else 0.0
            opt_feats[i, 43] = 1.0 if ot == 14 else 0.0
            opt_feats[i, 44] = 1.0 if ot == 12 else 0.0

            if area is not None and idx is not None:
                c = self._get_card(cur, pid, area, idx, sel)
                opt_card[i] = c
                target = self._get_pokemon(cur, pid, area, idx)
                opt_feats[i, 14] = self._damage_ratio(target)
                opt_feats[i, 15] = self._energy_count(target) / 10.0
                self._fill_option_extras(opt_feats[i], c, target, pid, you, area)
                self._fill_plan_target_extras(opt_feats[i], target, pid, you, area)
                if ot == 4:  # ToolCard: choose an attached tool, not only its Pokemon.
                    attached = self._get_attached_card(cur, pid, area, idx, "tools", o.get("toolIndex"))
                    if attached:
                        opt_card[i] = attached
                        opt_card2[i] = c
                elif ot in (5, 6):  # EnergyCard/Energy: choose an attached energy card.
                    attached = self._get_attached_card(cur, pid, area, idx, "energyCards", o.get("energyIndex"))
                    if attached:
                        opt_card[i] = attached
                        opt_card2[i] = c
            if o.get("inPlayArea") is not None and o.get("inPlayIndex") is not None:
                c2 = self._get_card(cur, pid, o.get("inPlayArea"), o.get("inPlayIndex"), sel)
                opt_card2[i] = c2
                target2 = self._get_pokemon(cur, pid, o.get("inPlayArea"), o.get("inPlayIndex"))
                if target2:
                    opt_feats[i, 14] = self._damage_ratio(target2)
                    opt_feats[i, 15] = self._energy_count(target2) / 10.0
                    self._fill_target_extras(opt_feats[i], target2, pid, you, o.get("inPlayArea"))
                    self._fill_plan_target_extras(opt_feats[i], target2, pid, you, o.get("inPlayArea"))
            if ot == 7 and o.get("index") is not None:  # Play: hand index only
                idx2 = int(o.get("index") or 0)
                opt_feats[i, 7] = 2.0 / 16.0
                opt_feats[i, 8] = float(idx2) / 64.0
                if idx2 < len(my_hand) and my_hand[idx2]:
                    opt_card[i] = my_hand[idx2]["id"]
                    self._fill_card_metadata(opt_feats[i], opt_card[i])
            if ot == 15:  # Skill: engine exposes card identity directly
                opt_card[i] = int(o.get("cardId") or 0)
                opt_feats[i, 8] = float(o.get("serial") or 0) / 64.0
                self._fill_card_metadata(opt_feats[i], opt_card[i])
            if ot == 16 and o.get("specialConditionType") is not None:
                opt_feats[i, 5] = float(o.get("specialConditionType") or 0) / 20.0
            if o.get("attackId") is not None:
                opt_attack[i] = o["attackId"]
                opt_feats[i, 30] = 1.0
            opt_feats[i, 1] = 1.0 if o.get("playerIndex") == you else 0.0

        return EncodedDecision(
            board_cards=board, hand_cards=hand, state_feats=s,
            opt_type=opt_type, opt_card=opt_card, opt_card2=opt_card2,
            opt_attack=opt_attack, opt_feats=opt_feats,
            min_count=sel.get("minCount", 0), max_count=sel.get("maxCount", 0),
        )

    @staticmethod
    def _damage_ratio(p: dict | None) -> float:
        if not p:
            return 0.0
        hp = float(p.get("hp", 0) or 0)
        max_hp = float(p.get("maxHp", 0) or 0)
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, (max_hp - hp) / max_hp))

    @staticmethod
    def _hp_ratio(p: dict | None) -> float:
        if not p:
            return 0.0
        return max(0.0, min(1.0, float(p.get("hp", 0) or 0) / MAX_HP))

    def _stage(self, p: dict | None) -> int:
        if not p:
            return 0
        cid = int(p.get("id") or 0)
        return int(self.card_stage[cid]) if 0 <= cid < len(self.card_stage) else 0

    def _is_ex(self, p: dict | None) -> float:
        if not p:
            return 0.0
        cid = int(p.get("id") or 0)
        return float(self.card_ex[cid]) if 0 <= cid < len(self.card_ex) else 0.0

    def _is_mega(self, p: dict | None) -> float:
        if not p:
            return 0.0
        cid = int(p.get("id") or 0)
        return float(self.card_mega[cid]) if 0 <= cid < len(self.card_mega) else 0.0

    def _retreat_cost(self, p: dict | None) -> int:
        if not p:
            return 0
        cid = int(p.get("id") or 0)
        return int(self.card_retreat[cid]) if 0 <= cid < len(self.card_retreat) else 0

    @staticmethod
    def _special_status(player: dict) -> float:
        return float(any(bool(player.get(k)) for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")))

    @staticmethod
    def _energy_count(p: dict | None) -> int:
        if not p:
            return 0
        energies = p.get("energies")
        if energies is not None:
            return len(energies)
        return len(p.get("energyCards") or [])

    @staticmethod
    def _in_play(player: dict) -> list[dict]:
        out = []
        active = player.get("active") or []
        if active and active[0]:
            out.append(active[0])
        out.extend([p for p in (player.get("bench") or []) if p])
        return out

    def _fill_card_metadata(self, feats: np.ndarray, card_id: int) -> None:
        cid = int(card_id or 0)
        if not (0 <= cid < len(self.card_hp)):
            return
        feats[16] = float(self.card_hp[cid]) / MAX_HP
        feats[17] = float(self.card_stage[cid]) / 2.0
        feats[18] = float(self.card_ex[cid])
        feats[19] = float(self.card_mega[cid])
        feats[20] = float(self.card_retreat[cid]) / 5.0

    def _fill_target_extras(self, feats: np.ndarray, target: dict | None,
                            player_idx: int, you: int, area: int | None) -> None:
        if not target:
            return
        feats[21] = self._hp_ratio(target)
        feats[22] = self._stage(target) / 2.0
        feats[23] = self._is_ex(target)
        feats[24] = self._is_mega(target)
        feats[25] = len(target.get("tools") or []) / 4.0
        feats[26] = 1.0 if area == 4 else 0.0
        feats[27] = 1.0 if area == 5 else 0.0
        feats[28] = 1.0 if player_idx == you else 0.0
        feats[29] = 1.0 if player_idx != you else 0.0

    def _fill_option_extras(self, feats: np.ndarray, card_id: int,
                            target: dict | None, player_idx: int,
                            you: int, area: int | None) -> None:
        self._fill_card_metadata(feats, card_id)
        self._fill_target_extras(feats, target, player_idx, you, area)

    def _fill_plan_target_extras(self, feats: np.ndarray, target: dict | None,
                                 player_idx: int, you: int, area: int | None) -> None:
        if not target:
            return
        target_energy = self._energy_count(target)
        feats[34] = 1.0 if player_idx == you and area == 4 else 0.0
        feats[35] = 1.0 if player_idx == you and area == 5 else 0.0
        feats[36] = 1.0 if player_idx != you and area == 4 else 0.0
        feats[37] = 1.0 if player_idx != you and area == 5 else 0.0
        feats[38] = target_energy / 10.0
        feats[39] = 1.0 if target_energy > 0 else 0.0
        feats[40] = 1.0 if self._damage_ratio(target) > 0 else 0.0
        cid = int(target.get("id") or 0)
        feats[45] = float(self.card_retreat[cid]) / 5.0 if 0 <= cid < len(self.card_retreat) else 0.0
        feats[46] = len(target.get("tools") or []) / 4.0
        feats[47] = len(target.get("energyCards") or target.get("energies") or []) / 10.0

    @staticmethod
    def _get_pokemon(cur: dict, player_idx: int, area: int, index: int) -> dict | None:
        ps = cur["players"][player_idx]
        if area == 4:  # ACTIVE
            a = ps.get("active", [])
            if a and index < len(a) and a[index]:
                return a[index]
        if area == 5:  # BENCH
            b = ps.get("bench", [])
            if index < len(b) and b[index]:
                return b[index]
        return None

    @classmethod
    def _get_attached_card(cls, cur: dict, player_idx: int, area: int, index: int,
                           key: str, attached_index: int | None) -> int:
        if attached_index is None:
            return 0
        target = cls._get_pokemon(cur, player_idx, area, index)
        if not target:
            return 0
        attached = target.get(key) or []
        ai = int(attached_index)
        if 0 <= ai < len(attached) and attached[ai]:
            return int(attached[ai].get("id") or 0)
        return 0

    @staticmethod
    def _get_card(cur: dict, player_idx: int, area: int, index: int, sel: dict | None = None) -> int:
        """Resolve (player, area, index) → card_id. 0 = not found."""
        ps = cur["players"][player_idx]
        # AreaType: 1=DECK, 2=HAND, 3=DISCARD, 4=ACTIVE, 5=BENCH, 6=PRIZE, 7=STADIUM, 11=LOOKING
        if area == 1:  # DECK
            deck = (sel or {}).get("deck")
            if deck and index < len(deck) and deck[index]:
                return deck[index]["id"]
            return 0
        elif area == 2:  # HAND
            h = ps.get("hand")
            if h and index < len(h) and h[index]:
                return h[index]["id"]
        elif area == 3:  # DISCARD
            d = ps["discard"]
            if index < len(d) and d[index]:
                return d[index]["id"]
        elif area == 4:  # ACTIVE
            a = ps.get("active", [])
            if a and index < len(a) and a[index]:
                return a[index]["id"]
        elif area == 5:  # BENCH
            b = ps.get("bench", [])
            if index < len(b) and b[index]:
                return b[index]["id"]
        elif area == 6:  # PRIZE
            p = ps.get("prize", [])
            if index < len(p) and p[index]:
                return p[index]["id"]
        elif area == 7:  # STADIUM
            s = cur.get("stadium", [])
            if index < len(s) and s[index]:
                return s[index]["id"]
        elif area == 11:  # LOOKING
            lk = cur.get("looking")
            if lk and index < len(lk) and lk[index]:
                return lk[index]["id"]
        return 0
