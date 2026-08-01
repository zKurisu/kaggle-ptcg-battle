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
STATE_FEAT_DIM = 32     # scalar state features
OPT_FEAT_DIM = 16       # per-option scalar features


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

        # ── Options ───────────────────────────────────────────────────────
        options = sel.get("option", [])
        n_opt = len(options)
        opt_type = np.zeros(n_opt, dtype=np.int64)
        opt_card = np.zeros(n_opt, dtype=np.int64)
        opt_card2 = np.zeros(n_opt, dtype=np.int64)
        opt_attack = np.zeros(n_opt, dtype=np.int64)
        opt_feats = np.zeros((n_opt, OPT_FEAT_DIM), dtype=np.float32)

        for i, o in enumerate(options):
            ot = o.get("type", 0)
            opt_type[i] = ot
            opt_feats[i, 0] = float(ot) / N_OPT_TYPES

            # Resolve the card this option acts on
            pid = o.get("playerIndex", you)
            area = o.get("area")
            idx = o.get("index", 0)

            if area is not None and idx is not None:
                c = self._get_card(cur, pid, area, idx)
                opt_card[i] = c
            if o.get("inPlayArea") is not None and o.get("inPlayIndex") is not None:
                c2 = self._get_card(cur, pid, o.get("inPlayArea"), o.get("inPlayIndex"))
                opt_card2[i] = c2
            if o.get("attackId") is not None:
                opt_attack[i] = o["attackId"]
            opt_feats[i, 1] = 1.0 if o.get("playerIndex") == you else 0.0

        return EncodedDecision(
            board_cards=board, hand_cards=hand, state_feats=s,
            opt_type=opt_type, opt_card=opt_card, opt_card2=opt_card2,
            opt_attack=opt_attack, opt_feats=opt_feats,
            min_count=sel.get("minCount", 0), max_count=sel.get("maxCount", 0),
        )

    @staticmethod
    def _get_card(cur: dict, player_idx: int, area: int, index: int) -> int:
        """Resolve (player, area, index) → card_id. 0 = not found."""
        ps = cur["players"][player_idx]
        # AreaType: 1=DECK, 2=HAND, 3=DISCARD, 4=ACTIVE, 5=BENCH, 6=PRIZE, 7=STADIUM, 11=LOOKING
        if area == 1:  # DECK
            return 0  # deck cards are not revealed
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
