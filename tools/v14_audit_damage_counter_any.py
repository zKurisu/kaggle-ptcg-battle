#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.seq.constants import DAMAGE_COUNTER_ANY_CONTEXT
from ptcg_rl.seq.data import discover_sequence_npz


def _split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _first_valid_action(action: object, nopt: int) -> int:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    arr = arr[(arr >= 0) & (arr < nopt)]
    return int(arr[0]) if arr.size else -1


def _group_rows(z, indices: list[int]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    cur: list[int] = []
    prev_game = None
    prev_dec = None
    for i in indices:
        game = str(z["game_key"][i]) if "game_key" in z else f"{z['episode_id'][i]}:{z['player_index'][i]}"
        dec = int(z["decision_index"][i]) if "decision_index" in z else i
        ctx = int(z["act_context"][i]) if "act_context" in z else -1
        contiguous = prev_game == game and prev_dec is not None and dec == prev_dec + 1
        if ctx == DAMAGE_COUNTER_ANY_CONTEXT and (not cur or contiguous):
            cur.append(i)
        else:
            if cur:
                groups.append(_summarize_group(z, cur))
                cur = []
            if ctx == DAMAGE_COUNTER_ANY_CONTEXT:
                cur.append(i)
        prev_game = game
        prev_dec = dec
    if cur:
        groups.append(_summarize_group(z, cur))
    return groups


def _group_dca_rows_fast(z, dca_indices: np.ndarray) -> list[dict[str, object]]:
    """Summarize extracted DamageCounterAny groups without scanning all rows.

    v14 extraction writes dca_group_index/dca_pos/dca_len directly. The original
    audit re-inferred groups by walking every row and repeatedly reading object
    columns, which is far too slow on multi-day corpora.
    """
    groups: list[dict[str, object]] = []
    if dca_indices.size == 0:
        return groups

    if "dca_group_index" not in z:
        return _group_rows(z, [int(i) for i in dca_indices])

    dca_pos = np.asarray(z["dca_pos"]) if "dca_pos" in z else np.zeros(len(z["dca_group_index"]), dtype=np.int16)
    starts = dca_indices[dca_pos[dca_indices] == 0]
    if starts.size == 0:
        starts = dca_indices[:1]
    for start in starts:
        start_i = int(start)
        length = int(z["dca_len"][start_i]) if "dca_len" in z else 1
        length = max(length, 1)
        groups.append(_summarize_group_fast(z, start_i, length))
    return groups


def _summarize_group_fast(z, first: int, length: int) -> dict[str, object]:
    end = min(first + length, len(z["dca_group_index"]))
    if "dca_selected_slot" in z:
        slots = [int(x) for x in np.asarray(z["dca_selected_slot"][first:end], dtype=np.int16).reshape(-1)]
    else:
        slots = []
    slot_counts = Counter(slots)
    unique_slots = int(z["dca_group_unique_slots"][first]) if "dca_group_unique_slots" in z else len([k for k in slot_counts if k >= 0])
    focus_frac = float(z["dca_group_focus_frac"][first]) if "dca_group_focus_frac" in z else max(slot_counts.values(), default=0) / max(length, 1)
    won = int(z["won"][first]) if "won" in z else 0
    return {
        "game_key": str(z["game_key"][first]) if "game_key" in z else f"{z['episode_id'][first]}:{z['player_index'][first]}",
        "episode_id": str(z["episode_id"][first]) if "episode_id" in z else "",
        "player_index": int(z["player_index"][first]) if "player_index" in z else -1,
        "deck_sig": str(z["deck_sig"][first]) if "deck_sig" in z else "",
        "team_name": str(z["team_name"][first]) if "team_name" in z else "",
        "score": float(z["score"][first]) if "score" in z else 0.0,
        "opponent_archetype": str(z["opponent_archetype"][first]) if "opponent_archetype" in z else "",
        "opponent_deck_sig": str(z["opponent_deck_sig"][first]) if "opponent_deck_sig" in z else "",
        "won": won,
        "length": length,
        "unique_slots": unique_slots,
        "unique_cards": 0,
        "focus_frac": focus_frac,
        "slot_sequence": " ".join(str(x) for x in slots),
        "card_sequence": "",
        "hp_before_sequence": "",
        "first_decision": int(z["decision_index"][first]) if "decision_index" in z else int(first),
    }


def _summarize_group(z, rows: list[int]) -> dict[str, object]:
    first = rows[0]
    slots = []
    target_cards = []
    hp_before = []
    for i in rows:
        ot = np.asarray(z["ot"][i], dtype=np.int64).reshape(-1)
        slot = _first_valid_action(z["action"][i], len(ot))
        slots.append(slot)
        if slot >= 0:
            oc = np.asarray(z["oc"][i], dtype=np.int64).reshape(-1)
            of_arr = np.asarray(z["of_arr"][i], dtype=np.float32)
            target_cards.append(int(oc[slot]) if slot < len(oc) else 0)
            hp_before.append(float(of_arr[slot, 21] * 400.0) if of_arr.ndim == 2 and slot < of_arr.shape[0] and of_arr.shape[1] > 21 else 0.0)
        else:
            target_cards.append(0)
            hp_before.append(0.0)
    slot_counts = Counter(slots)
    card_counts = Counter(target_cards)
    won = int(z["won"][first]) if "won" in z else 0
    return {
        "game_key": str(z["game_key"][first]) if "game_key" in z else f"{z['episode_id'][first]}:{z['player_index'][first]}",
        "episode_id": str(z["episode_id"][first]) if "episode_id" in z else "",
        "player_index": int(z["player_index"][first]) if "player_index" in z else -1,
        "deck_sig": str(z["deck_sig"][first]) if "deck_sig" in z else "",
        "team_name": str(z["team_name"][first]) if "team_name" in z else "",
        "score": float(z["score"][first]) if "score" in z else 0.0,
        "opponent_archetype": str(z["opponent_archetype"][first]) if "opponent_archetype" in z else "",
        "opponent_deck_sig": str(z["opponent_deck_sig"][first]) if "opponent_deck_sig" in z else "",
        "won": won,
        "length": len(rows),
        "unique_slots": len([k for k in slot_counts if k >= 0]),
        "unique_cards": len([k for k in card_counts if k > 0]),
        "focus_frac": max(slot_counts.values(), default=0) / max(len(rows), 1),
        "slot_sequence": " ".join(str(x) for x in slots),
        "card_sequence": " ".join(str(x) for x in target_cards),
        "hp_before_sequence": " ".join(f"{x:.0f}" for x in hp_before),
        "first_decision": int(z["decision_index"][first]) if "decision_index" in z else int(first),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["600-699", "700-799", "800-899", "900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=100000)
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    deck_sigs = set(_split_csv(args.deck_sig))
    paths = discover_sequence_npz(
        args.corpus,
        args.archetype,
        _split_csv(args.score_bands),
        date_from=args.date_from,
        date_to=args.date_to,
    )
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.corpus, args.archetype.replace(" ", "_"), "*", "*.npz")))
    print(f"paths={len(paths)} archetype={args.archetype}", flush=True)

    all_groups: list[dict[str, object]] = []
    rows_seen = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            n = len(z["board"])
            limit_n = n if not args.max_rows else min(n, max(args.max_rows - rows_seen, 0))
            if limit_n <= 0:
                break
            rows_seen += limit_n
            if "dca_group_index" in z:
                dca_group = np.asarray(z["dca_group_index"][:limit_n])
                idx = np.flatnonzero(dca_group >= 0)
                if deck_sigs and idx.size:
                    deck_arr = z["deck_sig"]
                    idx = np.asarray([int(i) for i in idx if str(deck_arr[int(i)]) in deck_sigs], dtype=np.int64)
                all_groups.extend(_group_dca_rows_fast(z, idx))
            else:
                keep: list[int] = []
                for i in range(limit_n):
                    if deck_sigs and str(z["deck_sig"][i]) not in deck_sigs:
                        continue
                    keep.append(i)
                all_groups.extend(_group_rows(z, keep))
            if args.progress_every and rows_seen % args.progress_every < n:
                print(f"  scanned rows={rows_seen} groups={len(all_groups)} path={path}", flush=True)
        if args.max_rows and rows_seen >= args.max_rows:
            break

    length_counts = Counter(int(g["length"]) for g in all_groups)
    unique_counts = Counter(int(g["unique_slots"]) for g in all_groups)
    focus_bins = Counter("focus>=0.84" if float(g["focus_frac"]) >= 0.84 else "focus>=0.50" if float(g["focus_frac"]) >= 0.50 else "spread" for g in all_groups)
    win_groups = [g for g in all_groups if int(g["won"]) == 1]
    loss_groups = [g for g in all_groups if int(g["won"]) == 0]
    opp_counts = Counter(str(g["opponent_archetype"]) for g in all_groups)

    print(f"rows_scanned={rows_seen} dca_groups={len(all_groups)}", flush=True)
    print(f"length_counts={dict(sorted(length_counts.items()))}", flush=True)
    print(f"unique_slot_counts={dict(sorted(unique_counts.items()))}", flush=True)
    print(f"focus_bins={dict(focus_bins)}", flush=True)
    if all_groups:
        print(f"mean_len={np.mean([g['length'] for g in all_groups]):.3f} mean_focus={np.mean([g['focus_frac'] for g in all_groups]):.3f}", flush=True)
    if win_groups:
        print(f"win_mean_focus={np.mean([g['focus_frac'] for g in win_groups]):.3f} n={len(win_groups)}", flush=True)
    if loss_groups:
        print(f"loss_mean_focus={np.mean([g['focus_frac'] for g in loss_groups]):.3f} n={len(loss_groups)}", flush=True)
    print("top_opponents", flush=True)
    for name, count in opp_counts.most_common(20):
        print(f"  {name:<24} {count:>8} {count / max(len(all_groups), 1):.3f}", flush=True)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "game_key", "episode_id", "player_index", "deck_sig", "team_name", "score",
            "opponent_archetype", "opponent_deck_sig", "won", "length",
            "unique_slots", "unique_cards", "focus_frac", "slot_sequence",
            "card_sequence", "hp_before_sequence", "first_decision",
        ]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_groups)
        print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
