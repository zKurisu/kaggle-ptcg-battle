#!/usr/bin/env python3
"""Compare action patterns between two BC subset npz files."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}

CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE",
    2: "SETUP_BENCH",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    13: "DAMAGE_COUNTER",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    35: "ATTACK",
    37: "EVOLVE",
    43: "ACTIVATE",
}

DEFAULT_CARD_NAMES = {
    96: "Teal Mask Ogerpon ex",
    245: "Alakazam",
    272: "Lillie's Clefairy ex",
    344: "Dwebble",
    345: "Crustle",
    431: "Team Rocket's Mewtwo ex",
    646: "Marnie's Impidimp",
    647: "Marnie's Morgrem",
    648: "Marnie's Grimmsnarl ex",
    756: "Mega Kangaskhan ex",
    1071: "Meowth ex",
}


def load_card_names(path: str) -> dict[int, str]:
    names = dict(DEFAULT_CARD_NAMES)
    p = Path(path) if path else _REPO / "data" / "EN_Card_Data.csv"
    if not p.exists():
        return names
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            card_id = (row.get("Card ID") or "").strip()
            card_name = (row.get("Card Name") or "").strip()
            if not card_id or not card_id.lstrip("-").isdigit() or not card_name:
                continue
            names[int(card_id)] = card_name
    return names


def read_rows(path: str, label: str) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    actions = z["action"]
    opt_types = z["ot"]
    opt_cards = z["oc"]
    feats_arr = z["feats"]
    rows: list[dict] = []
    for i in range(len(actions)):
        action = np.asarray(actions[i], dtype=np.int64)
        if len(action) == 0:
            first = -1
            opt_type = -1
            opt_card = -1
        else:
            first = int(action[0])
            ot = np.asarray(opt_types[i], dtype=np.int64)
            oc = np.asarray(opt_cards[i], dtype=np.int64)
            opt_type = int(ot[first]) if 0 <= first < len(ot) else -1
            opt_card = int(oc[first]) if 0 <= first < len(oc) else -1
        feats = np.asarray(feats_arr[i], dtype=np.float32)
        n_opts = len(opt_types[i])
        rows.append({
            "label": label,
            "context": int(round(float(feats[17]) * 64.0)) if len(feats) > 17 else -1,
            "turn_bucket": min(int(float(feats[0]) * 30.0) // 4, 9) if len(feats) > 0 else -1,
            "turn_action_bucket": min(int(float(feats[1]) * 50.0) // 4, 9) if len(feats) > 1 else -1,
            "nopt_bucket": 1 if n_opts <= 1 else 2 if n_opts <= 2 else 5 if n_opts <= 5 else 10 if n_opts <= 10 else 99,
            "type": opt_type,
            "card": opt_card,
            "action_len": len(action),
        })
    return rows


def summarize(
    rows_a: list[dict],
    rows_b: list[dict],
    group_cols: tuple[str, ...],
    min_count: int,
    card_names: dict[int, str],
) -> list[dict]:
    def counts(rows):
        total = Counter()
        pair = Counter()
        for r in rows:
            g = tuple(r[c] for c in group_cols)
            total[g] += 1
            pair[g, r["type"], r["card"]] += 1
        return total, pair

    total_a, pair_a = counts(rows_a)
    total_b, pair_b = counts(rows_b)
    keys = set(pair_a) | set(pair_b)
    out = []
    for key in keys:
        g, typ, card = key
        na = pair_a.get(key, 0)
        nb = pair_b.get(key, 0)
        ta = total_a.get(g, 0)
        tb = total_b.get(g, 0)
        if na + nb < min_count or ta == 0 or tb == 0:
            continue
        ra = na / ta
        rb = nb / tb
        row = {
            "group": "|".join(str(x) for x in g),
            "type": typ,
            "type_name": TYPE_NAMES.get(typ, str(typ)),
            "card": card,
            "card_name": card_names.get(card, ""),
            "a_count": na,
            "a_total": ta,
            "a_rate": f"{ra:.6f}",
            "b_count": nb,
            "b_total": tb,
            "b_rate": f"{rb:.6f}",
            "rate_delta": f"{ra - rb:.6f}",
        }
        if group_cols and group_cols[0] == "context":
            row["context_name"] = CONTEXT_NAMES.get(g[0], str(g[0]))
        out.append(row)
    out.sort(key=lambda r: abs(float(r["rate_delta"])), reverse=True)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0])
    else:
        fields = ["group", "type", "type_name", "card", "card_name", "a_count", "a_total", "a_rate", "b_count", "b_total", "b_rate", "rate_delta"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first subset npz, usually success")
    p.add_argument("--a-label", default="success")
    p.add_argument("--b", required=True, help="second subset npz, usually loss")
    p.add_argument("--b-label", default="loss")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument(
        "--card-data",
        default="",
        help="card metadata CSV; default uses data/EN_Card_Data.csv when present",
    )
    args = p.parse_args()

    card_names = load_card_names(args.card_data)
    rows_a = read_rows(args.a, args.a_label)
    rows_b = read_rows(args.b, args.b_label)
    out = Path(args.out_dir)

    summaries = {
        "by_context_type_card.csv": summarize(rows_a, rows_b, ("context",), args.min_count, card_names),
        "by_context_turn_type_card.csv": summarize(rows_a, rows_b, ("context", "turn_bucket"), args.min_count, card_names),
        "by_context_nopt_type_card.csv": summarize(rows_a, rows_b, ("context", "nopt_bucket"), args.min_count, card_names),
    }
    for name, rows in summaries.items():
        write_csv(out / name, rows)

    print(f"a={args.a_label} rows={len(rows_a)} b={args.b_label} rows={len(rows_b)} out={out}")
    for name, rows in summaries.items():
        print(f"\n{name}")
        for r in rows[:20]:
            ctx = f" {r.get('context_name', '')}" if r.get("context_name") else ""
            print(
                f"{r['group']}{ctx} {r['type_name']} card={r['card']} {r['card_name']} "
                f"a={r['a_rate']}({r['a_count']}/{r['a_total']}) "
                f"b={r['b_rate']}({r['b_count']}/{r['b_total']}) "
                f"delta={r['rate_delta']}"
            )


if __name__ == "__main__":
    main()
