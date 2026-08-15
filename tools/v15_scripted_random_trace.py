#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.encoder import FastEncoder
from ptcg_rl.policy_loader import load_policy
from tools.trace_matchup_decisions import encode_decision, safe_int
from tools.v15_trace_game import format_decision


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def legal_random(sel: dict) -> list[int]:
    opts = sel.get("option") or []
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if not opts or mx <= 0:
        return []
    hi = min(mx, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []


def legalize(action: list[int], sel: dict) -> list[int]:
    n = len(sel.get("option") or [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    action = [int(x) for x in action if 0 <= int(x) < n]
    action = list(dict.fromkeys(action))
    if mn <= len(action) <= mx:
        return action[:mx]
    return legal_random(sel)


def choose_policy(policy, obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    try:
        action = policy.select(obs, greedy=True, update_history=False)
    except Exception:
        action = []
    action = legalize(action, sel)
    try:
        policy.remember_decision(obs, action)
    except Exception:
        pass
    return action


def option_fingerprints(encoder: FastEncoder, obs: dict) -> list[list[int]]:
    try:
        enc = encoder.encode(obs)
    except Exception:
        return []
    out: list[list[int]] = []
    raw_opts = (obs.get("select") or {}).get("option") or []
    feats = np.asarray(enc.opt_feats, dtype=np.float32)
    for i in range(len(enc.opt_type)):
        feat = feats[i] if feats.ndim == 2 and i < feats.shape[0] else np.zeros(0, dtype=np.float32)

        def fidx(idx: int, scale: float) -> int:
            return int(round(float(feat[idx]) * scale)) if idx < feat.shape[0] else 0

        raw = raw_opts[i] if i < len(raw_opts) and isinstance(raw_opts[i], dict) else {}
        out.append([
            int(enc.opt_type[i]),
            int(enc.opt_card[i]),
            int(enc.opt_card2[i]),
            int(enc.opt_attack[i]),
            fidx(3, 64.0),
            fidx(4, 16.0),
            fidx(7, 16.0),
            fidx(8, 64.0),
            fidx(9, 16.0),
            fidx(10, 10.0),
            safe_int(raw.get("type")),
            safe_int(raw.get("cardId")),
            safe_int(raw.get("attackId")),
            safe_int(raw.get("playerIndex"), -1),
            safe_int(raw.get("area")),
            safe_int(raw.get("index")),
            safe_int(raw.get("inPlayArea")),
            safe_int(raw.get("inPlayIndex")),
        ])
    return out


def script_action(
    records: list[dict[str, Any]],
    ptr: int,
    obs: dict,
    encoder: FastEncoder,
    *,
    strict: bool,
) -> tuple[list[int], dict[str, Any]]:
    sel = obs.get("select") or {}
    cur_fps = option_fingerprints(encoder, obs)
    if ptr >= len(records):
        if strict:
            raise RuntimeError("script exhausted")
        return legal_random(sel), {"match": "fallback", "reason": "script_exhausted", "script_ptr": ptr}
    rec = records[ptr]
    rec_fps = rec.get("option_fps") or []
    rec_action = [int(x) for x in rec.get("action") or []]
    if cur_fps == rec_fps:
        return legalize(rec_action, sel), {"match": "exact", "script_ptr": ptr}

    used: set[int] = set()
    mapped: list[int] = []
    for old_idx in rec_action:
        if not (0 <= old_idx < len(rec_fps)):
            continue
        fp = rec_fps[old_idx]
        found = -1
        for i, cur_fp in enumerate(cur_fps):
            if i in used:
                continue
            if cur_fp == fp:
                found = i
                break
        if found < 0:
            if strict:
                raise RuntimeError(f"script mismatch at ptr={ptr}: cannot map chosen option {old_idx}")
            return legal_random(sel), {
                "match": "fallback",
                "reason": "option_mismatch",
                "script_ptr": ptr,
                "script_step": rec.get("step"),
                "script_turn": rec.get("turn"),
                "current_options": len(cur_fps),
                "script_options": len(rec_fps),
            }
        used.add(found)
        mapped.append(found)
    return legalize(mapped, sel), {
        "match": "mapped",
        "script_ptr": ptr,
        "script_step": rec.get("step"),
        "script_turn": rec.get("turn"),
        "current_options": len(cur_fps),
        "script_options": len(rec_fps),
    }


def play_game(
    args: argparse.Namespace,
    *,
    game: int,
    seed: int,
    policy,
    encoder: FastEncoder,
    script: dict[str, Any] | None,
) -> tuple[str, list[tuple[dict, dict]], dict[str, Any], int]:
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    np.random.seed(int(seed) & 0xFFFFFFFF)
    if script is None:
        candidate_side = game % 2
    else:
        candidate_side = int(script.get("candidate_side", game % 2))
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    first_deck = args.opponent_deck_cards if candidate_side == 1 else args.deck_cards
    second_deck = args.deck_cards if candidate_side == 1 else args.opponent_deck_cards
    obs, _ = battle_start(first_deck, second_deck)
    trace: list[tuple[dict, dict]] = []
    candidate_snapshots: list[dict[str, Any]] = []
    opponent_records: list[dict[str, Any]] = []
    replay_notes: list[dict[str, Any]] = []
    script_records = list((script or {}).get("opponent_actions") or [])
    script_ptr = 0
    result = 2
    steps = 0
    try:
        if obs is None:
            return "draw", trace, {}, steps
        for steps in range(args.max_turns):
            cur = obs.get("current") or {}
            res = safe_int(cur.get("result"), -1)
            if res != -1:
                result = res if res in (0, 1) else 2
                break
            sel = obs.get("select")
            if sel is None:
                result = 2
                break
            side = safe_int(cur.get("yourIndex"))
            if side == candidate_side:
                action = choose_policy(policy, obs)
                candidate_snapshots.append({
                    "step": steps,
                    "turn": safe_int(cur.get("turn")),
                    "turn_action_count": safe_int(cur.get("turnActionCount")),
                    "action": list(action),
                    "obs": deepcopy(obs),
                })
                try:
                    row = encode_decision(encoder, obs, action, game, steps, candidate_side, policy=policy)
                except Exception as exc:
                    row = {
                        "step": steps,
                        "turn": safe_int(cur.get("turn")),
                        "turn_action_count": safe_int(cur.get("turnActionCount")),
                        "chosen_card_names": f"encode_error:{exc}",
                    }
                trace.append((row, obs))
            elif script is None:
                fps = option_fingerprints(encoder, obs)
                action = legal_random(sel)
                opponent_records.append({
                    "step": steps,
                    "turn": safe_int(cur.get("turn")),
                    "turn_action_count": safe_int(cur.get("turnActionCount")),
                    "side": side,
                    "context": safe_int(sel.get("context")),
                    "select_type": safe_int(sel.get("type")),
                    "min_count": safe_int(sel.get("minCount")),
                    "max_count": safe_int(sel.get("maxCount")),
                    "option_fps": fps,
                    "action": list(action),
                })
            else:
                action, note = script_action(script_records, script_ptr, obs, encoder, strict=args.strict_script)
                note.update({
                    "step": steps,
                    "turn": safe_int(cur.get("turn")),
                    "turn_action_count": safe_int(cur.get("turnActionCount")),
                })
                replay_notes.append(note)
                script_ptr += 1
            obs = battle_select(action)
            if obs is None:
                result = 2
                break
        else:
            result = 2
    finally:
        battle_finish()
    if result == 2:
        outcome = "draw"
    elif result == candidate_side:
        outcome = "win"
    else:
        outcome = "loss"
    meta = {
        "version": 1,
        "mode": "record" if script is None else "replay",
        "policy": args.policy,
        "deck": args.deck,
        "opponent_deck": args.opponent_deck or args.deck,
        "game": game,
        "seed": seed,
        "candidate_side": candidate_side,
        "outcome": outcome,
        "steps": steps,
        "candidate_decisions": len(trace),
        "candidate_snapshots": candidate_snapshots,
        "opponent_actions": opponent_records,
        "replay_notes": replay_notes,
        "script_actions_used": script_ptr,
        "script_actions_total": len(script_records),
    }
    return outcome, trace, meta, steps


def _action_text(row: dict[str, Any]) -> str:
    return (
        f"{row.get('chosen_type_names', '')} {row.get('chosen_card_names', '')} "
        f"target={row.get('chosen_target_name', '')} rank={row.get('chosen_first_rank', '')}"
    ).strip()


def replay_recorded_states(args: argparse.Namespace, policy, encoder: FastEncoder, script: dict[str, Any]) -> tuple[list[tuple[dict, dict, dict]], dict[str, Any]]:
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    snapshots = list(script.get("candidate_snapshots") or [])
    rows: list[tuple[dict, dict, dict]] = []
    changed = 0
    first_change: dict[str, Any] | None = None
    for i, snap in enumerate(snapshots):
        obs = snap.get("obs") or {}
        original = [int(x) for x in snap.get("action") or []]
        sel = obs.get("select") or {}
        try:
            new_action = policy.select(obs, greedy=True, update_history=False)
        except Exception:
            new_action = []
        new_action = legalize(new_action, sel)
        try:
            original_row = encode_decision(encoder, obs, original, int(script.get("game", 0)), int(snap.get("step", i)), int(script.get("candidate_side", 0)), policy=None)
        except Exception as exc:
            original_row = {"step": snap.get("step", i), "turn": snap.get("turn", 0), "chosen_card_names": f"original_encode_error:{exc}"}
        try:
            new_row = encode_decision(encoder, obs, new_action, int(script.get("game", 0)), int(snap.get("step", i)), int(script.get("candidate_side", 0)), policy=policy)
        except Exception as exc:
            new_row = {"step": snap.get("step", i), "turn": snap.get("turn", 0), "chosen_card_names": f"new_encode_error:{exc}"}
        same = list(original) == list(new_action)
        original_row["_same_action"] = int(same)
        new_row["_same_action"] = int(same)
        if not same:
            changed += 1
            if first_change is None:
                first_change = {
                    "snapshot": i,
                    "step": snap.get("step"),
                    "turn": snap.get("turn"),
                    "original": _action_text(original_row),
                    "new": _action_text(new_row),
                }
        rows.append((original_row, new_row, obs))
        if args.state_history == "original":
            try:
                policy.remember_decision(obs, original)
            except Exception:
                pass
        elif args.state_history == "new":
            try:
                policy.remember_decision(obs, new_action)
            except Exception:
                pass
    meta = {
        "version": 1,
        "mode": "state_replay",
        "policy": args.policy,
        "script_policy": script.get("policy", ""),
        "deck": args.deck,
        "script_game": script.get("game"),
        "script_seed": script.get("seed"),
        "script_outcome": script.get("outcome"),
        "candidate_side": script.get("candidate_side"),
        "snapshots": len(snapshots),
        "changed": changed,
        "same": len(snapshots) - changed,
        "state_history": args.state_history,
        "first_change": first_change,
    }
    return rows, meta


def write_trace(path: Path, args: argparse.Namespace, meta: dict[str, Any], trace: list[tuple[dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = meta.get("replay_notes") or []
    counts: dict[str, int] = {}
    for note in notes:
        key = str(note.get("match", ""))
        counts[key] = counts.get(key, 0) + 1
    with path.open("w") as f:
        f.write("# scripted random trace\n\n")
        f.write(f"policy: `{args.policy}`\n\n")
        f.write(f"deck: `{args.deck}`\n\n")
        f.write(f"mode: `{meta.get('mode')}` outcome={meta.get('outcome')} game={meta.get('game')} seed={meta.get('seed')}\n\n")
        f.write(
            f"candidate_side={meta.get('candidate_side')} steps={meta.get('steps')} "
            f"candidate_decisions={meta.get('candidate_decisions')}\n\n"
        )
        if meta.get("mode") == "record":
            f.write(f"recorded_opponent_actions={len(meta.get('opponent_actions') or [])}\n\n")
        else:
            f.write(
                f"script_actions_used={meta.get('script_actions_used')}/{meta.get('script_actions_total')} "
                f"match_counts={counts}\n\n"
            )
            first_bad = next((n for n in notes if n.get("match") != "exact"), None)
            if first_bad:
                f.write(f"first_non_exact_script_step={first_bad}\n\n")
        for row, obs in trace:
            for line in format_decision(row, obs):
                f.write(line + "\n")
            f.write("\n")


def write_state_replay(path: Path, args: argparse.Namespace, meta: dict[str, Any], rows: list[tuple[dict, dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# scripted random state replay\n\n")
        f.write(f"policy: `{args.policy}`\n\n")
        f.write(
            f"script_game={meta.get('script_game')} script_seed={meta.get('script_seed')} "
            f"script_outcome={meta.get('script_outcome')} state_history={meta.get('state_history')}\n\n"
        )
        f.write(f"snapshots={meta.get('snapshots')} same={meta.get('same')} changed={meta.get('changed')}\n\n")
        if meta.get("first_change"):
            f.write(f"first_change={meta.get('first_change')}\n\n")
        for i, (orig, new, obs) in enumerate(rows):
            changed = not bool(new.get("_same_action", 0))
            f.write(
                f"- snapshot={i} step={orig.get('step')} turn={orig.get('turn')} "
                f"changed={int(changed)}\n"
            )
            f.write(f"  original: {_action_text(orig)}\n")
            f.write(f"  new: {_action_text(new)}\n")
            for line in format_decision(new, obs):
                f.write("  " + line + "\n")
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-deck", default="")
    p.add_argument("--script-in", default="")
    p.add_argument("--script-out", default="")
    p.add_argument("--state-replay", action="store_true",
                   help="do not step the engine; replay policy decisions on exact recorded candidate observations")
    p.add_argument("--state-history", choices=["original", "new", "none"], default="original",
                   help="which action to remember between recorded observations during --state-replay")
    p.add_argument("--target-outcome", choices=["loss", "win", "draw", "any"], default="loss")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--start-game", type=int, default=0,
                   help="first game index to scan; seed is still computed as --seed + game")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--strict-script", action="store_true")
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    args.deck_cards = load_deck(args.deck)
    args.opponent_deck_cards = load_deck(args.opponent_deck) if args.opponent_deck else list(args.deck_cards)
    policy = load_policy(args.policy, device="cpu")
    encoder = FastEncoder()
    script = json.loads(Path(args.script_in).read_text()) if args.script_in else None
    if args.state_replay:
        if script is None:
            raise SystemExit("--state-replay requires --script-in")
        rows, meta = replay_recorded_states(args, policy, encoder, script)
        write_state_replay(Path(args.out_md), args, meta, rows)
        print(f"Wrote state replay {args.out_md}", flush=True)
        return
    t0 = time.time()
    selected: tuple[str, list[tuple[dict, dict]], dict[str, Any]] | None = None
    counts = {"win": 0, "loss": 0, "draw": 0}
    if script is not None:
        outcome, trace, meta, _ = play_game(args, game=int(script.get("game", 0)), seed=int(script.get("seed", args.seed)), policy=policy, encoder=encoder, script=script)
        counts[outcome] += 1
        selected = (outcome, trace, meta)
    else:
        start_game = max(0, int(args.start_game))
        end_game = start_game + max(0, int(args.games))
        for scanned, g in enumerate(range(start_game, end_game), 1):
            outcome, trace, meta, _ = play_game(args, game=g, seed=args.seed + g, policy=policy, encoder=encoder, script=None)
            counts[outcome] += 1
            if args.progress_every and (scanned == 1 or scanned % args.progress_every == 0):
                print(
                    f"{scanned}/{args.games} game={g} win={counts['win']} loss={counts['loss']} draw={counts['draw']} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if args.target_outcome == "any" or outcome == args.target_outcome:
                selected = (outcome, trace, meta)
                break
    out_md = Path(args.out_md)
    if selected is None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(f"No game matched target_outcome={args.target_outcome}; counts={counts}\n")
        print(f"Wrote {out_md}", flush=True)
        return
    _, trace, meta = selected
    write_trace(out_md, args, meta, trace)
    if args.script_out:
        script_data = dict(meta)
        script_data.pop("replay_notes", None)
        Path(args.script_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.script_out).write_text(json.dumps(script_data, indent=2, ensure_ascii=False))
        print(f"Wrote script {args.script_out}", flush=True)
    print(f"Wrote trace {out_md}", flush=True)


if __name__ == "__main__":
    main()
