#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.deck_registry import deck_signature
from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.seq.constants import DAMAGE_COUNTER_ANY_CONTEXT, DEFAULT_FUTURE_HORIZON, FEATURE_VERSION, FUTURE_PLAN_DIM, LEDGER_FEAT_DIM
from ptcg_rl.seq.features import SequenceLedger, future_plan_targets, option_type_for_action, selected_action_event

ARCHETYPES = {
    "Marnie Grimmsnarl": [648],
    "Alakazam": [743, 245, 741, 742],
    "Crustle Wall": [345, 344],
    "Dragapult": [121],
    "Mega Lucario": [678],
    "Archaludon": [190],
    "Cynthia Garchomp": [381],
    "Mega Lopunny": [849],
    "Teal Mask Ogerpon": [96],
    "Team Rocket Mewtwo": [431],
    "Festival Lead": [93],
    "Mega Starmie": [1031, 367],
    "Iono Bellibolt": [269],
    "Mega Abomasnow": [723],
    "N's Zoroark": [293, 320],
    "Hop Trevenant": [879],
    "Raging Bolt": [1065],
}


def classify(deck: list[int]) -> str:
    cnt = Counter(deck)
    best, best_score = "Other", 0
    for name, ids in ARCHETYPES.items():
        score = sum(cnt.get(cid, 0) for cid in ids)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 2 else "Other"


def score_band(score: float) -> str:
    if score >= 1200:
        return "1200+"
    if score >= 1100:
        return "1100-1199"
    if score >= 1000:
        return "1000-1099"
    if score >= 900:
        return "900-999"
    if score >= 800:
        return "800-899"
    if score >= 700:
        return "700-799"
    return "600-699"


def load_leaderboard_scores(path: str = "") -> dict[str, float]:
    if not path:
        return {}
    out: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("TeamName") or row.get("team_name") or row.get("name") or ""
            score_raw = row.get("Score") or row.get("score") or ""
            if not name:
                continue
            try:
                out[name] = float(score_raw)
            except Exception:
                out[name] = 0.0
    return out


def valid_action(action: object, sel: dict) -> bool:
    if not isinstance(action, list) or len(action) == 60:
        return False
    n_opt = len(sel.get("option", []))
    mn = int(sel.get("minCount", 0))
    mx = int(sel.get("maxCount", 0))
    if len(action) < mn or len(action) > mx:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(a, int) and 0 <= a < n_opt for a in action)


def _prev_event_fields(ledger: SequenceLedger) -> dict[str, int | float]:
    ev = ledger.last_event or {}
    return {
        "prev_type": int(ev.get("type", 0) or 0),
        "prev_card": int(ev.get("card", 0) or 0),
        "prev_card2": int(ev.get("card2", 0) or 0),
        "prev_attack": int(ev.get("attack", 0) or 0),
        "prev_context": int(ev.get("context", 0) or 0),
        "prev_select_type": int(ev.get("select_type", 0) or 0),
        "prev_count": float(ev.get("count", 0.0) or 0.0),
    }


def _first_action_index(row: dict[str, object]) -> int:
    arr = np.asarray(row.get("action", []), dtype=np.int64).reshape(-1)
    return int(arr[0]) if arr.size else -1


def _annotate_damage_counter_any_groups(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["dca_group_index"] = -1
        row["dca_pos"] = -1
        row["dca_len"] = 0
        row["dca_remaining"] = 0
        row["dca_selected_slot"] = _first_action_index(row)
        row["dca_prior_same_slot"] = 0
        row["dca_prior_unique_slots"] = 0
        row["dca_prior_max_repeat"] = 0
        row["dca_group_unique_slots"] = 0
        row["dca_group_focus_frac"] = 0.0

    group_index = 0
    i = 0
    while i < len(rows):
        if int(rows[i].get("act_context", -1)) != DAMAGE_COUNTER_ANY_CONTEXT:
            i += 1
            continue
        j = i
        while j < len(rows) and int(rows[j].get("act_context", -1)) == DAMAGE_COUNTER_ANY_CONTEXT:
            j += 1
        block = rows[i:j]
        slots = [_first_action_index(r) for r in block]
        total_counts: dict[int, int] = {}
        for slot in slots:
            if slot >= 0:
                total_counts[slot] = total_counts.get(slot, 0) + 1
        group_unique = len(total_counts)
        group_focus = max(total_counts.values(), default=0) / max(len(block), 1)
        prior: dict[int, int] = {}
        for pos, row in enumerate(block):
            slot = slots[pos]
            row["dca_group_index"] = group_index
            row["dca_pos"] = pos
            row["dca_len"] = len(block)
            row["dca_remaining"] = len(block) - pos
            row["dca_selected_slot"] = slot
            row["dca_prior_same_slot"] = prior.get(slot, 0) if slot >= 0 else 0
            row["dca_prior_unique_slots"] = len(prior)
            row["dca_prior_max_repeat"] = max(prior.values(), default=0)
            row["dca_group_unique_slots"] = group_unique
            row["dca_group_focus_frac"] = group_focus
            if slot >= 0:
                prior[slot] = prior.get(slot, 0) + 1
        group_index += 1
        i = j


def _make_row(
    *,
    encoded,
    action: list[int],
    ledger_feats: np.ndarray,
    prev: dict[str, int | float],
    deck: list[int],
    deck_sig: str,
    team_name: str,
    score: float,
    opponent_deck: list[int],
    opponent_deck_sig: str,
    opponent_team_name: str,
    opponent_score: float,
    episode_id: str,
    player_index: int,
    reward: float,
    won: int,
    draw: int,
    final_status: str,
    game_steps: int,
    step_index: int,
    decision_index: int,
) -> dict[str, object]:
    act_event = selected_action_event(encoded, action)
    ot = np.asarray(encoded.opt_type, dtype=np.int16)
    return {
        "board": np.asarray(encoded.board_cards, dtype=np.int16),
        "hand": np.asarray(encoded.hand_cards, dtype=np.int16),
        "feats": np.asarray(encoded.state_feats, dtype=np.float16),
        "state_token_feats": np.asarray(encoded.state_token_feats, dtype=np.float16),
        "ledger_feats": np.asarray(ledger_feats, dtype=np.float16),
        "ot": ot,
        "oc": np.asarray(encoded.opt_card, dtype=np.int16),
        "oc2": np.asarray(encoded.opt_card2, dtype=np.int16),
        "oa": np.asarray(encoded.opt_attack, dtype=np.int16),
        "of_arr": np.asarray(encoded.opt_feats, dtype=np.float16),
        "action": np.asarray(action, dtype=np.int16),
        "act_type": int(option_type_for_action(ot, action)),
        "act_card": int(act_event.get("card", 0) or 0),
        "act_card2": int(act_event.get("card2", 0) or 0),
        "act_attack": int(act_event.get("attack", 0) or 0),
        "act_context": int(act_event.get("context", 0) or 0),
        "act_select_type": int(act_event.get("select_type", 0) or 0),
        "act_count": float(act_event.get("count", 0.0) or 0.0),
        **prev,
        "min_c": int(encoded.min_count),
        "max_c": int(encoded.max_count),
        "deck_sig": deck_sig,
        "archetype": classify(deck),
        "team_name": team_name,
        "score": float(score),
        "score_band": score_band(score),
        "opponent_deck_sig": opponent_deck_sig,
        "opponent_archetype": classify(opponent_deck),
        "opponent_team_name": opponent_team_name,
        "opponent_score": float(opponent_score),
        "opponent_score_band": score_band(opponent_score),
        "episode_id": episode_id,
        "player_index": int(player_index),
        "game_key": f"{episode_id}:{player_index}",
        "reward": float(reward),
        "won": int(won),
        "draw": int(draw),
        "final_status": final_status,
        "game_steps": int(game_steps),
        "step_index": int(step_index),
        "decision_index": int(decision_index),
    }


def process_zip(
    zip_path: str,
    out_dir: str,
    name_to_score: dict[str, float],
    *,
    future_horizon: int,
    min_rows: int,
    progress_every: int,
    max_episodes: int,
) -> int:
    from ptcg_rl.encoder import FastEncoder

    encoder = FastEncoder()
    zip_path_obj = Path(zip_path)
    all_data: dict[str, list[dict[str, object]]] = defaultdict(list)
    total_rows = 0
    bad_actions = 0
    errors = 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if max_episodes > 0:
            names = names[:max_episodes]
        print(f"{zip_path_obj.name}: {len(names)} eps", flush=True)
        for ep_i, name in enumerate(names, 1):
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                steps = data.get("steps") or []
                if len(steps) < 2:
                    continue
                decks_raw = steps[0][0].get("visualize", [{}])[0].get("action", [])
                if len(decks_raw) != 2:
                    continue
                decks = [list(decks_raw[0]), list(decks_raw[1])]
                if len(decks[0]) != 60 or len(decks[1]) != 60:
                    continue
                deck_sigs = [deck_signature(decks[0]), deck_signature(decks[1])]
                info = data.get("info", {})
                teams = list(info.get("TeamNames") or ["", ""])[:2]
                while len(teams) < 2:
                    teams.append("")
                scores = [float(name_to_score.get(teams[i], 0.0)) for i in range(2)]
                rewards_raw = data.get("rewards") or [0.0, 0.0]
                rewards = [float(rewards_raw[i] if i < len(rewards_raw) and rewards_raw[i] is not None else 0.0) for i in range(2)]
                statuses_raw = data.get("statuses") or ["", ""]
                statuses = [str(statuses_raw[i]) if i < len(statuses_raw) else "" for i in range(2)]
                wins = [int(rewards[i] > rewards[1 - i]) for i in range(2)]
                draws = [int(rewards[i] == rewards[1 - i]) for i in range(2)]
                episode_id = str(data.get("id") or info.get("EpisodeId") or Path(name).stem)

                ledgers = [SequenceLedger(), SequenceLedger()]
                pending: list[dict[str, object] | None] = [None, None]
                player_rows: list[list[dict[str, object]]] = [[], []]
                player_actions: list[list[np.ndarray]] = [[], []]
                player_types: list[list[int]] = [[], []]

                for step_index, step in enumerate(steps[1:], 1):
                    for pi, pd in enumerate(step[:2]):
                        if not isinstance(pd, dict):
                            continue
                        action = pd.get("action")
                        if pending[pi] is not None and isinstance(action, list) and len(action) != 60:
                            pend = pending[pi]
                            obs_prev = pend["obs"]
                            sel_prev = obs_prev.get("select") if isinstance(obs_prev, dict) else None
                            if isinstance(sel_prev, dict) and valid_action(action, sel_prev):
                                encoded = pend["encoded"]
                                row = _make_row(
                                    encoded=encoded,
                                    action=action,
                                    ledger_feats=pend["ledger_feats"],
                                    prev=pend["prev"],
                                    deck=decks[pi],
                                    deck_sig=deck_sigs[pi],
                                    team_name=teams[pi],
                                    score=scores[pi],
                                    opponent_deck=decks[1 - pi],
                                    opponent_deck_sig=deck_sigs[1 - pi],
                                    opponent_team_name=teams[1 - pi],
                                    opponent_score=scores[1 - pi],
                                    episode_id=episode_id,
                                    player_index=pi,
                                    reward=rewards[pi],
                                    won=wins[pi],
                                    draw=draws[pi],
                                    final_status=statuses[pi],
                                    game_steps=len(steps),
                                    step_index=int(pend["step_index"]),
                                    decision_index=int(pend["decision_index"]),
                                )
                                player_rows[pi].append(row)
                                player_actions[pi].append(np.asarray(action, dtype=np.int16))
                                player_types[pi].append(int(row["act_type"]))
                                ledgers[pi].update(encoded, action)
                            else:
                                bad_actions += 1
                            pending[pi] = None

                        obs = pd.get("observation")
                        obs = obs if isinstance(obs, dict) else None
                        sel = obs.get("select") if obs else None
                        if pd.get("status") == "ACTIVE" and isinstance(sel, dict) and len(sel.get("option", [])) > 0:
                            try:
                                encoded = encoder.encode(obs)
                                pending[pi] = {
                                    "obs": obs,
                                    "encoded": encoded,
                                    "ledger_feats": ledgers[pi].features(encoded),
                                    "prev": _prev_event_fields(ledgers[pi]),
                                    "step_index": step_index,
                                    "decision_index": len(player_rows[pi]),
                                }
                            except Exception:
                                errors += 1
                                pending[pi] = None

                for pi in (0, 1):
                    rows = player_rows[pi]
                    if len(rows) < 2:
                        continue
                    _annotate_damage_counter_any_groups(rows)
                    reward_vec = [rewards[pi]] * len(rows)
                    for pos, row in enumerate(rows):
                        row["future_plan"] = future_plan_targets(
                            player_types[pi],
                            player_actions[pi],
                            reward_vec,
                            wins[pi],
                            pos,
                            len(rows),
                            horizon=future_horizon,
                        ).astype(np.float16)
                        key = f"{row['archetype']}|{row['score_band']}"
                        all_data[key].append(row)
                        total_rows += 1
            except Exception:
                errors += 1

            if progress_every and (ep_i == 1 or ep_i % progress_every == 0 or ep_i == len(names)):
                elapsed = time.time() - t0
                rate = ep_i / max(elapsed, 1e-9)
                eta = (len(names) - ep_i) / max(rate, 1e-9)
                print(
                    f"  {zip_path_obj.name} {ep_i}/{len(names)} eps rows={total_rows} "
                    f"bad={bad_actions} err={errors} {rate:.1f} eps/s eta={eta:.0f}s",
                    flush=True,
                )

    written = 0
    for key, rows in sorted(all_data.items()):
        if len(rows) < min_rows:
            continue
        arch, band = key.split("|", 1)
        out_path = Path(out_dir) / arch.replace(" ", "_") / band.replace(" ", "_")
        out_path.mkdir(parents=True, exist_ok=True)
        dest = out_path / zip_path_obj.name.replace(".zip", ".v14seq.npz")
        _save_rows(dest, rows, future_horizon=future_horizon)
        written += len(rows)
        mb = dest.stat().st_size / 1024**2
        print(f"  {key}: {len(rows)} rows -> {dest} ({mb:.1f}MB)", flush=True)
    print(f"  Done {zip_path_obj.name}: written={written} total={total_rows} err={errors}\n", flush=True)
    return written


def _save_rows(dest: Path, rows: list[dict[str, object]], *, future_horizon: int) -> None:
    def obj(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in rows], dtype=object)

    def stack(name: str, dtype) -> np.ndarray:
        return np.stack([np.asarray(r[name]) for r in rows]).astype(dtype)

    np.savez_compressed(
        dest,
        board=stack("board", np.int16),
        hand=stack("hand", np.int16),
        feats=stack("feats", np.float16),
        state_token_feats=stack("state_token_feats", np.float16),
        ledger_feats=stack("ledger_feats", np.float16),
        ot=obj("ot"),
        oc=obj("oc"),
        oc2=obj("oc2"),
        oa=obj("oa"),
        of_arr=obj("of_arr"),
        action=obj("action"),
        act_type=np.asarray([r["act_type"] for r in rows], dtype=np.int16),
        act_card=np.asarray([r["act_card"] for r in rows], dtype=np.int16),
        act_card2=np.asarray([r["act_card2"] for r in rows], dtype=np.int16),
        act_attack=np.asarray([r["act_attack"] for r in rows], dtype=np.int16),
        act_context=np.asarray([r["act_context"] for r in rows], dtype=np.int16),
        act_select_type=np.asarray([r["act_select_type"] for r in rows], dtype=np.int16),
        act_count=np.asarray([r["act_count"] for r in rows], dtype=np.float16),
        dca_group_index=np.asarray([r["dca_group_index"] for r in rows], dtype=np.int16),
        dca_pos=np.asarray([r["dca_pos"] for r in rows], dtype=np.int16),
        dca_len=np.asarray([r["dca_len"] for r in rows], dtype=np.int16),
        dca_remaining=np.asarray([r["dca_remaining"] for r in rows], dtype=np.int16),
        dca_selected_slot=np.asarray([r["dca_selected_slot"] for r in rows], dtype=np.int16),
        dca_prior_same_slot=np.asarray([r["dca_prior_same_slot"] for r in rows], dtype=np.int16),
        dca_prior_unique_slots=np.asarray([r["dca_prior_unique_slots"] for r in rows], dtype=np.int16),
        dca_prior_max_repeat=np.asarray([r["dca_prior_max_repeat"] for r in rows], dtype=np.int16),
        dca_group_unique_slots=np.asarray([r["dca_group_unique_slots"] for r in rows], dtype=np.int16),
        dca_group_focus_frac=np.asarray([r["dca_group_focus_frac"] for r in rows], dtype=np.float16),
        prev_type=np.asarray([r["prev_type"] for r in rows], dtype=np.int16),
        prev_card=np.asarray([r["prev_card"] for r in rows], dtype=np.int16),
        prev_card2=np.asarray([r["prev_card2"] for r in rows], dtype=np.int16),
        prev_attack=np.asarray([r["prev_attack"] for r in rows], dtype=np.int16),
        prev_context=np.asarray([r["prev_context"] for r in rows], dtype=np.int16),
        prev_select_type=np.asarray([r["prev_select_type"] for r in rows], dtype=np.int16),
        prev_count=np.asarray([r["prev_count"] for r in rows], dtype=np.float16),
        future_plan=stack("future_plan", np.float16),
        min_c=np.asarray([r["min_c"] for r in rows], dtype=np.int16),
        max_c=np.asarray([r["max_c"] for r in rows], dtype=np.int16),
        deck_sig=np.asarray([r["deck_sig"] for r in rows], dtype=object),
        archetype=np.asarray([r["archetype"] for r in rows], dtype=object),
        team_name=np.asarray([r["team_name"] for r in rows], dtype=object),
        score=np.asarray([r["score"] for r in rows], dtype=np.float32),
        score_band=np.asarray([r["score_band"] for r in rows], dtype=object),
        opponent_deck_sig=np.asarray([r["opponent_deck_sig"] for r in rows], dtype=object),
        opponent_archetype=np.asarray([r["opponent_archetype"] for r in rows], dtype=object),
        opponent_team_name=np.asarray([r["opponent_team_name"] for r in rows], dtype=object),
        opponent_score=np.asarray([r["opponent_score"] for r in rows], dtype=np.float32),
        opponent_score_band=np.asarray([r["opponent_score_band"] for r in rows], dtype=object),
        episode_id=np.asarray([r["episode_id"] for r in rows], dtype=object),
        player_index=np.asarray([r["player_index"] for r in rows], dtype=np.int8),
        game_key=np.asarray([r["game_key"] for r in rows], dtype=object),
        reward=np.asarray([r["reward"] for r in rows], dtype=np.float32),
        won=np.asarray([r["won"] for r in rows], dtype=np.int8),
        draw=np.asarray([r["draw"] for r in rows], dtype=np.int8),
        final_status=np.asarray([r["final_status"] for r in rows], dtype=object),
        game_steps=np.asarray([r["game_steps"] for r in rows], dtype=np.int16),
        step_index=np.asarray([r["step_index"] for r in rows], dtype=np.int16),
        decision_index=np.asarray([r["decision_index"] for r in rows], dtype=np.int16),
        feature_version=np.asarray(FEATURE_VERSION, dtype=object),
        state_feat_dim=np.asarray(STATE_FEAT_DIM, dtype=np.int16),
        opt_feat_dim=np.asarray(OPT_FEAT_DIM, dtype=np.int16),
        state_token_feat_dim=np.asarray(STATE_TOKEN_FEAT_DIM, dtype=np.int16),
        ledger_feat_dim=np.asarray(LEDGER_FEAT_DIM, dtype=np.int16),
        future_plan_dim=np.asarray(FUTURE_PLAN_DIM, dtype=np.int16),
        future_horizon=np.asarray(future_horizon, dtype=np.int16),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--lb-csv", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--future-horizon", type=int, default=DEFAULT_FUTURE_HORIZON)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--max-episodes", type=int, default=0)
    args = p.parse_args()

    scores = load_leaderboard_scores(args.lb_csv)
    if scores:
        print(f"Leaderboard scores loaded: {len(scores)} teams", flush=True)
    else:
        print("WARNING: no leaderboard CSV supplied; all score bands default to 600-699", flush=True)
    zips = sorted(str(p) for p in Path(args.episodes_dir).glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no episode zip files found under {args.episodes_dir}")
    if args.workers <= 1:
        total = 0
        for zp in zips:
            total += process_zip(
                zp,
                args.out,
                scores,
                future_horizon=args.future_horizon,
                min_rows=args.min_rows,
                progress_every=args.progress_every,
                max_episodes=args.max_episodes,
            )
        print(f"All done: {total} rows", flush=True)
        return

    workers = min(int(args.workers), len(zips))
    print(f"Processing {len(zips)} zips with {workers} workers", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                process_zip,
                zp,
                args.out,
                scores,
                future_horizon=args.future_horizon,
                min_rows=args.min_rows,
                progress_every=args.progress_every,
                max_episodes=args.max_episodes,
            )
            for zp in zips
        ]
        for done, fut in enumerate(as_completed(futs), 1):
            total += int(fut.result())
            print(f"Finished {done}/{len(futs)} zips total_rows={total}", flush=True)
    print(f"All done: {total} rows", flush=True)


if __name__ == "__main__":
    main()
