#!/usr/bin/env python3
"""
Extract state→action pairs from Kaggle episode ZIPs, grouped by archetype.

Usage:
    python tools/bc_extract_v2.py raw_episode/ --out data/bc_corpus/

Output per zip × per archetype:
    data/bc_corpus/<Archetype>/<date>.npz
"""

import sys, os, json, zipfile, time, argparse, tempfile, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))

from ptcg_rl.deck_registry import deck_signature
from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM
from ptcg_rl.history_features import (
    BOARD_HISTORY_FEAT_DIM,
    DEFAULT_ACTION_HISTORY_K,
    DEFAULT_BOARD_HISTORY_K,
    DEFAULT_LOG_HISTORY_K,
    HISTORY_SUMMARY_DIM,
    ACTION_FIELDS,
    LOG_FIELDS,
    action_event_from_encoded,
    board_snapshot_from_encoded,
    history_summary_from_arrays,
    pack_action_history,
    pack_board_history,
    pack_log_history_from_obs,
)

FEATURE_VERSION = "v12_multistream_history"

ARCHETYPES = {
    "Marnie Grimmsnarl": [648], "Alakazam": [743, 245, 741, 742],
    "Crustle Wall": [345, 344], "Dragapult": [121], "Mega Lucario": [678],
    "Archaludon": [190], "Cynthia Garchomp": [381], "Mega Lopunny": [849],
    "Teal Mask Ogerpon": [96], "Team Rocket Mewtwo": [431], "Festival Lead": [93],
    "Mega Starmie": [1031, 367], "Iono Bellibolt": [269], "Mega Abomasnow": [723],
    "N's Zoroark": [293, 320], "Hop Trevenant": [879], "Raging Bolt": [1065],
}

def _find_leaderboard_csv(lb_csv_path: str = None) -> str | None:
    if lb_csv_path and os.path.exists(lb_csv_path):
        return lb_csv_path

    for p in [
        _REPO / "pokemon-tcg-ai-battle.zip",
        _REPO.parent / "pokemon-tcg-ai-battle.zip",
        Path("/tmp/lb/pokemon-tcg-ai-battle.zip"),
    ]:
        if p.exists():
            tmp = tempfile.mkdtemp(prefix="ptcg_lb_")
            with zipfile.ZipFile(p) as zf:
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
                if csv_files:
                    raw = zf.read(csv_files[0]).decode('utf-8')
                    out = os.path.join(tmp, csv_files[0])
                    with open(out, 'w') as f:
                        f.write(raw)
                    return out
    return None


def load_leaderboard_scores(lb_csv_path: str = None) -> dict[str, float]:
    """Load team_name → score from leaderboard CSV."""
    lb_csv_path = _find_leaderboard_csv(lb_csv_path)
    if lb_csv_path is None:
        # Download latest leaderboard
        import subprocess, glob
        tmp = tempfile.mkdtemp()
        subprocess.run(["kaggle", "competitions", "leaderboard",
                       "pokemon-tcg-ai-battle", "--download", "-p", tmp],
                       capture_output=True)
        zips = glob.glob(f"{tmp}/*.zip")
        if zips:
            import zipfile
            with zipfile.ZipFile(zips[0]) as zf:
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
                if csv_files:
                    raw = zf.read(csv_files[0]).decode('utf-8')
                    lb_csv_path = os.path.join(tmp, csv_files[0])
                    with open(lb_csv_path, 'w') as f: f.write(raw)

    name_to_score = {}
    if lb_csv_path and os.path.exists(lb_csv_path):
        import csv
        with open(lb_csv_path) as f:
            for r in csv.DictReader(f):
                name = r.get('TeamName', '')
                score = float(r.get('Score', 0)) if r.get('Score') else 0
                if name: name_to_score[name] = score
    return name_to_score


def score_band(score: float) -> str:
    if score >= 1200: return "1200+"
    if score >= 1100: return "1100-1199"
    if score >= 1000: return "1000-1099"
    if score >= 900:  return "900-999"
    if score >= 800:  return "800-899"
    if score >= 700:  return "700-799"
    return "600-699"


def classify(deck):
    cnt = Counter(deck); best, bs = "Other", 0
    for n, ks in ARCHETYPES.items():
        s = sum(cnt.get(k, 0) for k in ks)
        if s > bs: bs, best = s, n
    return best if bs >= 2 else "Other"


def _valid_action(action: list, sel: dict) -> bool:
    n_opt = len(sel.get('option', []))
    mn = int(sel.get('minCount', 0))
    mx = int(sel.get('maxCount', 0))
    if len(action) == 60:
        return False
    if len(action) < mn or len(action) > mx:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(a, int) and 0 <= a < n_opt for a in action)


def _append_decision(all_data, encoder, obs: dict, action: list,
                     deck: list[int], band: str, *, deck_sig: str = "",
                     team_name: str = "", score: float = 0.0,
                     opponent_deck: list[int] | None = None,
                     opponent_deck_sig: str = "",
                     opponent_team_name: str = "",
                     opponent_score: float = 0.0,
                     opponent_score_band: str = "",
                     episode_id: str = "", player_index: int = -1,
                     reward: float = 0.0, won: int = 0, draw: int = 0,
                     final_status: str = "", game_steps: int = 0,
                     step_index: int = -1, decision_index: int = -1,
                     encoded=None, history: dict | None = None) -> bool:
    sel = obs.get('select')
    if sel is None or len(sel.get('option', [])) == 0:
        return False
    if not _valid_action(action, sel):
        return False
    arch = classify(deck)
    opponent_archetype = classify(opponent_deck or []) if opponent_deck else "Other"
    key = f"{arch}|{band}"
    ed = encoded if encoded is not None else encoder.encode(obs)
    row = {
        'board': ed.board_cards.astype(np.int16),
        'hand': ed.hand_cards.astype(np.int16),
        'feats': ed.state_feats.astype(np.float16),
        'ot': ed.opt_type.astype(np.int16),
        'oc': ed.opt_card.astype(np.int16),
        'oc2': ed.opt_card2.astype(np.int16),
        'oa': ed.opt_attack.astype(np.int16),
        'of': ed.opt_feats.astype(np.float16),
        'action': np.array(action, dtype=np.int16),
        'min_c': ed.min_count, 'max_c': ed.max_count,
        'deck_sig': deck_sig,
        'team_name': team_name,
        'score': float(score),
        'opponent_deck_sig': opponent_deck_sig,
        'opponent_archetype': opponent_archetype,
        'opponent_team_name': opponent_team_name,
        'opponent_score': float(opponent_score),
        'opponent_score_band': opponent_score_band,
        'episode_id': episode_id,
        'player_index': int(player_index),
        'reward': float(reward),
        'won': int(won),
        'draw': int(draw),
        'final_status': final_status,
        'game_steps': int(game_steps),
        'step_index': int(step_index),
        'decision_index': int(decision_index),
    }
    if history:
        for prefix in ("own_hist", "opp_hist"):
            hist = history.get(prefix) or {}
            for field in ACTION_FIELDS:
                row[f"{prefix}_{field}"] = hist.get(field)
        log_hist = history.get("log_hist") or {}
        for field in LOG_FIELDS:
            row[f"log_hist_{field}"] = log_hist.get(field)
        board_hist = history.get("board_hist") or {}
        row["board_hist_cards"] = board_hist.get("cards")
        row["board_hist_feats"] = board_hist.get("feats")
        row["board_hist_mask"] = board_hist.get("mask")
        row["history_summary"] = history_summary_from_arrays(
            own_hist=history.get("own_hist") or {},
            opp_hist=history.get("opp_hist") or {},
            log_hist=log_hist,
            board_hist=board_hist,
            dim=HISTORY_SUMMARY_DIM,
        )
    all_data[key].append(row)
    return True


def process_zip(zip_path, out_dir, name_to_score: dict, progress_every: int = 500,
                action_history_k: int = DEFAULT_ACTION_HISTORY_K,
                log_history_k: int = DEFAULT_LOG_HISTORY_K,
                board_history_k: int = DEFAULT_BOARD_HISTORY_K,
                board_history_feat_dim: int = BOARD_HISTORY_FEAT_DIM,
                max_episodes: int = 0):
    from ptcg_rl.encoder import FastEncoder
    encoder = FastEncoder()

    with zipfile.ZipFile(str(zip_path)) as zf:
        fnames = [n for n in zf.namelist() if n.endswith('.json')]
        if max_episodes > 0:
            fnames = fnames[:max_episodes]
        print(f"{zip_path.name}: {len(fnames)} eps")
        t0 = time.time()

        all_data = defaultdict(list)  # key: "archetype|band"
        bad_actions = 0
        errors = 0
        for i, fname in enumerate(fnames):
            try:
                raw = zf.read(fname).decode('utf-8')
                data = json.loads(raw); steps = data['steps']
                if len(steps) < 2: continue
                decks_raw = steps[0][0].get('visualize', [{}])[0].get('action', [])
                if len(decks_raw) != 2: continue
                decks = [decks_raw[0], decks_raw[1]]
                if len(decks[0]) != 60 or len(decks[1]) != 60: continue
                deck_sigs = [deck_signature(decks[0]), deck_signature(decks[1])]

                # Get scores from team names
                info = data.get('info', {})
                teams = info.get('TeamNames', [])
                scores = [name_to_score.get(t, 0) for t in teams[:2]]
                bands = [score_band(s) for s in scores]
                episode_id = str(data.get("id") or info.get("EpisodeId") or fname.rsplit("/", 1)[-1].split(".")[0])
                rewards = data.get("rewards") or [0.0, 0.0]
                statuses = data.get("statuses") or ["", ""]
                rewards = [float(rewards[j]) if j < len(rewards) and rewards[j] is not None else 0.0 for j in range(2)]
                statuses = [str(statuses[j]) if j < len(statuses) else "" for j in range(2)]
                draws = [int(rewards[j] == rewards[1 - j]) for j in range(2)]
                wins = [int(rewards[j] > rewards[1 - j]) for j in range(2)]

                # Kaggle episode rows store the action that answered the
                # previous ACTIVE observation for that player.
                pending = [None, None]
                action_history = [[], []]
                board_history = [[], []]
                decision_count = [0, 0]
                for step_index, step in enumerate(steps[1:], 1):
                    for pi, pd in enumerate(step[:2]):
                        if not isinstance(pd, dict):
                            continue
                        has_action = 'action' in pd
                        action = pd.get('action')
                        if pending[pi] is not None and has_action and isinstance(action, list) and len(action) != 60:
                            pend = pending[pi]
                            obs_prev = pend["obs"]
                            ed_prev = pend.get("encoded")
                            band = bands[pi] if pi < len(bands) else "unknown"
                            try:
                                ok = _append_decision(
                                    all_data, encoder, obs_prev, action, decks[pi], band,
                                    deck_sig=deck_sigs[pi],
                                    team_name=teams[pi] if pi < len(teams) else "",
                                    score=scores[pi] if pi < len(scores) else 0.0,
                                    opponent_deck=decks[1 - pi],
                                    opponent_deck_sig=deck_sigs[1 - pi],
                                    opponent_team_name=teams[1 - pi] if 1 - pi < len(teams) else "",
                                    opponent_score=scores[1 - pi] if 1 - pi < len(scores) else 0.0,
                                    opponent_score_band=bands[1 - pi] if 1 - pi < len(bands) else "unknown",
                                    episode_id=episode_id,
                                    player_index=pi,
                                    reward=rewards[pi],
                                    won=wins[pi],
                                    draw=draws[pi],
                                    final_status=statuses[pi],
                                    game_steps=len(steps),
                                    step_index=pend.get("step_index", -1),
                                    decision_index=pend.get("decision_index", -1),
                                    encoded=ed_prev,
                                    history=pend.get("history"),
                                )
                                bad_actions += 0 if ok else 1
                                if ok:
                                    event = action_event_from_encoded(ed_prev, action)
                                    if event is not None:
                                        action_history[pi].append(event)
                                        if len(action_history[pi]) > max(action_history_k, 1) * 4:
                                            del action_history[pi][:-max(action_history_k, 1) * 4]
                            except Exception:
                                errors += 1
                            pending[pi] = None

                        obs = pd.get('observation')
                        obs = obs if isinstance(obs, dict) else None
                        sel = obs.get('select') if obs else None
                        if (pd.get('status') == 'ACTIVE' and sel is not None
                                and len(sel.get('option', [])) > 0):
                            try:
                                ed = encoder.encode(obs)
                                hist = {
                                    "own_hist": pack_action_history(action_history[pi], action_history_k),
                                    "opp_hist": pack_action_history(action_history[1 - pi], action_history_k),
                                    "log_hist": pack_log_history_from_obs(obs, log_history_k),
                                    "board_hist": pack_board_history(
                                        board_history[pi],
                                        board_history_k,
                                        board_history_feat_dim,
                                    ),
                                }
                                pending[pi] = {
                                    "obs": obs,
                                    "encoded": ed,
                                    "history": hist,
                                    "step_index": step_index,
                                    "decision_index": decision_count[pi],
                                }
                                decision_count[pi] += 1
                                board_history[pi].append(
                                    board_snapshot_from_encoded(ed, board_history_feat_dim)
                                )
                                if len(board_history[pi]) > max(board_history_k, 1) * 4:
                                    del board_history[pi][:-max(board_history_k, 1) * 4]
                            except Exception:
                                errors += 1
                                pending[pi] = None
            except Exception:
                errors += 1

            if progress_every and ((i+1) % progress_every == 0 or (i+1) == len(fnames)):
                total = sum(len(v) for v in all_data.values())
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(fnames)-i-1)
                rate = (i + 1) / max(elapsed, 1e-9)
                print(
                    f"  {zip_path.name} {i+1}/{len(fnames)} eps | {total} decs | "
                    f"bad {bad_actions} | err {errors} | {rate:.1f} eps/s | eta {eta:.0f}s",
                    flush=True,
                )

    # Save — directory: <Archetype>/<ScoreBand>/<date>.npz
    total = 0
    for key, decs in sorted(all_data.items()):
        n = len(decs); total += n
        if n < 100: continue
        parts = key.split('|')
        arch, band = parts[0], parts[1] if len(parts) > 1 else "unknown"
        arch_dir = os.path.join(out_dir, arch.replace(' ', '_'), band.replace(' ', '_'))
        os.makedirs(arch_dir, exist_ok=True)
        fbase = zip_path.name.replace('.zip', '')
        def stack(name, dtype):
            return np.stack([np.asarray(d[name]) for d in decs]).astype(dtype)
        np.savez_compressed(
            os.path.join(arch_dir, f'{fbase}.npz'),
            board=np.array([d['board'] for d in decs], dtype=object),
            hand=np.array([d['hand'] for d in decs], dtype=object),
            feats=np.array([d['feats'] for d in decs], dtype=object),
            ot=np.array([d['ot'] for d in decs], dtype=object),
            oc=np.array([d['oc'] for d in decs], dtype=object),
            oc2=np.array([d['oc2'] for d in decs], dtype=object),
            oa=np.array([d['oa'] for d in decs], dtype=object),
            of_arr=np.array([d['of'] for d in decs], dtype=object),
            action=np.array([d['action'] for d in decs], dtype=object),
            min_c=np.array([d['min_c'] for d in decs], dtype=np.int16),
            max_c=np.array([d['max_c'] for d in decs], dtype=np.int16),
            deck_sig=np.array([d['deck_sig'] for d in decs], dtype=object),
            team_name=np.array([d['team_name'] for d in decs], dtype=object),
            score=np.array([d['score'] for d in decs], dtype=np.float32),
            opponent_deck_sig=np.array([d['opponent_deck_sig'] for d in decs], dtype=object),
            opponent_archetype=np.array([d['opponent_archetype'] for d in decs], dtype=object),
            opponent_team_name=np.array([d['opponent_team_name'] for d in decs], dtype=object),
            opponent_score=np.array([d['opponent_score'] for d in decs], dtype=np.float32),
            opponent_score_band=np.array([d['opponent_score_band'] for d in decs], dtype=object),
            episode_id=np.array([d['episode_id'] for d in decs], dtype=object),
            player_index=np.array([d['player_index'] for d in decs], dtype=np.int8),
            reward=np.array([d['reward'] for d in decs], dtype=np.float32),
            won=np.array([d['won'] for d in decs], dtype=np.int8),
            draw=np.array([d['draw'] for d in decs], dtype=np.int8),
            final_status=np.array([d['final_status'] for d in decs], dtype=object),
            game_steps=np.array([d['game_steps'] for d in decs], dtype=np.int16),
            step_index=np.array([d['step_index'] for d in decs], dtype=np.int16),
            decision_index=np.array([d['decision_index'] for d in decs], dtype=np.int16),
            own_hist_type=stack('own_hist_type', np.int16),
            own_hist_card=stack('own_hist_card', np.int16),
            own_hist_card2=stack('own_hist_card2', np.int16),
            own_hist_attack=stack('own_hist_attack', np.int16),
            own_hist_context=stack('own_hist_context', np.int16),
            own_hist_select_type=stack('own_hist_select_type', np.int16),
            own_hist_count=stack('own_hist_count', np.float16),
            own_hist_mask=stack('own_hist_mask', np.float16),
            opp_hist_type=stack('opp_hist_type', np.int16),
            opp_hist_card=stack('opp_hist_card', np.int16),
            opp_hist_card2=stack('opp_hist_card2', np.int16),
            opp_hist_attack=stack('opp_hist_attack', np.int16),
            opp_hist_context=stack('opp_hist_context', np.int16),
            opp_hist_select_type=stack('opp_hist_select_type', np.int16),
            opp_hist_count=stack('opp_hist_count', np.float16),
            opp_hist_mask=stack('opp_hist_mask', np.float16),
            log_hist_type=stack('log_hist_type', np.int16),
            log_hist_player=stack('log_hist_player', np.int8),
            log_hist_card=stack('log_hist_card', np.int16),
            log_hist_card2=stack('log_hist_card2', np.int16),
            log_hist_attack=stack('log_hist_attack', np.int16),
            log_hist_serial=stack('log_hist_serial', np.int16),
            log_hist_serial2=stack('log_hist_serial2', np.int16),
            log_hist_from_area=stack('log_hist_from_area', np.int8),
            log_hist_to_area=stack('log_hist_to_area', np.int8),
            log_hist_value=stack('log_hist_value', np.float16),
            log_hist_mask=stack('log_hist_mask', np.float16),
            board_hist_cards=stack('board_hist_cards', np.int16),
            board_hist_feats=stack('board_hist_feats', np.float16),
            board_hist_mask=stack('board_hist_mask', np.float16),
            history_summary=stack('history_summary', np.float16),
            feature_version=np.array(FEATURE_VERSION, dtype=object),
            state_feat_dim=np.array(STATE_FEAT_DIM, dtype=np.int16),
            opt_feat_dim=np.array(OPT_FEAT_DIM, dtype=np.int16),
            history_summary_dim=np.array(HISTORY_SUMMARY_DIM, dtype=np.int16),
            action_history_k=np.array(action_history_k, dtype=np.int16),
            log_history_k=np.array(log_history_k, dtype=np.int16),
            board_history_k=np.array(board_history_k, dtype=np.int16),
            board_history_feat_dim=np.array(board_history_feat_dim, dtype=np.int16),
        )
        mb = os.path.getsize(os.path.join(arch_dir, f'{fbase}.npz')) / 1024**2
        print(f"  {key}: {n} decs, {mb:.0f}MB")

    elapsed = time.time() - t0
    print(f"  Done: {total} decs in {elapsed:.0f}s ({total/max(elapsed,1):.0f} dec/s)\n")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes_dir")
    p.add_argument("--out", default="data/bc_corpus")
    p.add_argument("--lb-csv", default=None, help="Leaderboard CSV path (auto-download if omitted)")
    p.add_argument("--workers", type=int, default=1,
                   help="number of episode zip files to process concurrently")
    p.add_argument("--progress-every", type=int, default=500,
                   help="print progress every N episodes per zip; 0 disables progress")
    p.add_argument("--action-history-k", type=int, default=DEFAULT_ACTION_HISTORY_K,
                   help="save this many previous own/opponent labeled action events per decision")
    p.add_argument("--log-history-k", type=int, default=DEFAULT_LOG_HISTORY_K,
                   help="save this many recent public observation log events per decision")
    p.add_argument("--board-history-k", type=int, default=DEFAULT_BOARD_HISTORY_K,
                   help="save this many previous board snapshots from the same player perspective")
    p.add_argument("--board-history-feat-dim", type=int, default=BOARD_HISTORY_FEAT_DIM,
                   help="number of scalar state features saved per board-history snapshot")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="debug/smoke-test limit per zip; 0 processes all episodes")
    args = p.parse_args()

    name_to_score = load_leaderboard_scores(args.lb_csv)
    print(f"Leaderboard: {len(name_to_score)} teams\n")
    if not name_to_score:
        print("WARNING: no leaderboard scores loaded; all episodes will fall into 600-699", flush=True)

    zips = sorted(Path(args.episodes_dir).glob("*.zip"))
    if args.workers <= 1:
        for zf in zips:
            process_zip(
                zf,
                args.out,
                name_to_score,
                args.progress_every,
                args.action_history_k,
                args.log_history_k,
                args.board_history_k,
                args.board_history_feat_dim,
                args.max_episodes,
            )
    else:
        workers = min(args.workers, len(zips))
        print(f"Processing {len(zips)} zips with {workers} workers\n", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    process_zip,
                    zf,
                    args.out,
                    name_to_score,
                    args.progress_every,
                    args.action_history_k,
                    args.log_history_k,
                    args.board_history_k,
                    args.board_history_feat_dim,
                    args.max_episodes,
                )
                for zf in zips
            ]
            done = 0
            t0 = time.time()
            for fut in as_completed(futs):
                fut.result()
                done += 1
                elapsed = time.time() - t0
                print(f"Finished {done}/{len(futs)} zip files in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
