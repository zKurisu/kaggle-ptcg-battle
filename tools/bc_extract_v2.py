#!/usr/bin/env python3
"""
Extract state→action pairs from Kaggle episode ZIPs, grouped by archetype.

Usage:
    python tools/bc_extract_v2.py ../episodes_raw/ --out data/bc_corpus/

Output per zip × per archetype:
    data/bc_corpus/<Archetype>/<date>.npz
"""

import sys, os, json, zipfile, time, argparse, numpy as np
from collections import defaultdict, Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent; _WS = _REPO.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_WS))

ARCHETYPES = {
    "Marnie Grimmsnarl": [648], "Alakazam": [743, 245, 741, 742],
    "Crustle Wall": [345, 344], "Dragapult": [121], "Mega Lucario": [678],
    "Archaludon": [190], "Cynthia Garchomp": [381], "Mega Lopunny": [849],
    "Teal Mask Ogerpon": [96], "Team Rocket Mewtwo": [431], "Festival Lead": [93],
    "Mega Starmie": [1031, 367], "Iono Bellibolt": [269], "Mega Abomasnow": [723],
    "N's Zoroark": [293, 320], "Hop Trevenant": [879], "Raging Bolt": [1065],
}

def load_leaderboard_scores(lb_csv_path: str = None) -> dict[str, float]:
    """Load team_name → score from leaderboard CSV."""
    if lb_csv_path is None:
        # Download latest leaderboard
        import subprocess, glob, tempfile
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
                     deck: list[int], band: str) -> bool:
    sel = obs.get('select')
    if sel is None or len(sel.get('option', [])) == 0:
        return False
    if not _valid_action(action, sel):
        return False
    arch = classify(deck)
    key = f"{arch}|{band}"
    ed = encoder.encode(obs)
    all_data[key].append({
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
    })
    return True


def process_zip(zip_path, out_dir, name_to_score: dict):
    from ptcg_rl.encoder import FastEncoder
    encoder = FastEncoder()

    with zipfile.ZipFile(str(zip_path)) as zf:
        fnames = [n for n in zf.namelist() if n.endswith('.json')]
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

                # Get scores from team names
                info = data.get('info', {})
                teams = info.get('TeamNames', [])
                scores = [name_to_score.get(t, 0) for t in teams[:2]]
                bands = [score_band(s) for s in scores]

                # Kaggle episode rows store the action that answered the
                # previous ACTIVE observation for that player.
                pending = [None, None]
                for step in steps[1:]:
                    for pi, pd in enumerate(step[:2]):
                        if not isinstance(pd, dict):
                            continue
                        action = pd.get('action', [])
                        if pending[pi] is not None and isinstance(action, list) and len(action) != 60:
                            obs_prev = pending[pi]
                            band = bands[pi] if pi < len(bands) else "unknown"
                            try:
                                ok = _append_decision(all_data, encoder, obs_prev, action, decks[pi], band)
                                bad_actions += 0 if ok else 1
                            except Exception:
                                errors += 1
                            pending[pi] = None

                        obs = pd.get('observation')
                        obs = obs if isinstance(obs, dict) else None
                        sel = obs.get('select') if obs else None
                        if (pd.get('status') == 'ACTIVE' and sel is not None
                                and len(sel.get('option', [])) > 0):
                            pending[pi] = obs
            except Exception:
                errors += 1

            if (i+1) % 500 == 0:
                total = sum(len(v) for v in all_data.values())
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(fnames)-i-1)
                print(f"  {i+1}/{len(fnames)} | {total} decs | bad {bad_actions} | err {errors} | eta {eta:.0f}s")

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
    args = p.parse_args()

    name_to_score = load_leaderboard_scores(args.lb_csv)
    print(f"Leaderboard: {len(name_to_score)} teams\n")

    for zf in sorted(Path(args.episodes_dir).glob("*.zip")):
        process_zip(zf, args.out, name_to_score)


if __name__ == "__main__":
    main()
