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

def classify(deck):
    cnt = Counter(deck); best, bs = "Other", 0
    for n, ks in ARCHETYPES.items():
        s = sum(cnt.get(k, 0) for k in ks)
        if s > bs: bs, best = s, n
    return best if bs >= 2 else "Other"


def process_zip(zip_path, out_dir):
    from ptcg_rl.encoder import FastEncoder
    encoder = FastEncoder()

    with zipfile.ZipFile(str(zip_path)) as zf:
        fnames = [n for n in zf.namelist() if n.endswith('.json')]
    print(f"{zip_path.name}: {len(fnames)} eps")
    t0 = time.time()

    all_data = defaultdict(list)
    for i, fname in enumerate(fnames):
        try:
            raw = zf.read(fname).decode('utf-8')
            data = json.loads(raw); steps = data['steps']
            if len(steps) < 2: continue
            decks_raw = steps[0][0].get('visualize', [{}])[0].get('action', [])
            if len(decks_raw) != 2: continue
            decks = [decks_raw[0], decks_raw[1]]
            if len(decks[0]) != 60 or len(decks[1]) != 60: continue

            for step in steps[1:]:
                for pi, pd in enumerate(step[:2]):
                    obs = pd.get('observation', {})
                    sel = obs.get('select')
                    act = pd.get('action', [])
                    if sel is None or not isinstance(act, list): continue
                    if len(sel.get('option', [])) == 0: continue
                    if len(act) == 60: continue  # deck selection

                    arch = classify(decks[pi])
                    ed = encoder.encode(obs)
                    all_data[arch].append({
                        'board': ed.board_cards.astype(np.int16),
                        'hand': ed.hand_cards.astype(np.int16),
                        'feats': ed.state_feats.astype(np.float16),
                        'ot': ed.opt_type.astype(np.int16),
                        'oc': ed.opt_card.astype(np.int16),
                        'oc2': ed.opt_card2.astype(np.int16),
                        'oa': ed.opt_attack.astype(np.int16),
                        'of': ed.opt_feats.astype(np.float16),
                        'action': np.array(act, dtype=np.int16),
                        'min_c': ed.min_count, 'max_c': ed.max_count,
                    })
        except: pass

        if (i+1) % 500 == 0:
            total = sum(len(v) for v in all_data.values())
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(fnames)-i-1)
            print(f"  {i+1}/{len(fnames)} | {total} decs | eta {eta:.0f}s")

    # Save
    total = 0
    for arch, decs in sorted(all_data.items()):
        n = len(decs); total += n
        if n < 100: continue
        arch_dir = os.path.join(out_dir, arch.replace(' ', '_'))
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
        print(f"  {arch}: {n} decs, {mb:.0f}MB")

    elapsed = time.time() - t0
    print(f"  Done: {total} decs in {elapsed:.0f}s ({total/max(elapsed,1):.0f} dec/s)\n")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes_dir")
    p.add_argument("--out", default="data/bc_corpus")
    args = p.parse_args()

    for zf in sorted(Path(args.episodes_dir).glob("*.zip")):
        process_zip(zf, args.out)


if __name__ == "__main__":
    main()
