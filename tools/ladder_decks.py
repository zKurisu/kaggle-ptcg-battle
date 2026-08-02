#!/usr/bin/env python3
"""
Pull decklists from the Kaggle leaderboard with scores and archetype labels.

Pipeline:
  1. Download leaderboard
  2. For each team, find a recent valid episode
  3. Extract the deck from the episode replay
  4. Classify archetype by key card IDs
  5. Output: deck_id, score, archetype, card_ids

Usage:
    python tools/ladder_decks.py pull --sample 100            # Pull 100 teams
    python tools/ladder_decks.py pull --score-min 800 --score-max 1000
    python tools/ladder_decks.py stats decks/ladder/          # Show distribution
"""

import os, sys, json, csv, time, hashlib, subprocess, random
from pathlib import Path
from collections import Counter, defaultdict

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO.parent))

# ── Archetype classification by key card IDs ────────────────────────────

ARCHETYPES = {
    "Mega Lucario ex":    {"keys": [678], "type": "fighting"},
    "Dragapult ex":       {"keys": [121], "type": "dragon"},
    "Alakazam":           {"keys": [743, 245, 741, 742], "type": "psychic"},
    "Iono Bellibolt ex":  {"keys": [269], "type": "lightning"},
    "Mega Abomasnow ex":  {"keys": [723], "type": "water"},
    "Crustle Wall":       {"keys": [345], "type": "fighting"},
    "Archaludon ex":      {"keys": [190], "type": "metal"},
    "Raging Bolt ex":     {"keys": [1065], "type": "dragon"},
    "Garchomp ex":        {"keys": [309], "type": "fighting"},
    "Slowking":           {"keys": [163], "type": "psychic"},
    "Lillie Clefairy ex": {"keys": [272], "type": "psychic"},
    "Marnie Grimmsnarl":  {"keys": [319], "type": "darkness"},
    "N Zoroark ex":       {"keys": [320], "type": "darkness"},
    "Hydrapple ex":       {"keys": [204], "type": "dragon"},
    "Mega Starmie ex":    {"keys": [367], "type": "water"},
    "Mega Greninja ex":   {"keys": [374], "type": "water"},
    "Beedrill ex":        {"keys": [365], "type": "grass"},
    "Sylveon Safeguard":  {"keys": [372], "type": "psychic"},
    "Mega Lopunny ex":    {"keys": [355], "type": "colorless"},
    "Metagross":          {"keys": [276], "type": "psychic"},
    "Ogerpon Box":        {"keys": [108], "type": "grass"},
    "Festival Lead":      {"keys": [308], "type": "psychic"},
    "Typhlosion ex":      {"keys": [302], "type": "fire"},
    "Charizard ex":       {"keys": [790, 928], "type": "fire"},
    "Gardevoir ex":       {"keys": [110, 325], "type": "psychic"},
    "Snorlax Stall":      {"keys": [143, 759, 760], "type": "colorless"},
    "Gholdengo ex":       {"keys": [194], "type": "metal"},
    "Roaring Moon ex":    {"keys": [318], "type": "darkness"},
    "Great Tusk":         {"keys": [339], "type": "fighting"},
}


def classify_deck(card_ids: list[int]) -> str:
    """Return archetype name based on key card presence."""
    cnt = Counter(card_ids)
    scores = {}
    for name, info in ARCHETYPES.items():
        score = sum(cnt.get(k, 0) for k in info["keys"])
        if score > 0:
            scores[name] = score
    if scores:
        return max(scores, key=scores.get)
    return "Other"


def deck_signature(card_ids: list[int]) -> str:
    """Short human-readable deck description."""
    try:
        from cg.api import all_card_data
        cards = {c.cardId: c for c in all_card_data()}
    except Exception:
        cards = {}
    cnt = Counter(card_ids)
    sig = []
    for cid, count in cnt.most_common(5):
        card = cards.get(cid)
        if card and card.cardType in (5, 6): continue
        sig.append(f"{card.name if card else '#'+str(cid)}×{count}")
    return ", ".join(sig[:3])


# ── Leaderboard + Episode pulling ────────────────────────────────────────

def get_leaderboard(limit: int = 500) -> list[dict]:
    """Download leaderboard via Kaggle CLI."""
    print("Downloading leaderboard...")
    subprocess.run(
        ["kaggle", "competitions", "leaderboard", "pokemon-tcg-ai-battle", "--download"],
        cwd="/tmp", capture_output=True, timeout=60,
    )
    # Find the downloaded file
    import glob
    files = glob.glob("/tmp/pokemon-tcg-ai-battle*.csv")
    if not files:
        print("ERROR: leaderboard download failed")
        return []

    rows = []
    with open(files[0]) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"  Got {len(rows)} teams")
    return rows


def get_team_episode(team_id: str) -> dict | None:
    """Find a recent valid episode for a team and extract deck info."""
    # Get episodes for latest submission
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "pokemon-tcg-ai-battle"],
            capture_output=True, text=True, timeout=30,
        )
        # This doesn't work per-team via CLI. Use API directly.
    except Exception:
        pass

    # Alternative: use the episodes API
    try:
        # Get recent submissions for this team
        import urllib.request, urllib.error
        # Kaggle API requires auth — use kaggle CLI wrapped
        pass
    except Exception:
        pass

    return None


def pull_decks_from_replays(replays_dir: str, score_map: dict = None) -> list[dict]:
    """Pull decks from a directory of episode JSON files.
    If score_map provided, it maps team_name → score."""
    results = []
    for fp in sorted(Path(replays_dir).glob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            steps = data.get("steps", [])
            if not steps: continue

            # Extract both decks
            step0 = steps[0]
            for pi, player_data in enumerate(step0[:2]):
                viz = player_data.get("visualize", [])
                if not viz: continue
                action = viz[0].get("action", [])
                if not isinstance(action, list) or len(action) != 2: continue

                deck_ids = action[pi]  # This player's deck
                if len(deck_ids) != 60: continue

                dhash = hashlib.md5(str(sorted(deck_ids)).encode()).hexdigest()[:8]
                arch = classify_deck(deck_ids)
                sig = deck_signature(deck_ids)

                results.append({
                    "hash": dhash,
                    "archetype": arch,
                    "signature": sig,
                    "filename": fp.name,
                    "player": pi,
                })
        except Exception:
            continue

    return results


def pull_with_kaggle_api(num_teams: int = 100,
                          min_score: float = 600,
                          max_score: float = 1300) -> list[dict]:
    """
    Full pipeline: leaderboard → team submission → episode → deck extraction.

    Uses Kaggle API directly (not CLI) for programmatic access.
    Rate-limited to ~1 req/sec to avoid 429 errors.
    """
    import requests
    import urllib.request
    import json as json_mod

    # Using Kaggle's public API (read-only, no auth needed for public data)
    BASE = "https://www.kaggle.com/api/v1"

    results = []

    # 1. Get leaderboard
    print(f"Fetching leaderboard (top {num_teams})...")
    try:
        url = f"{BASE}/competitions/pokemon-tcg-ai-battle/leaderboard/view"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            lb_data = json_mod.loads(resp.read())
        teams = lb_data.get("submissions", [])[:num_teams]
        print(f"  Got {len(teams)} teams on leaderboard")
    except Exception as e:
        print(f"  Leaderboard fetch failed: {e}")
        print(f"  Falling back to Kaggle CLI leaderboard download")
        # Fallback: use CLI
        subprocess.run(["kaggle", "competitions", "leaderboard",
                       "pokemon-tcg-ai-battle", "--download"],
                       cwd="/tmp", capture_output=True)
        import glob
        files = glob.glob("/tmp/pokemon-tcg-ai-battle*.csv")
        if not files:
            return []
        with open(files[0]) as f:
            reader = csv.DictReader(f)
            teams = list(reader)[:num_teams]

    print(f"\nPulling episodes for {len(teams)} teams (rate-limited, ~{len(teams)}s)...")

    for i, team in enumerate(teams):
        team_id = team.get("teamId") or team.get("TeamId")
        team_name = team.get("teamName") or team.get("TeamName", f"team_{team_id}")
        score = float(team.get("score") or team.get("Score") or team.get("publicScore") or 0)

        if score < min_score or score > max_score:
            continue

        # 2. Get team's latest submission
        try:
            url = f"{BASE}/competitions/pokemon-tcg-ai-battle/submissions?teamId={team_id}"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                subs = json_mod.loads(resp.read())

            if not subs:
                continue

            # Take the active submission closest to current score
            sub = subs[0]  # Most recent
            sub_id = sub.get("ref") or sub.get("id")
            if not sub_id:
                continue

        except Exception:
            continue

        # 3. Get episode for this submission
        try:
            url = f"{BASE}/competitions/pokemon-tcg-ai-battle/submissions/{sub_id}/episodes"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                eps = json_mod.loads(resp.read())

            if not eps:
                continue

            # Pick a completed episode
            ep = None
            for e in eps:
                if e.get("state") == "EpisodeState.COMPLETED":
                    ep = e; break
            if not ep:
                ep = eps[0]

            ep_id = ep.get("id")
            if not ep_id:
                continue

        except Exception:
            continue

        # 4. Download episode replay
        try:
            url = f"{BASE}/competitions/pokemon-tcg-ai-battle/episodes/{ep_id}/replay"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                replay_data = json_mod.loads(resp.read())

            # 5. Extract deck from replay
            steps = replay_data.get("steps", [])
            if not steps:
                continue

            step0 = steps[0]
            for pi, player_data in enumerate(step0[:2]):
                viz = player_data.get("visualize", [])
                if not viz: continue
                action = viz[0].get("action", [])
                if not isinstance(action, list) or len(action) != 2: continue

                deck_ids = action[pi]
                if len(deck_ids) != 60: continue

                dhash = hashlib.md5(str(sorted(deck_ids)).encode()).hexdigest()[:8]
                arch = classify_deck(deck_ids)
                sig = deck_signature(deck_ids)

                results.append({
                    "hash": dhash,
                    "archetype": arch,
                    "signature": sig,
                    "score": score,
                    "team": team_name,
                    "player": pi,
                })

        except Exception:
            continue

        # Rate limiting
        if i % 10 == 0:
            print(f"  {i}/{len(teams)} teams processed, {len(results)} decks found")
        time.sleep(1.1)  # ~1 req/sec

    return results


def show_stats(decks: list[dict]):
    """Print deck distribution by archetype and score band."""
    if not decks:
        print("No decks to analyze.")
        return

    # Score band distribution
    bands = [(600, 700), (700, 800), (800, 900), (900, 1000),
             (1000, 1100), (1100, 1200), (1200, 1300)]

    print(f"\n=== Deck Distribution ({len(decks)} decks) ===")

    for lo, hi in bands:
        in_band = [d for d in decks if lo <= d.get("score", 0) < hi]
        if not in_band: continue
        archs = Counter(d["archetype"] for d in in_band)
        print(f"\n{lo}-{hi} ({len(in_band)} decks):")
        for arch, count in archs.most_common(10):
            pct = count / len(in_band) * 100
            print(f"  {arch:25s} {count:>3} ({pct:5.1f}%)")

    print(f"\n=== Overall Archetype Distribution ===")
    archs = Counter(d["archetype"] for d in decks)
    for arch, count in archs.most_common(20):
        avg_score = sum(d["score"] for d in decks if d["archetype"] == arch) / max(count, 1)
        print(f"  {arch:25s} {count:>3} decks, avg score {avg_score:.0f}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "pull":
        num = 100
        min_s, max_s = 600, 1300
        args = sys.argv[2:]
        while args:
            if args[0] == "--sample" and len(args) > 1:
                num = int(args[1]); args = args[2:]
            elif args[0] == "--score-min" and len(args) > 1:
                min_s = float(args[1]); args = args[2:]
            elif args[0] == "--score-max" and len(args) > 1:
                max_s = float(args[1]); args = args[2:]
            else:
                args = args[1:]

        decks = pull_with_kaggle_api(num, min_s, max_s)

        # Save
        out_path = _REPO / "decks" / "ladder_decks.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(decks, f, indent=2)
        print(f"\nSaved {len(decks)} decks to {out_path}")

        show_stats(decks)

    elif cmd == "stats":
        path = sys.argv[2] if len(sys.argv) > 2 else str(_REPO / "decks" / "ladder_decks.json")
        with open(path) as f:
            decks = json.load(f)
        show_stats(decks)

    elif cmd == "classify":
        # Quick classify a single deck CSV
        deck_path = sys.argv[2]
        with open(deck_path) as f:
            ids = [int(l.strip()) for l in f if l.strip()]
        print(f"Archetype: {classify_deck(ids)}")
        print(f"Signature: {deck_signature(ids)}")

    else:
        print(f"Unknown: {cmd}")
