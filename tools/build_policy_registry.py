#!/usr/bin/env python3
"""Build a checkpoint -> deck registry to avoid policy/deck mismatch."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_registry import signature_for_deck


ARCH_PATTERNS = [
    ("Marnie Grimmsnarl", ("marnie_grimmsnarl", "marnie")),
    ("Alakazam", ("alakazam",)),
    ("Crustle Wall", ("crustle_wall", "crustle")),
    ("Team Rocket Mewtwo", ("team_rocket_mewtwo", "rocket_mewtwo", "mewtwo")),
    ("Teal Mask Ogerpon", ("teal_mask_ogerpon", "ogerpon")),
    ("Mega Lopunny", ("mega_lopunny", "lopunny")),
    ("Dragapult", ("dragapult",)),
    ("Festival Lead", ("festival_lead", "festival")),
    ("Cynthia Garchomp", ("cynthia_garchomp", "cynthia")),
    ("Mega Lucario", ("mega_lucario", "lucario")),
    ("Mega Abomasnow", ("mega_abomasnow", "abomasnow")),
    ("Mega Starmie", ("mega_starmie", "starmie")),
    ("Archaludon", ("archaludon",)),
    ("Hop Trevenant", ("hop_trevenant", "trevenant")),
]


def infer_archetype(policy: Path) -> str:
    name = policy.stem.lower()
    for arch, slugs in ARCH_PATTERNS:
        if any(slug in name for slug in slugs):
            return arch
    return ""


def load_manifest(path: str) -> list[dict[str, str]]:
    if not path:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_stats(paths: list[str]) -> dict[str, str]:
    """Return archetype -> old/top deck sig from bc_corpus_stats CSVs."""
    out: dict[str, str] = {}
    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows:
            continue
        stem = Path(path).stem
        arch = stem
        arch = arch.replace("bc_corpus_stats_", "").replace("_v7sig", "").replace("_", " ")
        arch = " ".join(w if w.lower() not in {"s"} else "s" for w in arch.split())
        out[arch.lower()] = rows[0].get("deck_sig", "")
    return out


def choose_manifest_row(policy: Path, arch: str, manifest: list[dict[str, str]]) -> dict[str, str] | None:
    name = policy.stem.lower()
    rows = [r for r in manifest if r.get("archetype") == arch]
    if not rows:
        return None
    # If checkpoint contains a canonical sig, prefer that exact deck.
    for r in rows:
        sig = (r.get("deck_sig") or "").lower()
        if sig and sig in name:
            return r
    # Heuristic for known deck-specific legacy Alakazam checkpoint.
    if arch == "Alakazam" and "cee" in name:
        for r in rows:
            if (r.get("deck_sig") or "").startswith("7f9a5389"):
                return r
    # Prefer highest manifest order/weight. Manifest is already sorted by build_ladder_pool.
    return rows[0]


def rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-glob", action="append", default=["checkpoints/*.npz"])
    p.add_argument("--manifest", default="logs/ladder_pool_v2/pool_manifest.csv")
    p.add_argument("--stats-csv", action="append", default=[],
                   help="optional bc_corpus_stats CSV; currently used for diagnostics only")
    p.add_argument("--out", default="logs/policy_deck_registry.csv")
    p.add_argument("--repo-root", default=".")
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    manifest = load_manifest(args.manifest)
    stats = load_stats(args.stats_csv)
    policies: list[Path] = []
    for pat in args.checkpoint_glob:
        policies.extend(sorted(Path(".").glob(pat)))
    seen = set()
    rows = []
    for policy in policies:
        if policy in seen or re.search(r"_ep\d{3}$", policy.stem):
            continue
        seen.add(policy)
        arch = infer_archetype(policy)
        row = choose_manifest_row(policy, arch, manifest) if arch else None
        deck_path = row.get("deck_path", "") if row else ""
        deck_sig = row.get("deck_sig", "") if row else ""
        status = "matched" if deck_path else "missing"
        if deck_path:
            try:
                actual_sig = signature_for_deck(deck_path)
                if deck_sig and actual_sig != deck_sig:
                    status = f"sig_mismatch:{actual_sig}"
            except Exception as exc:
                status = f"bad_deck:{exc}"
        rows.append({
            "policy_path": rel(str(policy), root),
            "deck_path": rel(deck_path, root) if deck_path else "",
            "deck_sig": deck_sig,
            "archetype": arch,
            "status": status,
            "source": "manifest_top",
            "note": f"top_old_sig={stats.get(arch.lower(), '')}" if arch else "",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = ["policy_path", "deck_path", "deck_sig", "archetype", "status", "source", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["status"] == "matched")
    print(f"Wrote {out}: {n_ok}/{len(rows)} matched")
    for r in rows:
        if r["status"] != "matched":
            print(f"  {r['status']}: {r['policy_path']} arch={r['archetype']}")


if __name__ == "__main__":
    main()
