from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "logs/ladder_pool_0812_all_v13_20260813/pool_manifest.csv"
OUT_DECKS = ROOT / "logs/v15_common_matchups_20260816/top_decks_by_arch.csv"
OUT_EVAL = ROOT / "logs/v15_common_matchups_20260816/eval_manifest_existing_models.csv"

ARCHES = [
    "Marnie Grimmsnarl",
    "Dragapult",
    "Mega Lopunny",
    "Alakazam",
    "Teal Mask Ogerpon",
    "Mega Lucario",
    "Crustle Wall",
    "Festival Lead",
    "Team Rocket Mewtwo",
    "Cynthia Garchomp",
]

# Prefer current v15 route-count/pilot when available, otherwise use v14
# sequence-pop models restored with the 0812 pool. This manifest is for
# diagnosis, not final submission.
POLICY_BY_ARCH = {
    "Dragapult": "checkpoints/v15_route_count_long_20260815/dragapult_cc2e_route_count_w512.pt",
    "Alakazam": "checkpoints/v15_route_count_long_20260815/alakazam_7f9_route_count_w512.pt",
    "Mega Lopunny": "checkpoints/v15_pilot_0812_0813_20260815/mega_lopunny_v15_plan.pt",
    "Marnie Grimmsnarl": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_marnie_grimmsnarl_b8f251a4_1.pt",
    "Teal Mask Ogerpon": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_teal_mask_ogerpon_8bc67715_1.pt",
    "Mega Lucario": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_mega_lucario_43d6d8b0_1.pt",
    "Crustle Wall": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_crustle_wall_7ee600c6_2.pt",
    "Festival Lead": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_festival_lead_41ffa789_1.pt",
    "Team Rocket Mewtwo": "checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_team_rocket_mewtwo_958e37d2_1.pt",
}

PREFERRED_SIG_BY_ARCH = {
    "Dragapult": "cc2e995b5ad0",
    "Alakazam": "7f9a538936e3",
    "Mega Lopunny": "f1445356c3a7",
    "Marnie Grimmsnarl": "b8f251a476e7",
    "Teal Mask Ogerpon": "8bc677152093",
    "Mega Lucario": "43d6d8b0fce9",
    "Crustle Wall": "7ee600c6f769",
    "Festival Lead": "41ffa7894f40",
    "Team Rocket Mewtwo": "958e37d2a5f6",
}


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def int_value(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0


def main() -> None:
    rows: list[dict[str, str]] = []
    with POOL.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("archetype") not in ARCHES:
                continue
            row["_weight"] = str(float_value(row, "weight"))
            row["_score"] = str(float_value(row, "score"))
            row["_games"] = str(int_value(row, "games"))
            rows.append(row)

    best: dict[str, dict[str, str]] = {}
    by_sig: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(
        rows,
        key=lambda r: (float(r["_score"]), float(r["_weight"]), int(r["_games"])),
        reverse=True,
    ):
        best.setdefault(row["archetype"], row)
        by_sig.setdefault((row["archetype"], row["deck_sig"]), row)
    for arch, sig in PREFERRED_SIG_BY_ARCH.items():
        if (arch, sig) in by_sig:
            best[arch] = by_sig[(arch, sig)]

    OUT_DECKS.parent.mkdir(parents=True, exist_ok=True)
    deck_fields = [
        "archetype",
        "deck_sig",
        "team_name",
        "score",
        "score_band",
        "games",
        "wins",
        "losses",
        "weight",
        "deck_path",
    ]
    with OUT_DECKS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=deck_fields)
        writer.writeheader()
        for arch in ARCHES:
            row = best.get(arch)
            if row:
                writer.writerow({k: row.get(k, "") for k in deck_fields})

    eval_fields = [
        "name",
        "eval_entry",
        "policy_path",
        "policy",
        "deck_path",
        "archetype",
        "deck_sig",
        "team_name",
        "score",
        "weight",
    ]
    with OUT_EVAL.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fields)
        writer.writeheader()
        for arch in ARCHES:
            row = best.get(arch)
            policy = POLICY_BY_ARCH.get(arch)
            if not row or not policy or not (ROOT / policy).exists():
                continue
            name = arch.lower().replace("'", "").replace(" ", "_")
            entry_name = f"{name}_{row.get('deck_sig', '')[:8]}"
            writer.writerow(
                {
                    "name": entry_name,
                    "eval_entry": f"{entry_name}={policy}:{row.get('deck_path', '')}",
                    "policy_path": policy,
                    "policy": policy,
                    "deck_path": row.get("deck_path", ""),
                    "archetype": arch,
                    "deck_sig": row.get("deck_sig", ""),
                    "team_name": row.get("team_name", ""),
                    "score": row.get("score", ""),
                    "weight": row.get("weight", ""),
                }
            )

    print(f"wrote {OUT_DECKS}")
    print(f"wrote {OUT_EVAL}")
    with OUT_EVAL.open(newline="", encoding="utf-8") as f:
        print(f"entries {max(sum(1 for _ in f) - 1, 0)}")


if __name__ == "__main__":
    main()
