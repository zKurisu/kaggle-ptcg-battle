#!/usr/bin/env python3
"""Plan deck-specific BC2 specialist training from bc_corpus_stats CSVs."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shlex
from pathlib import Path


SLUG_TO_ARCH = {
    "alakazam": "Alakazam",
    "crustle": "Crustle Wall",
    "crustle_wall": "Crustle Wall",
    "cynthia": "Cynthia Garchomp",
    "cynthia_garchomp": "Cynthia Garchomp",
    "dragapult": "Dragapult",
    "festival": "Festival Lead",
    "festival_lead": "Festival Lead",
    "lopunny": "Mega Lopunny",
    "mega_lopunny": "Mega Lopunny",
    "marnie": "Marnie Grimmsnarl",
    "marnie_grimmsnarl": "Marnie Grimmsnarl",
    "mewtwo": "Team Rocket Mewtwo",
    "team_rocket_mewtwo": "Team Rocket Mewtwo",
    "ogerpon": "Teal Mask Ogerpon",
    "teal_mask_ogerpon": "Teal Mask Ogerpon",
}


FIELDS = [
    "job_name",
    "archetype",
    "variant",
    "deck_sigs",
    "n_decks",
    "kept",
    "total_kept",
    "share",
    "cumulative_share",
    "top_team",
    "avg_score",
    "save",
    "log",
]


def slugify(text: str) -> str:
    text = text.lower().replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "policy"


def infer_slug(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"^bc_corpus_stats_", "", stem)
    stem = re.sub(r"_v\d+.*$", "", stem)
    return slugify(stem)


def infer_archetype(path: str) -> str:
    slug = infer_slug(path)
    return SLUG_TO_ARCH.get(slug, slug.replace("_", " ").title())


def expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    out = []
    for part in m.group(1).split(","):
        out.extend(expand_braces(pattern[: m.start()] + part + pattern[m.end() :]))
    return out


def expand_stats_glob(pattern: str) -> list[str]:
    paths: list[str] = []
    for pat in expand_braces(pattern):
        paths.extend(glob.glob(pat))
    return sorted(dict.fromkeys(paths))


def read_stats(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                kept = int(row.get("kept") or 0)
            except ValueError:
                kept = 0
            row = dict(row)
            row["_kept"] = kept
            rows.append(row)
    rows.sort(key=lambda r: (r["_kept"], float(r.get("avg_score") or 0.0)), reverse=True)
    return rows


def choose_variants(rows: list[dict], *, total: int, top1_threshold: float, cover_threshold: float,
                    max_k: int, force_top1: bool) -> list[tuple[str, list[dict]]]:
    if total <= 0 or not rows:
        return []
    variants: list[tuple[str, list[dict]]] = []
    top1_share = rows[0]["_kept"] / total
    if force_top1 or top1_share >= top1_threshold:
        variants.append(("top1", rows[:1]))
    if top1_share < top1_threshold:
        kept = 0
        selected = []
        for row in rows[:max_k]:
            selected.append(row)
            kept += row["_kept"]
            if kept / total >= cover_threshold:
                break
        if selected and len(selected) > 1:
            variants.append((f"top{len(selected)}", selected))
        elif selected and not variants:
            variants.append(("top1", selected))
    return variants


def make_plan(args: argparse.Namespace) -> list[dict]:
    plan: list[dict] = []
    seen_arch: set[str] = set()
    for path in expand_stats_glob(args.stats_glob):
        arch = infer_archetype(path)
        if arch in seen_arch and not args.allow_duplicate_archetype:
            print(f"Skip duplicate stats for {arch}: {path}", flush=True)
            continue
        seen_arch.add(arch)
        slug = slugify(arch)
        all_rows = read_stats(path)
        total = sum(r["_kept"] for r in all_rows)
        if total < args.min_total_decisions:
            print(f"Skip {path}: total kept {total} < {args.min_total_decisions}", flush=True)
            continue
        rows = [r for r in all_rows if r["_kept"] >= args.min_deck_decisions]
        if not rows:
            print(f"Skip {path}: no deck has >= {args.min_deck_decisions} kept decisions", flush=True)
            continue
        variants = choose_variants(
            rows,
            total=total,
            top1_threshold=args.top1_threshold,
            cover_threshold=args.cover_threshold,
            max_k=args.max_k,
            force_top1=args.force_top1,
        )
        for variant, selected in variants:
            kept = sum(r["_kept"] for r in selected)
            deck_sigs = [r["deck_sig"] for r in selected]
            job_name = f"{slug}_{variant}_{args.tag}"
            save = str(Path(args.checkpoint_dir) / f"bc2_{job_name}.npz")
            log = str(Path(args.log_dir) / f"train_{job_name}.log")
            plan.append(
                {
                    "job_name": job_name,
                    "archetype": arch,
                    "variant": variant,
                    "deck_sigs": " ".join(deck_sigs),
                    "n_decks": len(selected),
                    "kept": kept,
                    "total_kept": total,
                    "share": kept / total,
                    "cumulative_share": kept / total,
                    "top_team": selected[0].get("top_team", ""),
                    "avg_score": selected[0].get("avg_score", ""),
                    "save": save,
                    "log": log,
                }
            )
    return plan


def write_plan(path: str, plan: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in plan:
            w.writerow(row)


def train_cmd(args: argparse.Namespace, row: dict) -> str:
    cmd = [
        "python3",
        "-u",
        "tools/bc2_train.py",
        "--corpus",
        args.corpus,
        "--archetype",
        row["archetype"],
        "--score-bands",
        *args.score_bands,
    ]
    for sig in str(row["deck_sigs"]).split():
        cmd.extend(["--deck-sig", sig])
    cmd.extend(
        [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--width",
            str(args.width),
            "--device",
            "cuda:0",
            "--first-action-weight",
            str(args.first_action_weight),
            "--option-weight",
            str(args.option_weight),
            "--win-weight",
            str(args.win_weight),
            "--loss-weight",
            str(args.loss_weight),
            "--draw-weight",
            str(args.draw_weight),
            "--value-weight",
            str(args.value_weight),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--save",
            row["save"],
        ]
    )
    if args.winner_only:
        cmd.append("--winner-only")
    if args.legacy_state_pool:
        cmd.append("--legacy-state-pool")
    return f"CUDA_VISIBLE_DEVICES=${{GPU_ID}} {shlex.join(cmd)} > {shlex.quote(row['log'])} 2>&1"


def write_script(path: str, plan: list[dict], args: argparse.Namespace) -> None:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        gpus = [""]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(args.log_dir)} {shlex.quote(args.checkpoint_dir)}",
        "",
        "JOB_NAMES=()",
        "JOB_CMDS=()",
    ]
    for row in plan:
        lines.append(f"JOB_NAMES+=({shlex.quote(row['job_name'])})")
        lines.append(f"JOB_CMDS+=({shlex.quote(train_cmd(args, row))})")
    lines.extend(
        [
            "",
            "TOTAL_JOBS=${#JOB_CMDS[@]}",
            'QUEUE_DIR="${TMPDIR:-/tmp}/ptcg_bc_queue_$$_$(date +%s%N)"',
            'IDX_FILE="$QUEUE_DIR/next"',
            'LOCK_DIR="$QUEUE_DIR/lock"',
            'mkdir -p "$QUEUE_DIR"',
            'echo 0 > "$IDX_FILE"',
            "",
            "claim_job() {",
            "  while ! mkdir \"$LOCK_DIR\" 2>/dev/null; do",
            "    sleep 0.05",
            "  done",
            "  local idx",
            "  idx=$(cat \"$IDX_FILE\")",
            "  if (( idx >= TOTAL_JOBS )); then",
            "    rmdir \"$LOCK_DIR\"",
            "    return 1",
            "  fi",
            "  echo $((idx + 1)) > \"$IDX_FILE\"",
            "  CLAIMED_IDX=\"$idx\"",
            "  rmdir \"$LOCK_DIR\"",
            "  return 0",
            "}",
            "",
            "worker() {",
            "  local GPU_ID=\"$1\"",
            "  while claim_job; do",
            "    local idx=\"$CLAIMED_IDX\"",
            "    local name=\"${JOB_NAMES[$idx]}\"",
            "    local cmd=\"${JOB_CMDS[$idx]}\"",
            "    echo \"==== $name gpu=${GPU_ID:-cpu}\"",
            "    if ! eval \"$cmd\"; then",
            "      echo \"FAILED $name gpu=${GPU_ID:-cpu}\" >&2",
            "      touch \"$QUEUE_DIR/failed\"",
            "    fi",
            "  done",
            "}",
            "",
        ]
    )
    for gpu in gpus:
        lines.append(f"worker {shlex.quote(gpu)} &")
    lines.extend(
        [
            "wait",
            'if [[ -f "$QUEUE_DIR/failed" ]]; then',
            '  rm -rf "$QUEUE_DIR"',
            "  exit 1",
            "fi",
            'rm -rf "$QUEUE_DIR"',
        ]
    )
    lines.append("echo 'All deck-specific BC jobs finished.'")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    os.chmod(out, 0o755)


def write_eval_script(path: str, plan: list[dict], args: argparse.Namespace) -> None:
    if not path:
        return
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(args.log_dir)}",
        "",
        "python3 tools/build_policy_registry.py \\",
        "  --checkpoint-glob 'checkpoints/*.npz' \\",
        f"  --manifest {shlex.quote(args.manifest)} \\",
        f"  --out {shlex.quote(args.registry)}",
        "",
        "OPPS=$(python3 tools/emit_ladder_pool_entries.py \\",
        f"  {shlex.quote(args.manifest)} \\",
        f"  --top {args.ladder_top} \\",
        "  --one-per-archetype)",
        "",
    ]
    random_log = Path(args.log_dir) / f"eval_random_{args.tag}.log"
    ladder_summary = Path(args.log_dir) / f"round_robin_{args.tag}_vs_ladder_pool_summary.log"
    lines.extend([
        f": > {shlex.quote(str(random_log))}",
        f": > {shlex.quote(str(ladder_summary))}",
        "",
    ])
    for row in plan:
        name = row["job_name"]
        policy = row["save"]
        ladder_csv = Path(args.log_dir) / f"round_robin_{name}_vs_ladder_pool.csv"
        ladder_log = Path(args.log_dir) / f"round_robin_{name}_vs_ladder_pool.log"
        lines.extend([
            f"echo '==== random {name}' | tee -a {shlex.quote(str(random_log))}",
            "python3 tools/eval_bc.py \\",
            f"  {shlex.quote(policy)} \\",
            f"  --registry {shlex.quote(args.registry)} \\",
            "  --auto-deck \\",
            f"  --games {args.random_games} \\",
            f"  --workers {args.workers} \\",
            f"  --max-turns {args.max_turns} \\",
            f"  --progress-every {args.progress_every} \\",
            f"  2>&1 | tee -a {shlex.quote(str(random_log))}",
            "",
            f"echo '==== ladder {name}' | tee -a {shlex.quote(str(ladder_summary))}",
            "python3 tools/eval_round_robin.py \\",
            f"  --registry {shlex.quote(args.registry)} \\",
            f"  --entry candidate={shlex.quote(policy)} \\",
            "  $OPPS \\",
            f"  --games {args.ladder_games} \\",
            f"  --workers {args.workers} \\",
            f"  --max-turns {args.max_turns} \\",
            f"  --progress-every {args.progress_every} \\",
            "  --skip-bad-entries \\",
            "  --candidate-only \\",
            f"  --out-csv {shlex.quote(str(ladder_csv))} \\",
            f"  2>&1 | tee {shlex.quote(str(ladder_log))}",
            "python3 tools/summarize_round_robin.py \\",
            f"  {shlex.quote(str(ladder_csv))} \\",
            f"  --manifest {shlex.quote(args.manifest)} \\",
            f"  --top 8 2>&1 | tee -a {shlex.quote(str(ladder_summary))}",
            "",
        ])
    lines.append("echo 'All deck-specific BC evals finished.'")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    os.chmod(out, 0o755)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stats-glob", required=True)
    p.add_argument("--allow-duplicate-archetype", action="store_true",
                   help="do not suppress duplicate stats files that infer to the same archetype")
    p.add_argument("--corpus", default="data/bc_corpus_banded_v7sig")
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--tag", default="v7sig_topdeck_w2")
    p.add_argument("--out", default="logs/deck_specific_bc_plan.csv")
    p.add_argument("--script", default="logs/train_deck_specific_bc.sh")
    p.add_argument("--eval-script", default="",
                   help="optional shell script for random smoke + candidate-only ladder-pool evals")
    p.add_argument("--registry", default="logs/policy_deck_registry_v7sig.csv")
    p.add_argument("--manifest", default="logs/ladder_pool_v2/pool_manifest.csv")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--top1-threshold", type=float, default=0.75)
    p.add_argument("--cover-threshold", type=float, default=0.80)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--force-top1", action="store_true",
                   help="always include a top1 specialist in addition to any topK mixed job")
    p.add_argument("--min-deck-decisions", type=int, default=5000)
    p.add_argument("--min-total-decisions", type=int, default=20000)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--first-action-weight", type=float, default=1.5)
    p.add_argument("--option-weight", type=float, default=0.15)
    p.add_argument("--win-weight", type=float, default=1.5)
    p.add_argument("--loss-weight", type=float, default=0.4)
    p.add_argument("--draw-weight", type=float, default=0.8)
    p.add_argument("--value-weight", type=float, default=0.0)
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--legacy-state-pool", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--random-games", type=int, default=500)
    p.add_argument("--ladder-games", type=int, default=100)
    p.add_argument("--ladder-top", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=50)
    args = p.parse_args()

    plan = make_plan(args)
    write_plan(args.out, plan)
    write_script(args.script, plan, args)
    if args.eval_script:
        write_eval_script(args.eval_script, plan, args)
    print(f"Wrote {args.out}: {len(plan)} jobs")
    print(f"Wrote {args.script}")
    if args.eval_script:
        print(f"Wrote {args.eval_script}")
    for row in plan:
        print(
            f"  {row['job_name']}: {row['archetype']} {row['variant']} "
            f"decks={row['n_decks']} kept={row['kept']} share={float(row['share']):.1%} "
            f"sigs={row['deck_sigs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
