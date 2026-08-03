# Specialist BC Recipes

This document records deck-specific BC recipes that should not be replaced by
the generic population trainer without checking data distribution first.

## Deck-Sig vs Mixed Policy

Default rule: use high-quality deck signatures for any policy that may become a
Kaggle submission candidate. Mixed policies are acceptable for broad population
opponents only when the archetype has large, homogeneous data and local/Kaggle
evidence shows the mixed plan is not diluted.

Current classification:

| Archetype | Default | Rationale |
| --- | --- | --- |
| Teal Mask Ogerpon | High-quality deck-sig/top-k required | `ogerpon_top2_v7sig` reached about 963 Kaggle, while `ogerpon_v8_mixed` stayed around 662 with the same registry deck. This is direct evidence that mixed training diluted the plan. |
| Alakazam | High-quality deck-sig required | Earlier low random win rate was mostly policy/deck mismatch. Correct signature plus registry deck reached about 90% vs random; wrong deck stayed near 25-32%. Always evaluate with registry `--auto-deck`. |
| Mega Lopunny | High-quality deck-sig/top-k required | Top2 v9 reaches about 97% vs random on both known decks. It is still weak in core round-robin, so mixed training should only be used for opponent population, not submission. |
| Mega Lucario | Single high-quality deck-sig required | Data is tiny and the usable signature can move between score bands. Mixed training or wrong bands can produce `kept=0` or a weak policy. |
| Dragapult | High-quality deck-sig required, not ready | Current models are weak vs random/core. Treat as a specialist target with failure-report-driven reweighting, not mixed training. |
| Festival Lead | Prefer high-quality deck-sig/top1 | Top1 specialist reached about 98% vs random. Mixed is acceptable as a population opponent, but not preferred for a submission candidate. |
| Crustle Wall | Prefer high-quality deck-sig/top-k | Useful as an Ogerpon counter and stable in round-robin, but Kaggle score has been moderate. Keep both top-k and population variants for matchup testing. |
| Marnie Grimmsnarl | Mixed acceptable | It has the largest data volume and mixed/winweighted variants have been stable local baselines and around 850 Kaggle. Keep a deck-sig/top-k ablation, but mixed is not disallowed. |
| Cynthia Garchomp | Mixed or top1 acceptable | Data is narrower and the plan is relatively linear. Use top1 if submitting; mixed is fine for core population. |
| Team Rocket Mewtwo | Mixed acceptable for population only | It beats random strongly but has not translated to strong Kaggle score. Do not prioritize as a submission until round-robin improves. |
| Mega Starmie / Mega Abomasnow / Archaludon / Hop Trevenant / N's Zoroark | Do not train BC yet | Current replay data is too small or missing. Keep these as random ladder-pool opponents until enough high-quality signatures appear. |

Operational implications:

- Build registries from policy filenames and ladder manifests before every eval.
- Use `--registry ... --auto-deck` for random tests and round-robin whenever a
  checkpoint was trained with `--deck-sig`.
- If a deck-sig specialist has high offline accuracy but poor random win rate,
  first suspect policy/deck mismatch, then inspect failure reports.
- Mixed policies should be labeled as population baselines, not as primary
  daily candidates, unless they survive full round-robin and Kaggle trend checks.
- For new daily replay data, run corpus stats first and select top signatures by
  `(score band, episode count, win_dec, known team strength)`, not by raw count
  alone.

## Mega Lucario

Mega Lucario has too little high-band data for normal 1000+ population training.
In v9, the main usable deck moved into `1200+` because leaderboard scores changed.

Use this stats command before training:

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" "800-899" \
  --out logs/bc_corpus_stats_lucario_v9_800plus.csv
```

Known useful signature:

```text
43d6d8b0fce9  Majkel1337  kept=8135  avg_score=1279.6  win_dec=0.76
```

Do not train Lucario with only `1100-1199 900-999 800-899` on v9; that misses
the main signature and produces `kept=0`. `bc2_train.py` now fails fast for this.

Current best Lucario recipe:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1200+" "900-999" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --epochs 30 \
  --batch-size 1024 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --context-weight MAIN=1.6 \
  --context-weight TO_HAND=2.0 \
  --context-weight ATTACH_FROM=3.0 \
  --context-weight ATTACH_TO=2.5 \
  --type-weight ATTACK=1.8 \
  --type-weight ATTACH=2.5 \
  --type-weight PLAY=1.4 \
  --type-weight EVOLVE=1.6 \
  --checkpoint-every 1 \
  --save checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz \
  2>&1 | tee logs/train_mega_lucario_top1_v9_gameplan_w2.log
```

Wider model control:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1200+" "900-999" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --epochs 40 \
  --batch-size 512 \
  --width 3.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --context-weight MAIN=1.6 \
  --context-weight TO_HAND=2.0 \
  --context-weight ATTACH_FROM=3.0 \
  --context-weight ATTACH_TO=2.5 \
  --type-weight ATTACK=1.8 \
  --type-weight ATTACH=2.5 \
  --type-weight PLAY=1.4 \
  --type-weight EVOLVE=1.6 \
  --checkpoint-every 1 \
  --save checkpoints/bc2_mega_lucario_top1_v9_gameplan_w3.npz \
  2>&1 | tee logs/train_mega_lucario_top1_v9_gameplan_w3.log
```

Observed results:

```text
v8 winner-only baseline:       random 91.2%
v8 reweight:                   random 93.8%
v8 reweight2:                  random 94.2-94.8%
v9 game-plan width 2.0:         random 95.8%
```

Lucario is still below the preferred daily-pool threshold. Keep it as a
secondary candidate unless it reaches at least `97%` vs random and survives core
round-robin.

Evaluation:

```bash
python3 tools/eval_bc.py checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/43d6d8b0fce9_mega_lucario_majkel1337.csv \
  --games 500 --workers 8 --max-turns 700 --progress-every 50

python3 tools/bc2_failure_report.py checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1200+" "900-999" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --max-samples 50000 \
  --batch-size 4096 \
  --progress-every 5000 \
  --out-prefix logs/bc2_failure_mega_lucario_top1_v9_gameplan_w2
```

Main remaining issues from failure reports:

```text
MAIN / ATTACH_FROM / TO_HAND are still weak.
PLAY <-> ATTACK and ATTACK <-> PLAY confusion is the main policy-level issue.
Samples are limited: only about 6214 winner-only labels for the main signature.
```

## Mega Lopunny

Lopunny has enough v9 data, but mixed training can dilute the plan. Use
deck-signature specialists instead of the generic population trainer.

Stats command:

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lopunny" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --out logs/bc_corpus_stats_lopunny_v9_900plus.csv
```

Known useful signatures:

```text
b0cb21e29406  lmaffei     kept=78046  avg_score=1025.0
276707c0fdb4  Majkel1337  kept=71369  avg_score=1230.5
f1445356c3a7  Luca        kept=38976  avg_score=1103.0
```

Current best recipe is top2 winner-only:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lopunny" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b0cb21e29406 \
  --deck-sig 276707c0fdb4 \
  --winner-only \
  --epochs 20 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.40 \
  --multi-select-weight 1.8 \
  --context-weight MAIN=1.8 \
  --context-weight TO_HAND=2.0 \
  --context-weight DISCARD=3.0 \
  --context-weight TO_BENCH=2.0 \
  --context-weight TO_ACTIVE=1.5 \
  --type-weight ATTACK=1.8 \
  --type-weight PLAY=1.6 \
  --type-weight EVOLVE=1.8 \
  --type-weight ATTACH=2.0 \
  --type-weight RETREAT=1.8 \
  --checkpoint-every 1 \
  --save checkpoints/bc2_mega_lopunny_top2_v9_gameplan_w2.npz \
  2>&1 | tee logs/train_mega_lopunny_top2_v9_gameplan_w2.log
```

Top3 control:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lopunny" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b0cb21e29406 \
  --deck-sig 276707c0fdb4 \
  --deck-sig f1445356c3a7 \
  --epochs 20 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.6 \
  --context-weight MAIN=1.6 \
  --context-weight TO_HAND=1.8 \
  --context-weight DISCARD=2.5 \
  --context-weight TO_BENCH=1.8 \
  --type-weight ATTACK=1.6 \
  --type-weight PLAY=1.5 \
  --type-weight EVOLVE=1.6 \
  --type-weight ATTACH=1.8 \
  --type-weight RETREAT=1.5 \
  --win-weight 1.8 \
  --loss-weight 0.25 \
  --draw-weight 0.8 \
  --checkpoint-every 1 \
  --save checkpoints/bc2_mega_lopunny_top3_v9_gameplan_w2.npz \
  2>&1 | tee logs/train_mega_lopunny_top3_v9_gameplan_w2.log
```

Observed random results:

```text
top2 on b0cb21e29406 deck: 97.0%
top2 on 276707c0fdb4 deck: 97.2%
top3 on b0cb21e29406 deck: 96.8%
top3 on 276707c0fdb4 deck: 96.2%
```

Top2 is better than top3. However, core round-robin shows it is not a good
Kaggle submission candidate yet:

```text
lopunny_v9 avg_no_random=0.269
vs ogerpon_v7 0.230
vs marnie_v8  0.085
vs cynthia_v8 0.085
```

Keep Lopunny in the training population, but do not spend Kaggle submissions on
it until it improves against Marnie/Ogerpon/Cynthia.
