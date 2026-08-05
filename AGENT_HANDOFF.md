# Agent Handoff

Last updated: 2026-08-05 09:50 Asia/Shanghai.

This file is the first place a new agent should read before touching the project. Keep it updated whenever the pipeline changes, a Kaggle submission is made, a long remote job is started/stopped, or the interpretation of current results changes. After updating it locally, sync it to the `ks` workspace and commit the change.

## Workspaces

- Local repo: `/home/jie/Do/0_PTCG/bak/ptcg_rl_git`
- Local branch: `v7-baseline-20260804`
- Remote training repo: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`
- Remote original workspace also exists in older notes: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git`
- Current remote training data: `data/bc_corpus_banded_v10_all_0803`
- Current remote raw episodes: `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git/episodes_raw` and adjacent workspace-level episode dirs may both exist; verify before launching extraction.

For long ad-hoc checks over SSH, first build a script under local `/tmp`, upload it to `ks:/tmp`, then execute it. Avoid large heredocs inside `ssh` commands; quoting already caused noisy failures.

## Current Git State

Recent relevant commits and current update:

- Latest update in this session:
  - `tools/rl_finetune_vs_pool.py`: new BC2-initialized PPO fine-tune loop against fixed `NumpyPolicy`/random opponent pools.
  - `tools/summarize_matchup_failures.py`: aggregate `trace_matchup_decisions.py` summaries into loss-vs-win failure priorities.
  - `docs/05_rl_training.md`: replaced legacy `train.py` PPO notes with the new shadow/failure-pool fine-tune workflow.
  - `README.md`: added RL fine-tune and trace summary usage; removed stale `--workers` from the trace command example.
- Previous local change: add `tools/eval_manifest_random.py` for batch random evaluation of manifest/shadow policies.
- `00f3a0b` Handle duplicate shadow manifest entries
- `c0e0366` Add agent handoff notes
- `1d421ad` Document current BC pipeline notes
- `edc0113` Add matchup-aware BC data pipeline features
- `950f7fa` Add matchup decision trace diagnostic

Remote smoke tests completed on `ks`:

- `/tmp/ptcg_rl_dry_run_0805.sh`: passed. It compiled the new tools, ran trace-summary aggregation, and loaded Marnie `state=64`/`option=48` checkpoint plus 1 explicit Ogerpon and 3 Ogerpon shadow opponents.
- `/tmp/ptcg_rl_tiny_rollout_0805.sh`: passed after fixing the single-opponent loader. It ran a 2-game CPU rollout/update against random Ogerpon deck, collected 106 decisions, completed PPO update, and wrote `/tmp/ptcg_rl_tiny_out.npz`.

These files were synced to `ks`. Check `git status` before starting the next change.

## Kaggle Accounts And Monitoring

Local Kaggle configs under `~/.kaggle`:

- `kaggle.json`
- `kaggle.json.jie`
- `kaggle.json.by`

Do not print token contents.

Remote `ks` configs:

- `/root/.kaggle/kaggle.json`: original `jie` account, left unchanged.
- `/root/.kaggle/by/kaggle.json`: uploaded `by` account, mode `600`.

Remote score monitoring currently running:

- `jie`: `python3 -u tools/track_kaggle_scores.py --watch --interval 60 --out logs/kaggle_submission_scores.csv`
- `by`: `KAGGLE_CONFIG_DIR=/root/.kaggle/by python3 -u tools/track_kaggle_scores.py --watch --interval 60 --out logs/kaggle_submission_scores_by.csv`
- `by` log: `logs/track_kaggle_scores_by.log`

Latest observed `by` score-monitor rows at `2026-08-04T22:57:08+00:00`:

| Submission | Description | Score |
| --- | --- | ---: |
| `55241705` | `bc: team_rocket_mewtwo_v10pop` | 715.1 |
| `55241683` | `bc: crustle_wall_v10pop` | 880.1 |
| `55234504` | `bc: ogerpon_v10_fixed_top2` | 956.3 |
| `55234481` | `bc: ogerpon_v10_fixed_top3` | 718.5 |

User noted `55234504` peaked around 1040 even though not all score changes are recorded.

## Current Shadow Training

Shadow manifest:

```text
logs/shadow_pool_manifest_v10_all0803_popinit_set.csv
```

Completed command on `ks`:

```bash
python3 -u tools/train_shadow_manifest.py \
  logs/shadow_pool_manifest_v10_all0803_popinit_set.csv \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --batch-size 1024 \
  --cuda-memory-gb 8 \
  --skip-existing \
  --log-dir logs/v10_pipeline/shadow_v10pop_set_mem8g
```

Status snapshot at 2026-08-04 23:37 Asia/Shanghai:

```text
manifest_rows=361
done=361
running=0
failed=0
pending=0
started_no_final=0
npz_all=2457
npz_final=351
```

Per-archetype snapshot:

| Archetype | Done / Total | Pending | Running |
| --- | ---: | ---: | ---: |
| Marnie Grimmsnarl | 151 / 151 | 0 | 0 |
| Alakazam | 56 / 56 | 0 | 0 |
| Crustle Wall | 38 / 38 | 0 | 0 |
| Team Rocket Mewtwo | 24 / 24 | 0 | 0 |
| Dragapult | 22 / 22 | 0 | 0 |
| Festival Lead | 18 / 18 | 0 | 0 |
| Mega Lopunny | 17 / 17 | 0 | 0 |
| Teal Mask Ogerpon | 15 / 15 | 0 | 0 |
| Cynthia Garchomp | 14 / 14 | 0 | 0 |
| Mega Lucario | 4 / 4 | 0 | 0 |
| Mega Starmie | 2 / 2 | 0 | 0 |

Running shadow jobs at the snapshot: none.

There were duplicate `shadow_*_unknown` names in the original manifest. Fixes synced to `ks` at 2026-08-04 23:37:

- `tools/eval_baseline_delta.py` skips exact duplicate manifest entries and suffixes same-name/different-entry opponents.
- `tools/train_shadow_manifest.py` skips duplicate checkpoint paths when reading a manifest for future runs.
- `tools/build_shadow_pool.py` suffixes shadow names on safe-name collisions for future manifests.

The completed v10 run used the older running process, so some duplicate checkpoint work was already done. For evaluation this is now handled by `eval_baseline_delta.py`; top120 manifest read check passed with `entries=120 unique_names=120 duplicate_names=0 duplicate_specs=0`.

Use the uploaded helper to re-check:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && python3 /tmp/check_shadow_status.py'
```

If `/tmp/check_shadow_status.py` is missing, rebuild it locally under `/tmp` and upload it. Do not rely only on `pgrep -af bc2_train.py`; it can miss jobs or include unrelated `bc2_train.py` runs.

Observed unrelated remote CPU job:

- There is a separate `/home/byer/PTCG/.../v11_crustle_refresh_r1` CPU `bc2_train.py` process visible in broad process scans.
- Treat it as unrelated unless the user explicitly asks to inspect/stop it.

## Data And Feature Compatibility

0803 v10 corpus:

- Example path: `data/bc_corpus_banded_v10_all_0803`
- Feature dims: `state=64`, `option=48`
- Existing v10 population/shadow checkpoints are 64/48.

Current encoder after `edc0113`:

- `STATE_FEAT_DIM = 80`
- `OPT_FEAT_DIM = 64`
- Added features are appended, so old checkpoints can still evaluate through `NumpyPolicy` truncation.

Training rules:

- Old v10 corpus without `--init`: pass `--state-feat-dim 64 --opt-feat-dim 48`.
- Exact `--init` from an old checkpoint auto-infers old dims unless `--init-partial` is set.
- v11 80/64 training initialized from v10 must use `--init-partial --state-feat-dim 80 --opt-feat-dim 64`.
- Opponent filters only work meaningfully on newly extracted corpus containing `opponent_*` metadata.
- Do not overwrite v10 corpus with v11 extraction.

## Recent Kaggle And Local Results

Ogerpon:

- `55234504`: `bc: ogerpon_v10_fixed_top2`, final observed 956.3/951.8 depending snapshot, peak around 1040. This is the main restored v7-level candidate.
- `55234481`: `bc: ogerpon_v10_fixed_top3`, around 718.5/720.5. Adding `5899c772bace` diluted the policy; do not prefer top3.

Probe submissions on `by` account:

- `55241683`: Crustle Wall v10pop, current monitored score around 880.1; earlier observed around 920.
- `55241705`: Team Rocket Mewtwo v10pop, current monitored score around 715.1; earlier observed around 635.

Local random 500:

- Alakazam: 497/500 = 99.4%
- Crustle Wall: 467/500 = 93.4%
- Marnie Grimmsnarl: 500/500 = 100%
- Team Rocket Mewtwo: 500/500 = 100%

Local category round-robin g200 vs core categories:

- Crustle Wall: avg 0.697, min 0.380 vs Marnie, losing 2/10
- Marnie Grimmsnarl: avg 0.689, min 0.135 vs Ogerpon fixed top2, losing 1/10
- Alakazam: avg 0.613, min 0.305 vs Crustle, losing 4/10
- Team Rocket Mewtwo: avg 0.505, min 0.100 vs Mega Lopunny, losing 3/10

Important conclusion: random is not enough. Use random only as a sanity check, then run category round-robin, ladder/failure pool evaluation, and matchup trace.

## 0804 Ladder Audit

The 0804 raw episodes are present on `ks` as:

```text
/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-04.zip
```

It contains 4,811 episode JSON files plus one CSV, about 743 MB. A 0804-only ladder pool was built without overwriting older pools:

```text
logs/ladder_pool_0804_all/pool_manifest.csv
logs/ladder_pool_0804_all/archetype_stats.csv
logs/ladder_pool_0804_all/decks/
logs/build_ladder_pool_0804_all.log
```

Build command used a temporary 0804-only symlink dir because `/home/jie/Do/0_PTCG/workspace/episodes_raw` contains 12 zip files:

```bash
mkdir -p /tmp/ptcg_episodes_0804_only
ln -sf /home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-04.zip \
  /tmp/ptcg_episodes_0804_only/pokemon-tcg-ai-battle-episodes-2026-08-04.zip
python3 -u tools/build_ladder_pool.py \
  --episodes-dir /tmp/ptcg_episodes_0804_only \
  --out logs/ladder_pool_0804_all \
  --top 0 \
  --min-games 1 \
  --progress-every 500 \
  2>&1 | tee logs/build_ladder_pool_0804_all.log
```

0804 pool summary:

```text
unique deck sigs: 112
deck-side games: 19,244
total weight: 36,097.3
score bands by games:
  1200+:      3,510 games, 4 decks
  1100-1199: 9,056 games, 16 decks
  1000-1099: 5,552 games, 30 decks
```

Top archetypes by 0804 weight:

| Archetype | Decks | Games | Weight | Max Score |
| --- | ---: | ---: | ---: | ---: |
| Marnie Grimmsnarl | 9 | 5,952 | 10,641.2 | 1,175.7 |
| Mega Lopunny | 5 | 2,860 | 6,233.7 | 1,205.5 |
| Teal Mask Ogerpon | 17 | 2,286 | 4,111.2 | 1,273.8 |
| Alakazam | 18 | 2,884 | 3,452.5 | 1,088.5 |
| Dragapult | 13 | 1,214 | 2,803.5 | 1,116.7 |
| Crustle Wall | 15 | 1,194 | 2,551.6 | 1,156.9 |
| Mega Lucario | 7 | 582 | 1,996.2 | 1,273.8 |
| Cynthia Garchomp | 3 | 882 | 1,548.5 | 1,115.7 |
| Festival Lead | 6 | 654 | 1,277.2 | 1,108.5 |
| Team Rocket Mewtwo | 9 | 264 | 584.4 | 1,156.9 |

Top 0804 deck signatures:

| Sig | Archetype | Games | Weight | Score | Team |
| --- | --- | ---: | ---: | ---: | --- |
| `b8f251a476e7` | Marnie Grimmsnarl | 5,074 | 9,336.3 | 1,175.7 | Raihan Ramadistra |
| `7f9a538936e3` | Alakazam | 2,588 | 2,989.3 | 1,084.9 | M Sato |
| `f1445356c3a7` | Mega Lopunny | 1,144 | 2,970.4 | 1,205.5 | ntumlnoob |
| `276707c0fdb4` | Mega Lopunny | 1,274 | 2,313.3 | 1,205.5 | www... |
| `43d6d8b0fce9` | Mega Lucario | 478 | 1,890.1 | 1,273.8 | Majkel1337 |
| `697a82e582d5` | Teal Mask Ogerpon | 614 | 1,483.9 | 1,273.8 | Majkel1337 |
| `52f467394857` | Cynthia Garchomp | 716 | 1,314.5 | 1,115.7 | Octavi Grau |
| `7ac0181b46a0` | Dragapult | 330 | 990.0 | 1,116.7 | LumenLiquidity |
| `5899c772bace` | Teal Mask Ogerpon | 518 | 958.3 | 1,133.8 | keidroid |
| `e82dcbe62260` | Festival Lead | 440 | 916.4 | 1,108.5 | __Taichicchi__ |
| `2c22fa761816` | Marnie Grimmsnarl | 640 | 885.2 | 1,108.5 | KawattaTaido |
| `47756cdfd20f` | Crustle Wall | 328 | 704.8 | 1,156.9 | M Sato |
| `3cd5039c59d2` | Crustle Wall | 246 | 692.4 | 1,120.0 | Oshbocker |
| `f0bac971c56d` | Team Rocket Mewtwo | 122 | 366.0 | 1,156.9 | flg |

Compared with `logs/ladder_pool_0802_all/pool_manifest.csv`:

```text
0804 rows: 112
0802 rows: 82
common sigs: 50
new-only sigs: 62
```

Important new/high-weight 0804-only signatures include `7ac0181b46a0` Dragapult, `1784e485688c` Ogerpon, `46ceec8cc5ae` Dragapult, `f63436c208ad` Mega Starmie, `0e532395fc46` Ogerpon, and `45eb2708a6d1` Crustle.

Interpretation:

- 0804 has enough high-quality data to justify v11 extraction immediately.
- Do not run a broad blind v11 training wave first. The 0804 ladder shifted materially: Mega Lopunny, Mega Lucario, Dragapult, Ogerpon, and Marnie all need explicit deck-sig/matchup treatment.
- Best next order: extract v11 from 0804 (and optionally 0803+0804 combined under a separate output), then run targeted recipe training for top signatures and matchup-conditioned BC. Use trace/RR to decide which archetypes get RL fine-tune.

## v11 Extraction Status

User completed steps 2-5 from the v11 workflow. Remote audit at 2026-08-05 09:39 Asia/Shanghai:

- `logs/build_ladder_pool_0804_all.log`: complete, 4,811 0804 episodes, 112 deck sigs, outputs under `logs/ladder_pool_0804_all/`.
- `logs/extract_v11_0804_only.log`: complete, 763,202 decisions, `bad=0`, `err=0`, 54 `.npz` files.
- `logs/extract_v11_0803_0804.log`: complete for both zip files, 111 `.npz` files. Log lines are interleaved because `--workers 2` processed 0803 and 0804 concurrently.
- User manually verified `data/bc_corpus_banded_v11_0804_only`: `state feat=(80,)`, `option feat=(2, 64)` in sample, `feature_version=v11_matchup_mechanic`, and opponent metadata keys are present.

Corpus summary:

| Corpus | Files | Decisions | Dims | Feature Version |
| --- | ---: | ---: | --- | --- |
| `data/bc_corpus_banded_v11_0804_only` | 54 | 763,202 | state 80 / option 64 | `v11_matchup_mechanic` |
| `data/bc_corpus_banded_v11_0803_0804` | 111 | ~1,528,400 | state 80 / option 64 | `v11_matchup_mechanic` |

0804-only top training bands/decks from stats:

| Archetype | Top Sig | Kept In Top Bands | Notes |
| --- | --- | ---: | --- |
| Marnie Grimmsnarl | `b8f251a476e7` | 140,101 | huge dominant sig; top3 bands total kept 162,050 |
| Mega Lopunny | `f1445356c3a7` | 31,232 | `276707c0fdb4` is also important |
| Alakazam | `7f9a538936e3` | 29,175 | strong data volume but Kaggle shadow probe was volatile |
| Dragapult | `cc2e995b5ad0` | 20,227 | also include 0804-new `7ac0181b46a0` later |
| Teal Mask Ogerpon | `697a82e582d5` | 16,112 | `5899c772bace` and `2a5072194fdf` remain relevant |
| Mega Lucario | `43d6d8b0fce9` | 14,498 | clean high-score single sig |
| Festival Lead | `e82dcbe62260` | 12,969 | enough for specialist |
| Cynthia Garchomp | `52f467394857` | 15,398 | enough for specialist |
| Crustle Wall | `47756cdfd20f` | 7,428 | top bands are thinner; include multiple sigs |
| Team Rocket Mewtwo | `f0bac971c56d` | 4,302 | thin; treat as probe/specialist |

Recommended next:

1. Generate v11 shadow/specialist manifest from `data/bc_corpus_banded_v11_0804_only`, but use `--min-decisions 2000` rather than 3000 if Team Rocket / thinner specialists should be included.
2. Train v11 shadows from that manifest.
3. For broad population baseline, use `data/bc_corpus_banded_v11_0803_0804`; for high-reactivity ladder specialists, use `v11_0804_only`.

## v11 Shadow 0804 Status

User completed:

```text
logs/build_shadow_pool_v11_0804_set.log
logs/v11_pipeline/shadow_0804_set_mem8g/*.log
```

Note: expected runner path `logs/v11_pipeline/train_shadow_v11_0804_set_mem8g.runner.log` was not present during audit; only per-job logs existed under `logs/v11_pipeline/shadow_0804_set_mem8g/`.

Manifest:

```text
logs/shadow_pool_manifest_v11_0804_set.csv
rows=50
missing_deck_path=0
missing_eval_entry=0
```

Manifest rows by archetype:

| Archetype | Rows | Sigs |
| --- | ---: | ---: |
| Mega Lopunny | 10 | 4 |
| Teal Mask Ogerpon | 9 | 5 |
| Marnie Grimmsnarl | 7 | 3 |
| Dragapult | 6 | 5 |
| Festival Lead | 5 | 2 |
| Crustle Wall | 5 | 4 |
| Alakazam | 4 | 2 |
| Cynthia Garchomp | 2 | 1 |
| Mega Lucario | 1 | 1 |
| Team Rocket Mewtwo | 1 | 1 |

Training status at audit:

```text
checkpoint_present=43
checkpoint_missing=7
bad_logs=7
sample checkpoint dims=(80, 64)
```

All failures were CUDA OOM under the 8 GB per-process cap, mainly because the first run used batch size 2048 and likely two jobs per GPU. Missing checkpoints:

```text
rank 003 shadow_mega_lopunny_276707c0_unknown
rank 007 shadow_dragapult_6763881e_third_ptcg_club
rank 009 shadow_alakazam_7f9a5389_m_sato
rank 027 shadow_alakazam_7f9a5389_team_kasa
rank 045 shadow_alakazam_7f9a5389_northstar
rank 046 shadow_mega_lopunny_b0cb21e2_insuperabilehart
rank 048 shadow_alakazam_d791eb8b_goonew
```

Important implication: do not start the main random audit yet if Alakazam quality matters; all Alakazam v11 shadows are currently missing. Rerun missing jobs with `--skip-existing`, `--jobs-per-gpu 1`, and `--batch-size 1024` first:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u tools/train_shadow_manifest.py \
  logs/shadow_pool_manifest_v11_0804_set.csv \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --batch-size 1024 \
  --cuda-memory-gb 8 \
  --skip-existing \
  --log-dir logs/v11_pipeline/shadow_0804_set_mem8g_retry1024 \
  2>&1 | tee logs/v11_pipeline/train_shadow_v11_0804_set_retry1024.runner.log
```

If any still OOM, retry the remaining jobs with `--batch-size 512`; keep `--skip-existing`.

Shadow top120 baseline-delta eval:

```text
logs/eval_shadow_v10/probes_vs_ogerpon_shadow_top120_g80.csv
```

Summary vs Ogerpon fixed top2 baseline over 120 shadow opponents, 80 games each:

| Candidate | Candidate Avg | Baseline Avg | Avg Delta | Weighted Delta | Lost Delta | Candidate WR < 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crustle Wall | 0.595 | 0.719 | -0.125 | -0.141 | 78 / 120 | 50 / 120 |
| Marnie Grimmsnarl | 0.636 | 0.719 | -0.084 | -0.093 | 82 / 120 | 16 / 120 |
| Team Rocket Mewtwo | 0.455 | 0.719 | -0.264 | -0.272 | 97 / 120 | 84 / 120 |

Interpretation:

- This shadow pool is not trivial: Ogerpon baseline averages only 0.719 and is nearly blanked by Crustle shadows, weak into Alakazam/Mega Lopunny, but strong into Marnie/TR.
- None of the three probes beats Ogerpon overall on this Marnie-heavy top120 pool.
- Marnie is the best of the three by average, but loses delta on 82/120 and is weak into Ogerpon shadows and Marnie mirrors.
- Crustle strongly beats Crustle/Alakazam/Ogerpon shadows but is crushed by Marnie shadows; this explains why it can show useful Kaggle score yet fail broad local ladder pressure.
- Team Rocket Mewtwo is clearly not ready as a broad candidate; high random does not translate here.
- Top120 contains 67 Marnie shadows, so treat it as a stress pool. Later build balanced-per-archetype and hard-pool views rather than relying on one aggregate.

Shadow all random 500 quality audit:

```text
logs/eval_shadow_v10/shadow_all_random_g500.csv
```

This CSV has 298 unique evaluated shadow entries, not 361 manifest rows, because duplicate `eval_entry` rows are skipped by the fixed manifest reader.

Overall:

```text
n=298 mean=0.977 median=0.994 p25=0.984 p10=0.941 min=0.350
100%=42  >=99%=199  >=97%=252  <95%=35  <90%=20
timeouts=294
```

By archetype:

| Archetype | n | mean | median | min | >=99 | <95 | timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Marnie Grimmsnarl | 136 | 0.996 | 0.996 | 0.984 | 132 | 0 | 0 |
| Alakazam | 49 | 0.988 | 0.990 | 0.966 | 26 | 0 | 0 |
| Crustle Wall | 34 | 0.929 | 0.943 | 0.564 | 1 | 21 | 0 |
| Mega Lopunny | 15 | 0.987 | 0.988 | 0.972 | 7 | 0 | 0 |
| Cynthia Garchomp | 14 | 0.982 | 0.984 | 0.962 | 3 | 0 | 0 |
| Festival Lead | 14 | 0.988 | 0.998 | 0.894 | 12 | 1 | 0 |
| Teal Mask Ogerpon | 14 | 0.917 | 0.991 | 0.350 | 9 | 3 | 294 |
| Dragapult | 10 | 0.873 | 0.873 | 0.780 | 0 | 8 | 0 |
| Team Rocket Mewtwo | 10 | 0.995 | 0.997 | 0.982 | 9 | 0 | 0 |
| Mega Lucario | 2 | 0.851 | 0.851 | 0.846 | 0 | 2 | 0 |

Interpretation:

- Random is a quality gate, not a ladder predictor. High random does not make TR Mewtwo good; shadow top120 already showed TR Mewtwo fails broader pressure.
- Random below 100% can be acceptable, but below 99% is a yellow flag for submission candidates unless a matchup trace shows real strategic tradeoff rather than missed attacks/setup.
- Marnie shadows look extremely clean by random. If selecting a shadow submission probe, start with high-weight Marnie variants, then prove them in shadow/balanced RR.
- Crustle/Dragapult/Mega Lucario shadow quality is much less reliable by random; use them as opponent diversity, but do not submit without additional RR/trace evidence.
- Ogerpon random rows with timeouts need separate timeout/trace inspection before treating their WR as real.

## Diagnostic Artifacts

Local/remote logs worth checking:

- `logs/eval_v10/category_rr_v10pop_20260804/random_summary.csv`
- `logs/eval_v10/category_rr_v10pop_20260804/candidate_summary.csv`
- `logs/eval_v10/category_rr_v10pop_20260804/round_robin_candidate_rows.csv`
- `logs/eval_v10/category_rr_v10pop_20260804/failure_traces_g100/`
- `logs/eval_v10/category_rr_v10pop_20260804/failure_trace_diagnostics.csv`
- `logs/eval_v10/category_rr_v10pop_20260804/failure_trace_setup_choices.csv`
- `logs/eval_v10/category_rr_v10pop_20260804/bc_failure/`
- `logs/eval_v10/category_rr_v10pop_20260804/bc_failure_digest.csv`
- `logs/eval_shadow_v10/probes_vs_ogerpon_shadow_top120_g80.csv`
- `logs/eval_shadow_v10/shadow_all_random_g500.csv`

Key diagnostic conclusions:

- Marnie: overall imitation is high, but `ATTACH_TO` and setup quality are weak against Ogerpon.
- Team Rocket Mewtwo: needs Team Rocket in-play count, Mewtwo Power Saver readiness, and setup priority.
- Alakazam: random is misleading; broad RR weakness comes from fewer attacks/evolves/abilities in losses and bad long matchup into Crustle.
- Crustle: best of the four probes locally, but still loses into Marnie/TR and random is only 93.4%.
- Failures are not mainly illegal moves or early END.

Useful tool:

```bash
python3 tools/trace_matchup_decisions.py \
  --candidate marnie=checkpoints/pop/bc2_marnie_grimmsnarl_v10pop_all0803_set_w2.npz:logs/ladder_pool_0802_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --opponent ogerpon=checkpoints/v10/bc2_ogerpon_v10_fixed_top2_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --games 100 \
  --max-turns 700 \
  --out-prefix logs/eval_v10/marnie_vs_ogerpon_trace_g100
```

Batch trace-summary priority ranking:

```bash
python3 tools/summarize_matchup_failures.py \
  "logs/eval_v10/category_rr_v10pop_20260804/failure_traces_g100/*.summary.csv" \
  --min-loss-decisions 20 \
  --out-csv logs/eval_v10/category_rr_v10pop_20260804/failure_trace_priority.csv
```

Batch random evaluation for manifest/shadow policies:

```bash
python3 tools/eval_manifest_random.py \
  --manifest logs/shadow_pool_manifest_v10_all0803_popinit_set.csv \
  --limit 120 \
  --games 200 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50 \
  --skip-bad-entries \
  --resume \
  --out-csv logs/eval_shadow_v10/shadow_top120_random_g200.csv \
  2>&1 | tee logs/eval_shadow_v10/shadow_top120_random_g200.log
```

Use `--games 500` and a different output name for the 500-game version. The script appends each policy result immediately, so `--resume` is safe after interruption.

New targeted RL infrastructure:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/rl_finetune_vs_pool.py \
  --policy-init checkpoints/pop/bc2_marnie_grimmsnarl_v10pop_all0803_set_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --opponent ogerpon_top2=checkpoints/v10/bc2_ogerpon_v10_fixed_top2_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --opponent-manifest logs/shadow_pool_manifest_v10_all0803_popinit_set.csv \
  --manifest-archetype-regex "Teal Mask Ogerpon" \
  --skip-bad-entries \
  --iterations 12 \
  --games-per-iter 64 \
  --ppo-epochs 3 \
  --minibatch 256 \
  --lr 3e-5 \
  --clip-eps 0.1 \
  --entropy-coef 0.003 \
  --bc-anchor-weight 0.15 \
  --bc-anchor-corpus data/bc_corpus_banded_v11_matchup_0803 \
  --bc-anchor-archetype "Marnie Grimmsnarl" \
  --bc-anchor-deck-sig b8f251a476e7 \
  --bc-anchor-opponent-archetype "Teal Mask Ogerpon" \
  --bc-anchor-batch-size 512 \
  --device cuda:0 \
  --cuda-memory-gb 8 \
  --max-turns 700 \
  --checkpoint-dir checkpoints/rl/marnie_vs_ogerpon_pilot \
  --metrics-csv logs/rl/marnie_vs_ogerpon_pilot_metrics.csv \
  --save checkpoints/rl/bc2_marnie_vs_ogerpon_rl_pilot_w2.npz \
  2>&1 | tee logs/rl/marnie_vs_ogerpon_pilot.log
```

Status and constraints:

- `rl_finetune_vs_pool.py` compiled locally and passed remote dry-run plus 2-game CPU rollout/update smoke test.
- It auto-infers BC2 width and feature dimensions from `--policy-init`, so v10 64/48 checkpoints remain compatible.
- Rollout is currently single-process and should be used for small targeted pilots first.
- Use BC anchor whenever v11 matchup corpus exists; without anchor, only run tiny sanity checks because PPO can overfit a fixed opponent pool quickly.
- Acceptance gate is external evaluation: random, core RR, balanced shadow/failure pool, and trace. Do not accept based on training-pool WR alone.

## Next Work

Immediate:

1. Build balanced-per-archetype and hard-pool views from the completed shadow pool; top120 is useful but Marnie-heavy.
2. For each target archetype, select weak matchups from RR/baseline-delta, run `trace_matchup_decisions.py`, and aggregate with `summarize_matchup_failures.py`.
3. Decide per weakness whether to use matchup-conditioned BC, rule overlay, deck-sig shadow/specialist, or RL fine-tune.
4. Run the first real RL pilot only after choosing a curated opponent pool and, preferably, after v11 matchup corpus exists for BC anchoring.
5. Deep-dive random losses/timeouts for Ogerpon/Crustle/Dragapult/Mega Lucario before using them as submission candidates or high-trust RL opponents.

Pipeline direction:

- Improve BC pipeline and shadow training first, not just select a currently strongest deck.
- Add matchup-conditioned data selection using new `opponent_*` metadata after v11 re-extraction.
- Consider deck-sig shadow policies, but evaluate quality per archetype and per matchup.
- For Marnie/Ogerpon, investigate setup and `ATTACH_TO` decisions rather than generic accuracy.
- For Team Rocket Mewtwo, test whether new 80/64 features improve setup and Power Saver decisions.

Kaggle:

- Daily submission limit matters. On 2026-08-04 the user has used submissions across two accounts; verify current limits before submitting.
- Do not submit Ogerpon top3 again unless there is a concrete fix.
- `Crustle Wall v10pop` is currently a useful probe around 920, not proof of final strength.

Remote operations:

- Avoid killing LLaMA Factory or `/home/byer/PTCG/...` jobs unless the user explicitly asks.
- Keep CUDA memory caps around `--cuda-memory-gb 8` while shared GPU jobs are active.
- If starting more long-running jobs, write logs under a distinct `logs/...` directory and record command/status here.

## Update Checklist

Whenever changing active state, update:

- Date/time at top of this file.
- Active remote jobs and output paths.
- Kaggle account/score monitor status.
- Latest submission IDs and observed scores.
- New corpus/checkpoint feature dimensions.
- New conclusions from round-robin, Kaggle replay, or failure traces.
- Any commands that the next agent should continue or avoid.
