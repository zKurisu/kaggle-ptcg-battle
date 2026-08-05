# Agent Handoff

Last updated: 2026-08-05 13:42 Asia/Shanghai.

This file is the first place a new agent should read before touching the project. Keep it updated whenever the pipeline changes, a Kaggle submission is made, a long remote job is started/stopped, or the interpretation of current results changes. After updating it locally, sync it to the `ks` workspace and commit the change.

## Workspaces

- Local repo: `/home/jie/Do/0_PTCG/bak/ptcg_rl_git`
- Local branch: `v7-baseline-20260804`
- Remote training repo: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`
- Remote original workspace also exists in older notes: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git`
- Current remote training data: `data/bc_corpus_banded_v10_all_0803`
- Current remote raw episodes: `/home/jie/Do/0_PTCG/workspace/episodes_raw`. Older notes may mention `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git/episodes_raw`; verify the intended path before launching extraction.

For long ad-hoc checks over SSH, first build a script under local `/tmp`, upload it to `ks:/tmp`, then execute it. Avoid large heredocs inside `ssh` commands; quoting already caused noisy failures.

## Episode Backfill

Remote raw zip status at 2026-08-05 13:42 Asia/Shanghai:

- Existing before the current backfill: `pokemon-tcg-ai-battle-episodes-2026-07-23.zip` through `2026-08-04.zip` under `/home/jie/Do/0_PTCG/workspace/episodes_raw`.
- User verified that Kaggle allows downloading earlier datasets such as `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-23`.
- Current backfill target is `2026-07-01` through `2026-07-22` into `/home/jie/Do/0_PTCG/workspace/episodes_raw`.
- The slow remote Kaggle download was stopped by request. No `ptcg_download_july_episodes` or `kaggle datasets download` process remained in the 13:42 check.
- Previously observed remote background parent PID `582836` and script PID `582838` are obsolete.
- Script path on `ks`: `/tmp/ptcg_download_july_episodes.sh`.
- Log: `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/download_july_episodes_20260701_20260722.log`.
- Status CSV: `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/download_july_episodes_20260701_20260722_status.csv`.
- At the 13:42 check, remote target dir had valid zips for `2026-07-01` through `2026-07-09`, `2026-07-11`, and `2026-07-23` through `2026-07-31`; `2026-07-09` and `2026-07-11` both passed `python3 -m zipfile -t`.
- Remaining missing July dates: `2026-07-10`, `2026-07-12`, `2026-07-13`, `2026-07-14`, `2026-07-15`, `2026-07-16`, `2026-07-17`, `2026-07-18`, `2026-07-19`, `2026-07-20`, `2026-07-21`, `2026-07-22`.
- User requested future backfill use local Kaggle download then upload to `ks`, because remote Kaggle download is too slow. Local `/tmp` is small, so use one-day or bounded-parallel caching and delete local zips after upload/remote validation.

Monitor with:

```bash
ssh ks 'tail -f /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/download_july_episodes_20260701_20260722.log'
ssh ks 'tail -30 /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/download_july_episodes_20260701_20260722_status.csv'
ssh ks 'pgrep -af "ptcg_download_july_episodes|kaggle datasets download" || true'
```

## Current Git State

Recent relevant commits and current update:

- Latest update in this session:
  - `tools/eval_round_robin.py`: now supports `--manifest`, `--manifest-limit`, and `--manifest-random`. It reads CSVs with `eval_entry` or `checkpoint_path`/`deck_path`, skips exact duplicate entries, and suffixes duplicate names.
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

Initial training status before retry:

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

Retry completed with `--batch-size 1024`; final audit:

```text
checkpoint_present=50
checkpoint_missing=0
Alakazam done=4 missing=0
all manifest archetypes done
```

Retry logs:

```text
logs/v11_pipeline/train_shadow_v11_0804_set_retry1024.runner.log
logs/v11_pipeline/shadow_0804_set_mem8g_retry1024/
```

The original `logs/v11_pipeline/shadow_0804_set_mem8g/*.log` still contains the seven stale OOM logs. Do not interpret those as final failures unless the checkpoint is missing.

Shadow random audit completed:

```bash
python3 tools/eval_manifest_random.py \
  --manifest logs/shadow_pool_manifest_v11_0804_set.csv \
  --games 500 \
  --workers 24 \
  --max-turns 700 \
  --progress-every 50 \
  --skip-bad-entries \
  --resume \
  --out-csv logs/eval_shadow_v11/shadow_v11_0804_random_g500.csv \
  2>&1 | tee logs/eval_shadow_v11/shadow_v11_0804_random_g500.log
```

Result:

```text
logs/eval_shadow_v11/shadow_v11_0804_random_g500.csv
n=50 mean=0.667 median=0.669 min=0.130 max=0.992
>=0.99: 1
>=0.97: 5
<0.95: 42
<0.90: 40
deck_sig_ok=50/50
manifest_init_nonempty=0/50
completed_logs=50/50
```

By archetype:

| Archetype | n | Mean WR | Median | Min | Max | <0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dragapult | 6 | 0.367 | 0.419 | 0.130 | 0.572 | 6 |
| Crustle Wall | 5 | 0.451 | 0.454 | 0.310 | 0.634 | 5 |
| Team Rocket Mewtwo | 1 | 0.496 | 0.496 | 0.496 | 0.496 | 1 |
| Marnie Grimmsnarl | 7 | 0.561 | 0.620 | 0.334 | 0.836 | 7 |
| Alakazam | 4 | 0.735 | 0.710 | 0.568 | 0.952 | 3 |
| Mega Lopunny | 10 | 0.761 | 0.778 | 0.656 | 0.862 | 10 |
| Festival Lead | 5 | 0.784 | 0.758 | 0.668 | 0.976 | 4 |
| Teal Mask Ogerpon | 9 | 0.813 | 0.958 | 0.202 | 0.992 | 3 |
| Cynthia Garchomp | 2 | 0.878 | 0.878 | 0.852 | 0.904 | 1 |
| Mega Lucario | 1 | 0.922 | 0.922 | 0.922 | 0.922 | 0 |

Diagnostics from `/tmp/ptcg_analyze_v11_shadow_random.py` on `ks`:

- Low random WR is not mainly sample-count limited: Pearson correlation with random WR was `decisions=0.09`, `episodes=0.21`, `trajectory_score=0.06`.
- Deck paths are not corrupt: 50/50 evaluated deck CSVs hash to the expected `deck_sig`. Most deck paths resolve to `logs/ladder_pool_0802_all/decks`, but matching signatures mean the 60-card list is equivalent.
- Training completed and did not show final errors. This is not a stale OOM issue.
- The v11 shadow manifest has no init paths. These were trained from scratch as single team/deck specialists on 0804-only data.

Interpretation:

- Do not treat the current v11 shadow pool as a reliable ladder pool yet.
- This result does not prove the new 80/64 v11 features are bad. Some v11 shadows still pass random, and the failure pattern is weakly related to data volume.
- The most likely issue is recipe/initialization: narrow from-scratch specialists overfit local trajectory imitation and lose basic game execution. Next v11 shadow wave should initialize from strong population checkpoints, preferably v11 population trained on `data/bc_corpus_banded_v11_0803_0804`; if v11 population is not ready, use v10 population checkpoints with `--init-partial --state-feat-dim 80 --opt-feat-dim 64`.
- Current v11 shadows can still be used for trace/debug diversity, but they should be low-trust opponents unless they pass random and RR gates.

## v11 Pop-Init Shadow Audit

Remote audit at 2026-08-05 10:40 Asia/Shanghai:

- Population training first failed for Alakazam, Mega Lopunny, and Cynthia Garchomp under `--cuda-memory-gb 8`, but retry with `--cuda-memory-gb 24` completed all three.
- Final `checkpoints/pop_v11/bc2_*_v11pop_0803_0804_set_w2.npz` population checkpoints are present for all 11 trained archetypes.
- `logs/shadow_pool_manifest_v11_0804_popinit_set.csv` has 46 rows, all with non-empty and existing `init_path`; all 46 shadow checkpoints exist.
- `logs/v11_pipeline/train_shadow_v11_0804_popinit_set_mem8g.runner.log` completed 46/46 with no failures.
- Initial pop-init random eval contained 43 rows because three 0804-new signatures had empty `deck_path` in the manifest and were skipped. The user rebuilt/patched with 0804 deck paths and resumed random eval; the CSV now contains 46/46 rows.

Current evaluated pop-init random result:

```text
n=46 mean=0.860 median=0.934 min=0.436 max=0.996
>=0.99: 5
>=0.97: 10
<0.95: 27
<0.90: 17
```

By archetype:

| Archetype | n | Mean WR | Median | Min | Max | <0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dragapult | 6 | 0.638 | 0.636 | 0.572 | 0.718 | 6 |
| Crustle Wall | 5 | 0.738 | 0.784 | 0.656 | 0.804 | 5 |
| Teal Mask Ogerpon | 8 | 0.840 | 0.984 | 0.436 | 0.996 | 3 |
| Mega Lucario | 1 | 0.898 | 0.898 | 0.898 | 0.898 | 1 |
| Festival Lead | 4 | 0.903 | 0.933 | 0.752 | 0.996 | 1 |
| Mega Lopunny | 9 | 0.935 | 0.930 | 0.904 | 0.976 | 0 |
| Alakazam | 4 | 0.943 | 0.943 | 0.938 | 0.948 | 0 |
| Marnie Grimmsnarl | 6 | 0.955 | 0.961 | 0.898 | 0.988 | 1 |
| Team Rocket Mewtwo | 1 | 0.968 | 0.968 | 0.968 | 0.968 | 0 |
| Cynthia Garchomp | 2 | 0.975 | 0.975 | 0.960 | 0.990 | 0 |

Interpretation:

- Pop init fixed a large part of the from-scratch v11 shadow failure: mean random improved from 0.667 to 0.860, and `<0.90` dropped from 40/50 to 17/46.
- It still does not recover the v10 shadow random reference (`mean=0.977`, `median=0.994`).
- v10's micro mean is inflated by many high-WR Marnie/Alakazam rows, but this does not explain the full gap. After archetype balancing:
  - v10 raw micro mean: `0.977`
  - v10 macro-by-archetype mean: `0.951`
  - v10 reweighted to v11 pop-init archetype distribution: `0.952`
  - v11 pop-init actual micro mean: `0.860`
  - v11 pop-init macro-by-archetype mean: `0.879`
- Biggest common-signature regressions:
  - Dragapult `cc2e995b5ad0`: v10 `0.826`, v11 pop-init `0.575`
  - Dragapult `6763881ee2d5`: v10 `0.878`, v11 `0.616`
  - Dragapult `d112d6fbe57d`: v10 `0.958`, v11 `0.718`
  - Crustle `7ee600c6f769`: v10 `0.945`, v11 `0.656`
  - Crustle `b141ae295739`: v10 `0.924`, v11 `0.662`
  - Crustle `47756cdfd20f`: v10 `0.958`, v11 `0.794`
- Do not launch full feature rollback yet. First patch missing deck paths and resume random for the 3 skipped rows, then run population-model random and balanced RR/baseline-delta. If v11 population itself is weak against random, run 64/48 rollback. If v11 population is fine but pop-init shadows remain weak, fix the specialist recipe instead of blaming features.
- Dragapult and Crustle are the clearest remaining failures; prioritize trace and recipe checks there.

Feature rollback targeted contrast for Dragapult/Crustle completed:

```text
logs/eval_shadow_v11/pop_v11_80_dragapult_crustle_random_g500.csv
logs/eval_shadow_v11/pop_v11_64_48_dragapult_crustle_random_g500.csv
logs/eval_shadow_v11/shadow_v11_0804_popinit_random_g500.csv
logs/eval_shadow_v11/shadow_v11_0804_feat64_48_popinit_dc_random_g500.csv
```

Training status:

- `logs/v11_pipeline/train_pop_v11_0803_0804_feat64_48_dc.runner.log`: Dragapult and Crustle 64/48 population completed, no failures.
- `logs/shadow_pool_manifest_v11_0804_feat64_48_popinit_dc.csv`: 11 rows.
- `logs/v11_pipeline/train_shadow_v11_0804_feat64_48_popinit_dc.runner.log`: 11/11 completed, no failures.

Random 500 summary for Dragapult + Crustle only:

| Variant | n | Mean | Median | Min | Max | <0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pop80` | 11 | 0.701 | 0.650 | 0.548 | 0.858 | 11 |
| `pop64` | 11 | 0.684 | 0.616 | 0.548 | 0.828 | 11 |
| `shadow80` | 11 | 0.683 | 0.662 | 0.572 | 0.804 | 11 |
| `shadow64` | 11 | 0.672 | 0.658 | 0.528 | 0.804 | 11 |
| `v10_shadow_ref` | 44 | 0.916 | 0.938 | 0.564 | 0.990 | 14 |

By archetype:

| Variant | Dragapult Mean | Crustle Mean |
| --- | ---: | ---: |
| `pop80` | 0.609 | 0.810 |
| `pop64` | 0.580 | 0.808 |
| `shadow80` | 0.638 | 0.738 |
| `shadow64` | 0.607 | 0.750 |
| `v10_shadow_ref` | 0.873 | 0.929 |

Interpretation:

- 64/48 rollback does not improve Dragapult/Crustle. It is slightly worse overall.
- The v11 population baselines are already weak against random for these archetypes, so the primary issue is not just specialist overfitting.
- Since `pop64` is not better than `pop80`, do not run full feature rollback as the next step.
- Treat this as a data/recipe/behavior problem. Next useful diagnostics are random-loss traces and BC failure reports for Dragapult/Crustle, plus a v10-corpus/v10-recipe reproduction for these two archetypes to isolate whether v11 corpus/data distribution caused the regression.

Corpus coverage audit:

- The user correctly noted that `data/bc_corpus_banded_v10_all_0803` is not just one or two days. It covers 11 daily episode files from 2026-07-24 through 2026-08-03.
- `data/bc_corpus_banded_v11_0803_0804` covers only 2026-08-03 and 2026-08-04.
- Therefore the `pop_v10_repro` result is not a fair feature-only comparison; it also benefits from much broader historical coverage.

Top-band target archetype coverage:

| Corpus | Dates | Dragapult Decisions | Crustle Decisions |
| --- | --- | ---: | ---: |
| `v10_all_0803` | 2026-07-24..2026-08-03 | 312,662 | 304,073 |
| `v11_0803_0804` | 2026-08-03..2026-08-04 | 119,806 | 60,604 |
| `v11_0804_only` | 2026-08-04 | 60,898 | 25,140 |

All-archetype corpus coverage:

| Corpus | Dates | Total Decisions |
| --- | --- | ---: |
| `v10_all_0803` | 11 days | about 8.37M |
| `v11_0803_0804` | 2 days | about 1.53M |
| `v11_0804_only` | 1 day | about 0.76M |

Interpretation update:

- The main next experiment should be a v11 multi-day extraction matching the 2026-07-24..2026-08-03 coverage of `v10_all_0803`, optionally plus 2026-08-04 as a separate corpus.
- If v11 multi-day recovers Dragapult/Crustle random quality, the regression was mostly data coverage/distribution.
- If v11 multi-day remains low while v10 11-day is high, then investigate v11 extraction/features/training target more directly.
- Do not treat the current 2-day v11 results as decisive evidence against v11 features.

v11 multi-day Dragapult/Crustle result:

```text
pop80_v11_2d      n=11 mean=0.701 median=0.650 dragapult=0.609 crustle=0.810 <0.90=11
shadow80_v11_2d   n=11 mean=0.683 median=0.662 dragapult=0.638 crustle=0.738 <0.90=11
pop_v10_11d       n=11 mean=0.844 median=0.836 dragapult=0.797 crustle=0.900 <0.90=8
pop_v11_12d       n=11 mean=0.911 median=0.906 dragapult=0.891 crustle=0.934 <0.90=2
shadow_v11_12d    n=11 mean=0.904 median=0.922 dragapult=0.880 crustle=0.932 <0.90=5
```

Interpretation update:

- This validates the data-coverage hypothesis. The 12-day v11 corpus recovers Dragapult/Crustle and is stronger than the v10 11-day population reproduction on the same 0804 deck set.
- Do not run full 64/48 rollback. The main v11 line should use multi-day extraction.
- Pulling earlier July episodes is useful for coverage and rare archetype stability, but do not blindly mix all dates into the main submission recipe. Keep date-window ablations such as 12d, 19d, and full-July-plus-0804, and consider recency weighting so older ladder distributions do not dilute 0804 matchups.
- Remaining shadow gap is much smaller and likely specialist/team coverage. One failure, `shadow_v11all_dragapult_cc2e995b_benarg`, was due to `--team-name Benarg` keeping zero samples in the 12-day corpus. The practical fix is to train that checkpoint as a deck-sig specialist without `--team-name`.
- Next scale-up should train full v11 multi-day population across all relevant archetypes, then rebuild shadow manifests from `v11_0724_0804` with `--known-decks-dir logs/ladder_pool_0804_all/decks` and guard against zero-sample team specialists.

v11 0724-0804 population/shadow training and random audits:

- User reported at 2026-08-05 13:42 that the v11 population training, v11 shadow training, and random audits are complete.
- Next remaining validation is candidate round-robin:

```bash
python3 tools/eval_round_robin.py \
  --manifest logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/eval_v11_0724_0804/rr_candidates_pop_top3_shadow_ge097_g100.csv \
  2>&1 | tee logs/eval_v11_0724_0804/rr_candidates_pop_top3_shadow_ge097_g100.log
```

- This command requires the `eval_round_robin.py` `--manifest` patch synced at 13:42. Remote smoke test passed with `--manifest-limit 2 --games 1` and wrote `/tmp/rr_manifest_smoke.csv`.

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
