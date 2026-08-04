# Agent Handoff

Last updated: 2026-08-04 23:15 Asia/Shanghai.

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

Recent relevant commits:

- `1d421ad` Document current BC pipeline notes
- `edc0113` Add matchup-aware BC data pipeline features
- `950f7fa` Add matchup decision trace diagnostic
- `dc79b90` Show failed jobs in BC runners
- `8b5354c` Add CUDA memory caps to BC training
- `0ed5d12` Improve BC shadow training pipeline

The worktree was clean before this file was added.

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

Latest observed `by` scores at `2026-08-04T15:15:43+00:00`:

| Submission | Description | Score |
| --- | --- | ---: |
| `55241705` | `bc: team_rocket_mewtwo_v10pop` | 635.1 |
| `55241683` | `bc: crustle_wall_v10pop` | 920.5 |
| `55234504` | `bc: ogerpon_v10_fixed_top2` | 956.3 |
| `55234481` | `bc: ogerpon_v10_fixed_top3` | 718.5 |

User noted `55234504` peaked around 1040 even though not all score changes are recorded.

## Current Shadow Training

Shadow manifest:

```text
logs/shadow_pool_manifest_v10_all0803_popinit_set.csv
```

Command currently running on `ks`:

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

Status snapshot at 2026-08-04 23:15 Asia/Shanghai:

```text
manifest_rows=361
done=289
running=3
failed=0
pending=69
started_no_final=0
npz_all=1953
npz_final=279
```

Per-archetype snapshot:

| Archetype | Done / Total | Pending | Running |
| --- | ---: | ---: | ---: |
| Marnie Grimmsnarl | 127 / 151 | 22 | 2 |
| Alakazam | 39 / 56 | 16 | 1 |
| Crustle Wall | 33 / 38 | 5 | 0 |
| Team Rocket Mewtwo | 18 / 24 | 6 | 0 |
| Dragapult | 17 / 22 | 5 | 0 |
| Festival Lead | 12 / 18 | 6 | 0 |
| Mega Lopunny | 14 / 17 | 3 | 0 |
| Teal Mask Ogerpon | 13 / 15 | 2 | 0 |
| Cynthia Garchomp | 12 / 14 | 2 | 0 |
| Mega Lucario | 3 / 4 | 1 | 0 |
| Mega Starmie | 1 / 2 | 1 | 0 |

Running shadow jobs at the snapshot:

- rank 284: `shadow_marnie_grimmsnarl_2c22fa76_mega_regigigas_ex_vmax`
- rank 287: `shadow_alakazam_7f9a5389_yumizu`
- rank 288: `shadow_marnie_grimmsnarl_2c22fa76_k_yoshida`

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

- `55241683`: Crustle Wall v10pop, currently around 920.5.
- `55241705`: Team Rocket Mewtwo v10pop, currently around 635.1.

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
  --workers 8 \
  --max-turns 700 \
  --out-prefix logs/eval_v10/marnie_vs_ogerpon_trace_g100
```

## Next Work

Immediate:

1. Let the current shadow manifest finish unless the user asks to stop it.
2. Re-run the shadow status helper and update this file when it completes.
3. Build/evaluate a larger local ladder pool using the completed shadow checkpoints. Do not filter out weak shadows yet; every archetype is useful for population quality.
4. Run round-robin/failure-pool eval for submitted probes and top shadow candidates.
5. Deep-dive bad matchups with `trace_matchup_decisions.py` and `bc2_failure_report.py`.

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

