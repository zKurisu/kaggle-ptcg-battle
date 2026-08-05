# Agent Handoff

Last updated: 2026-08-05 08:52 Asia/Shanghai.

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
