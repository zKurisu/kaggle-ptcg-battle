# 05 — RL Fine-tuning

The old `train.py` / `ptcg_rl.trainer.PPOTrainer` path is legacy self-play code.
For the current BC2 pipeline, use `tools/rl_finetune_vs_pool.py`.

RL is only for targeted fine-tuning from a strong BC/shadow checkpoint against a
known weakness pool. Do not use it for from-scratch training, and do not accept a
checkpoint only because training-pool win rate rises.

## Pilot Command

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

For another archetype, keep the same structure and replace:

- `--policy-init` and `--deck` with the target checkpoint/deck.
- `--opponent` / `--opponent-manifest` filters with its weak matchup pool.
- `--bc-anchor-*` filters with the corresponding target archetype and opponent.

## RL v2 Notes

As of 2026-08-09, `tools/rl_finetune_vs_pool.py` is the active RL path.

- It can infer `pointer` or `cross_attn` architecture from the `.npz`
  checkpoint and build the matching torch model.
- PPO re-evaluation now passes per-decision history snapshots into
  `evaluate_actions()`, so history-enabled checkpoints are not silently trained
  as stateless policies.
- Rollout can run through CPU actor workers with `--rollout-workers`.
  Actors use `NumpyPolicy` sampling, then the GPU learner refreshes old
  log-probs/value estimates before PPO.
- Use `--rollout-temperature` and `--rollout-top-k` for real exploration.
  Greedy rollout is only for debugging.
- `--shaping-weight` adds a coarse dense potential reward. Keep it modest and
  verify with baseline-delta, because it is heuristic.
- BC anchor should be low weight for RL experiments. It is a drift guard, not
  the main learning signal.

Current long-run template is stored as `/tmp/run_rl_v2_wave1_20260809.sh` on
`ks`. It launches Marnie, Mega Lucario, Ogerpon, and Dragapult weak-pool PPO in
parallel and then runs random plus paired baseline-delta checks.

## PPO v3 Options

The wave1 result showed that plain PPO can improve one matchup while damaging
general play, especially for Dragapult and Ogerpon. The script now supports a
more constrained PPO recipe:

- `--ref-kl-coef`: action-level squared log-prob drift from the initial policy.
  This is a cheap trust region across iterations.
- `--target-kl`: stop the current PPO update early when the approximate KL is
  already too large.
- `--value-clip-eps`: PPO2-style value clipping to keep the newly trained value
  head from amplifying noisy returns.
- `--advantage-normalization opponent`: normalize advantages within each
  opponent bucket, useful for mixed weakness pools.
- `--advantage-clip`: bound extreme GAE samples.
- `--reward-weight-mode opponent_inverse_winrate`: emphasize opponents that are
  still hard in the current rollout batch and de-emphasize already-easy ones.
- `--save-policy best --save-final ...` or `--save-policy both`: preserve the
  best rollout checkpoint instead of overwriting it with the final policy.

Current wave2 template is stored as `/tmp/run_rl_v3_wave2_20260809.sh` on `ks`.
It runs on GPU0 only because GPU1-3 are occupied by unrelated ARC jobs.

## Aggressive Self-Exploration PPO

As of 2026-08-09, success-trajectory imitation is no longer the default path.
Directly overfitting weak-matchup wins did not reliably improve the target
matchups, which suggests many wins were either opponent bricks, one-off tactical
errors, or otherwise not causal demonstrations. Use those trajectories for
diagnosis, not as the main objective.

The aggressive PPO path instead lets the policy explore against a hard league:

- `--opponent-weight-mode adaptive_lossrate` or `adaptive_inverse_winrate`
  updates opponent sampling weights from the latest rollout batch.
- `--opponent-stats-csv` records per-opponent games, win rate, and updated
  weight every iteration.
- `--league-bootstrap-current` adds the initial policy as a league opponent.
- `--league-snapshot-every N` periodically adds the current policy as a fixed
  future opponent, approximating self-play without changing submission format.
- `--rollout-temperature-final`, `--rollout-top-k-final`,
  `--entropy-final-coef`, `--ref-kl-final-coef`, and
  `--bc-anchor-final-weight` linearly schedule exploration and constraints over
  `--schedule-iters`.

This should be run at a much larger scale than the early PPO pilots. Community
reports for this competition point to tens of thousands of games for weak
models and millions for strong RL. Treat 5k games as a smoke test, not a
verdict.

User direction as of 2026-08-09: BC is locked. Do not spend more iterations on
BC fine-tuning, including weak-matchup specialist fine-tuning. It is expected to
damage general play while failing to repair structural weaknesses. For
structural weaknesses, use from-scratch or otherwise major training changes. In
this mode:

- run `tools/rl_train_league.py`;
- pass `--init-mode random`;
- use `--policy-init` only as an architecture/checkpoint-format template;
- keep `--bc-anchor-weight 0` and `--ref-kl-coef 0`;
- train against curriculum/league opponents at scale.

Existing BC checkpoints are still useful as baselines, fixed opponents,
submission references, and model-shape templates. They should not be updated by
small PPO/BC fine-tunes.

Kaggle constraints still matter: submission inference is CPU-only with 1.6 vCPU,
8GB RAM, and 600 seconds total per team per game. Keep the submitted policy
lightweight; use GPUs and large actor counts only during training.

## Validation Gate

Evaluate every saved checkpoint before treating it as useful:

```bash
python3 tools/eval_bc.py CHECKPOINT.npz \
  --deck TARGET_DECK.csv \
  --games 500 \
  --workers 8 \
  --max-turns 700

python3 tools/eval_baseline_delta.py \
  --baseline base=BASELINE.npz:TARGET_DECK.csv \
  --candidate rl=CHECKPOINT.npz:TARGET_DECK.csv \
  --opponent-manifest FAILURE_OR_SHADOW_POOL.csv \
  --manifest-limit 120 \
  --skip-bad-entries \
  --games 80 \
  --workers 8 \
  --max-turns 700 \
  --out-csv logs/rl/rl_vs_baseline_shadow_probe.csv
```

Then trace the worst remaining matchup:

```bash
python3 tools/trace_matchup_decisions.py \
  --candidate rl=CHECKPOINT.npz:TARGET_DECK.csv \
  --opponent weak=OPPONENT.npz:OPPONENT_DECK.csv \
  --games 100 \
  --max-turns 700 \
  --out-prefix logs/rl/trace_target_vs_weak_g100
```

Aggregate multiple trace summaries with:

```bash
python3 tools/summarize_matchup_failures.py \
  "logs/rl/*.summary.csv" \
  --min-loss-decisions 20 \
  --out-csv logs/rl/failure_trace_priority.csv
```

## Current Limits

- Opponents are fixed `NumpyPolicy`/random entries. The trainable side is a
  torch policy/value model initialized from BC2 `.npz`.
- Self-play is still targeted weak-pool PPO, not a full ladder simulator.
- The value head starts weak because BC mostly trains policy. Treat early PPO
  value loss and rollout win-rate cautiously.
- Dense shaping is coarse and may reward local board progress while missing a
  matchup-specific long plan. Always validate against greedy baseline-delta and
  random before using a checkpoint.
- Full action/strategy hierarchy is still outside this PPO script. Current
  history support uses checkpoint history features; it does not maintain a
  separate learned high-level planner during RL.
