# v14 Sequence-First BC Pipeline

Last updated: 2026-08-14.

This branch introduces a separate sequence-first pipeline. It is not another
small extension of `bc2_train.py`.

## Why This Exists

The old BC path trains one legal-action classification problem per decision.
History, trajectory, rule, and matchup tags were attached to that single-step
objective, but the sample unit remained one decision point. That made it easy
for the model to ignore prefix information because most labels are locally
predictable from the current legal options.

v14 changes the sample unit to a game/window. Each training example is a
causal prefix of decisions. The model must predict:

- current action among legal options,
- multi-select option set,
- multi-select count and ordered selected option sequence,
- current action type,
- future behavior over the next horizon,
- game outcome.

The future-plan and outcome heads are not for submission scoring directly.
They are pressure on the representation so sequence information cannot be a
decorative feature.

## New Code

- `tools/v14_extract_sequences.py`: extracts game-contiguous v14 corpus.
- `ptcg_rl/seq/data.py`: NPZ-backed game/window dataset.
- `ptcg_rl/seq/model.py`: causal transformer sequence policy.
- `ptcg_rl/seq/torch_policy.py`: local/live inference wrapper for `.pt`.
- `tools/v14_train_sequence_policy.py`: single model trainer.
- `tools/v14_train_population.py`: multi-GPU population runner.
- `tools/v14_audit_sequence_corpus.py`: data audit before training.
- `tools/v14_probe_sequence_policy.py`: sequence-use probe after training.
- `tools/v14_build_train_manifest.py`: create population jobs from corpus.

`tools/eval_bc.py` and `tools/eval_round_robin.py` now load both legacy `.npz`
and v14 `.pt` checkpoints via `ptcg_rl/policy_loader.py`.

## Smoke Result

Local smoke extraction with the first 20 episodes of each local zip succeeded:

- rows: `52099`
- bad actions: `0`
- extraction errors: `0`

Marnie smoke training on CPU with only 5 mini-batches produced a valid checkpoint
and non-NaN validation metrics after fixing left-padding/causal-mask handling.

Important probe result from the smoke checkpoint:

- `last_only`, `last_zero_prefix`, and `last_shuffle_prefix` were nearly equal.
- This is expected for a 5-batch smoke model and proves the probe can detect
when the model has not really learned sequence dependence.

For a real run, do not claim sequence learning unless zero/shuffle prefix
ablations make last-step loss/top1/plan meaningfully worse.

## Extract

Use a leaderboard CSV whenever possible; otherwise every row falls into
`600-699`.

```bash
python3 -u tools/v14_extract_sequences.py /data/jie/episodes_raw \
  --out data/seq_corpus_v14_0801_0812 \
  --lb-csv logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv \
  --workers 12 \
  --future-horizon 12 \
  --progress-every 500 \
  2>&1 | tee logs/v14_sequence/extract_v14_0801_0812.log
```

## Audit Corpus

```bash
python3 tools/v14_audit_sequence_corpus.py \
  --corpus data/seq_corpus_v14_0801_0812 \
  --archetype "Dragapult" \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --out-csv logs/v14_sequence/audit_dragapult_games.csv
```

Check:

- game length distribution,
- action type distribution,
- multi-select count distribution,
- opponent archetype coverage,
- top deck signatures,
- future-plan means.

Run the integrity checker before launching long jobs:

```bash
python3 tools/v14_check_sequence_integrity.py \
  --corpus data/seq_corpus_v14_0801_0812 \
  --samples 20000
```

## Train One Model

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/v14_train_sequence_policy.py \
  --corpus data/seq_corpus_v14_0801_0812 \
  --archetype "Dragapult" \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --deck-sig cc2e995b5ad0 \
  --seq-len 32 \
  --width 384 \
  --layers 4 \
  --heads 6 \
  --batch-size 256 \
  --epochs 10 \
  --amp \
  --out checkpoints/v14_sequence/bc2_dragapult_cc2e995b_v14seq.pt \
  2>&1 | tee logs/v14_sequence/train_dragapult_cc2e995b_v14seq.log
```

For A800 with idle memory, increase `--batch-size` to `512` or `768` after
watching `nvidia-smi`.

Training progress is diagnostic by default. Each progress line should include
loss parts, top1/type/count accuracy, count MAE, predicted/target selection
count, set-F1, ordered-selection accuracy, outcome accuracy, samples/s, ETA,
and CUDA allocated/reserved memory. If `pred_k` collapses away from `target_k`,
`setF1` stays flat, or `orderAcc` does not move, stop the run and inspect the
pipeline before waiting for RR.

## Population

Build a manifest:

```bash
python3 tools/v14_build_train_manifest.py \
  --corpus data/seq_corpus_v14_0801_0812 \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --top-per-archetype 2 \
  --min-rows 5000 \
  --out logs/v14_sequence/train_manifest_top2.csv
```

Run multi-GPU:

```bash
python3 -u tools/v14_train_population.py \
  --manifest logs/v14_sequence/train_manifest_top2.csv \
  --corpus data/seq_corpus_v14_0801_0812 \
  --out-dir checkpoints/v14_sequence/pop_top2 \
  --log-dir logs/v14_sequence/pop_top2 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --seq-len 32 \
  --width 384 \
  --layers 4 \
  --heads 6 \
  --batch-size 256 \
  --epochs 10 \
  --amp \
  2>&1 | tee logs/v14_sequence/pop_top2.runner.log
```

The population runner prints each running job plus the latest training progress
line from its log on every poll. A status line that only says a job is running
without metric movement is not sufficient for long overnight runs.

## Probe Sequence Use

```bash
python3 tools/v14_probe_sequence_policy.py \
  --checkpoint checkpoints/v14_sequence/bc2_dragapult_cc2e995b_v14seq.pt \
  --corpus data/seq_corpus_v14_0801_0812 \
  --archetype "Dragapult" \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --deck-sig cc2e995b5ad0 \
  --seq-len 32 \
  --samples 512 \
  --device cuda \
  --prefixes 4,8,16,24,32
```

Interpretation:

- `last_zero_prefix_keep_ledger` close to `last_only`: current ledger dominates.
- `last_zero_prefix_zero_ledger` close to `last_only`: model is not using
  explicit ledger/previous action either.
- `last_shuffle_prefix` close to `last_only`: causal prefix order is not used.
- Full/base much better than shuffled/zero-prefix: sequence dependence exists.
- Plan loss should improve with longer prefixes; otherwise the future-plan head
  is not doing useful work.

## Local Evaluation

Legacy eval scripts now accept `.pt`:

```bash
python3 tools/eval_bc.py \
  checkpoints/v14_sequence/bc2_dragapult_cc2e995b_v14seq.pt \
  --deck logs/ladder_pool_0812_all_v13_20260813/decks/cc2e995b5ad0_dragapult_flg.csv \
  --games 200 \
  --workers 16 \
  --max-turns 700
```

Round robin uses the same `--entry name=checkpoint.pt:deck.csv` format.

## Submission Caveat

`main.py` and `tools/package_submission.py` can now load/package `.pt` files,
but v14 submission has not been validated for Kaggle runtime/size. Treat v14
as a local validation pipeline first. If v14 clearly improves random/RR and
probe metrics prove sequence use, then optimize/export for submission.

Compatibility notes:

- `TorchSequencePolicy.select()` accepts the legacy `temperature` and
  `update_history` arguments used by old eval/submission paths.
- `TorchSequencePolicy.select_mcts()` is only a compatibility shim; v14 does
  not train a value head for MCTS yet, so this shim returns the greedy sequence
  policy instead of falling back to random actions.
- `tools/eval_round_robin.py` parallel workers must use
  `ptcg_rl.policy_loader.load_policy()` so `.pt` checkpoints are evaluated
  correctly with `--workers > 1`.
- Logit masks must stay fp16-safe under AMP. Do not change `NEG_INF` back to
  `-1e9`.
- Ordered multi-select scoring must not materialize `[B,T,K,N,W]`; population
  training relies on the low-memory dot-product scorer.
