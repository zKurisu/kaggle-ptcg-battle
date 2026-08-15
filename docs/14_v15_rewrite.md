# v15 Turn-Block Plan Pipeline

v15 is a probe-first rewrite after v14 showed that auxiliary history/plan heads
could learn labels without changing the live action logits.

## Non-Negotiable Gates

- Training must print corpus signal density before any long run:
  history slots, public log slots, known opponent hand rate, plan slots,
  same-turn continuation, multi-select rate, and DCA split/focus.
- Training must print action-logit ablations every epoch:
  `no_history`, `reverse_history`, `no_known`, and `no_plan`.
- A run with `no_plan_agree` near 1.0 means the plan branch is decorative.
- A run with `no_history_agree` near 1.0 means the model is still current-board
  BC.
- A run with `reverse_history_agree` near 1.0 means event order is not used,
  even if event counts are used.

## Design

- Corpus unit: one live decision row with explicit event history and same-turn
  future plan labels.
- Event history: own selected actions plus public logs visible from the current
  observation. Opponent hidden actions are not copied in directly.
- Known opponent hand: maintained only from public card movement logs.
- Plan labels: next `P` same-turn actions with type/card/attack/context,
  turn-continuation flag, block length/position, and coarse plan mode.
- Model: state tokens, option tokens, known-card tokens, and event-history
  tokens feed a plan latent. The legal-option scorer uses that latent directly.
- Explicit priors:
  - current type prior from plan latent;
  - history-conditioned type prior from event history.
  - known-card type prior from public reveal memory.

## Smoke Result 2026-08-15

Local Dragapult smoke on 12 episodes from 2026-08-12:

- Corpus signals were nonzero:
  `event_slots_mean=46.89`, `public_log_slots_mean=31.51`,
  `known_rate=0.879`, `plan_rate=0.909`, `turn_continue_rate=0.909`,
  `dca_rate=0.137`.
- After adding type prior and history type prior, action logits responded to
  `no_plan` and `no_history`.
- `reverse_history` still remains too weak. This is intentionally logged as
  `history_order_not_affecting_action`; do not hide it with RR-only evaluation.

## Starter Commands

Small local/remote extraction:

```bash
PYTHONPATH=/data/jie/ptcg_rl_git_v7_baseline_20260804:$PYTHONPATH \
python3 -u tools/v15_extract_blocks.py /data/jie/episodes_raw \
  --out data/seq_corpus_v15_smoke \
  --date-from 2026-08-12 --date-to 2026-08-12 \
  --max-episodes 20 --min-rows 1 \
  --history-k 48 --plan-steps 4 \
  --progress-every 5 --workers 1
```

Short diagnostic training:

```bash
PYTHONPATH=/data/jie/ptcg_rl_git_v7_baseline_20260804:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python3 -u tools/v15_train_plan_policy.py \
  --corpus data/seq_corpus_v15_smoke \
  --archetype Dragapult --score-band 600-699 \
  --history-k 48 --plan-steps 4 \
  --batch-size 128 --epochs 2 \
  --max-train-batches 40 --max-val-batches 10 \
  --width 256 --layers 2 --heads 4 \
  --type-prior-scale 1.50 \
  --history-type-prior-scale 0.35 \
  --known-type-prior-scale 0.25 \
  --progress-every 10 \
  --out checkpoints/v15_smoke/dragapult_v15_plan.pt
```
