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
- Dragapult and Alakazam must pass random at exactly 100% before RR/Kaggle-style
  interpretation. If random is below 100%, run `tools/v15_random_gate.py`; it
  writes a first-loss step-by-step trace. The trace must be read and the
  concrete failed setup/route/resource decisions must be recorded before the
  next long training run.

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
- Explicit route targets:
  - Dragapult mainline: Dreepy -> Drakloak -> Dragapult ex.
  - Alakazam mainline: Abra -> Kadabra -> Alakazam.
  - Training logs report `val_route`, `route_label`, and `route_rate`.
    `route_label` below roughly `0.55` means raw BC labels often conflict with
    the strategic mainline and route/rule supervision must override imitation.

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

Remote v15 pilot after the smoke:

- Dragapult and Alakazam both reached only `19/20` random in a small smoke.
- Dragapult first-loss trace: active Budew, Poké Pad did not take Dreepy, no
  Dragapult line established, then early end/attach misses. Root cause:
  setup-route failure.
- Alakazam first-loss trace: Abra/Kadabra route started, but Alakazam evolution
  and attacks were repeatedly delayed by resource/retreat/end choices. Root
  cause: evolution route priority failure.
- Therefore v15 now adds route supervision and within-type action loss; random
  100% plus readable first-loss trace is the hard loop.

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

Random gate with first-loss trace:

```bash
python3 tools/v15_random_gate.py checkpoints/v15_smoke/dragapult_v15_plan.pt \
  --deck logs/ladder_pool_0812_all_v13_20260813/decks/cc2e995b5ad0_dragapult_kh0a.csv \
  --games 300 --workers 8 --progress-every 50 \
  --out-dir logs/v15_gate/dragapult_smoke
```
