# Agent Handoff

Last updated: 2026-08-15 14:04 Asia/Shanghai.

This file is the first place a new agent should read before touching the project. Keep it updated whenever the pipeline changes, a Kaggle submission is made, a long remote job is started/stopped, or the interpretation of current results changes. After updating it locally, sync it to the `ks` workspace and commit the change.

## Update: v14 Failure Summary And v15 Rewrite Gate 2026-08-15

Current hard direction from the user:

- Do not continue fine-tuning as the main path. The old BC family is treated as
  locked for submission/baseline comparison. New work should be large
  architecture/data-objective rewrites, explicit rule/plan integration, or
  from-scratch retraining. Small loss-weight tweaks and weak-matchup
  fine-tunes have repeatedly consumed time without changing structure.
- Do not discover failures only after random/RR/Kaggle. Every new signal must be
  visible during training. A run that cannot prove it uses sequence/global/plan
  information in training logs should be stopped before long evaluation.
- v15 should be rewritten after testing the current v14 changes fails the signal
  gate. The rewrite must be probe-first: the data object, model, loss, and
  diagnostics must all make multi-turn decisions observable.

Historical version lessons:

- v7: strongest simple baseline. Direct BC with older compact features and
  deck-sig/top2 filtering produced useful Kaggle results, especially Ogerpon.
  It had limited strategic generalization, but its simplicity made it stable.
- v8/v9: adding gameplan/trajectory-style features did not improve online
  performance. Offline metrics could look fine while Kaggle ladder dropped
  sharply. Lesson: adding features without proving the action head uses them is
  actively risky.
- v10: fixed several v8/v9 extraction/feature issues and recovered much of v7
  for Ogerpon fixed top2 (`951.8` final shown, temporarily about `1040`).
  Top3 was much worse. Lesson: cleaner data/signature selection matters, but it
  still did not solve multi-option/global/weak-matchup reasoning.
- v11: opponent/matchup features and more dates helped random quality only when
  enough dates were used. Two-day v11 was poor; all-date/12-day v11 rose toward
  v10 random quality. Structural matchup weaknesses remained.
- high1100/team-specific/winner-only/900+ mixes: results were unstable. Some
  high-only models missed 600-1100 climb opponents; some winner-only mixtures
  overfit sparse lucky wins or lost broad coverage. DVH/old strong submissions
  are local comparison baselines only and must not be reused for submission.
- v12/v13 history/ledger/teacher/trajectory attempts: history streams, summary
  features, success-trajectory mining, and weak-matchup fine-tunes did not show
  stable broad gains. Directly fitting rare successful weak-matchup games did
  not reliably improve those matchups, suggesting many "success" rows are luck,
  opponent mistakes, or non-transferable lines.
- v14: sequence-first rewrite added current-row weighting, DCA labels,
  same-turn continuation, opportunity losses, rank losses, known-info/reveal
  probes, and ablation diagnostics. Local random/RR and Kaggle submissions still
  showed pipeline-level failure: v14 submitted Lucario/Alakazam/Lopunny could
  not stably hold strong ladder scores.

Concrete v14 failures now known:

- The policy is still mostly a current-board next-action imitator. Even after
  sequence inputs, `revhist_delta` stayed around `-0.005` to `-0.006` and
  `revhist_agree` stayed about `0.95-0.98` in rank/turn-next diagnostics. If
  reversing chronological history barely changes the live action, the model is
  not learning a true multi-turn policy.
- Latest completed rank/turn-next evidence:
  - Rank-margin diagnostic: Alakazam baseline `val_cur=0.762/0.877`, rank
    margin `0.35` `val_cur=0.761/0.873`; Dragapult baseline
    `val_cur=0.680/0.862`, rank margin `0.35` `val_cur=0.676/0.860`.
    Exact rank margin did not improve the live action.
  - Turn-next diagnostic: Alakazam loss-only `val_cur=0.708/0.857`, conditioned
    `0.704/0.855`; Dragapult loss-only `0.619/0.846`, conditioned
    `0.616/0.845`. Conditioning was active but did not improve current action.
- Same-turn/turn-next labels are learnable but not controlling the action head.
  The 2026-08-15 turn-next diagnostic had `turn_next_type_pos_acc` roughly
  `0.59-0.62` on current rows and attack prediction high, but conditioning on
  this signal did not improve current action accuracy or smoke rollouts.
- Opportunity/rank losses were negative or no-op. Type-level opportunity
  pressure, exact rank margin, and target-type margins either degraded
  `val_cur`/random smoke or changed only auxiliary metrics. They did not repair
  attach/evolve/attack misses.
- Multi-select remains rare and underpowered. Typical target count is close to
  one, with true multi rows often only a few percent. A row-level BCE/order loss
  cannot make the model understand turn-block or damage-allocation strategy by
  itself.
- Dragapult-specific DCA is still too shallow. DCA labels are focus-dominated;
  training DCA accuracy can become high while validation and RR remain weak.
  Per-step DCA supervision does not teach the broader prize-map/damage-plan
  required by Dragapult.
- Known opponent hand/reveal information is not currently proven useful. Some
  audits found `known_opp_rate=0`, so the corpus/extractor must be verified
  before any claim that the model remembers revealed cards.
- Random and local shadow RR are unreliable gates. Several models had
  near-perfect random and strong local RR but failed badly on Kaggle, proving
  the shadow pool is too weak or too unlike real submitted policies.

Signals expected before any long v15 training:

- Data integrity signals: per-deck rows by date/score/team, winner/loss split,
  opponent archetype coverage, target count distribution, same-turn block
  length, reveal/known-info nonzero rate, DCA focus/split/spread distribution,
  and leak-safe train/val split by game/team/date.
- Live-decision signals: current-row top1/topk, ambiguous-option top1, per-type
  action top1, attach/evolve/ability/attack miss rates, terminal-while-continue
  rate, target margin against best legal alternative, and multi-target exact/F1.
- Sequence-use signals: removing, zeroing, shuffling, or reversing historical
  events must change current logits. Track `delta`, `agree`, and KL for
  no-history/no-actions/reversed-history/no-known-info/no-plan ablations.
  A healthy sequence model should not have `revhist_agree > 0.90` on contexts
  where turn history is relevant.
- Ordered-plan signals: model should predict the next 2-4 actions in the same
  turn or next turn with stepwise type/card/context/attack metrics, and those
  plan tokens must measurably affect current action logits.
- Global resource signals: expose remaining deck/discard/prize/resource state,
  revealed opponent cards, used supporter/energy/retreat flags, prize race, and
  key engine status as audited fields. If a field is claimed useful, log its
  nonzero rate and ablation impact.
- Online proxies only after training signals pass: random, filtered RR, live
  ladder-pool RR, and Kaggle replay analysis should validate a model that has
  already shown real signal use, not act as the first diagnostic.

v15 rewrite requirements:

- Start a new branch from the current project state, but do not keep extending
  v14 with more auxiliary heads unless they are tied to action-logit probes.
- Design the corpus around canonical events and turn blocks, not only
  independent prefix rows. The unit of learning should include "what plan was
  executed over this turn/phase" and "which current legal action advances that
  plan".
- Build a clean model path for state tokens, option tokens, event-history
  tokens, known-opponent-card tokens, and plan tokens. The action scorer must be
  conditioned on the plan representation, not merely share an auxiliary loss.
- Include rule/plan hooks as first-class inputs or teachers. For structural
  weaknesses, pure BC imitation has not been enough; v15 should allow explicit
  strategic overlays and teacher/search-generated plans to be trained/evaluated.
- Add a small mandatory diagnostic-train target before any overnight run. It
  should finish quickly and fail loudly if labels are zero, losses are nonfinite,
  chronological ablation has no effect, or the action head ignores plan tokens.
- Current v14 local code contains work-in-progress ordered same-turn plan
  fields. Treat that as experimental scaffolding only. It is not yet proof that
  v14 is fixed; the likely next step is v15 rewrite after a compile/smoke check.

## Update: v14 Same-Turn And Opportunity Diagnostics 2026-08-15

Current direction from the user: stop treating random/RR as the first place to
discover broken ideas. Every new sequence/global/history signal must be visible
in the training log, and bad runs should be stopped from training metrics alone.
Avoid local loops around tiny hyperparameter tweaks; use diagnostics to decide
whether the pipeline actually changed the decision being made.

Completed diagnostic conclusions:

- Same-turn planner signal is valid but insufficient. `turn_continue` learned
  quickly (`turn_f1` around `0.98` on Alakazam), proving same-turn targets are
  extractable and trainable. However, the action head still ended turns during
  same-turn continuation opportunities around `8-11%`, and attach/attack misses
  stayed high. The planner head alone does not force the chosen legal action.
- `turn_terminal_weight` sweep did not solve this. We tried weights
  `0/0.5/1.0/1.5`; `pterm` shifted only slightly and attach/attack miss did not
  materially improve. Do not keep tuning terminal mass as the main lever.
- `opportunity_type_weight` is currently a negative result. In
  `logs/v14_opportunity_diag_20260815`, Alakazam epoch 1 already showed higher
  opportunity type weight reduced `val_cur` and did not improve attach:
  baseline `val_cur=0.718`, attach miss `0.418`; weight `0.5` had
  `val_cur=0.701`, attach miss `0.473`; weight `1.0` had `val_cur=0.691`,
  attach miss `0.494`; weight `2.0 + terminal0.5` had `val_cur=0.670`.
  Interpretation: the model often assigns some mass to the right action type,
  but the exact target action still loses to another legal option. Type-level
  pressure alone is too coarse.

Code changes now present locally and synced to ks:

- `ptcg_rl/seq/data.py`: sequence batches include same-turn continuation
  targets: `turn_continue`, `turn_remaining`, `turn_future_types`.
- `tools/v14_extract_sequences.py`: extracted rows now save `turn_number` and
  `turn_action_count`.
- `ptcg_rl/seq/model.py`: same-turn heads and conditioning exist; loss/metrics
  include `turn_plan`, `turn_terminal`, `opportunity_type`, and the new
  `opportunity_margin` objective. The margin objective penalizes key current
  targets (`attach/evolve/ability/attack`) when the target action logit is not
  above the best other-type legal action by a configurable margin.
- `tools/v14_train_sequence_policy.py`: training logs now print
  `oppMargin`, `val_oppMargin`, `val_oppMarginL`, and `signal_health`
  fields `cur_opp_margin` plus `opp_margin_loss`.
- `tools/v14_train_population.py`: population training now passes
  `--opportunity-margin-weight` and `--opportunity-margin` through to the
  sequence trainer.

Active/queued remote work:

- Active on ks: `/tmp/run_v14_opportunity_diag_20260815.sh`, log dir
  `logs/v14_opportunity_diag_20260815`, checkpoints
  `checkpoints/v14_opportunity_diag_20260815`. Four Alakazam 7f9 jobs are
  running on GPUs 0-3. This run started before the margin code was synced, so it
  does not contain `opportunity_margin` metrics.
- Queued on ks: `/tmp/run_v14_margin_diag_20260815.sh`. Start it only after the
  opportunity run finishes. It runs Alakazam `7f9a538936e3` and Dragapult
  `cc2e995b5ad0` baseline/margin pairs over
  `data/seq_corpus_v14_0801_0813_hq`, with log dir
  `logs/v14_margin_diag_20260815` and checkpoints
  `checkpoints/v14_margin_diag_20260815`.

2026-08-15 12:55 follow-up:

- The first margin diagnostic launch accidentally included `--amp`. Training
  logs immediately exposed a degenerate run: epoch 1 top1 around `0.38-0.44`,
  `next` near zero, turn prediction collapsed toward always-continue, and
  action type metrics were far below the previous no-AMP diagnostics. The bad
  run was stopped from training signals before waiting for RR.
- `tools/v14_train_sequence_policy.py` now refuses `--amp` by default. Use
  `--allow-amp` only for an explicit AMP repair experiment. This is a hard
  guard because AMP has repeatedly produced nonfinite or degenerate v14
  training signals.
- `/tmp/run_v14_margin_diag_20260815.sh` was fixed to no-AMP and relaunched.
  Early no-AMP batch metrics returned to the expected learning curve, so the
  immediate failure was AMP misuse, not the margin implementation itself.
- The no-AMP margin run was restarted once more with `--diagnostic-ablation`,
  because the first no-AMP restart omitted ablation and therefore could not
  validate history/ledger/order sensitivity. The current valid run is in
  `logs/v14_margin_diag_20260815` and should be interpreted from the run whose
  process command includes `--diagnostic-ablation`.
- Early epoch-1 valid margin signals: ablation fields are real again. Current
  action depends somewhat on history/ledger (`stateless_delta` and `noact_delta`
  are nonzero), but `revhist_delta` remains near zero, so chronological order is
  still weak for live current decisions. Opportunity margin weight `0.5`
  improves some attach/type margin numbers but lowers `val_cur`/ambiguous
  top-1 and hurts early random smoke, especially Dragapult. Treat
  opportunity-margin as a negative or at best inconclusive direction unless
  later epochs reverse this.
- New exact-rank diagnostic code has been added and synced to ks:
  `current_rank_margin_weight`, `current_rank_margin`, and
  `current_rank_margin_min_options`. This directly requires the current target
  action to beat the best non-target legal action, covering same-type target
  mistakes that type-level losses miss. Training logs now include
  `cur_rank_margin`, rank violation rate, ambiguous-target rank margin, and
  `rank_margin_loss`.
- Queued script for the next GPU-free slot:
  `/tmp/run_v14_rank_diag_20260815.sh`, log dir
  `logs/v14_rank_diag_20260815`, checkpoints
  `checkpoints/v14_rank_diag_20260815`. It compares baseline vs rank weight
  `0.35` on Alakazam `7f9a538936e3` and Dragapult `cc2e995b5ad0`, with
  `--diagnostic-ablation` and no AMP.

Final margin diagnostic result:

- `logs/v14_margin_diag_20260815` completed at
  `2026-08-15T13:22:03+08:00`; logs/checkpoints were pulled locally under
  `artifacts/ks_pull_20260815_v14_margin_diag/`.
- Alakazam baseline `m00`: epoch 3 `val=1.9605`, `val_top1=0.768`,
  `val_cur=0.762/0.877`, random smoke `46/48`.
- Alakazam opportunity-margin `m05`: epoch 3 `val=2.3129`,
  `val_top1=0.750`, `val_cur=0.744/0.851`, random smoke `44/48`.
- Dragapult baseline `m00`: epoch 3 `val=2.2010`, `val_top1=0.670`,
  `val_cur=0.680/0.862`, random smoke `35/48`.
- Dragapult opportunity-margin `m05`: epoch 3 `val=2.5581`,
  `val_top1=0.667`, `val_cur=0.661/0.836`, random smoke `29/48`.
- Conclusion: opportunity-margin made target-type margins look better but
  degraded the primary action objective and rollout smoke. Do not promote it.
  The next active run is exact rank margin, not opportunity margin.

Immediate interpretation rule:

- If margin weight improves `cur_attach/evolve/ability/attack_target_margin_other`
  and reduces corresponding `miss_if_target` without reducing `val_cur`,
  promote it to a broader multi-archetype diagnostic.
- If margin improves only the metric but not random smoke or ambiguous-current
  top1, the action-ranking problem is likely not type-vs-other; move to exact
  action-set/listwise supervision or grouped turn-block imitation.
- If history probes still show `revhist_delta≈0` and `noact_delta≈0`, the model
  remains mostly current-board BC despite sequence inputs. Do not run more large
  history jobs until the loss forces chronology to matter.

## Update: Training-Time Signal Diagnostics 2026-08-14

Latest v14 Kaggle diagnosis after three poor submissions:

- Current v14 submissions on `jie` were all poor despite some local metrics:
  - `55501985` `submission_oldv14_mega_lucario_43d6`: public `569.1`.
  - `55502017` `submission_oldv14_alakazam_7f9a`: public `799.4`.
  - `55505115` `submission_v14_lopunny_f144`: public `881.8`.
  Treat this as a pipeline-level failure, not a single-card matchup anomaly.
- Old v14 local random was misleading. Mega Lucario `43d6` had random g300
  `100.0%`, Alakazam `7f9a` had `99.7%`, and Lopunny `f144` had `99.7%`,
  yet Kaggle did not hold 900+. Random only proves the agent can beat weak legal
  play; it is not a quality gate for v14.
- Live-tail local RR was also misleading. Files:
  `logs/kaggle_live_20260814/oldv14_submit_analysis/rr_lucario_vs_live_tail_g40.csv`
  and `rr_alakazam_vs_live_tail_g40.csv` predicted oldv14 Lucario/Alakazam at
  about `98-99%` vs the extracted live opponent deck pool, while Kaggle scores
  were `569.1/799.4`. This means the current shadow/live-tail agents are too
  weak or too unlike real submitted policies. Do not use these local shadows as
  decisive submission selectors.
- Filtered old-v14 RR on local strong-random entries was internally coherent
  but not enough for Kaggle. Top local entries were:
  `Mega Lopunny f144` mean `0.702`, `Alakazam 7f9` mean `0.698`,
  `Marnie b8f` mean `0.667`; `Dragapult cc2e` was only `0.371`.
  Full matrix:
  `logs/v14_sequence_0808_0812/rr_v14_pop_filtered_random96_20260814/archetype_matrix.csv`.
- Label audit on `data/seq_corpus_v14_0801_0813_hq` shows why larger v14 models
  did not become strategic:
  - Dragapult `cc2e995b5ad0`: `569727` rows, row win rate `0.591`,
    `multi_rate=0.032`. DCA has `95796` rows but most groups are focus labels:
    `unique_slot_counts={1:74754, 2:18318, 3:2322, 4:372, 5:30}` and
    `focus_1.0=74754`. This teaches "put counters on one target" far more than
    tactical 60-damage allocation.
  - Alakazam `7f9a538936e3`: `1250512` rows but row win rate only `0.470`;
    many rows come from `600-899`. More data here is not automatically cleaner.
  - Marnie `b8f251a476e7`: `3562957` rows, row win rate `0.501`; all-band
    deck-sig BC directly imitates a large amount of losing/low-quality play.
  - Lopunny `f1445356c3a7`: `732811` rows, row win rate `0.563`, local random
    strong but Kaggle only `881.8`.
- DCA-specific training did not solve strategy. `v14dca_dragapult_cc2e_w512_dca6`
  reached training DCA top1 about `0.997`, but validation DCA top1 only about
  `0.861`; RR vs old-v14 filtered pool remained weak, e.g. vs Crustle
  `0.1625`, vs Marnie b8f `0.175`. This is stronger evidence that per-step DCA
  labels are insufficient. Need block-level/plan-level targets or search/rules,
  not just feature/weight tweaks.
- Practical conclusion for next work:
  - v14 must be diagnosed during training, not after random/RR.
  - Do not restart large v14 training unless the training log reports history
    sensitivity (`noact`/`revhist`), current-step metrics, ambiguous option
    metrics, DCA split/focus metrics, per-action-type top1, and nonfinite/grad
    checks.
  - Next architecture/data change should stop treating the sequence as many
    independent next-action rows. Use grouped tactical units for DCA and other
    multi-step resolutions, and add an objective that requires chronology to
    matter. If `revhist_delta≈0`, the model is still a single-step policy with
    decorative history.

Latest critical diagnostic result:

- The first no-guard v14 diagnostic run
  `logs/v14_signal_diag_20260814/diag_*.log` exposed two hard failures before
  random/RR:
  - `loss`/`val` became `nan` during epoch 1, driven by the order/ranking loss
    path.
  - The old sequence action loss supervised every prefix row almost equally;
    the live inference row had only about `0.039-0.040` raw decision weight.
    Validation ablation showed current-only input was essentially identical
    (`cur1_delta` near zero, `cur1_agree` around `0.95-0.99`), so the model was
    not measurably using history.
- A second smoke isolated a separate numerical failure: with `--amp`, the first
  backward pass produced `nonfinite_grad` even when order loss or DCA/multi
  boost were disabled. The same smoke without AMP completed. Until bf16/fp16 is
  explicitly repaired, **do not use `--amp` for v14 sequence training**.
- The current local code changes align the objective with live inference:
  `SequenceLossConfig` now has `current_action_weight`,
  `prefix_action_weight`, and `order_weight`. Action CE is dominated by the
  rightmost/current decision row; prefix CE is only a small auxiliary. Order CE
  is applied only on true `target_k>1` rows and logits are clamped for loss
  stability.
- `tools/v14_train_sequence_policy.py` now hard-stops on nonfinite loss or
  nonfinite gradient and logs `grad`/pre-clip gradient norm in progress lines.
  `signal_health` now includes `current_row_share`,
  `current_weight_share`, and `large_preclip_grad` warnings. Treat very large
  pre-clip gradients as a real training failure signal, not harmless noise.
- Active diagnostic runner on ks:
  - script: `/tmp/run_v14_lossfix_diagnostics_20260814.sh`
  - log dir: `logs/v14_lossfix_diag_20260814`
  - checkpoints: `checkpoints/v14_lossfix_diag_20260814`
  - jobs: Dragapult `cc2e995b5ad0`, Alakazam `7f9a538936e3`,
    Mega Lopunny `f1445356c3a7`, Marnie `b8f251a476e7`
  - config: no AMP, `width=256`, `layers=3`, `heads=8`, `batch=384`,
    `epochs=2`, `max_train_batches=160`, `max_val_batches=40`,
    `--diagnostic-ablation`, current/prefix/order weights `1.0/0.10/0.05`.
  - Completed status `0` at `2026-08-14T21:35:04+08:00`.
  - It showed no `nan` without AMP and aligned current-step supervision:
    `current_weight_share=0.909`, raw `current_row_share≈0.037-0.040`.
  - It also showed very large first-batch pre-clip gradients (`1e7` range),
    falling to small finite values after a few dozen batches. Keep the new
    `large_preclip_grad` warning for future runs.
  - Final diagnostic metrics, epoch 2:
    - Alakazam `7f9a`: `val_top1=0.756`, `val_cur=0.752/0.870`,
      `cur1_delta=-0.031`, `cur1_agree=0.843`, `m2F1=0.827`,
      warnings `ambiguous_options_weak,no_val_dca_signal`.
    - Dragapult `cc2e`: `val_top1=0.672`, `val_cur=0.683/0.867`,
      `cur1_delta=-0.050`, `cur1_agree=0.755`, `m2F1=0.769`,
      `dcaTop1=0.874`, `dcaSplit=0.796/0.891/0.425`, warning
      `attach_decision_weak`.
    - Mega Lopunny `f144`: `val_top1=0.653`, `val_cur=0.657/0.828`,
      `cur1_delta=-0.038`, `cur1_agree=0.831`, `m2F1=0.762`.
    - Marnie `b8f`: `val_top1=0.728`, `val_cur=0.731/0.899`,
      `cur1_delta=-0.037`, `cur1_agree=0.846`, `m2F1=0.767`.
  - Interpretation: the loss/numerical path is now trainable, and current-only
    ablation is no longer identical, but Dragapult attach remains weak and DCA
    labels are still focus-heavy. Do not claim strategic improvement from these
    diagnostics alone.
- Added next-generation training-time history probes to
  `tools/v14_train_sequence_policy.py` after this runner started. Future runs
  with `--diagnostic-ablation` now also print:
  - `val_noact=top1/delta/agree/kl`: zeroes action ledger/prev-action inputs
    while keeping board/hand history.
  - `val_revhist=top1/delta/agree/kl`: reverses historical prefix order while
    preserving the live/current row.
  Use these to distinguish "uses current board only" from "uses action ledger"
  and "uses chronological history".

Hard constraint from the user: do not wait for random/RR to discover whether a
new signal worked. Any new signal and any existing critical signal must be
visible during training, with enough train/validation metrics to stop a bad run
early. Treat this as a design requirement for all future training scripts.

The v14 DCA/focus training that was running on ks was paused before launching
full-data training. The old orchestrator was stopped; the full-date extraction
was allowed to finish. As of this update there are no active v14 training
processes on ks.

Completed extraction:

- Remote corpus: `data/seq_corpus_v14_0801_0813_hq`
- Log: `logs/v14_next_hq_20260814/extract_0801_0813.log`
- Result: `11,389,640` rows, all `15/15` zips completed. Note that the source
  directory contained `2026-07-24` and `2026-07-25` in addition to August
  files, so this corpus is not strictly August-only unless rebuilt through a
  symlink-filtered episode directory.

Code changes made locally and synced to
`ks:/data/jie/ptcg_rl_git_v7_baseline_20260804`:

- `ptcg_rl/seq/data.py`
  now keeps DCA block annotations in `SequenceBatch`: `dca_pos`, `dca_len`,
  `dca_remaining`, `dca_prior_*`, `dca_group_unique_slots`, and
  `dca_group_focus_frac`. `Corpus stats` now reports signal coverage before
  training: `target_k_mean`, `multi_target_rate`, `capable_multi_rate`,
  `dca_rate`, `dca_groups`, `dca_spread_rate`, `dca_focus_mean`, and
  `dca_unique_mean`.
- `ptcg_rl/seq/model.py`
  now reports signal-specific loss and accuracy metrics, including
  `weight_boost`, `dca_weight_share`, `multi_weight_share`,
  `dca_action/non_dca_action`, DCA count accuracy, DCA pred/target k, DCA
  focus/spread/prior statistics, and DCA first/late/spread top-1 splits.
- `tools/v14_train_sequence_policy.py`
  now prints `Signal stats` at startup, signal metrics on progress lines, and a
  per-epoch `signal_health` line with train/val gaps and warnings such as
  `single_step_dominated`, `dca_overfit_gap`, `dca_labels_focus_dominated`,
  `dca_underweighted`, and `dca_late_sequence_weak`.
- `ptcg_rl/seq/torch_policy.py`
  was updated so live/eval dummy batches still construct correctly with the new
  dataclass fields.

Smoke test on ks:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 tools/v14_train_sequence_policy.py \
  --corpus data/seq_corpus_v14_dca_0808_0813 \
  --archetype Dragapult \
  --score-bands 1000-1099 1100-1199 \
  --deck-sig cc2e995b5ad0 \
  --min-score 900 \
  --seq-len 32 \
  --width 128 \
  --layers 2 \
  --heads 4 \
  --batch-size 64 \
  --epochs 1 \
  --max-train-batches 3 \
  --max-val-batches 2 \
  --progress-every 1 \
  --multi-target-weight 3.0 \
  --damage-counter-weight 4.0 \
  --amp \
  --out /tmp/v14_signal_smoke_dragapult.pt
```

Smoke log: `logs/v14_signal_smoke_20260814/dragapult_cc2e_smoke.log`.
The smoke correctly surfaced `dca_rate=0.168`,
`dca_spread_rate=0.245`, `dca_focus_mean=0.926`, per-batch
`boost/dcaW/dcaA/dcaSplit`, and epoch warning
`warnings=dca_labels_focus_dominated`.

Interpretation of first DCA attempt:

- DCA code was present but too shallow. First DCA training showed `dca6` and
  `nomulti` behaved almost identically, so the added weights did not materially
  change optimization.
- Most Dragapult DCA labels are still single-slot/focus-heavy. In the smoke,
  `dca_focus_mean=0.926`; this means a naive "learn DCA" objective can just
  learn to pile counters on one target, not necessarily learn prize-map
  planning.
- Future DCA work should be block-level/strategy-level, not just per-step
  cross-entropy weighting. If training logs show `single_step_dominated` or
  `dca_labels_focus_dominated`, stop early and redesign the target or data
  selection before running full RR.

## Update: RR Long-Tail Coverage 2026-08-14

The user pointed out a concrete local-RR blind spot: a high-score Alakazam
submission lost early Kaggle climb games to Archaludon, but previous filtered RR
pools often excluded Archaludon and other low-frequency archetypes. Do not rank
future candidates solely on high-quality/high-random RR pools that omit tail
decks.

New tool:

```bash
python3 tools/build_coverage_rr_manifest.py \
  --policy-manifest <candidate_or_shadow_manifest.csv> \
  --random-audit <eval_manifest_random.csv> \
  --ladder-manifest <latest_ladder_pool>/pool_manifest.csv \
  --policy-per-archetype 2 \
  --random-tail-per-archetype 1 \
  --include-other \
  --out logs/<run>/coverage_rr_manifest.csv
```

This emits an eval-compatible manifest with two kinds of opponent rows:

- `source_kind=policy`: usable trained shadow/pop entries, filtered by random
  audit quality.
- `source_kind=random_tail`: live ladder deck entries played by legal-random.
  These are not strong shadows, but they force low-frequency archetypes such as
  Archaludon/Hop Trevenant/Iono/N's Zoroark/Raging Bolt into coverage RR so
  early-climb risks are visible instead of silently dropped.

Use the coverage manifest with:

```bash
python3 tools/eval_round_robin.py \
  --manifest logs/<run>/coverage_rr_manifest.csv \
  --games 80 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/<run>/coverage_rr_g80.csv
```

Then summarize by archetype and climb weights. Treat `random_tail` rows as risk
coverage, not as proof that a candidate beats a strong opponent model. If an
archetype repeatedly matters in Kaggle replay, train a real shadow/pop model for
that deck signature next.

Default population/shadow archetype lists were also expanded to match the
current extractor coverage by adding `Iono Bellibolt`, `N's Zoroark`, and
`Raging Bolt`.

## Workspaces

- Local repo: `/home/jie/Do/0_PTCG/bak/ptcg_rl_git`
- Local branch currently checked out in `/home/jie/Do/0_PTCG/bak/ptcg_rl_git`:
  `sequence-decision-pipeline-v14`.
  This branch intentionally starts a separate sequence-first pipeline instead
  of continuing to patch the old single-step BC stack.
  Older handoff entries and the remote workspace name still refer to
  `v7-baseline-20260804`; do not switch or rewrite branches unless the user
  explicitly asks.
- Old remote training repo before the 2026-08-13 ks reset:
  `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`
- Current remote recovery root after the 2026-08-13 ks reset: `ks:/data/jie`.
  Restore code to `/data/jie/ptcg_rl_git_v7_baseline_20260804` unless the user
  asks for a different directory.
- Remote original workspace also exists in older notes: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git`
- Current remote training data for active BC experiments:
  - Completed stable baseline: `data/bc_corpus_banded_v11_0701_0804`
  - Completed v12 history corpus: `data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12`
  - Current refreshed v12 corpus: `data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12`
- Current remote raw episodes after the reset should be
  `/data/jie/episodes_raw`. Older notes may mention
  `/home/jie/Do/0_PTCG/workspace/episodes_raw` or
  `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git/episodes_raw`; verify the
  intended path before launching extraction.

For long ad-hoc checks over SSH, first build a script under local `/tmp`, upload it to `ks:/tmp`, then execute it. Avoid large heredocs inside `ssh` commands; quoting already caused noisy failures.

## Recovery Note: ks Ephemeral Disk Loss 2026-08-13

The `ks` server unexpectedly powered off on 2026-08-13. Its workspace/data disk
is ephemeral, so remote corpora, checkpoints, logs, and `/tmp` scripts from the
active 2026-08-13 jobs should be treated as lost unless they were already
synced back to the local repo/machine.

Critical implications:

- Future long training jobs must periodically sync important outputs back to
  local persistent storage, especially `checkpoints/**`, eval CSVs, and runner
  logs.
- Raw Kaggle episode zip files should be treated as a local persistent cache.
  Prefer downloading once to local `/home/jie/Do/0_PTCG/raw_episode`, then
  using `scp`/`rsync` to restore `ks:/home/jie/Do/0_PTCG/workspace/episodes_raw`
  after a server rebuild. Avoid depending on remote Kaggle downloads as the
  only copy.
- Use only the `jie` Kaggle account for downloads/API calls. The local machine
  now defaults to `jie`, so plain `kaggle ...` is enough locally. Do not use the
  `by` Kaggle API. On ks, verify the active Kaggle config before running any
  API command.

Local raw episode cache after recovery:

- `/home/jie/Do/0_PTCG/raw_episode` now has all August 2026 zip files from
  `2026-08-01` through `2026-08-12`.
- It also currently has only partial July coverage locally: `2026-07-24` through
  `2026-07-27`. July still needs refill for `2026-07-01..2026-07-23` and
  `2026-07-28..2026-07-31` if reproducing all-date v11/v12/v13 experiments.
- A local July refill script was started at
  `/tmp/download_july_episodes_20260813.sh`, logging to
  `/home/jie/Do/0_PTCG/raw_episode/download_july_20260813.log`. Recheck the
  log and `ls -lh /home/jie/Do/0_PTCG/raw_episode/pokemon-tcg-ai-battle-episodes-2026-07-*.zip`
  before assuming July is complete.
- Uploading 20+ large zip files from local to ks was too slow. The preferred
  short-term restore path is to run `kaggle datasets download` directly on ks
  under `/data/jie/episodes_raw`, using only the jie Kaggle config.
- User will handle slow episode zip downloads manually because Kaggle has
  throttled large downloads. Do not start more bulk episode downloads unless
  asked. It is still useful to pull small Kaggle metadata such as submissions,
  leaderboard CSVs, topics, and selected replay JSONs.

Remote restore state at `2026-08-13 17:20`:

- Code archive from local branch `dragapult-pipeline-v13` was uploaded to
  `ks:/data/jie/ptcg_rl_git_v13_code_20260813.tar` and extracted into
  `/data/jie/ptcg_rl_git_v7_baseline_20260804`.
- Auxiliary small logs/decks archive was uploaded as
  `ks:/data/jie/ptcg_v13_aux_logs_20260813.tar` and extracted into the same
  repo. Large corpora/checkpoints were not restored.
- Remote `py_compile` passed for:
  `tools/bc_extract_v2.py`, `tools/bc2_train.py`, `ptcg_rl/model.py`,
  `ptcg_rl/numpy_policy.py`, and `ptcg_rl/bc2/data.py`.
- All GPU memory was free when checked after reset. Recheck `nvidia-smi` before
  launching training and increase batch/CUDA caps if A800 memory is idle.

Active remote restore job started `2026-08-13 18:19`:

- User restored August episode zips on ks under `/data/jie/episodes_raw`.
  The files `2026-08-01` through `2026-08-12` were verified present and
  full-sized. A v13 extraction/training recovery job is running from
  `/data/jie/ptcg_rl_git_v7_baseline_20260804`.
- The `cg` engine was missing after the ks reset. It was restored by uploading
  local `data/sample_submission/sample_submission/cg` to the remote repo root.
  Remote smoke test succeeded with `from cg.api import all_card_data` and
  `1267` cards.
- Long-run script:
  `ks:/tmp/run_restore_0812_dragapult_v13_20260813.sh`.
- Main log:
  `logs/restore_0812_dragapult_v13_20260813/runner.log`.
- The script uses a symlink-only episode directory
  `/tmp/ptcg_episodes_0801_0812_v13_20260813` so the v13 corpus is August-only
  even if `/data/jie/episodes_raw` also contains July files.
- Leaderboard snapshot used for the 0812-ish shadow RR recovery:
  `logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv`.
  There was no saved true 2026-08-12 public leaderboard snapshot after the
  reset, so this latest saved snapshot is the recovery source.
- Corpus target:
  `data/bc_corpus_banded_v13_0801_0812_state_token_20260813`.
  Extraction command uses `--action-history-k 32 --log-history-k 128
  --board-history-k 12 --board-history-feat-dim 32 --workers 12`.
- Dragapult v13 manifest:
  `logs/restore_0812_dragapult_v13_20260813/dragapult_v13_train_manifest.csv`.
  It contains four `cc2e995b5ad0` Dragapult jobs: cross-attn width 3
  state-token, width 4 state-token, width 4 no-state-token control, and recent
  `0805-0812` width 4 state-token.
- 0812 shadow manifest target:
  `logs/restore_0812_dragapult_v13_20260813/shadow_manifest_0812_v13_cross_w3.csv`.
  It is built with `tools/build_shadow_pool.py` top 96 / top-per-archetype 8 /
  max-per-deck-sig 3 and emits v13 train args including state-token dim 24,
  raw-policy loss, sequence loss, `--best-metric policy_raw`, and
  `--split-by-game`.
- `tools/build_shadow_pool.py` was updated before the manifest build to preserve
  climb context in shadow outputs. New manifests include `source_score`,
  `score_label`, and `score_tier`, and `shadow_name` defaults to a score prefix
  such as `shadow_s1218_dragapult_cc2e995b_flg`. This matters because RR CSVs
  otherwise keep only row/column names and lose the source team's score band.
- Training runner logs after extraction/manifest build:
  `logs/restore_0812_dragapult_v13_20260813/dragapult_train_runner.log` and
  `logs/restore_0812_dragapult_v13_20260813/shadow_train_runner.log`.
  Dragapult uses GPUs `0,1`; shadow uses GPUs `2,3` with two jobs per GPU.
- v13 extraction completed successfully: `729` npz files. Smoke sample showed
  `state=(80,)`, `option=(3,64)`, `state_token=(37,24)`, and feature version
  `v13_state_token_multistream_history`.
- 0812 ladder pool was rebuilt at `logs/ladder_pool_0812_all_v13_20260813`.
  It selected `220` decks and wrote deck CSVs under its `decks/` directory.
  Top entries included Marnie `b8f251a476e7` score `1230.3`, Dragapult
  `cc2e995b5ad0` score `1218.2`, Mega Lucario `43d6d8b0fce9` score `1230.3`,
  Alakazam `7f9a538936e3` score `1204.1`, and Crustle `47756cdfd20f` score
  `1218.2`.
- Shadow manifest generation completed with `81` rows. Training runners were
  launched at `2026-08-13T18:58:17+08:00`:
  Dragapult runner PID `356850`, shadow runner PID `356852`.
- `tools/enrich_shadow_manifest.py` was added to join a shadow manifest against
  a ladder `pool_manifest.csv`. It preserves both trajectory/team source score
  and deck-signature pool score. The enriched 0812 shadow manifest is:
  `logs/restore_0812_dragapult_v13_20260813/shadow_manifest_0812_v13_cross_w3_enriched.csv`.
  It matched `80/81` rows to the 0812 ladder pool. Use this enriched file for
  climb-aware analysis, especially when a concrete team trajectory has a lower
  score than the best current deck-sig result.
- At the time of this update, Dragapult v13 training had started two jobs:
  `cross_w3_state_token` and `cross_w4_state_token`, both on `cc2e995b5ad0`
  with `438634` kept decisions from August 1-12. Shadow training had started
  the first four entries. Logs already showed epoch progress, so the runners
  were not stuck at data loading.
- The old failed `ModuleNotFoundError: cg` remains earlier in the appended
  runner log; check the latest timestamped block, not only the first traceback.

Was July data used by the lost jobs?

- Marnie history-summary repair: yes. The active 2026-08-13 Marnie experiment
  used `data/bc_corpus_banded_v12_0701_0811_merged_hist32_log128_board12_20260812`
  plus `data/bc_corpus_high900_winners_20260812`, so reproducing it requires
  July 1 through August 11 episode coverage.
- Dragapult v13 state-token pipeline: the launched v13 extraction was
  `0801-0812` only, so that specific lost v13 extraction did not require July.
  Earlier Dragapult corpus audits and specialist sweeps did use the v12 merged
  `0701-0811` corpus, so reproducing those older comparisons does require July.

Handoff recoverability:

- The handoff has enough information to recreate the important experiment
  recipes, code changes, corpus names, and intended output paths.
- The exact trained model weights from the lost remote jobs cannot be recovered
  from handoff alone unless they were synced locally before shutdown. The last
  observed Marnie history-summary state before loss was: four of five branches
  completed, `cross_plan_h8_w4` was in epoch 6/6. No final eval files had begun
  because the eval watcher was still waiting for all training processes.
- The lost Dragapult v13 job was still in the extraction/watch stage at last
  known state; no reliable completed v13 training/eval artifacts should be
  assumed to exist.

Do not restart the Marnie history-summary repair until July `2026-07-01..31`
and August `2026-08-01..11` data have been restored, because that run depends
on the all-date v12 history corpus. Record explicitly: as of this note, the

## Update: v14 HQ Round 2026-08-14

Goal for the current session:

- Rebuild a higher-quality sequence corpus and retrain key deck-specific
  sequence policies without touching the locked BC baseline stack.
- Keep GPU usage high while the ks workspace is live again.

Current remote orchestrator:

- Script: `ks:/tmp/start_next_v14_hq_round.sh`
- Main log: `logs/v14_next_hq_20260814/orchestrator.nohup.log`

What is running now on `ks`:

- Full sequence extraction:
  `python3 tools/v14_extract_sequences.py /data/jie/episodes_raw --out data/seq_corpus_v14_0801_0813_hq --lb-csv logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv --workers 10 --future-horizon 16 --min-rows 100`
- Immediate high-quality DCA-focused training on existing corpus:
  - Dragapult `cc2e995b5ad0`
  - Alakazam `7f9a538936e3`
  - Mega Lopunny `f1445356c3a7`
  - Mega Lucario `43d6d8b0fce9`
  - Marnie `b8f251a476e7` is queued and will start when a GPU frees up

Current immediate-train settings:

- Corpus: `data/seq_corpus_v14_dca_0808_0813`
- Width/layers/heads: `512 / 5 / 8`
- Batch size: `1024`
- Epochs: `8`
- Loss weights emphasize high-quality sequence signals:
  `win=1.8 loss=0.20 draw=0.8 plan=0.60 multi_target=3.0 damage_counter=4.0`

Follow-up phase already scripted:

- Once extraction and the immediate focus train both finish, the orchestrator
  will build:
  `logs/v14_next_hq_20260814/train_manifest_full0801_0813_high900_top2.csv`
- Then it will launch full-date training on
  `data/seq_corpus_v14_0801_0813_hq` with the same high-quality settings.

Important note:

- The current extraction scans every zip in `/data/jie/episodes_raw`, which at
  the moment includes `2026-07-24` and `2026-07-25` in addition to
  `2026-08-01..2026-08-13`.
- If strict `0801..0813` only is required later, narrow the input set before
  re-running extraction.
post-reset Marnie history repair has not been retrained.

## Submission Stability Analysis 2026-08-13

Latest `jie` Kaggle submissions were queried with the jie API only. The newest
two 2026-08-13 submissions are repeats of 2026-08-12 packages:

- `submission_lossopt_alakazam_7f9_20260812`: 929.4 on 2026-08-12, repeated
  as 806.8 on 2026-08-13.
- `submission_high900win_old_marnie_b8f_20260812`: 953.4 on 2026-08-12,
  repeated as 890.1 on 2026-08-13.

Small info pull saved locally:

- Summary:
  `logs/info_pull_20260813/info_pull_summary.md`.
- Submissions snapshot:
  `logs/info_pull_20260813/jie_submissions_20260813.csv`.
- Leaderboard:
  `logs/info_pull_20260813/leaderboard_top200_20260813.csv` and full
  `logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv`.
- Topic index:
  `logs/info_pull_20260813/topics_time_filtered_20260813.md`.
- Topic details:
  `logs/info_pull_20260813/topic_*.json`.
- Lightweight episode metadata:
  `logs/info_pull_20260813/episodes_meta/*.json`.
- Selected replay pulls for latest repeats:
  `logs/info_pull_20260813/replay_55475933_marnie_rows.csv`,
  `logs/info_pull_20260813/replay_55475933_marnie_by_deck.csv`,
  `logs/info_pull_20260813/replay_55478159_alakazam_rows.csv`,
  `logs/info_pull_20260813/replay_55478159_alakazam_by_deck.csv`.
- Opponent deck CSVs from those replay pulls:
  `logs/info_pull_20260813/opponent_decks/`.

Current leaderboard snapshot at `2026-08-13T09:36:23`:

- full rows: 6791 teams.
- score >= 1200: 4 teams.
- score >= 1100: 22 teams.
- score >= 1000: 108 teams.
- score >= 900: 394 teams.
- score >= 800: 986 teams.
- score >= 700: 2180 teams.
- score >= 600: 3688 teams.
- top 1200+ teams: Luca, flg, Dominic Peel & Rory Neville, ANDPAD kaggler team.

Latest replay pull results:

- `55475933` Marnie high900 repeat: analyzed 25 non-mirror-identifiable games,
  `16-9`, WR `0.640`. Same-deck mirrors were skipped when deck matching could
  not identify our side. Chronological losses include Ogerpon `5899c772bace`,
  Ogerpon `697a82e582d5`, Crustle `96d572411df0`, and unknown/new signatures
  `ae759424cfd9`, `22aed7613909`, `90abbfb0eee0`, `56ff24f629ed`,
  `82a7789211b6`, `f2c3693911b6`.
- `55478159` Alakazam lossopt repeat: analyzed 17 non-mirror-identifiable
  games, `13-4`, WR `0.765`. Losses were to unknown/new signatures
  `a92ef7135802`, `2878a7405032`, and Marnie b8f twice.

Topic timing rule:

- Current-meta claims should use topics on or after `2026-08-02`, preferably
  `2026-08-08` or later.
- Late-July topics are method/background unless validated against 0809+ episode
  or replay data.
- June and early-July topics are durable only for official rules, engine
  behavior, API limits, and broad method lessons. Do not treat them as current
  meta evidence.

This strongly supports the user's observation that public score is unstable and
early climb matters. A package that can sit near 950 can still fail early when
matched into another strong fresh submission climbing from 600. A single mean
RR score is therefore not sufficient; validation must estimate low-tail climb
risk.

## v14 Sequence-First Pipeline 2026-08-14

User explicitly requested a new branch and aggressive redesign because the old
BC pipeline is too focused on single next-step decisions. The new v14 branch is
not a micro-tweak of `bc2_train.py`.

New files:

- `ptcg_rl/seq/constants.py`
- `ptcg_rl/seq/features.py`
- `ptcg_rl/seq/data.py`
- `ptcg_rl/seq/model.py`
- `ptcg_rl/seq/torch_policy.py`
- `ptcg_rl/policy_loader.py`
- `tools/v14_extract_sequences.py`
- `tools/v14_train_sequence_policy.py`
- `tools/v14_train_population.py`
- `tools/v14_audit_sequence_corpus.py`
- `tools/v14_probe_sequence_policy.py`
- `tools/v14_eval_random.py`
- `tools/v14_build_train_manifest.py`
- `docs/12_sequence_pipeline_v14.md`

Changed existing files:

- `main.py`: can load `policy.pt`/`policy.pth` through `policy_loader`, falling
  back to `policy.npz`.
- `tools/package_submission.py`: copies `.pt` checkpoints as `policy.pt`.
- `tools/eval_bc.py` and `tools/eval_round_robin.py`: load `.npz` legacy or
  `.pt` v14 policies through `ptcg_rl/policy_loader.py`.

Design:

- Corpus unit is now a game/window, not one independent decision row.
- Each step stores current board/hand/options, explicit ledger features, previous
  action event, action label, action type label, outcome, and a future-plan
  label over the next horizon.
- Model is `SequencePolicyNet`, a causal Transformer over decision tokens.
  It trains action, ordered multi-select, selection-count, action-type,
  future-plan, and outcome heads.
- The future-plan/outcome heads are deliberately part of the objective so the
  representation has pressure to encode long-game behavior.

Important implementation note:

- A first smoke train produced NaN validation loss because left padding plus
  causal mask plus key-padding mask caused all-masked attention rows. This was
  fixed by not passing `src_key_padding_mask` to the sequence encoder; padded
  input/output tokens are explicitly zeroed and loss uses `step_mask`.
- The 2026-08-14 audit found and fixed three hard defects that could make v14
  look ineffective even if training ran:
  1. Initial multi-select inference had no count head, so live play defaulted
     to one option even when labels selected multiple options.
  2. Initial multi-select training had set BCE but no ordered target; v14 now
     stores `target_order` and trains per-position option logits.
  3. `TorchSequencePolicy.select()` initially did not accept legacy
     `temperature` kwargs from `main.py`, and `select_mcts()` did not exist.
     Both are now compatible; `select_mcts()` is only a greedy shim, not real
     MCTS.
- The same audit found an RR infrastructure bug: `tools/eval_round_robin.py`
  parallel workers were still trying to call an undefined `NumpyPolicy.load`.
  They now use `ptcg_rl.policy_loader.load_policy()`, which is required for
  `.pt` v14 checkpoints and fixes `--workers > 1`.
- New audit file: `docs/13_v14_sequence_audit_checklist.md`.
- New integrity checker: `tools/v14_check_sequence_integrity.py`.
- Training telemetry was expanded because user explicitly requested progress
  and in-training diagnostics by default. `tools/v14_train_sequence_policy.py`
  now prints batch totals, seen samples, ETA, CUDA allocated/reserved memory,
  loss parts, top1/type/count accuracy, count MAE, predicted/target selection
  count, set-F1, ordered-selection accuracy, and outcome accuracy. Treat flat
  `setF1`, bad `pred_k/target_k`, or stagnant `orderAcc` as early stop signals.
- `tools/v14_train_population.py` now passes loss-head weights through to
  single-model training and prints the latest metric line from each running
  job's log on every poll.
- `tools/v14_audit_sequence_corpus.py` and
  `tools/v14_check_sequence_integrity.py` now support progress output; long
  audit commands should use `--max-rows` or visible progress so they do not
  silently block GPU training.
- Two additional hard bugs were found only after CUDA/AMP population launch:
  1. Masking logits with `-1e9` overflows fp16 under AMP. `NEG_INF` is now
     fp16-safe (`-1e4`).
  2. The initial ordered multi-select scorer expanded `[B,T,K,N,W]` activations
     and OOMed with two jobs on one A800. It now uses a low-memory
     position-conditioned dot-product scorer and keeps only `[B,T,K,N]` order
     logits.

Active v14 remote training as of 2026-08-14 11:29 Asia/Shanghai:

- Running script:
  `ks:/tmp/run_v14_population_parallel3_20260814.sh`.
- Main log:
  `logs/v14_sequence_0808_0812/pop_parallel3.nohup.log`.
- Runner log:
  `logs/v14_sequence_0808_0812/train_pop_top2_allbands_parallel3.runner.log`.
- Checkpoints:
  `checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3`.
- Per-job logs:
  `logs/v14_sequence_0808_0812/pop_top2_allbands_parallel3/*.log`.
- Config: v14 sequence corpus `data/seq_corpus_v14_0808_0812`, manifest
  `logs/v14_sequence_0808_0812/train_manifest_top2_allbands.csv`, 23 jobs,
  8 running slots (`jobs-per-gpu=2`), `seq_len=32`, `width=384`, `layers=4`,
  `heads=6`, `batch_size=256`, `epochs=8`, AMP on.
- First healthy poll showed all 8 jobs running with no failures and memory
  around 25-27 GiB per GPU. Telemetry lines include `pred_k/target_k`,
  `setF1`, `orderAcc`, `outAcc`, ETA, and CUDA memory.
- Ignore failed attempts under:
  `pop_top2_allbands_parallel` and `pop_top2_allbands_parallel2`.
  They failed before useful training because of the fp16 mask and high-memory
  order-head bugs.

Local smoke validation:

- Command used local `PYTHONPATH=/home/jie/Do/0_PTCG/bak/para` because this
  local repo does not directly contain `cg`.
- `tools/v14_extract_sequences.py` ran over local raw episodes with
  `--max-episodes 20 --min-rows 1` and wrote `/tmp/ptcg_v14_seq_smoke`.
- Smoke extraction result: `52099` rows, `bad=0`, `err=0`.
- Smoke Marnie CPU train ran one epoch with only 5 train batches and saved
  `/tmp/ptcg_v14_smoke_marnie.pt`; validation was finite after the mask fix.
- Smoke probe correctly showed the tiny model had not learned prefix dependence:
  `last_only`, `last_zero_prefix`, and `last_shuffle_prefix` were nearly equal.
  This is expected for smoke and proves the probe can detect non-use of sequence.

Required v14 acceptance standard before claiming improvement:

- Run `tools/v14_probe_sequence_policy.py` on real trained checkpoints.
- Full/base or longer-prefix metrics must beat `last_only`.
- `last_zero_prefix_zero_ledger` and `last_shuffle_prefix` should be meaningfully
  worse than full/base for models that truly use sequence.
- Plan loss should improve with longer prefixes. If it does not, the model is
  still acting mostly as a single-step classifier.

Canonical remote extraction:

```bash
python3 -u tools/v14_extract_sequences.py /data/jie/episodes_raw \
  --out data/seq_corpus_v14_0801_0812 \
  --lb-csv logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv \
  --workers 12 \
  --future-horizon 12 \
  --progress-every 500 \
  2>&1 | tee logs/v14_sequence/extract_v14_0801_0812.log
```

Canonical single-model train:

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

Population manifest + runner:

```bash
python3 tools/v14_build_train_manifest.py \
  --corpus data/seq_corpus_v14_0801_0812 \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --top-per-archetype 2 \
  --min-rows 5000 \
  --out logs/v14_sequence/train_manifest_top2.csv

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

Do not submit v14 until local random/RR and sequence-probe results justify it.
The code can package `.pt`, but Kaggle runtime/size has not been validated.

Stable-ish families from `jie` submissions:

- Best repeated family remains Marnie `b8f251a4` old/w4/high900/v12hist:
  visible scores include 986.2, 975.4, 964.6, 959.1, 953.4, 946.2, 931.9, but
  repeats also show 916.7, 890.1, 838.6. This is the most recoverable baseline,
  but not robust enough for 1200+ without reducing early-loss risk.
- Crustle has mid-stable results around 903/904 for some signatures, but no
  clear 1000+ evidence from jie submissions.
- Alakazam has strong data volume and one 929.4 result, but repeat 806.8 and
  high1100 833.1 indicate a poor low-tail profile or bad matchup coverage.
- Ogerpon `v7sig` was strong on 2026-08-03/04 around 959/966, but recent
  resubmits fell to 672/715; treat it as environment- or signature-stale.
- History/sequence/trajectory submissions have not shown a stable improvement:
  `historyk_marnie` 916.7, `v12_seqplan_festival` 813.9,
  `v12_seqplan_lucario` 634.0, `ogerpon_v9_gameplan` 545.0. Keep these as
  architectural ablations, not submission defaults.

Training/validation implication:

- Optimize for early-climb survival, not only final average. Add a validation
  report that simulates the first 7-10 ladder games against a weighted
  600-1100 live-climb opponent pool built from the last 3-5 episode days and
  recent replay opponents. Report probabilities of `7-0`, `6-1`, `5-2`, and
  any early loss in first 3 games.
- Keep a filtered high-quality RR pool, but add a separate "fresh submit"
  opponent pool. New strong submissions start at 600, so low score bands are
  not purely weak opponents.
- Candidate selection should use a low-tail metric: minimum/worst-quartile
  archetype score, CVaR over opponent entries, and early-climb weighted score.
  Do not rank solely by mean RR.
- For training, avoid small finetunes as the default. If changing BC, prefer
  full retrains with explicit controls and large structural changes. For
  Dragapult v13, use aggressive controlled comparisons rather than tiny
  parameter nudges.

## Dragapult v13 Restore Plan After ks Reset

Priority after `/data/jie/episodes_raw` has August data restored:

1. Re-extract v13 corpus for August `2026-08-01..2026-08-12`:

```bash
cd /data/jie/ptcg_rl_git_v7_baseline_20260804
python3 -u tools/bc_extract_v2.py /data/jie/episodes_raw \
  --out data/bc_corpus_banded_v13_0801_0812_state_token_20260813 \
  --workers 12 \
  --action-history-k 32 \
  --log-history-k 128 \
  --board-history-k 12 \
  --board-history-feat-dim 32 \
  --progress-every 200
```

`bc_extract_v2.py` extracts all available zip files under the episodes
directory; score/date filtering happens in `tools/bc2_train.py`.

2. Recreate the Dragapult specialist sweep manifest and run three first-line
models plus controls:

- `cross_w3_state_token`: cross-attn, width 3, state-token dim 24.
- `cross_w4_state_token`: cross-attn, width 4, state-token dim 24.
- `cross_w4_no_state_token`: same as above but `--state-token-feat-dim 0`,
  the required control to prove state-token extraction matters.
- Optional larger/control branches if GPUs remain idle: raw-policy best metric,
  recent-only 0805-0812, Kh0a team-specific, and top-sig mix.

Shared Dragapult settings for the first restore run:

- Corpus:
  `data/bc_corpus_banded_v13_0801_0812_state_token_20260813`.
- Archetype/signature:
  `--archetype "Dragapult" --deck-sig cc2e995b5ad0`.
- Score/date:
  `--score-bands 1200+ 1100-1199 1000-1099 900-999 --date-from 2026-08-01 --date-to 2026-08-12`.
- Weights:
  `--win-weight 1.5 --loss-weight 0.4 --draw-weight 0.8 --include-empty --split-by-game`.
- Set/multi-select emphasis:
  `--set-loss-weight 0.8 --set-loss-min-count 2 --set-loss-negative-weight 0.15 --multi-select-sequence-weight 0.25`.
- Hardware:
  A800s can likely handle `--batch-size 1536` to `2048` and
  `--cuda-memory-gb 32` to `48`; check `nvidia-smi` first.

Minimum eval after each restored v13 checkpoint:

- random g500 on `logs/ladder_pool_0805_all/decks/cc2e995b5ad0_dragapult_flg.csv`;
- filtered high-quality RR matrix, not an unfiltered weak pool;
- fresh-climb weighted pool from recent 600-1100 opponents;
- first-10-game climb-risk metric.

If v13 is still close to old Dragapult results, treat it as a bottom-pipeline
problem, not a data shortage: inspect encoding of bench damage counters, target
choice, attack selection, evolution lines, and whether multi-select labels are
represented as an order-free set where appropriate.

## Marnie History Repair Restore Config

This job was lost in the ks reset and has not been retrained after reset.
Restore only after the all-date v12 history corpus can be rebuilt.

Required data:

- Main corpus:
  `data/bc_corpus_banded_v12_0701_0811_merged_hist32_log128_board12_20260812`.
- Aux corpus:
  `data/bc_corpus_high900_winners_20260812`.
- Deck:
  `logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv`.
- Init checkpoint:
  `checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_marnie_grimmsnarl_b8f251a4_w4.npz`.
  If this checkpoint is unavailable after reset, first retrain or restore the
  old high900win b8f baseline before running history repair.

Key intended branches:

- `hist_summary_pointer_h0_fixed`: summary-only pointer control.
- `hist_summary_pointer_raw_h8`: pointer + raw h8 + summary.
- `hist_summary_cross_raw_h8`: cross-attn + raw h8 + summary.
- `hist_summary_cross_plan_h8`: cross-attn + raw h8 + summary + step plan and
  hierarchical plan.

Most important settings:

```bash
--history-summary \
--history-gate-init-bias -2.0 \
--history-sensitivity-weight 0.15 \
--history-sensitivity-margin 0.05 \
--history-augment \
--history-stream-drop-prob 0.10 \
--history-event-drop-prob 0.05 \
--history-tail-drop-prob 0.20
```

Do not submit a history checkpoint merely because validation loss improves.
It must beat the old high900win b8f baseline on random, filtered RR, live
opponents, history gate inspection, and early-climb risk. Previous history
submissions did not prove stable live improvement.

Kaggle mirror/download note:

- No trusted official Kaggle mirror is known for these dataset zips. Kaggle CLI
  officially supports `kaggle datasets download`; third-party mirror/proxy
  tricks are not reliable enough for the competition data chain.
- Practical workaround is local caching plus restore-to-ks sync, not relying on
  a public mirror. Kaggle download speed can drop after repeated large files.

## Active Remote Job: Dragapult Specialist Sweep 2026-08-13

User asked to investigate why Dragapult has been one of the weakest trained
archetypes and to start a parallel specialist sweep without repeating known bad
settings.

Important Kaggle/API constraint:

- Use only the jie Kaggle account for any future `kaggle` CLI/API call.
- Do not call the by Kaggle API. Static by logs may exist, but current audits
  should avoid them unless the user explicitly asks.

Trajectory/history/plan evidence audit:

- Script uploaded/run on ks:
  `/tmp/summarize_jie_trajectory_effect_20260813.py`.
- Outputs:
  `logs/dragapult_specialists_20260813/history_effect_jie_only/summary.md`,
  `kaggle_score_grouped_traj_history_plan_rl_jie.csv`,
  `kaggle_score_grouped_related_baselines_jie.csv`,
  and `local_traj_history_eval_summary.csv`.
- Read only jie sources:
  `logs/kaggle_submission_scores.csv` and refreshed current submissions from
  jie via `KAGGLE_CONFIG_DIR=/root/.kaggle`.
- Conservative result:
  - Marnie `v12hist_bigbatch` had good but not clearly superior Kaggle scores
    (visible 975/959, one 839), while non-history w4/high900 Marnie also had
    986/965/945-class results.
  - `historyk_marnie` around 917, `historyk_lucario` around 644,
    `v12_seqplan_festival` around 814, `v12_seqplan_lucario` around 634,
    `ogerpon_v9_gameplan` weak, and early `rl: dragapult with mcts50` around
    178.
  - Local weakup/success/trajectory/RL CSVs do not show stable broad positive
    deltas; Dragapult success/trajectory variants remained poor.
- Interpretation:
  trajectory mining is diagnostic and should be kept as an ablation, not the
  main trusted improvement path until it beats strong non-history baselines in
  random and coverage RR.

Dragapult corpus audit:

- Corpus:
  `data/bc_corpus_banded_v12_0701_0811_merged_hist32_log128_board12_20260812`.
- Script:
  `/tmp/audit_dragapult_corpus_20260813.py`.
- Output dir:
  `logs/dragapult_specialists_20260813/corpus_audit`.
- Key result: `cc2e995b5ad0` has enough data, so failure is not simple sample
  shortage:
  435,029 decisions, 3,856 games, 20 dates, top team Kh0a, game WR about 0.57.
- Strong cc2e teams have enough samples for style-purity tests:
  Kh0a 67,116 decisions / 582 games, 213tubo 64,116 / 555,
  flg 62,446 / 558.

Trajectory mining for Dragapult cc2e:

- Command already run:
  `tools/mine_strategy_trajectories.py` over Dragapult cc2e 900+ all dates.
- Output dir:
  `logs/dragapult_specialists_20260813/trajectory/cc2e_all_900plus`.
- Useful correlations, not proof:
  winning games have more Dragapult ex board/active turns, more Drakloak bench
  turns, fewer early END decisions, and high positive gaps for
  `primary_board_by_6`, `board_by_4_drakloak`, and Munkidori/Drakloak ability
  action patterns.

Parallel training currently running on ks:

- Manifest generator:
  `/tmp/create_dragapult_specialist_manifest_20260813.py`.
- Train manifest:
  `logs/dragapult_specialists_20260813/train_manifest.csv`.
- Eval manifest:
  `logs/dragapult_specialists_20260813/eval_manifest.csv`.
- Training runner:
  `python3 -u tools/train_shadow_manifest.py
  logs/dragapult_specialists_20260813/train_manifest.csv --gpus 0,1,2,3
  --jobs-per-gpu 1 --cuda-memory-gb 24 --log-dir
  logs/dragapult_specialists_20260813/train --poll-seconds 60 --skip-existing`.
- Runner log:
  `logs/dragapult_specialists_20260813/train_runner.log`.
- Per-job logs:
  `logs/dragapult_specialists_20260813/train/*.log`.
- Checkpoints:
  `checkpoints/dragapult_specialists_20260813/*.npz`.

Training jobs:

- `dragapult_cc2e_all900_old_w4_ep16`: main scratch old-recipe baseline,
  full 900+ all dates.
- `dragapult_cc2e_all900_rawmetric_w4_ep16`: same data, extra raw policy loss
  and `--best-metric policy_raw`.
- `dragapult_cc2e_recent0805_0811_w4_ep14`: recent-meta-only window.
- `dragapult_cc2e_high900aux_old_w4_ep14`: extends prior best high900
  oldmethod recipe from 8 to 14 epochs.
- `dragapult_cc2e_trajplan_w4_ep12`: trajectory/step-plan ablation only.
- `dragapult_cc2e_histsummary_w4_ep12`: history-summary ablation only.
- `dragapult_cc2e_all900_old_w6_ep12`: larger pointer model.
- `dragapult_cc2e_kh0a_all900_w4_ep14`: team-specific Kh0a style-purity test.
- `dragapult_7ac_all900_w4_ep14`: second current/high Dragapult signature.
- `dragapult_topsig_mix_all900_w4_ep12_on_cc2e`: risky multi-sig mix, also
  evaluated on a 7ac deck through an eval alias.

At launch, first four jobs indexed successfully:

- `cc2e_all900` variants: 402,063 kept decisions.
- `recent0805_0811`: 252,212 kept decisions.
- `high900aux`: 240,659 kept decisions.

Evaluation script uploaded but not yet run:

- `/tmp/run_dragapult_eval_20260813.sh`.
- It runs random g500, coverage baseline-delta versus previous local Dragapult
  baseline
  `checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_dragapult_cc2e995b_w4.npz`,
  archetype matrix, and 0812 ladder-distribution weighted score.

Monitor:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
tail -f logs/dragapult_specialists_20260813/train_runner.log
for f in logs/dragapult_specialists_20260813/train/*.log; do echo "===$f==="; tail -30 "$f"; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

After all Dragapult jobs finish:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
bash /tmp/run_dragapult_eval_20260813.sh
```

## Active Remote Job: History Summary Marnie Parallel 2026-08-13

User asked to run the next training stage in parallel. A five-way Marnie b8f
history-summary experiment is now running on `ks`:

- Main runner script: `/tmp/run_history_summary_20260813.sh`.
- Main runner log: `logs/history_summary_20260813/runner.log`.
- Main checkpoint dir: `checkpoints/history_summary_20260813`.
- Base corpus:
  `data/bc_corpus_banded_v12_0701_0811_merged_hist32_log128_board12_20260812`.
- Auxiliary corpus: `data/bc_corpus_high900_winners_20260812`.
- Baseline/init checkpoint:
  `checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_marnie_grimmsnarl_b8f251a4_w4.npz`.
- Deck:
  `logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv`.

Training jobs launched:

- GPU0 `hist_summary_pointer_h0`
  - Checkpoint:
    `checkpoints/history_summary_20260813/bc2_marnie_b8f_histsummary_pointer_h0_w4.npz`.
  - Log:
    `logs/history_summary_20260813/train/marnie_b8f_histsummary_pointer_h0_w4.log`.
  - Note: this is now known to be an invalid/weak control on old v12 corpus,
    because v12 npz files do not store `history_summary` and the original
    loader fallback used only enabled raw-history streams; with all raw streams
    disabled, summary was all zero. Keep it as a no-effective-history control.
- GPU0 `hist_summary_pointer_h0_fixed`
  - Supplemental runner script:
    `/tmp/run_history_summary_h0_fixed_20260813.sh`.
  - Checkpoint:
    `checkpoints/history_summary_20260813/bc2_marnie_b8f_histsummary_pointer_h0_fixed_w4.npz`.
  - Log:
    `logs/history_summary_20260813/train/marnie_b8f_histsummary_pointer_h0_fixed_w4.log`.
  - This is the corrected summary-only experiment.
- GPU1 `hist_summary_pointer_raw_h8`
  - Checkpoint:
    `checkpoints/history_summary_20260813/bc2_marnie_b8f_histsummary_pointer_raw_h8_w4.npz`.
  - Log:
    `logs/history_summary_20260813/train/marnie_b8f_histsummary_pointer_raw_h8_w4.log`.
- GPU2 `hist_summary_cross_raw_h8`
  - Checkpoint:
    `checkpoints/history_summary_20260813/bc2_marnie_b8f_histsummary_cross_raw_h8_w4.npz`.
  - Log:
    `logs/history_summary_20260813/train/marnie_b8f_histsummary_cross_raw_h8_w4.log`.
- GPU3 `hist_summary_cross_plan_h8`
  - Checkpoint:
    `checkpoints/history_summary_20260813/bc2_marnie_b8f_histsummary_cross_plan_h8_w4.npz`.
  - Log:
    `logs/history_summary_20260813/train/marnie_b8f_histsummary_cross_plan_h8_w4.log`.

Evaluation watcher:

- Script: `/tmp/eval_history_summary_all_20260813.sh`.
- Nohup log: `logs/history_summary_eval_all_20260813.nohup.log`.
- Watch log: `logs/history_summary_20260813/eval_all/watch.log`.
- It waits until all `history_summary_20260813` `bc2_train.py` processes end,
  then evaluates all five candidates plus old b8f baseline on:
  - random g500:
    `logs/history_summary_20260813/eval_all/random_g500.csv`;
  - coverage RR baseline-delta:
    `logs/history_summary_20260813/eval_all/vs_coverage_g100.csv`;
  - archetype matrix:
    `logs/history_summary_20260813/eval_all/vs_coverage_g100_arch_matrix.csv`;
  - live-opponent replay manifest:
    `logs/history_summary_20260813/eval_all/vs_liveopponents_g100.csv`;
  - gate/summary weight inspection:
    `logs/history_summary_20260813/eval_all/gate_summary.txt`.

Code correction made after initial launch:

- `ptcg_rl/bc2/data.py`: if an old v12 npz lacks `history_summary`, the loader
  now computes summary directly from the raw `own_hist_*`, `opp_hist_*`,
  `log_hist_*`, and `board_hist_*` arrays even when raw GRU stream lengths are
  set to zero. This makes summary an independent feature stream.
- `ptcg_rl/numpy_policy.py`: submission-time policy now records own action and
  board history whenever a model has a summary stream, even if it has no raw
  history GRU weights. Summary always uses up to 32 own actions, 128 public log
  events, and 12 board snapshots, matching the intended offline summary scale.
- Remote smoke after syncing the fix:
  `history_k=0`, `log_history_k=0`, `board_history_k=0`,
  `history_summary_dim=48` on an old v12 Marnie b8f file produced
  `summary_shape=(32,48)`, `summary_abs_mean=0.2293`,
  `summary_nonzero_rows=32`.

At `2026-08-13 08:12 Asia/Shanghai`, all five training jobs and the eval
watcher were alive. GPU memory was roughly GPU0 30GB, GPU1 16GB, GPU2 31GB,
GPU3 48GB.

Monitor:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
tail -f logs/history_summary_20260813/eval_all/watch.log
for f in logs/history_summary_20260813/train/*.log; do echo "===$f==="; tail -30 "$f"; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## Active Remote Job: History-Safe Marnie Repair 2026-08-12

User asked how to repair history models and requested a complete implementation
and test pass rather than small Kaggle-submission-consuming tweaks.

Implemented locally and synced to ks:

- `ptcg_rl/model.py`: history GRUs now use `_SafeGRU`, which preserves normal
  checkpoint parameter names but disables cuDNN `flatten_parameters`. This fixes
  the `CUDNN_STATUS_NOT_INITIALIZED` failure seen when history models were moved
  to CUDA on the shared A800 host.
- `PolicyValueNet` and `CrossAttentionPolicyValueNet`: history is now merged
  through a residual gate instead of a full replacement concat MLP:
  `h + sigmoid(gate([h,hist])) * tanh(delta(hist))`. The gate is initialized to
  about 1.8%, so history starts as a small correction to the strong stateless
  state encoder and can grow only if useful.
- `ptcg_rl/bc2/data.py`: training-only history corruption is available through
  stream/event/tail dropping. This targets the teacher-forcing gap: offline
  history prefixes come from replay/expert play, while live history prefixes
  come from the model's own prior decisions.
- `ptcg_rl/bc2/losses.py` and `tools/bc2_train.py`: added first-step KL
  consistency between clean history, corrupted history, and zero history. This
  discourages brittle history shortcuts that improve validation loss but damage
  ladder play.
- `tools/bc2_train.py`: any enabled history stream now disables cuDNN
  automatically, in addition to supporting `PTCG_DISABLE_CUDNN=1`.
- `ptcg_rl/numpy_policy.py`: NumPy inference understands the new residual-gated
  history merge and remains backward-compatible with old checkpoints.

Validation before long run:

- Local `py_compile` and `git diff --check` passed.
- Local synthetic forward/backward passed for pointer and cross-attn history
  models.
- Remote `py_compile` passed.
- Remote CUDA `.to()` smoke passed:
  `pointer _SafeGRU cuda:0`, `cross_attn _SafeGRU cuda:0`.
- A previous history-safe runner failed before this fix with
  `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`; those failed logs
  were backed up to `logs/history_safe_20260812/failed_cudnn_2000/`.

Remote runner completed:

- Script: `/tmp/run_history_safe_20260812.sh`.
- Main log: `logs/history_safe_20260812/runner.log`.
- Nohup log: `logs/history_safe_20260812.nohup.log`.
- Train logs:
  - `logs/history_safe_20260812/train/marnie_b8f_pointer_init_h16.log`
  - `logs/history_safe_20260812/train/marnie_b8f_cross_init_h16.log`
  - `logs/history_safe_20260812/train/marnie_b8f_cross_scratch_h32.log`
- Checkpoints:
  `checkpoints/history_safe_20260812/bc2_marnie_b8f_historysafe_*.npz`.
- Eval outputs:
  `logs/history_safe_20260812/eval/random_g500.csv`,
  `logs/history_safe_20260812/eval/vs_coverage_g100.csv`,
  `logs/history_safe_20260812/eval/vs_coverage_g100_arch_matrix.csv`,
  weighted 0810 ladder CSVs, and `summary.txt`.
- Completion times:
  - pointer init h16: 2026-08-12 23:27.
  - cross-attn init h16: 2026-08-13 00:15.
  - cross-attn scratch h32: 2026-08-13 04:03.
  - runner eval done: 2026-08-13 04:19.
- Coverage/current_rr evaluation:
  - random g500: pointer 499/500, cross init 498/500, cross scratch 500/500.
  - old high900win b8f macro 0.780, players-weighted all 0.7551.
  - pointer init h16 macro 0.785, all 0.7673.
  - cross init h16 macro 0.788, all 0.7740, best coverage-pool score.
  - cross scratch h32 macro 0.766, all 0.7269, broad regression.
- Extra recent live-opponent replay manifest evaluation:
  `logs/history_safe_20260812/eval/historysafe_vs_old_liveopponents_g100.csv`.
  Results against old high900win b8f:
  - pointer init h16: candidate 0.740 vs old 0.741,
    avg_delta -0.001, lost 9/14.
  - cross init h16: candidate 0.726 vs old 0.741,
    avg_delta -0.014, lost 8/14.
  - cross scratch h32: candidate 0.709 vs old 0.741,
    avg_delta -0.031, lost 10/14.
- Interpretation:
  - History-safe fixed the CUDA/runtime brittleness but did not improve live-like
    opponent performance.
  - Coverage/current_rr overstates history benefits, especially for cross init.
    On recent live opponents, history mostly trades wins across matchups and
    damages important ones such as Alakazam, Dragapult, Cynthia/Lucario, or
    mirror.
  - Do not submit these history-safe Marnie models. Keep old high900win b8f as
    the current Marnie baseline unless a different ladder environment makes the
    coverage-pool metric more relevant than recent live opponents.

## Implemented: Explicit History Summary And Sensitivity Loss 2026-08-13

Motivation:

- History-safe models fixed runtime issues but did not improve live-like
  opponents. Checkpoint inspection showed `history_gate_fc.bias` stayed near
  -4.0, so the history residual gate remained around 1.8%. The models mostly
  learned to ignore raw history.
- User requested two stronger changes:
  1. explicitly aggregate history so the agent can use it more clearly;
  2. add loss pressure that makes the model pay attention to history.

Implemented locally and synced to ks:

- `ptcg_rl/history_features.py`
  - Added `HISTORY_SUMMARY_DIM = 48`.
  - Added `history_summary_from_arrays(...)`, a ledger-style summary from own
    actions, opponent actions, public logs, and board snapshots.
  - Summary includes coarse action tempo, attach/evolve/attack/ability ratios,
    public log player/move/damage signals, board-history occupancy/change, and
    recent key-action flags.
- `tools/bc_extract_v2.py`
  - New v13-style extra key: `history_summary`.
  - New metadata key: `history_summary_dim`.
  - Old v12 corpus can still be used because the loader can generate summary
    from existing raw history streams.
- `ptcg_rl/bc2/data.py`
  - Added `history_summary_dim`.
  - Collate now emits `batch.history["summary"]`.
  - If the npz lacks `history_summary`, it computes it from `own_hist`,
    `opp_hist`, `log_hist`, and `board_hist`.
  - If history augmentation corrupts streams, summary is recomputed after
    corruption so the forward pass sees consistent information.
- `ptcg_rl/model.py`
  - Pointer and cross-attn policies now support a dedicated summary stream:
    `history_summary_fc1/fc2`, projected to the same history latent size as
    GRU streams.
  - Added `history_gate_init_bias`. Old default remains -4.0. For new
    experiments use -2.0 or -1.5 so the history gate starts open enough to
    receive useful gradient.
  - Added `checkpoint_history_summary_dim(...)`.
- `ptcg_rl/numpy_policy.py`
  - Submission-time NumPy inference now auto-detects `history_summary_fc1`.
  - It builds the same summary from live own history, public logs, board
    history, and zero-filled opponent-labeled history.
- `ptcg_rl/bc2/losses.py`
  - Added `history_sensitivity_weight` and `history_sensitivity_margin`.
  - This is a margin/ranking loss: real-history first-action label NLL should
    beat zero-history first-action label NLL by the configured margin.
  - Logs `history_sensitivity` and `history_sensitivity_delta`.
- `tools/bc2_train.py`
  - New flags:
    - `--history-summary` shorthand for summary dim 48.
    - `--history-summary-dim N`.
    - `--history-gate-init-bias`.
    - `--history-sensitivity-weight`.
    - `--history-sensitivity-margin`.
  - `--best-metric history_sensitivity` is available for diagnostics, but do
    not select submission checkpoints solely by that metric.

Validation:

- Local `py_compile` passed for modified files.
- Local `git diff --check` passed.
- `tools/bc2_train.py --help` parses.
- Remote ks `py_compile` passed.
- Remote real-v12-corpus smoke passed:
  - `BCCorpus(... history_k=4, log_history_k=8, board_history_k=3,
    history_summary_dim=48)` produced `summary_shape=(16,48)` and nonzero
    summary mean.
  - Tiny pointer model forward/backward with `history_sensitivity_weight=0.1`
    passed.
  - Temporary checkpoint loaded through `NumpyPolicy`; detected
    `np_history_summary_dim=48`.

Recommended next experiment:

- Do not reuse the conservative history-safe settings unchanged.
- Train from old high900win b8f with partial init, but reset/new history heads
  are allowed:
  - `--history-summary`
  - `--history-gate-init-bias -2.0`
  - `--history-sensitivity-weight 0.15`
  - `--history-sensitivity-margin 0.05`
  - avoid `--history-consistency-zero-weight` initially because it explicitly
    pushes real and zero history to be similar.
- Watch training logs:
  - `history_sensitivity_delta` should become positive on train and ideally
    non-negative on val.
  - after training, inspect `history_gate_fc.bias`; sigmoid mean should be
    meaningfully above 1.8%. If it stays near 1.8%, the model still ignores
    history.

Three models started in parallel at about `2026-08-12T20:05:18+08:00`:

- GPU0 `marnie_b8f_pointer_init_h16`: old strong pointer checkpoint init,
  `width=4`, `history_k=16`, `log_history_k=64`, `board_history_k=8`,
  `epochs=6`, `batch=1024`, `lr=5e-5`, 32GB cap.
- GPU1 `marnie_b8f_cross_init_h16`: old strong pointer checkpoint partial init,
  cross-attn `width=4`, `state_layers=2`, same h16 history, `epochs=6`,
  `batch=1536`, `lr=4e-5`, 56GB cap.
- GPU2 `marnie_b8f_cross_scratch_h32`: scratch cross-attn `width=4`,
  `history_k=32`, `log_history_k=128`, `board_history_k=12`, `epochs=8`,
  `batch=1536`, `lr=3e-5`, 56GB cap.

At the first post-fix check all three had loaded the full corpus and entered
epoch 1. GPU memory/utilization were roughly GPU0 `18.8GB/92%`, GPU1
`30.3GB/55%`, GPU2 `44.2GB/96%`.

Use these monitor commands:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
tail -f logs/history_safe_20260812/runner.log
for f in logs/history_safe_20260812/train/*.log; do echo "===$f==="; tail -30 "$f"; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Submission rule for this experiment: do not submit merely because validation
loss improves. Only consider a history-safe checkpoint if random remains strong
and `eval_baseline_delta.py`/archetype matrix beats the current old-method
Marnie baseline on the filtered RR pool and the 0810 ladder-weighted scores.

## Active Remote Job: Marnie Old-Recipe Sig Specialists 2026-08-12

User asked to recover two sig-specific Marnie old-version trainings while GPUs
are idle, using all August 900+ data. This is an explicit exception to the
earlier "BC locked/no micro-finetune" preference: the goal is to recover the
old live Kaggle behavior of `marnie_b8f_v12hist_bigbatch_b1536_20260809`, which
reached about 1050 and held 950+.

Remote workspace:
`ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`.

Old successful recipe recovered from
`logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.b1536.train.log`:

- `arch=cross_attn`, `width=4`, `state_layers=2`, `history_k=32`,
  `log_history_k=128`, `board_history_k=12`,
  `board_history_feat_dim=32`.
- Init:
  `checkpoints/v12_history_pilots_20260807/bc2_marnie_b8f_v12hist_cross_init.npz`.
- `batch_size=1536`, `lr=3e-5`, `epochs=8`, `split_by_game=True`,
  `checkpoint_every=1`.
- `win/loss/draw=1.5/0.4/0.8`, `first_action_weight=1.5`,
  `option_weight=0.15`, `set_loss_weight=0`, non-winner-only.
- `PTCG_DISABLE_CUDNN=1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Data selection for the active run:

- Main corpus:
  `data/bc_corpus_banded_v12_0701_0810_merged_hist32_log128_board12_20260811`.
- Aux corpus:
  `data/bc_corpus_banded_v12_0811_only_hist32_log128_board12_20260812`.
- Date filter: `2026-08-01..2026-08-11`.
- Score bands: `1200+`, `1100-1199`, `1000-1099`, `900-999`.
- This does not mix July decisions because the trainer date filter is active.

Sig audit output:

- Script: `/tmp/audit_marnie_aug900_sigs.py` on local and ks.
- Log: `logs/marnie_aug900_sig_audit_20260812.txt` on ks.
- 8月 900+ Marnie rows/games:
  - `b8f251a476e7`: `2,401,840` rows, `25,556` games, all 11 days.
  - `2c22fa761816`: `297,595` rows, `3,047` games, all 11 days.
  - next sigs are much smaller, e.g. `b9c5f5744723` only `51,580` rows.
- No useful DVH checkpoint/deck-sig was found in current ks paths; DVH remains
  a local comparison baseline only and must not be submitted.

Active run:

- Runner script on ks: `/tmp/run_marnie_sig_oldrecipe_aug900_20260812.sh`.
- First launch failed immediately because current `bc2_train.py` requires
  `--device cuda:0` when setting `--cuda-memory-gb`; logs kept under
  `logs/marnie_sig_oldrecipe_aug900_20260812`.
- Corrected run id: `marnie_sig_oldrecipe_aug900_20260812_r2`.
- Runner log:
  `logs/marnie_sig_oldrecipe_aug900_20260812_r2/runner.log`.
- Checkpoint dir:
  `checkpoints/marnie_sig_oldrecipe_aug900_20260812_r2`.
- Training logs:
  - `logs/marnie_sig_oldrecipe_aug900_20260812_r2/marnie_b8f251a4_v12hist_aug900_b1536_seed25.train.log`
  - `logs/marnie_sig_oldrecipe_aug900_20260812_r2/marnie_2c22fa76_v12hist_aug900_b1536_seed26.train.log`
- Started at about `2026-08-12T16:32:44+08:00`.
- At 16:35 CST both jobs were training:
  - `b8f`: `2,391,647` kept rows, train/val `2,154,227/237,420`,
    `1403` steps/epoch, GPU0, 56GB cap, seed 25.
  - `2c22`: `295,893` kept rows, train/val `266,753/29,140`,
    `174` steps/epoch, GPU1, 40GB cap, seed 26.

Monitor commands:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
tail -f logs/marnie_sig_oldrecipe_aug900_20260812_r2/runner.log
tail -f logs/marnie_sig_oldrecipe_aug900_20260812_r2/marnie_b8f251a4_v12hist_aug900_b1536_seed25.train.log
tail -f logs/marnie_sig_oldrecipe_aug900_20260812_r2/marnie_2c22fa76_v12hist_aug900_b1536_seed26.train.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Expected duration:

- `2c22` is much smaller and should finish first.
- `b8f` is close to the old successful run size; expect roughly 3 hours from
  launch, not counting any later evaluation.
- After completion, run random first and then current filtered RR/coverage
  against the latest pool before considering any submission.

## Active Remote Job: High900 Winner Old-Method Specialists 2026-08-12

User asked to train `high1100+` specialists with "previous-version method",
adding 900+ winner-only data, and explicitly not using history or later
history/cross-attention additions.

This is separate from both:

- `marnie_sig_oldrecipe_aug900_20260812_r2`, which is a history/cross-attn
  Marnie recovery run.
- `high_climbwin_20260812`, which used 600-1099 winner-only aux and batch
  1024. That earlier recipe degraded and should not be conflated with this one.

Definition of this run:

- Main corpus:
  `data/bc_corpus_banded_v12_0701_0811_merged_hist32_log128_board12_20260812`.
  Built from 0701-0810 merged plus 0811-only via symlinks.
- Main score bands: `1200+`, `1100-1199`, all outcomes.
- Aux corpus:
  `data/bc_corpus_high900_winners_20260812`.
- Aux score bands: `900-999`, `1000-1099`, `won=1` only, date
  `2026-08-01..2026-08-11`.
- Architecture: legacy `pointer`, `width=4`, no `history_k`,
  no `log_history_k`, no `board_history_k`, no `cross_attn`.
- Training params: `batch_size=512`, `lr=8e-5`, `win/loss/draw=1.5/0.4/0.8`,
  `first_action_weight=1.6`, `option_weight=0.2`, `set_loss=0.12/2/0.15`,
  `multi_select_weight=1.6`, context weights `MAIN=1.10`,
  `ATTACH_FROM=1.35`, `ATTACH_TO=1.25`, `SKILL_ORDER=1.40`, type weights
  `EVOLVE=1.15`, `ATTACK=1.10`, `include_empty=True`,
  `split_by_game=True`, checkpoint every epoch.
- The batch size is intentionally 512 even though GPUs have room. Previous
  diagnosis showed batch 1024 with fixed epochs reduced update steps and
  degraded this old-method family.

Scripts:

- Aux builder: `/tmp/build_high900_winner_aux_20260812.py` on ks.
- Main train runner: `/tmp/run_high900win_oldmethod_20260812.py` on ks.
- Post-eval watcher: `/tmp/run_high900win_oldmethod_eval_20260812.sh` on ks.
- Spare seed runner: `/tmp/run_high900win_spare_marnie_2c22_20260812.sh` on ks.

Outputs:

- Build log:
  `logs/high900win_oldmethod_20260812/build_aux.log`.
- Aux summaries:
  `logs/high900win_oldmethod_20260812/aux_file_summary.csv`,
  `logs/high900win_oldmethod_20260812/aux_sig_summary.csv`.
- Main runner log:
  `logs/high900win_oldmethod_20260812/runner.log`.
- Main train manifest:
  `logs/high900win_oldmethod_20260812/train_manifest.csv`.
- Main checkpoints:
  `checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_*_w4.npz`.
- Spare seed checkpoint:
  `checkpoints/high900win_oldmethod_20260812/w4_spare/bc2_high900win_old_marnie_grimmsnarl_2c22fa76_seed61_w4.npz`.
- Post-eval log:
  `logs/high900win_oldmethod_20260812/post_eval.log`.
- Post-eval outputs after training:
  `logs/high900win_oldmethod_20260812/random_g500.csv`,
  `logs/high900win_oldmethod_20260812/vs_coverage_g100.csv`,
  `logs/high900win_oldmethod_20260812/vs_coverage_g100_arch_matrix.csv`,
  `logs/high900win_oldmethod_20260812/vs_coverage_g100_ladder0810_players_weighted.csv`,
  `logs/high900win_oldmethod_20260812/vs_coverage_g100_ladder0810_rows_weighted.csv`,
  `logs/high900win_oldmethod_20260812/top_summary.txt`.

Aux builder result:

- Total aux: `115` files, `1,691,106` rows.
- By target sig:
  - Marnie `b8f251a476e7`: `888,640` rows, 11 dates.
  - Alakazam `7f9a538936e3`: `294,454` rows, 11 dates.
  - Mega Lopunny `f1445356c3a7`: `204,557` rows, 10 dates.
  - Dragapult `cc2e995b5ad0`: `141,222` rows, 11 dates.
  - Marnie `2c22fa761816`: `132,854` rows, 11 dates.
  - Mega Lucario `43d6d8b0fce9`: `21,422` rows, 6 dates.
  - Ogerpon `2bd9da52c43a`: `6,694` rows, 1 date.
  - Ogerpon `90abbfb0eee0`: `1,263` rows, 2 dates.
- Interpret Ogerpon results carefully: their aux is too thin for a fair
  old-method+900winner conclusion.

Active status as of 2026-08-12 17:31 CST:

- Main runner pid family includes `1274972`.
- Post-eval watcher pid family includes `1279023`; it only evaluates after the
  main runner exits successfully.
- Main queue:
  - Done: `high900win_old_dragapult_cc2e995b`, best val/loss `0.9358`.
  - Done: `high900win_old_marnie_grimmsnarl_b8f251a4`, best val/loss `0.6643`.
  - Running GPU2: `high900win_old_alakazam_7f9a5389`; at last check epoch
    4/6 completed, best val/loss `0.7273`.
  - Running GPU3: `high900win_old_marnie_grimmsnarl_2c22fa76`; corpus loaded
    `335,040` kept rows, train/val `300,841/34,199`.
- Spare GPU1 run:
  - Running: `Marnie 2c22` seed61, separate output under `w4_spare`.

Operational note:

- The custom main runner assigns GPUs by `len(running)`, not by the actual
  freed GPU id. With the current timing this is acceptable because GPU3 is
  processing smaller jobs while Alakazam holds GPU2, but if a future run uses
  this script, improve scheduling before relying on all GPUs.

Monitor:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
tail -f logs/high900win_oldmethod_20260812/runner.log
tail -f logs/high900win_oldmethod_20260812/post_eval.log
cat logs/high900win_oldmethod_20260812/train_manifest.csv
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## Active Remote Job: 0811 Team-Specific History BC 2026-08-12

User asked to parallel-start team-specific training and noted 2026-08-11 episode data was pulled.

Remote workspace: `ks:/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`.

Data and setup completed:

- 0811 raw zip exists at `/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-11.zip`.
- Extracted 0811-only v12 history corpus:
  `data/bc_corpus_banded_v12_0811_only_hist32_log128_board12_20260812`.
  Extraction log: `logs/team_specific_0811_20260812/extract_0811_only.log`.
  It processed `4622` episodes, `755255` decisions, bad/err `0`, output size about `129M`.
- Base corpus for these jobs:
  `data/bc_corpus_banded_v12_0701_0810_merged_hist32_log128_board12_20260811`.
- Alakazam trajectory targets built and reused:
  - `logs/team_specific_0811_parallel_20260812/majkel_7f9_0701_0810_targets.csv`: `9320` games, WR `0.562`.
  - `logs/team_specific_0811_parallel_20260812/majkel_7f9_0811_targets.csv`: exists; no meaningful 0811 Majkel 7f rows were found.
  - `logs/team_specific_0811_parallel_20260812/liamk_ca08_0701_0810_targets.csv`: `2748` games, WR `0.490`.
  - `logs/team_specific_0811_parallel_20260812/liamk_ca08_0811_targets.csv`: exists; no meaningful 0811 LiamK ca08 rows were found.

Important training-launch lessons:

- For history models, set `PTCG_DISABLE_CUDNN=1`; otherwise `nn.GRU` can fail during CUDA initialization with `CUDNN_STATUS_NOT_INITIALIZED`.
- `cross_attn + history_k=32 + log_history_k=128 + board_history_k=12 + width=4` with batch `1536` and `--cuda-memory-gb 24` OOMs on first epoch despite A800 total VRAM. It hits the per-process allocator cap, not physical GPU capacity.
- Working restart uses `batch-size=1024`, `--cuda-memory-gb 32`, `PTCG_DISABLE_CUDNN=1`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Active/finished training status as of 2026-08-12 12:39 CST:

- Completed successfully:
  - `checkpoints/team_specific_0811_train_20260812/bc2_ogerpon_jch_2bd9_w4_hist.npz`
  - Log: `logs/team_specific_0811_train_20260812/ogerpon_jch_2bd9_w4_hist.log`
  - It trained 12 epochs with best checkpoint saved around epoch 6 (`val=0.9068`); later epochs overfit upward.
- Active b1024/32GB restart runner:
  - Script: `/tmp/restart_failed_team_specific_0811_b1024.py` on ks.
  - Runner log: `logs/team_specific_0811_train_b1024_20260812/runner.log`.
  - Status CSV: `logs/team_specific_0811_train_b1024_20260812/train_status.csv`.
  - Checkpoint dir: `checkpoints/team_specific_0811_train_b1024_20260812`.
  - Jobs:
    - GPU0: `alakazam_majkel_7f9_histplan_w4_0811_b1024`, PID around `966071`, log `logs/team_specific_0811_train_b1024_20260812/alakazam_majkel_7f9_histplan_w4_0811_b1024.log`. At last check it entered epoch 1 (`1/725`, then `25/725`).
    - GPU1: `alakazam_liamk_ca08_histplan_w4_0811_b1024`, PID around `966188`, log `logs/team_specific_0811_train_b1024_20260812/alakazam_liamk_ca08_histplan_w4_0811_b1024.log`. At last check it entered epoch 1 (`50/202`).
    - GPU2: `ogerpon_jch_2a507_w4_hist_b1024`, PID around `966304`, log `logs/team_specific_0811_train_b1024_20260812/ogerpon_jch_2a507_w4_hist_b1024.log`. At last check it reached epoch 5; epoch 4 had best val `0.7552`.
  - GPU3 is currently free after 2bd9 finished. Do not start extra IO-heavy jobs until the two Alakazam jobs are safely past early epochs, unless the user asks to maximize throughput.

Team-specific data counts relevant to these jobs:

- Alakazam Majkel 7f9: `831044` kept rows, `9320` games in 0701-0810 target CSV; includes climb games because corpus `score_band` is static leaderboard snapshot, not per-game ladder score.
- Alakazam LiamK ca08: `228715` kept rows for training, `2748` target games.
- Ogerpon 2a507 JCH: about `97394` kept training rows for `James Cox & Henry Chao`; 0811 has no 2a507 rows.
- Ogerpon 2bd9 JCH: `50727` kept rows with 0811 aux repeated; completed successfully.
- Additional 2a507 rows exist for team `James Cox` (`52695` rows/`791` games) and `zoroark190` (`31573` rows/`466` games`) in the base corpus, but the active 2a507 job is strict `James Cox & Henry Chao` only. A future contrast can train deck-sig-only or JCH+James Cox if needed.

Use this monitor command on ks:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,cmd | grep -E "restart_failed_team_specific_0811_b1024.py|bc2_train.py" | grep -v grep || true
for f in logs/team_specific_0811_train_b1024_20260812/*.log; do echo ===$f; grep -E "^BC2:|^Corpus:|^Split:|epoch|done epoch|saved best|Traceback|OutOfMemory|RuntimeError" "$f" | tail -14; done
```

After these finish, evaluate before considering submissions:

- Random 500 for each checkpoint.
- Current filtered RR/coverage matrix, especially Alakazam vs TRM/Marnie/Festival and Ogerpon vs Crustle/current ladder pool.
- Compare against old `high1100_alakazam_7f9`, `cur0810_alakazam_7f9`, and existing Ogerpon baselines.

## Completed: Loss-Optimized BC Contrast 2026-08-12

User asked whether val loss can be optimized and retrained for comparison.

Local code changed and synced to ks:

- `ptcg_rl/bc2/losses.py`
  - Added `sequence_loss_parts`.
  - The total training objective still includes weighted autoregressive
    action NLL plus optional set/value/trajectory/step-plan losses.
  - It now also reports separable metrics: `policy`, `policy_raw`,
    `first_action_raw`, `first_action_acc`, `set`, `trajectory`,
    `step_plan`, and `value`.
  - Added `raw_policy_loss_weight`, which adds unweighted action-sequence NLL
    to the training objective. This is intended to prevent sample weighting and
    plan/set auxiliaries from hiding ordinary action imitation quality.
- `tools/bc2_train.py`
  - Added `--raw-policy-loss-weight`.
  - Added `--best-metric` with options such as `loss`, `policy`,
    `policy_raw`, `first_action_raw`, `trajectory`, and `step_plan`.
  - Epoch logs now print total loss and action-first metrics.
  - Validation aggregation now uses metric-specific weighted sums rather than
    a simple mean of per-batch losses.
- `ptcg_rl/bc2/data.py`
  - When `split_by_game` is active, game groups are now merged globally by
    `episode_id:player_index`. This prevents `aux_repeat` copies of the same
    game from being split across train and validation.

Remote runner:

- Script: `/tmp/run_lossopt_bc_20260812.py` on ks.
- Runner log: `logs/team_specific_0811_lossopt_20260812.runner.log`.
- Per-job logs: `logs/team_specific_0811_lossopt_20260812/*.log`.
- Checkpoints: `checkpoints/team_specific_0811_lossopt_20260812/*.npz`.
- Runner PID at launch: `1008388`.
- Started four parallel jobs, one per GPU:
  - `lossopt_ogerpon_jch_2a507_w4_hist`
  - `lossopt_ogerpon_jch_2bd9_w4_hist`
  - `lossopt_alakazam_liamk_ca08_histplan_w4`
  - `lossopt_alakazam_majkel_7f9_histplan_w4`

Common lossopt settings:

- `--raw-policy-loss-weight 0.35`
- `--best-metric policy_raw`
- `--batch-size 1024`
- `--cuda-memory-gb 48`
- `PTCG_DISABLE_CUDNN=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Completed training outputs:

- `checkpoints/team_specific_0811_lossopt_20260812/bc2_lossopt_ogerpon_jch_2a507_w4_hist.npz`
  - Best `policy_raw=0.6178` at epoch 6/8.
- `checkpoints/team_specific_0811_lossopt_20260812/bc2_lossopt_ogerpon_jch_2bd9_w4_hist.npz`
  - Best `policy_raw=0.8757` at epoch 6/12; later epochs overfit badly.
- `checkpoints/team_specific_0811_lossopt_20260812/bc2_lossopt_alakazam_liamk_ca08_histplan_w4.npz`
  - Best `policy_raw=0.7584` at epoch 7/8.
- `checkpoints/team_specific_0811_lossopt_20260812/bc2_lossopt_alakazam_majkel_7f9_histplan_w4.npz`
  - Best `policy_raw=0.5196` at epoch 6/8.

Evaluation outputs:

- Random logs: `logs/team_specific_0811_lossopt_20260812_eval/random_*.log`.
- Coverage delta CSVs:
  - `logs/team_specific_0811_lossopt_20260812_eval/lossopt_alakazam_7f9_vs_baseline_coverage_g80.csv`
  - `logs/team_specific_0811_lossopt_20260812_eval/lossopt_alakazam_ca08_vs_baseline_coverage_g80.csv`
  - `logs/team_specific_0811_lossopt_20260812_eval/lossopt_ogerpon_2a507_vs_baseline_coverage_g80.csv`
  - `logs/team_specific_0811_lossopt_20260812_eval/lossopt_ogerpon_2bd9_vs_baseline_coverage_g80.csv`
- Runner summary had a harmless post-eval parsing crash:
  `AttributeError: 'NoneType' object has no attribute 'group'`. The actual
  random logs and coverage CSVs were generated correctly.

Random gate:

- `lossopt_alakazam_7f9`: `100.0%` vs random, `500/500`.
- `lossopt_alakazam_ca08`: `99.0%` vs random, `495/500`.
- `lossopt_ogerpon_2a507`: `89.6%` vs random, `448/500`.
- `lossopt_ogerpon_2bd9`: `86.6%` vs random, `433/500`.

Coverage delta summary:

- `lossopt_alakazam_7f9` vs `high1100_alakazam_7f9`:
  - Candidate avg `0.762`, weighted `0.680`.
  - Baseline avg `0.712`, weighted `0.618`.
  - Delta avg `+0.051`, weighted `+0.062`, lost `2/13`.
  - Worst absolute matchup remains current Marnie `0.487`, but this is still
    better than baseline `0.463`.
  - Team Rocket Mewtwo improved from `0.412` to `0.588`, though its current
    environment weight is tiny in the 0810 weighted pool.
- `lossopt_alakazam_ca08`:
  - Candidate avg `0.676`, weighted `0.536`.
  - Baseline was random, so the large positive delta is only a sanity check.
  - Worst matchup is current Marnie `0.338`, so it is not a primary submission
    candidate despite `99%` random.
- `lossopt_ogerpon_2a507`:
  - Candidate avg `0.327`, weighted `0.318`; too weak despite positive delta
    over random.
  - Worst Crustle `0.087`.
- `lossopt_ogerpon_2bd9`:
  - Candidate avg `0.303`, weighted `0.261`.
  - Absolute strength is too low; random gate also failed.

Interpretation:

- Only `lossopt_alakazam_7f9` is a plausible local candidate from this batch.
  It has real positive coverage delta over the submitted/high1100 7f9 baseline
  and fixes the old TRM weakness locally.
- The online result for `submission_high1100_alakazam_7f9a5389` staying around
  800 means the current local coverage pool is still too optimistic for Kaggle
  climb conditions. Do not assume `lossopt_alakazam_7f9` will restore old
  scores without a live probe.
- Both lossopt Ogerpon models should be rejected for submission unless later
  Ogerpon deck/signature analysis changes the deck choice.
- Marnie was not part of this lossopt batch. That was a scope choice because
  the batch targeted Alakazam underperformance and Ogerpon sig issues after
  Marnie had just been continued/submitted. Given the live score of
  `submission_high1100_marnie_b8f_cont_b1024_seed25` only holding about 900,
  this is an incomplete experimental axis. Future Marnie work should be from
  scratch or DVH-style reproduction, not another small continuation.

Follow-up Alakazam trace audit:

- Remote outputs:
  - `logs/alakazam_lossopt_trace_20260812/*.games.csv`
  - `logs/alakazam_lossopt_trace_20260812/*.summary.csv`
  - `logs/alakazam_lossopt_trace_20260812/failure_summary.csv`
  - `logs/alakazam_lossopt_trace_20260812/plan_failure/plan_win_loss_gaps.csv`
  - `logs/alakazam_lossopt_trace_20260812/plan_failure/defer_classification.csv`
  - `logs/alakazam_lossopt_trace_20260812/majkel_7f9_targets_summary.txt`
- Trace results for `lossopt_alakazam_7f9`:
  - vs current Marnie: `89-71`, WR `0.556`.
  - vs current TRM: `82-78`, WR `0.512`.
  - vs current Cynthia: `96-64`, WR `0.600`.
  - vs current Festival: `100-60`, WR `0.625`.
  - High1100 baseline controls: Marnie `83-77`, WR `0.519`; TRM `66-94`,
    WR `0.412`.
- Interpretation:
  - `lossopt_alakazam_7f9` is genuinely better than the submitted/high1100
    Alakazam in local trace matchups, especially TRM, but it is still only
    near-even to small-favored into high-weight current Marnie/TRM pressure.
    This explains why it may still look worse than a strong Marnie submission.
  - Do not treat raw `attack/evolve/attach miss` as directly trainable errors.
    The defer audit shows most skips are `same_turn_later`, meaning the policy
    often correctly performs setup/search/resource actions before attacking or
    evolving in the same turn.
  - Useful failure signals are matchup-specific:
    - vs TRM: losses have much higher MAIN ability miss (`+0.137`), more
      late attack/evolve/attach skip, more play/evolve/ability churn, and
      more Xerosic/Abra-line/Rare Candy option exposure. This suggests TRM
      losses are not simply "never saw the card"; the policy enters complex
      long-turn states and fails ordering/resource conversion.
    - vs Cynthia/Festival: losses show delayed first attack/first evolve and
      more early-end; targeted pressure-window rules or training labels may
      help.
    - vs Marnie: loss/win differences are small. The matchup is close and
      likely needs stronger deck/strategy or opponent-conditioned planning,
      not simple no-miss filtering.
  - Training data target CSV
    `logs/team_specific_0811_parallel_20260812/majkel_7f9_0701_0810_targets.csv`
    confirms that successful Alakazam games generally have earlier first
    attack, earlier primary board, higher attack counts, and fewer early ends.
    For Marnie specifically, wins have first attack turn about `3.81` vs
    losses `42.04`, attack count `4.93` vs `3.00`, and primary board turn
    `4.37` vs `40.27`.
  - If building a new data-filtered or rule-guided Alakazam model, filter on
    game-level success signals such as `strategy_success`, `attack_by_4/6`,
    `primary_board_by_4/6`, and `no_early_end`, not per-decision
    "no attack miss" or "no evolve miss".

## Alakazam 7f9 Underperformance Audit 2026-08-12

User asked why Alakazam performs poorly locally despite abundant high-score
samples, and how to diagnose/adjust it.

Key remote audit outputs:

- `logs/alakazam_7f9_matchup_data_audit_20260812_allscore.log`
- `logs/alakazam_7f9_matchup_data_audit_20260812_score1100p.log`
- `logs/alakazam_trace_outcome_gap_report_20260812.csv`

Local RR evidence:

- `cur0810_alakazam_7f9a5389`
  (`logs/current_rr_pool_20260811/rr_coverage_g100_arch_matrix.csv`):
  macro `0.614`, worst `Team Rocket Mewtwo 0.040`, Marnie `0.440`,
  Festival `0.420`, Crustle `0.650`, Ogerpon `0.870`, Lucario `0.670`.
- `high1100_alakazam_7f9a5389`
  (`logs/current_rr_pool_20260811/high_vs_coverage/high_vs_coverage_arch_matrix.csv`):
  macro `0.732`, worst `Team Rocket Mewtwo 0.380`, Marnie `0.530`,
  Festival `0.560`, Crustle `0.680`, Ogerpon `0.900`, Lucario `0.740`.
- Therefore the `7f9a538936e3` deck signature is not inherently unusable.
  Training recipe/data-window quality changes TRM performance from near-zero
  to roughly 38% in the current coverage pool.

Corpus evidence for `7f9a538936e3` in
`data/bc_corpus_banded_v12_0701_0810_merged_hist32_log128_board12_20260811`:

- All scores: `3,466,735` decision rows for this sig.
- All-score game WR by opponent:
  - Marnie: `6055/14024 = 0.432`
  - Crustle: `3100/4794 = 0.647`
  - Team Rocket Mewtwo: `1056/3373 = 0.313`
  - Dragapult: `805/1919 = 0.419`
  - Festival: `433/897 = 0.483`
  - Ogerpon: `881/1359 = 0.648`
  - Mega Lucario: `206/438 = 0.470`
- Score >=1100 only: `1,324,654` decision rows for this sig.
  - TRM: `637/1569 = 0.406`
  - Marnie: `2371/4761 = 0.498`
  - Crustle: `1432/2038 = 0.703`
  - Dragapult: `247/509 = 0.485`
  - Festival: `175/284 = 0.616`
  - Ogerpon: `167/229 = 0.729`
  - Mega Lucario: `64/130 = 0.492`
- Important split: score band `1100-1199` Alakazam 7f vs TRM is only
  `106/571 = 0.186`, while `1200+` is `531/998 = 0.532`. This suggests the
  good TRM plan may be concentrated in very top/team-specific games, while
  normal high-score rows include many losing TRM patterns.

Deck facts for the current 7f9 deck
(`logs/ladder_pool_0805_0809_fast_all/decks/7f9a538936e3_alakazam_team_rot-wei.csv`):

- It includes resource/tempo tools: `4 Enhanced Hammer`, `3 Xerosic's
  Machinations`, `3 Boss's Orders`, `4 Dawn`, `2 Nighttime Mine`.
- The TRM opponent deck used in current coverage includes Team Rocket's
  Tarountula/Spidops, Team Rocket's Mewtwo ex, Articuno, Factory, Transceiver,
  Ariana/Archer/Giovanni/Proton, and Team Rocket's Energy.
- Because Alakazam already has disruption cards, very low TRM WR should not be
  blamed only on missing deck tools. First inspect whether the policy uses
  these tools in the correct matchup windows.

Trace evidence from older v10 failure traces:

- MAIN phase miss rates for Alakazam vs TRM were high:
  attack available/chosen `5273/1551`, miss `0.706`; attach miss `0.854`;
  evolve miss `0.680`; play miss `0.637`.
- `trace_outcome_gap_report.py` highlights recurring loss-heavy contexts:
  - vs TRM: choosing `ATTACK` when ability was available; attaching
    `Telepath Psychic Energy` while play options existed; active/activate
    decisions around Alakazam vs Team Rocket's Mewtwo ex; late missed ability
    or missed play while facing Team Rocket's Spidops/Mimikyu.
  - vs Marnie: many loss-heavy contexts around Froslass skill order and
    repeated ability use where play/attach/attack/retreat options existed.
  - vs Festival/Crustle: multiple active selection and MAIN-phase priority
    errors, especially when active Alakazam/Kadabra faces high-pressure board
    states.

Interpretation:

- Alakazam is a strong high-ladder archetype, but its local weakness is not
  random/legal-action incompetence; random WR is high. It is matchup-specific
  sequencing and resource management under many legal options.
- Current global BC averages across many opponents and many team styles. That
  can wash out the rare, correct TRM/Marnie/Festival plan.
- Ordinary winner-only/success fine-tuning previously did not solve this
  reliably; many "wins" are noisy/luck-based. Do not revive small finetunes as
  the main fix.

Recommended next diagnostic path:

1. Build exact top-team cohorts for Alakazam 7f9, especially teams/players that
   reach `1200+` and actually beat TRM/Marnie/Festival. Compare their action
   n-grams and card timing against 1100-1199 losses.
2. Run fresh `trace_matchup_decisions.py` for `cur0810`, `high1100`, and any
   strategy-conditioned Alakazam against the same TRM/Marnie/Festival shadows,
   then feed decisions into `trace_outcome_gap_report.py`.
3. Convert repeated, verified errors into opponent-aware routing or hard rule
   guards. Do not hand-code generic "always play/attach/attack" rules; require
   a specific opponent archetype, turn/window, active matchup, and card/tool
   condition.
4. If training, prefer a new routed Alakazam policy with explicit opponent
   archetype inference and top-team/1200+ TRM-plan data, not a small finetune.

Follow-up 2026-08-12 team-specific / trajectory audit:

- Important bug found and fixed locally and on ks in `ptcg_rl/deck_plans.py`:
  Alakazam's deck plan still used old IDs `109/245` but current 7f9 deck uses
  `741 Abra`, `742 Kadabra`, `743 Alakazam`. This made
  `build_trajectory_targets.py` report `setup_success=0.000` for Alakazam,
  weakening any prior Alakazam trajectory/strategy-label experiments.
- The plan now includes both legacy and current IDs:
  - `signature_ids=(245, 742, 743)`
  - `primary_attackers=(245, 743)`
  - `setup_basics=(109, 741)`
  - `evolution_chain=(109, 741, 742, 245, 743)`
  - Alakazam draw/search/disruption tools now include Rare Candy, Poffin, Poke
    Pad, Hilda, Dawn, Enhanced Hammer, Boss, Xerosic, and Nighttime Mine.

New data audit outputs:

- `logs/alakazam_7f9_matchup_data_audit_20260812_score1200p.log`
- `logs/alakazam_team_data_audit_20260812.log`
- `logs/alakazam_trajectory_target_audit_20260812/summary_planfix.log`
- `logs/team_climb_coverage_20260812/*.csv`

Important correction on score bands:

- The `score` and `score_band` fields in the extracted BC corpus are not true
  per-game ladder scores. `tools/bc_extract_v2.py` obtains scores by looking up
  `TeamNames` in a leaderboard CSV snapshot during extraction, then assigns the
  same score to all episodes from that team in the extracted window.
- Raw Kaggle episode JSONs checked from
  `/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-07-17.zip`
  contain `TeamNames`, `EpisodeId`, rewards/statuses, and steps, but no
  per-episode ladder score.
- Therefore, a team/deck appearing only under corpus band `1200+` does not mean
  the submitted agent only played high-ladder games. It may include the full
  climb from 600 upward; the extract just labels all of that team's rows using
  the leaderboard snapshot score.
- For climb-stage analysis of other teams, use proxies:
  chronological order within team/deck/date, opponent archetype/deck coverage,
  opponent static score band, and Kaggle score-monitor traces when available
  for our own accounts.

Alakazam 7f9 score `1200+` only:

- `831,044` decision rows, `9,320` games, all from `Majkel1337` in this corpus.
  These rows are labeled `1200+` because Majkel's extracted leaderboard
  snapshot score is `1273.8`, not because each game happened at 1200+.
- Overall WR `5242/9320 = 0.562`.
- Matchups:
  - Alakazam mirror `1398/2799 = 0.499`
  - Marnie `1142/2352 = 0.486`
  - Crustle `1188/1716 = 0.692`
  - Team Rocket Mewtwo `531/998 = 0.532`
  - Cynthia `383/545 = 0.703`
  - Dragapult `134/258 = 0.519`
  - Festival `136/200 = 0.680`
- This is much better than `1100-1199` vs TRM (`106/571 = 0.186` from the
  earlier audit), so training/eval should try a strict 1200+ Majkel specialist.

Team-specific Alakazam data:

- `Majkel1337` Alakazam:
  - total `12,227` games, `1,074,939` rows, WR `0.569`.
  - deck sigs: `7f9a538936e3` `9,320` games, `91f48d8ef728` `1,853`, `5a5ea26c99b9` `1,054`.
  - corpus score-band coverage for 7f9 is only `1200+`, but that is a static
    leaderboard label. Do not interpret it as absence of 600-1100 climb games.
    Opponent static score bands for these rows include 600/700/800/900/1000/
    1100 ranges, so the environment coverage is broader than the own band
    suggests.
- `LiamK` Alakazam:
  - `ca082d1c1eab`, `2,748` games, `229,827` rows, WR `0.490`.
  - corpus score-band coverage is only `1100-1199`, again a static leaderboard
    label, not true per-game climb phase.
  - Broader low/mid ladder coverage still needs other teams/sigs, not only
    Majkel/LiamK.
- `James Cox & Henry Chao` are not Alakazam in the scanned data:
  - Ogerpon `2a5072194fdf`: `1,439` games, `98,441` rows, `1000-1099`.
  - Ogerpon `2bd9da52c43a`: `579` games, `39,018` rows, `1000/1100` bands.
  - Useful for Ogerpon team-specific work, not Alakazam.

Trajectory labels after Alakazam plan fix:

- Majkel 7f9 1200+:
  - all: `setup_success=0.295`, `tempo_success=0.289`,
    `strategy_success=0.205`.
  - wins: `setup_success=0.364`, `tempo_success=0.361`.
  - TRM wins: `setup_success=0.230`, `tempo_success=0.226`.
- LiamK ca08 1100:
  - all: `setup_success=0.486`, `tempo_success=0.458`.
  - wins: `setup_success=0.548`, `tempo_success=0.533`.
  - TRM wins: `setup_success=0.500`, `tempo_success=0.500`.
- Therefore "correct sample" filtering should be trajectory-level
  (`strategy_success`, `setup_success`, `tempo_success`, no early-end, clean
  opponent) instead of single-step attach/evolve heuristics. Attach/evolve
  miss from rollout trace is a diagnostic signal, not a direct filter label.

Training implication:

- Try at least three Alakazam experiments before concluding:
  1. strict `Majkel1337 + 7f9 + 1200+` specialist.
  2. strict `LiamK + ca08 + 1100-1199` specialist, evaluated as a separate
     Alakazam deck, not mixed into 7f9.
  3. 7f9 routed mix: Majkel 7f9 as a strong team-specific core plus broader
     7f9/Alakazam data for opponent coverage. Because own score bands are
     static, build climb proxies from chronological order and opponent
     coverage rather than filtering by own score band alone.

## DVH Marnie Historical Submission Recheck 2026-08-11

User reported `/tmp/submission_marnie_dvh.tar.gz` on `ks` maintained around
1000 Kaggle score for roughly four days and asked whether that was just an
early-climb advantage or whether it truly holds up against the local high pool.

Remote extraction/check:

```text
submission: /tmp/submission_marnie_dvh.tar.gz
unpacked policy/deck:
  logs/dvh_vs_high_20260811/unpack/policy.npz
  logs/dvh_vs_high_20260811/unpack/deck.csv
deck_sig: 3121746f2b28
manifest: logs/dvh_vs_high_20260811/high_plus_dvh_manifest.csv
```

DVH random:

```text
logs/dvh_vs_high_20260811/random_dvh_g500.csv
random_wr=1.000 in eval_manifest_random.py g500
standalone eval_bc log had 499/500 = 99.8%
```

DVH plus high1100 RR:

```text
rr: logs/dvh_vs_high_20260811/rr_dvh_plus_high_g100.csv
matrix: logs/dvh_vs_high_20260811/rr_dvh_plus_high_g100_arch_matrix.csv
players-weighted: logs/dvh_vs_high_20260811/rr_dvh_plus_high_g100_ladder0810_players_weighted.csv
rows-weighted: logs/dvh_vs_high_20260811/rr_dvh_plus_high_g100_ladder0810_rows_weighted.csv
games: 100 per pair, 9 entries, 36 matchups
```

Key DVH row from the RR matrix:

```text
dvh_marnie_3121746f2b28
macro_avg=0.753
weighted_avg=0.758
worst_archetype=Mega Lopunny
worst_wr=0.680
vs Alakazam=0.690
vs Dragapult=0.920
vs Mega Lopunny=0.680
vs Mega Lucario=0.690
vs Teal Mask Ogerpon=0.785
```

0810 ladder distribution weighted comparison:

```text
player_entries climb_weighted:
  dvh_marnie_3121746f2b28              0.7442
  high1100_marnie_grimmsnarl_b8f251a4 0.7253

decision_rows climb_weighted:
  dvh_marnie_3121746f2b28              0.7515
  high1100_marnie_grimmsnarl_b8f251a4 0.7345
```

Live-loss shadow comparison, using shadows trained from recent opponents that
beat the submitted high1100 Marnie:

```text
log: logs/live_loss_shadows_20260811/dvh_vs_high_marnie_live_loss_trained_g200.log
csv: logs/live_loss_shadows_20260811/dvh_vs_high_marnie_live_loss_trained_g200.csv
baseline: high_marnie_b8f
candidate: dvh_marnie_3121746f2b28
opponents: 6 trained live-loss shadows
games: 200 per opponent

summary:
  dvh avg_delta=+0.027
  dvh candidate_wr=0.732
  high_marnie_b8f baseline_wr=0.706
  dvh lost 1/6, worst live_loss_mega_lucario_43d6d8b0fce9 delta=-0.065
```

Interpretation: DVH's four-day 1000-ish Kaggle behavior is unlikely to be only
an early-climb artifact. In the local high1100 pool it is at least competitive
with, and slightly better than, the newly trained high1100 Marnie b8f. The
important caveat is that DVH uses a different Marnie deck signature
`3121746f2b28`, so the gain may be deck-list quality as much as policy quality.
Future Marnie recovery should include this sig/list as a first-class training
target instead of only retraining `b8f251a476e7`.

### DVH Forensics 2026-08-12

User asked to recover what DVH actually was, likely from a very early version.
Remote scan covered `/home/byer/PTCG` and `/home/jie/Do/0_PTCG`.

Exact DVH submission facts:

```text
submission: /tmp/submission_marnie_dvh.tar.gz
tar sha256: 11311ccd15f877c36db166d900cb647b6dcb9c44c7782dbc99a666f63022ed36
tar size: 9,592,530 bytes
tar mtime: 2026-08-11 21:48:17
policy sha256: b7b1e386e5ee2d03be15e0f8bd14b7a35c9301bc37c1dad990069b3d1d939c97
policy size: 7,580,139 bytes
deck sha256: 92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d
deck sig: 3121746f2b28
main.py sha256: 0935fc939163bfecf62b266ecba4c0b87145dbca10d48a21615a692f1cd9cedc
```

Policy architecture:

```text
legacy pointer MLP, not cross_attn
no history/log-history/board-history arrays
no hierarchical plan head
main.py has USE_MCTS=False and greedy NumPy policy selection
width=2.0
slot_state=True
state_feat_dim=64
opt_feat_dim=48
state_fc1.weight=(1024,704)
opt_fc.weight=(256,528)
score_fc1.weight=(256,1024)
approx params: 2,043,778
```

Important negative result: the exact DVH `policy.npz` hash was not found in
retained checkpoint trees. The only exact match found was the unpacked audit
copy at:

```text
/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/dvh_vs_high_20260811/unpack/policy.npz
```

The old byer submission
`/home/byer/PTCG/workspace/ptcg_rl_git/submissions/marnie_v7sig_winweighted.tar.gz`
has the same `deck.csv` and same `main.py` as DVH, but a different policy:

```text
old policy sha256: fe2eda9da96fa77ee2c49a0526c1a43293ca8c87d17567a93a77715b5a89da44
old policy shape: state_fc1=(1024,688), opt_fc=(256,512)
old inferred dims: state_feat_dim=48, opt_feat_dim=32
```

Closest recoverable early training logs are from byer, but they are not exact
DVH because their retained checkpoints are `48/32`, not DVH's `64/48`:

```text
logs/train_marnie_grimmsnarl_top1_v8_topdeck_w2.log
  BC2: Marnie Grimmsnarl ['1200+', '1100-1199', '1000-1099']
  width=2.0 slot_state=True deck_sigs=['b8f251a476e7']
  winner_only=False win/loss/draw=1.5/0.4/0.8
  files=30 raw=3,196,393 kept=2,416,614
  split train=2,293,129 val=123,485 batch=4096
  epochs=8, best val=0.7046
  retained checkpoint sha256=e885f822b7df28e06956c0e79b8dd9ef32516b0d902a09c9db6c83924f975e24
  retained checkpoint shape: state_fc1=(1024,688), opt_fc=(256,512)

logs/bc2_marnie_grimmsnarl_v8_mixed_1000_w2.log
  BC2: Marnie Grimmsnarl ['1200+', '1100-1199', '1000-1099']
  width=2.0 slot_state=True deck_sigs=all
  winner_only=False win/loss/draw=1.5/0.4/0.8
  files=30 raw=3,196,393 kept=3,181,893
  split train=2,948,828 val=233,065 batch=4096
  epochs=8, best val=0.7492
  retained checkpoint sha256=809ec2fb0d2c8aca7a08466696791130691251d5ecf41a50ebf4dd34220a2396
  retained checkpoint shape: state_fc1=(1024,688), opt_fc=(256,512)
```

Current best interpretation:

- DVH is not an RL/MCTS/history/attention model. It is a simple early BC
  pointer policy in the old submission wrapper.
- DVH used the old Marnie deck list `3121746f2b28`, but its policy is a later
  `64/48` feature policy, so it was probably repacked into the old wrapper/deck
  after the original v7/v8 submission.
- The original training checkpoint was probably deleted, overwritten, or
  generated in a temporary path. Treat the exact parameters as unrecovered.
- To reproduce DVH behavior, run ablations with the old simple BC recipe:
  width 2, batch 4096, lr 1e-4, score bands `1000+`, no winner-only,
  win/loss/draw `1.5/0.4/0.8`, no set/history/plan/cross-attn, but force
  `state_feat_dim=64` and `opt_feat_dim=48`. Package each resulting policy with
  both deck `3121746f2b28` and deck `b8f251a476e7` to isolate deck-list vs
  policy effects.

## Ladder Distribution 2026-08-10

User asked why Ogerpon is still struggling on Kaggle ladder. A single-day
distribution analysis over the extracted v12 0801-0810 corpus was run for
`pokemon-tcg-ai-battle-episodes-2026-08-10`.

Important caveat: scores/bands are from the leaderboard snapshot used in the
0811 extraction, not necessarily the exact score at episode time.

Remote script:

```text
/tmp/analyze_0810_ladder_distribution.py
```

Outputs:

```text
logs/ladder_distribution_0810_20260811/summary.log
logs/ladder_distribution_0810_20260811/archetype_summary_0810.csv
logs/ladder_distribution_0810_20260811/band_archetype_players_0810.csv
logs/ladder_distribution_0810_20260811/band_archetype_rows_0810.csv
logs/ladder_distribution_0810_20260811/deck_sig_summary_0810.csv
```

0810 coverage:

```text
players_600+: 9065 episode-player entries
decision_rows_600+: 743737
max_score in this snapshot: 1185.7
no 1200+ band in this extracted snapshot
```

Archetype distribution by episode-player entries, 600+:

```text
Marnie Grimmsnarl      2664 29.39%, max 1182.0
Alakazam               1761 19.43%, max 1175.1
Teal Mask Ogerpon      1067 11.77%, max 1182.0
Mega Lopunny           1028 11.34%, max 1131.8
Dragapult               836  9.22%, max 1163.0
Crustle Wall            526  5.80%, max 1119.5
Mega Lucario            366  4.04%, max 1185.7
Festival Lead           364  4.02%, max 1103.8
```

High band 1100-1199:

```text
Mega Lucario          23.79%
Marnie Grimmsnarl     20.89%
Teal Mask Ogerpon     19.20%
Dragapult             12.32%
Mega Lopunny          11.02%
Alakazam               3.60%
Crustle Wall           2.83%
```

Mid band 1000-1099:

```text
Marnie Grimmsnarl     28.08%
Alakazam              23.15%
Dragapult             12.90%
Mega Lopunny          12.51%
Teal Mask Ogerpon      9.25%
Festival Lead          7.09%
Crustle Wall           3.98%
```

Ogerpon-specific finding:

```text
Top 0810 Ogerpon signatures by max score:
90abbfb0eee0  max=1182.0, players=116, WR=0.690, team=palsystem
2bd9da52c43a  max=1164.3, players= 95, WR=0.600, team=James Cox & Henry Chao
d17573abc0e3  max=1163.0, players= 63, WR=0.667, team=Sixth Sense
cc81a7ad2d24  max=1152.8, players= 12, WR=0.583, team=Sixth Sense
85d941c82b94  max=1059.5, players=122, WR=0.598
6205fc379380  max=1058.9, players=186, WR=0.575
4fc9a6206c93  max=1048.9, players= 98, WR=0.577

Old/local Ogerpon sigs:
697a82e582d5  max=1039.6, players=175, WR=0.474
5899c772bace  max= 965.2, players= 33, WR=0.303
```

Interpretation: Ogerpon as an archetype is not dead; it is 19.2% of the
1100-1199 band in this snapshot. The problem is that the old `5899/697`
signatures are no longer representative of the current high-performing Ogerpon
population. Future Ogerpon work should prioritize extracting/training/evaluating
`90abbfb0eee0`, `2bd9da52c43a`, `d17573abc0e3`, and `cc81a7ad2d24` before
spending more submission attempts on old `5899/697`.

## Active Remote Job: BC Recovery Retrain 2026-08-11

User asked to recover historical strong BC behavior after the ladder shifted and
new August episodes were added. This is BC retraining on an August-only window,
not weak-matchup fine-tuning. The target is to reproduce/refresh previously good
Marnie `b8f251a476e7` lines:

- old `w4_marnie_sig1_b8f251a4`
- old `marnie_b8f_v12hist_bigbatch_b1536_20260809`

Remote runner:

```text
script: /tmp/run_bc_retrain_0801_0810_20260811.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
raw episodes: /home/jie/Do/0_PTCG/workspace/episodes_raw
episode window: 2026-08-01 through 2026-08-10, 10 zip files
leaderboard: /tmp/lb_0811_bc_retrain/leaderboard.csv
leaderboard source: pokemon-tcg-ai-battle-publicleaderboard-2026-08-11T01:19:26.csv, 6717 teams
corpus: data/bc_corpus_banded_v12_0801_0810_hist32_log128_board12_20260811
log dir: logs/bc_retrain_0801_0810_20260811
checkpoint dir: checkpoints/bc_retrain_0801_0810_20260811
runner log: logs/bc_retrain_0801_0810_20260811/runner.log
extract log: logs/bc_retrain_0801_0810_20260811/extract_v12_0801_0810_hist32_log128_board12.log
```

The runner first extracts a fresh v12 multistream-history corpus from only
`2026-08-01..2026-08-10` with:

```text
workers=10
action_history_k=32
log_history_k=128
board_history_k=12
board_history_feat_dim=32
```

After extraction completes, it launches three parallel Marnie `b8f251a476e7`
BC trainings:

```text
1. w4legacy_b256 on GPU0
   save: checkpoints/bc_retrain_0801_0810_20260811/bc2_marnie_b8f_w4legacy_0801_0810_b256.npz
   recipe: old v11all35 w4 objective, pointer/width=4, batch=256, epochs=7,
           lr=8e-5, score bands 900+, win/loss/draw=1.5/0.4/0.8,
           first_action=1.6, option=0.2, set_loss=0.12,
           MAIN/ATTACH/SKILL_ORDER and EVOLVE/ATTACK weights preserved.

2. v12hist_init_b1536 on GPU1
   save: checkpoints/bc_retrain_0801_0810_20260811/bc2_marnie_b8f_v12hist_cross_init_0801_0810_b1536.npz
   recipe: historical 2026-08-09 big-batch objective, cross_attn/width=4,
           state_layers=2, history/log/board history enabled, batch=1536,
           epochs=8, lr=3e-5, 56GiB cap, split-by-game.
   init: checkpoints/v12_history_pilots_20260807/bc2_marnie_b8f_v12hist_cross_init.npz
   note: this init variant is included because it exactly reproduces the old
         good bigbatch recipe; it should not be confused with weak-matchup
         fine-tuning.

3. v12hist_scratch_b1536 on GPU2
   save: checkpoints/bc_retrain_0801_0810_20260811/bc2_marnie_b8f_v12hist_cross_scratch_0801_0810_b1536.npz
   recipe: same v12 history/cross_attn data path but no init, epochs=10,
           lr=8e-5, batch=1536, split-by-game.
```

Automatic post-eval after training:

```text
random g500 for old w4 and all new checkpoints:
  logs/bc_retrain_0801_0810_20260811/random_*.log

paired baseline-delta vs old w4 over strict 2026-08-10 primary pool:
  logs/bc_retrain_0801_0810_20260811/marnie_0801_0810_vs_old_w4_primary_g80.csv
  logs/bc_retrain_0801_0810_20260811/marnie_0801_0810_vs_old_w4_primary_g80.log

focused Marnie vs Ogerpon 5899/697 w4, 300 games each:
  logs/bc_retrain_0801_0810_20260811/marnie_0801_0810_vs_ogerpon_w4_g300.csv
  logs/bc_retrain_0801_0810_20260811/marnie_0801_0810_vs_ogerpon_w4_g300.log
```

### Status Update 2026-08-11 10:05

Important: the original runner did **not** correctly wait for its background
training jobs. `run_train` returned PIDs from a command substitution/subshell,
then `wait` reported:

```text
/tmp/run_bc_retrain_0801_0810_20260811.sh: line 246: wait: pid ... is not a child of this shell
```

The follow-on `eval_baseline_delta.py` error:

```text
error: the following arguments are required: --candidate
```

was caused by the eval stage starting before new checkpoints existed, so no
candidate args were appended. Treat that as a runner/eval orchestration bug, not
as evidence that extraction or training failed.

Extraction completed successfully:

```text
corpus: data/bc_corpus_banded_v12_0801_0810_hist32_log128_board12_20260811
summary_files: 562
summary_rows: 7611041
feature_version: v12_multistream_history
state_feat: (80,)
option_feat: (1, 64)
own_history: (32,)
log_history: (128,)
board_history_cards/features: (12, 12)/(12, 32)
```

Top extracted rows by archetype:

```text
Marnie_Grimmsnarl 3,093,272
Alakazam 1,042,480
Mega_Lopunny 910,303
Teal_Mask_Ogerpon 700,478
Dragapult 662,126
Crustle_Wall 393,796
```

Actual training status at 2026-08-11 10:05:

```text
Marnie w4legacy_b256: running, epoch 5/7 in progress.
Marnie v12hist_init_b1536: running, epoch 2/8 in progress; epoch 1 val=0.5195.
Marnie v12hist_scratch_b1536: running, epoch 2/10 in progress; epoch 1 val=0.6304.
Ogerpon 5899 w4legacy: complete, best val=0.6389 at epoch 6/7.
Ogerpon 697 w4legacy: complete, best val=0.9626 at epoch 6/7.
```

Ogerpon random g500 on the freshly retrained models:

```text
5899 new 0801-0810: 497/500 = 99.4%
697 new 0801-0810:  487/500 = 97.4%
old w4 5899 in the premature runner eval: 499/500 = 99.8%
old w4 697 in the premature runner eval:  492/500 = 98.4%
```

A corrected post-eval watcher has been uploaded and started:

```text
remote script: /tmp/run_bc_retrain_0801_0810_posteval_20260811.sh
runner log: logs/bc_retrain_0801_0810_20260811/posteval_runner.log
posteval dir: logs/bc_retrain_0801_0810_20260811/posteval
```

It is idempotent: existing logs are skipped. It immediately runs Ogerpon strict
pool/focused deltas, then waits until all Marnie training processes exit before
running Marnie random and paired delta tests.

Ogerpon g80/g300 post-eval results already produced:

```text
5899 vs old w4, strict primary pool g80:
  avg_delta=+0.0508, candidate=0.5808, baseline=0.5300, lost=3/15
  worst=marnie_grimmsnarl_sig2_2c22fa76:-0.0375
  best=alakazam_sig3_91f48d8e:+0.2000

697 vs old w4, strict primary pool g80:
  avg_delta=-0.0175, candidate=0.5192, baseline=0.5367, lost=9/15
  worst=festival_lead_sig1_e82dcbe6:-0.1625
  best=marnie_grimmsnarl_sig1_b8f251a4:+0.0875

5899 vs Crustle focused g300:
  avg_delta=-0.0133, candidate=0.0250, baseline=0.0383, lost=2/2

697 vs Crustle focused g300:
  avg_delta=+0.0100, candidate=0.0550, baseline=0.0450, lost=1/2
```

Interpretation: the 0801-0810 Ogerpon 5899 retrain improves broad strict-pool
RR signal but does not solve the Crustle weakness. The 697 retrain is weaker on
the broad pool, although it slightly improves one Crustle focused row.

A higher-confidence Ogerpon paired delta finished:

```text
remote script: /tmp/run_ogerpon_rr_g200_20260811.sh
runner log: logs/bc_retrain_0801_0810_20260811/posteval/ogerpon_primary_g200_runner.log
outputs:
  logs/bc_retrain_0801_0810_20260811/posteval/ogerpon5899_0801_0810_vs_old_w4_primary_g200.{csv,log}
  logs/bc_retrain_0801_0810_20260811/posteval/ogerpon697_0801_0810_vs_old_w4_primary_g200.{csv,log}
status at 2026-08-11 10:17:
  complete, both rc=0.

5899 vs old w4, strict primary pool g200:
  avg_delta=-0.0043, weighted_delta=-0.0144
  candidate=0.5537, baseline=0.5580, lost=8/15
  worst=cynthia_garchomp_sig1_52f46739:-0.0900
  other losses include Ogerpon 697 mirror -0.0750 and Alakazam sig3 -0.0450
  best=festival_lead_sig3 +0.0750, marnie_sig2 +0.0750

697 vs old w4, strict primary pool g200:
  avg_delta=+0.0073, weighted_delta=+0.0037
  candidate=0.5340, baseline=0.5267, lost=6/15
  worst=marnie_sig1 -0.0650 and alakazam_sig3 -0.0650
  best=Ogerpon 697 mirror +0.1200 and Ogerpon 5899 mirror +0.0750
```

Interpretation update: the earlier 5899 g80 improvement was mostly variance.
At g200, the 0801-0810 retrain is not a clear replacement for old 5899 w4. The
new 697 has a tiny broad-pool gain, but its random was lower (`97.4%`) and it
did not materially solve Crustle. For submission, treat both as probes, not as a
strong upgrade over old w4.

Marnie `b8f251a476e7` w4legacy early eval finished before the history models:

```text
checkpoint:
  checkpoints/bc_retrain_0801_0810_20260811/bc2_marnie_b8f_w4legacy_0801_0810_b256.npz
training:
  complete epoch 7/7, best val=0.6629
random:
  logs/bc_retrain_0801_0810_20260811/posteval/marnie_w4legacy_early_random_g500.log
  500/500 = 100.0%
strict primary pool:
  logs/bc_retrain_0801_0810_20260811/posteval/marnie_w4legacy_early_vs_old_w4_primary_g200.csv
  avg_delta=+0.029, weighted_delta=+0.036
  candidate=0.623, old_w4_baseline=0.594, lost=3/15
focused Ogerpon:
  logs/bc_retrain_0801_0810_20260811/posteval/marnie_w4legacy_early_vs_ogerpon_g300.csv
  avg_delta=+0.005, candidate=0.163, baseline=0.158, lost=2/4
```

Strict g200 notable rows:

```text
largest gains:
  festival_lead_sig1: +0.115
  cynthia_garchomp_sig1: +0.100
  festival_lead_sig3: +0.080
  mega_lopunny_sig2: +0.075
  teal_mask_ogerpon_sig3_5899: +0.055

largest losses:
  mega_lopunny_sig3: -0.075
  alakazam_sig1: -0.060
  marnie_sig2: -0.010
```

Focused Ogerpon g300 rows:

```text
ogerpon5899_old: +0.013, 0.190 vs 0.177
ogerpon697_old:  -0.010, 0.163 vs 0.173
ogerpon5899_new: -0.017, 0.113 vs 0.130
ogerpon697_new:  +0.033, 0.187 vs 0.153
```

Interpretation: the Marnie 0801-0810 w4legacy retrain is a plausible broad-pool
upgrade over old w4, unlike Ogerpon 5899 where g200 erased the apparent g80
gain. It still does not solve the Ogerpon structural weakness; focused WR
remains around 16%.

Monitor the corrected Marnie watcher:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/bc_retrain_0801_0810_20260811/posteval_runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "bc2_train.py.*bc_retrain_0801_0810_20260811|run_bc_retrain_0801_0810_posteval" || true'
```

Initial status at launch:

```text
started: 2026-08-11 09:19 Asia/Shanghai
runner pid observed: 3228423
extract parent observed: 3228486
GPU before train: all four A800 effectively free, ~81GiB free each
RAM before extract: ~39GiB used, ~962GiB available
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/bc_retrain_0801_0810_20260811/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/bc_retrain_0801_0810_20260811/extract_v12_0801_0810_hist32_log128_board12.log'
ssh ks 'pgrep -af "run_bc_retrain_0801_0810_20260811|bc_extract_v2.py|bc2_train.py"'
ssh ks 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits'
```

### Ogerpon Add-On 2026-08-11

User also asked whether any locally strong Ogerpon decks should be retrained on
the same `2026-08-01..2026-08-10` data. Local strict RR evidence:

```text
logs/eval_deck_sig_specialists_v11all35_20260806/random_w4_all39_g200_joined.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w4_random_ge097_g80_summary.txt
logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_primary.csv

5899c772bace:
  old w4 random: 200/200 = 1.000
  old strict RR mean: 0.597656
  best Ogerpon row in the strict pool.

697a82e582d5:
  old w4 random: 197/200 = 0.985
  old strict RR mean: 0.579688
  second-best Ogerpon row in the strict pool.

2a5072194fdf:
  old w4 random: 177/200 = 0.885
  excluded from the strict pool. It remains strategically interesting for some
  Crustle/Ogerpon route analysis, but is not the local-RR-good retrain target.
```

Remote watcher:

```text
script: /tmp/run_bc_retrain_0801_0810_ogerpon_20260811.sh
log dir: logs/bc_retrain_0801_0810_ogerpon_20260811
runner log: logs/bc_retrain_0801_0810_ogerpon_20260811/runner.log
status at launch: waiting for logs/bc_retrain_0801_0810_20260811/extract.done
```

When the base extraction is done, it launches two scratch BC retrains using the
old w4 objective on the fresh v12 corpus:

```text
1. Ogerpon 5899, GPU3
   save: checkpoints/bc_retrain_0801_0810_20260811/bc2_ogerpon_5899_w4legacy_0801_0810_b256.npz

2. Ogerpon 697, GPU0
   save: checkpoints/bc_retrain_0801_0810_20260811/bc2_ogerpon_697_w4legacy_0801_0810_b256.npz

recipe for both:
  corpus: data/bc_corpus_banded_v12_0801_0810_hist32_log128_board12_20260811
  archetype: Teal Mask Ogerpon
  score bands: 1200+ 1100-1199 1000-1099 900-999
  date: 2026-08-01..2026-08-10
  width=4 pointer/legacy w4 objective, batch=256, epochs=7, lr=8e-5
  win/loss/draw=1.5/0.4/0.8
  first_action=1.6, option=0.2, set_loss=0.12
  MAIN/ATTACH/SKILL_ORDER and EVOLVE/ATTACK weights preserved
```

Automatic Ogerpon post-eval:

```text
random g500 for old/new 5899 and old/new 697:
  logs/bc_retrain_0801_0810_ogerpon_20260811/random_*.log

paired baseline-delta vs old corresponding w4 over strict primary pool:
  logs/bc_retrain_0801_0810_ogerpon_20260811/ogerpon_5899_0801_0810_vs_old_w4_primary_g80.csv
  logs/bc_retrain_0801_0810_ogerpon_20260811/ogerpon_697_0801_0810_vs_old_w4_primary_g80.csv

focused Ogerpon vs Crustle 3cd/b141 w4, 300 games each:
  logs/bc_retrain_0801_0810_ogerpon_20260811/ogerpon_5899_0801_0810_vs_crustle_w4_g300.csv
  logs/bc_retrain_0801_0810_ogerpon_20260811/ogerpon_697_0801_0810_vs_crustle_w4_g300.csv
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 100 logs/bc_retrain_0801_0810_ogerpon_20260811/runner.log'
ssh ks 'pgrep -af "run_bc_retrain_0801_0810_ogerpon_20260811|bc2_train.py.*ogerpon.*0801_0810"'
```

## Completed Remote Job: Search Teacher Wave1 2026-08-10

Reason: scratch PPO learned high random win rate but became worse than BC on
hard fixed opponents. The current direction is no longer PPO hyperparameter
tuning. It builds replayable weak-matchup state banks, then uses engine search
rollouts to generate planner/teacher labels for specific decision states.

New tools:

```text
tools/build_weakness_state_bank.py
  Plays candidate vs opponent, records candidate-side raw observations,
  baseline actions, policy rankings, policy-history arrays, and compact
  board/option summaries. Output is JSONL/JSONL.GZ plus CSV indices.

tools/search_action_teacher.py
  Reads state bank JSONL, enumerates legal root actions, uses cg.api
  search_begin/search_step to roll each action forward, and scores actions by
  terminal result plus horizon heuristic score. Output includes action-level
  CSV, best-action CSV, and teacher JSONL.
```

Pilot result before wave1:

```text
pair: Ogerpon 2a vs Crustle 3cd
midgame filters: loss states, context=0, turn>=4
teacher: 8 states, 4 rollouts/action, horizon 160
result: improved_score=6/8, mean delta_score=+0.5524
example signal:
  baseline Teal Mask Ogerpon ability was often worse than attach/Energy Switch
  or bench setup in searched midgame states.
interpretation:
  search teacher now produces non-trivial decision signals. This is the first
  promising signal after BC/PPO plateau, but it is not yet validated in actual
  policy training or rule-overlay evaluation.
```

Active remote wave:

```text
remote script: /tmp/run_search_teacher_wave1_20260810.sh
runner PID: 1290943, completed status=0 at 2026-08-10 07:27 CST
logs root: logs/search_teacher_20260810

pairs:
  wave1_og2a_vs_crustle3
  wave1_marnie_vs_og5899
  wave1_lucario43_vs_marnie
  wave1_dragcc2_vs_marnie

per pair:
  build bank: up to 80 loss states, context=0, turn>=4, min options=3
  teacher: first 24 states, 8 rollouts/action, horizon 180, max 16 actions,
           4 worker processes
```

Wave1 summary:

```text
summary file: logs/search_teacher_20260810/wave1_summary.csv
motifs file:  logs/search_teacher_20260810/wave1_motifs.csv

dragcc2_vs_marnie:
  states=24, improved_score=18/24, strong>=0.20=14
  mean_delta_score=+0.276, mean WR 0.052 -> 0.156

lucario43_vs_marnie:
  states=24, improved_score=20/24, strong>=0.20=18
  mean_delta_score=+0.450, mean WR 0.240 -> 0.458

marnie_vs_og5899:
  states=24, improved_score=17/24, strong>=0.20=16
  mean_delta_score=+0.349, mean WR 0.240 -> 0.406

og2a_vs_crustle3:
  states=24, improved_score=19/24, strong>=0.20=19
  mean_delta_score=+0.341, mean WR 0.156 -> 0.328
```

Important negative validation:

```text
tools/eval_search_guided.py was added to apply the search teacher online during
real local battles. Two Ogerpon-vs-Crustle smoke runs failed:

ungated online search:
  logs/search_teacher_20260810/online_og2a_vs_crustle3_search_guided_g4.log
  result: 0/4 wins
  issue: search changed setup/YES-NO/early decisions and introduced bad choices.

gated online search:
  logs/search_teacher_20260810/online_og2a_vs_crustle3_search_guided_gated_g8.log
  gate: turn>=4, context=0, option_count>=3
  result: 0/8 wins
  changed decisions: 205/373, positive delta_score on 205/373, mean +0.273
  issue: many local positive deltas did not compose into whole-game wins; some
  best actions are still suspicious, e.g. END or delay actions.
```

Current interpretation:

```text
The search teacher is promising as an offline signal generator, not as a direct
online greedy policy. Do not use online search-guided policy for submission.
Next step should be high-confidence teacher filtering and route/rule conversion:
  - exclude setup/YES-NO/non-context=0 states;
  - exclude END/STOP best actions unless terminal rollout WR improves strongly;
  - require delta_score >= 0.20 and either best WR improves or motif repeats;
  - mine repeated motifs per matchup and encode them as route rules or
    scratch distillation labels.
```

## Active Remote Job: Rule-Guided Rollout Stage1b 2026-08-10

Current user direction: rules must be used for training, and rule hypotheses
must be grounded in Kaggle/community/top-strategy ideas. BC is still locked:
do not fine-tune existing BC checkpoints. Use generated rule/search/planner
data only for scratch/retraining experiments.

Sources incorporated:

- Kaggle topic `724362`: top methods inferred from 30k games; top appears more
  like model + bounded search/RL, but search needs a good value estimate.
- Kaggle topic `717697`: RL journey notes emphasize representation, replay
  analysis, broad card exposure, and curriculum; random self-play alone is not
  enough.
- Kaggle topic `708586`: simulator behavior is authoritative over official TCG
  differences.
- PTCG strategy/community notes: official Mega Lucario route
  (Solrock/Lunatone/Fighting Gong/Poke Pad/Premium Power Pro), Marnie
  Grimmsnarl engine, Ogerpon/Kangaskhan box secondary/search route, and Crustle
  anti-ex wall.

Code/data changes made:

- `tools/generate_rollout_bc.py`: `+rules:resource_plan` now instantiates
  `ResourcePlanner` per game and uses true stateful route decisions. Previously
  this was effectively a no-op because it called stateless `apply_rule_overlay`.
- `tools/build_weakness_state_bank.py`: same `resource_plan` support for
  state-bank/action-source traces.
- `ptcg_rl/resource_planner.py`: expanded routes beyond Ogerpon/Marnie/Lucario:
  Crustle anti-ex wall, Team Rocket board-count route, generic stage/setup
  route, and Mega Starmie route.
- `ptcg_rl/deck_plans.py`: added Mega Starmie plan:
  Staryu/Mega Starmie ex + Duskull/Dusclops/Dusknoir + Hilda/Grand Tree/Wally.
- `tools/filter_search_teacher_labels.py`: new high-confidence filter for
  `*_teacher_best.csv`. Default excludes STOP/END/YES/NO/NUMBER/CARD and keeps
  only clear `delta_score` labels.
- `data/matchup_strategy_seeds_v1.csv` and
  `data/matchup_strategy_seed_cards_v1.csv`: added Lucario route, Ogerpon box
  route, and Mega Starmie route seeds/card mappings.
- `docs/11_human_matchup_strategy.md`: updated with Kaggle constraints,
  no-finetune rule, rule-guided rollout commands, and search-label filtering.

Remote sync status:

- Synced changed source/data/docs to:
  `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`.
- A mistaken `scp` placed four duplicate files in the remote repo root; they
  were removed. Official files under `data/`, `docs/`, and `tools/` were
  verified present after cleanup.

Important operational note:

- First attempt `/tmp/run_rule_guided_stage1_20260810.sh` used
  `parallel_jobs=4`, `workers=24`, spawned about 100 rollout processes, and
  pushed ks load above 1700. It was killed before producing NPZ files. Do not
  repeat that concurrency.
- Active safer run:
  - script: `/tmp/run_rule_guided_stage1b_20260810.sh`
  - runner log: `logs/rule_guided_20260810b/stage1b.runner.log`
  - rollout plan: `logs/rule_guided_20260810b/rollout_teacher/rollout_teacher_plan.csv`
  - generated corpus: `data/generated_rollout_bc_resource_plan_20260810b`
  - settings: `max_jobs=18`, `games=400`, `workers=12`, `parallel_jobs=2`,
    `keep_outcomes=nonloss`, `rule_mode=resource_plan`.
  - coverage in plan: Dragapult 3, Festival Lead 3, Marnie 3, Mega Lucario 1,
    Mega Starmie 3, Ogerpon 3, Team Rocket Mewtwo 2.
  - At first check it had 2 rollout jobs and 27 matching processes. CPU was
    busy but not overloaded. No NPZ had been flushed yet because workers were
    still around 1/33 games.

Stage1b pre-results:

- Search teacher filter read 96 labels and kept 12:
  - rejects: low_best_score=50, low_delta_score=28, excluded_type:END=6.
  - top kept motifs: Lucario vs Marnie Premium Power Pro, ATTACK, Spikemuth
    Gym; one Ogerpon vs Crustle Boss's Orders; one Marnie vs Ogerpon darkness
    attach.
- Interpretation: search labels are useful as offline motifs, not direct online
  greedy policy. Filtered label count is small, so the main data source is the
  `resource_plan` rollout teacher, not search labels alone.

Next actions after Stage1b:

1. Check `logs/rule_guided_20260810b/rollout_teacher/rollout_teacher_summary_final.csv`
   and generated NPZ actor modes. Confirm `actor_mode` contains `resource:...`
   labels; otherwise planner still did not trigger.
2. If per-job nonloss/win rates are too low or rows are tiny, do not train.
   Tighten route rules or target only the archetypes with generated wins.
3. If rows are adequate, train scratch policies only, e.g. v12 full corpus plus
   `--aux-corpus data/generated_rollout_bc_resource_plan_20260810b --aux-repeat 8`.
   Do not use `--init` from BC; at most compare pointer/cross-attn/history
   scratch runs.
4. Validate against random, focused weakness pairs, balanced shadow RR, and
   Kaggle replay-derived opponents before considering a submission candidate.

## Completed Remote Job: Scratch League RL 2026-08-09/10

Hard constraint from the user: **BC is locked. Do not fine-tune BC anymore.**
Existing BC checkpoints are only templates/baselines/fixed opponents. Current RL
jobs use `tools/rl_train_league.py` with:

```text
phase1: --init-mode random
phase2: --init-mode resume from the phase1 scratch checkpoint
no BC anchor
no ref-KL
```

Remote roots:

```text
logs:        logs/rl_scratch_league_20260809
checkpoints: checkpoints/rl_scratch_league_20260809
main script: /tmp/run_scratch_league_rl_20260809.sh
Ogerpon retry: /tmp/run_ogerpon_scratch_retry_20260809.sh
Lucario retry: /tmp/run_lucario_scratch_retry_20260809.sh
```

Final status at 2026-08-10 07:07 CST:

```text
All jobs finished. No matching rl_train_league/eval processes remain.
All four A800 GPUs are idle at ~2 MiB used.

Marnie b8f:
  phase1 completed, final rollout WR 0.8125
  phase2 completed 72/72, final rollout WR 0.354
  random300: 99.3% (298/300)
  baseline-delta vs Ogerpon pool:
    og5899: 15.0% -> 1.25%, -13.75pp
    og697:  17.5% -> 0.0%,  -17.5pp
    og2a:   81.25% -> 33.75%, -47.5pp
  best checkpoint:
    checkpoints/rl_scratch_league_20260809/marnie_b8f_scratch_phase2_best.npz
  interpretation: random preserved, but substantially worse than BC on target
    matchups. Do not submit as an improvement candidate.

Dragapult cc2:
  phase1 completed, final rollout WR 0.8099
  phase2 completed 72/72, final rollout WR 0.281
  random300: 99.7% (299/300)
  baseline-delta vs core:
    marnie:   11.25% -> 0.625%, -10.625pp
    crustle3: 14.375% -> 5.625%, -8.75pp
    lucario:  43.125% -> 35.625%, -7.5pp
  interpretation: random preserved, but hard-matchup policy is worse than BC.

Ogerpon 2a:
  original 24GB job OOMed.
  48GB retry completed:
    runner log: logs/rl_scratch_league_20260809/ogerpon_2a_scratch_mem48.runner.log
  phase1 completed, final rollout WR 0.719
  phase2 completed 72/72, final rollout WR 0.328
  random300: 97.7% (293/300)
  baseline-delta vs Crustle:
    crustle3:  4.375% -> 5.0%, +0.625pp
    crustle96: 4.375% -> 5.0%, +0.625pp
    crustleb:  5.625% -> 1.25%, -4.375pp
  interpretation: not a meaningful breakthrough; one Crustle sig regressed.

Mega Lucario 43d:
  original 24GB phase1/phase2 both OOMed.
  48GB retry completed:
    runner log: logs/rl_scratch_league_20260809/lucario_43d_scratch_mem48.runner.log
  phase1 completed, final rollout WR 0.602
  phase2 completed 72/72, final rollout WR 0.383
  random300: 98.0% (294/300)
  baseline-delta vs core:
    marnie:   11.25% -> 0.625%, -10.625pp
    crustle3: 15.0% -> 3.75%, -11.25pp
    og5899:   37.5% -> 4.375%, -33.125pp
  interpretation: random preserved, but target matchups are much worse than BC.
```

Quick status command:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && for f in logs/rl_scratch_league_20260809/*scratch*metrics.csv; do echo ===$f===; tail -3 "$f"; done'
```

High-level interpretation: this scratch PPO wave proves the pipeline can train
from random to high random win rate, but it did not learn stronger policies than
BC against fixed high-quality opponents. The next aggressive direction should
not be BC fine-tuning; focus on changing the RL objective/curriculum/search
signal, e.g. opponent-specific terminal decomposition, self-play league with
much longer exploration before hard BC opponents, or planner/search-generated
targets.

## RL v3 Parallel Results 2026-08-09

User preference for scheduling: before starting GPU training, check `nvidia-smi`;
if idle memory is available, use multiple GPUs in parallel instead of single-GPU
serial execution. Keep CPU worker count reasonable, but do not serialize merely
out of habit.

Most of the PPO v3 parallel wave has completed. Current `ks` status at
2026-08-09 20:55 CST:

```text
GPU0: duplicate serial Lucario job still running from /tmp/run_rl_v3_wave2_20260809.sh
GPU1-3: idle
```

The active duplicate is:

```text
remote script: /tmp/run_rl_v3_wave2_20260809.sh
runner PID: 428410
current child: Mega Lucario 43d vs Marnie/Crustle/Ogerpon
train PID around check time: 540816 plus rollout workers
metrics: logs/rl_v3_wave2_20260809/lucario_43d_vs_marnie_crustle_og_rlv3.metrics.csv
checkpoint root: checkpoints/rl_v3_wave2_20260809/
note: a no-cuDNN parallel Lucario run already completed, so this serial job is
      mostly a duplicate control unless intentionally kept.
```

Completed v3 outcomes:

```text
Marnie b8f v3:
  metrics rows=24, rollout best WR=0.184 at iter22, final rollout WR=0.109
  random500=99.8% (499/500)
  delta vs Ogerpon pool: avg -2.1pp
    og5899 -2.3pp, og697 -2.0pp, og2a -2.0pp
  interpretation: stable random, but weak-pool PPO made matchups worse.

Dragapult cc2 v3p:
  metrics rows=20, rollout best WR=0.117 at iter13, final rollout WR=0.082
  random500=80.4% (402/500)
  delta vs pool: avg -1.4pp
    marnie -1.7pp, crustle -3.3pp, lucario +0.8pp
  interpretation: not usable; random remains poor and matchup avg worsened.

Mega Lucario 43d v3p no-cuDNN:
  metrics rows=20, rollout best/final WR=0.125
  random500=98.2% (491/500)
  delta vs pool: avg +3.6pp
    marnie +0.0pp, crustle +3.3pp, ogerpon +7.5pp
  interpretation: best signal in this wave; modest positive, not a large
  breakthrough.

Ogerpon 2a v3p no-cuDNN:
  metrics rows=24, rollout best WR=0.039 at iter19, final rollout WR=0.020
  random500=83.0% (415/500)
  delta vs Crustle pool: avg +3.8pp
    crustle3cd +0.6pp, crustleb141 +6.9pp
  interpretation: weak matchup improved slightly, but random collapsed too far
  for submission use.
```

Summary: PPO v3 is a stable constrained PPO baseline, not a major improvement
yet. The constraints prevented catastrophic KL drift, but also made updates too
small for structural weaknesses. Next RL attempts should be more aggressive:
curriculum / rule-shaped rewards / success-trajectory replay / staged KL
relaxation, especially for Ogerpon-vs-Crustle style matchups.

`tools/rl_finetune_vs_pool.py` now honors `PTCG_DISABLE_CUDNN=1`, matching
`tools/bc2_train.py` and `tools/bc2_accuracy.py`. This is required for
history/GRU checkpoints on the remote A800 environment; otherwise `.to(cuda)`
can fail before training starts.

Monitor:

```bash
ssh ks 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
ssh ks 'pgrep -af "rl_finetune_vs_pool.py|run_rl_v3|eval_bc.py|eval_baseline_delta.py"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/rl_v3_wave2_20260809/lucario_43d_vs_marnie_crustle_og_rlv3.train.log'
```

## Completed Remote Job: RL v2 Wave1 2026-08-09

Reason: BC/history/cross-attn variants have plateaued. The current experiment
switches from BC tuning to targeted weak-pool PPO with real large-model support.

Implementation now in use:

- `ptcg_rl/model.py`
  - `PolicyValueNet.evaluate_actions()` now forwards stored per-decision history
    into `encode_state()`.
  - `CrossAttentionPolicyValueNet.evaluate_actions()` was added, so PPO can
    actually fine-tune cross-attention/history checkpoints.
- `tools/rl_finetune_vs_pool.py`
  - infers checkpoint architecture/width/features/history dims;
  - supports parallel CPU rollout actors via `--rollout-workers`;
  - exports the current torch model to actor `.npz`, collects sampled games
    with `NumpyPolicy`, then refreshes old log-probs/value on GPU before PPO;
  - supports modest dense shaping and low-weight BC anchor.

Remote smoke test passed before the long run:

```text
script: /tmp/run_rl_v2_smoke_20260809.sh
log: logs/rl_v2_smoke_20260809/train_smoke.log
checkpoint: checkpoints/rl_v2_smoke_20260809/marnie_vs_og5899_rl_v2_smoke.npz
result: 1 iter, 4 games, W/L/D=2/2/0, saved successfully
```

Wave1 script:

```text
local script source: /tmp/run_rl_v2_wave1_20260809.sh
remote script: ks:/tmp/run_rl_v2_wave1_20260809.sh
remote runner PID: 153826, completed
root logs: logs/rl_v2_wave1_20260809
root checkpoints: checkpoints/rl_v2_wave1_20260809
started: 2026-08-09 15:08 CST
ended: 2026-08-09 17:40 CST, status=0
```

Wave1 jobs:

```text
GPU0 Marnie b8f big history/cross-attn vs Ogerpon 5899/697/2a
  init: checkpoints/marnie_nightly_20260809/bc2_marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_b1536.npz
  save: checkpoints/rl_v2_wave1_20260809/marnie_b8f_vs_ogerpon_rlv2.npz
  log:  logs/rl_v2_wave1_20260809/marnie_b8f_vs_ogerpon_rlv2.train.log

GPU1 Mega Lucario 43d history/cross-attn vs Marnie/Crustle/Ogerpon
  init: checkpoints/v12_0701_0807_history_baselines_20260808/bc2_mega_lucario_43d_v12_0701_0807_hist_cross_init.npz
  save: checkpoints/rl_v2_wave1_20260809/lucario_43d_vs_marnie_crustle_rlv2.npz
  log:  logs/rl_v2_wave1_20260809/lucario_43d_vs_marnie_crustle_rlv2.train.log

GPU2 Ogerpon 2a history/cross-attn vs Crustle 3cd/96d/b141
  init: checkpoints/v12_0701_0807_history_baselines_20260808/bc2_ogerpon_2a507_v12_0701_0807_hist_cross_init.npz
  save: checkpoints/rl_v2_wave1_20260809/ogerpon_2a_vs_crustle_rlv2.npz
  log:  logs/rl_v2_wave1_20260809/ogerpon_2a_vs_crustle_rlv2.train.log

GPU3 Dragapult cc2 W4 pointer vs Marnie/Crustle/Lucario
  init: checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_dragapult_sig2_cc2e995b_v11all35_sigpure_top3_w4.npz
  save: checkpoints/rl_v2_wave1_20260809/dragapult_cc2_vs_marnie_crustle_rlv2.npz
  log:  logs/rl_v2_wave1_20260809/dragapult_cc2_vs_marnie_crustle_rlv2.train.log
```

Final metrics:

```text
Marnie b8f vs Ogerpon:
  iterations=32, best rollout WR=0.414 at iter31, final WR=0.387
  random500=99.6%
  paired delta avg=+0.0478
  og5899: 15.3% -> 20.0%, +4.7pp
  og697:  19.7% -> 19.3%, -0.3pp
  og2a:   80.3% -> 90.3%, +10.0pp
  interpretation: useful but not universal; Ogerpon 697 did not improve.

Mega Lucario 43d vs Marnie/Crustle/Ogerpon:
  iterations=28, best/final rollout WR=0.164
  random500=98.4%
  paired delta avg=+0.0022
  marnie: 14.7% -> 14.0%, -0.7pp
  crustle: 17.7% -> 19.7%, +2.0pp
  og5899: 39.3% -> 38.7%, -0.7pp
  interpretation: random preserved, no meaningful matchup gain.

Dragapult cc2 vs Marnie/Crustle/Lucario:
  iterations=28, best rollout WR=0.102 at iter21, final WR=0.094
  random500=78.2%
  paired delta avg=-0.0500
  marnie: 10.7% -> 11.3%, +0.7pp
  crustle: 10.7% -> 5.0%, -5.7pp
  lucario: 46.0% -> 36.0%, -10.0pp
  interpretation: failed, policy quality collapsed relative to baseline.

Ogerpon 2a vs Crustle:
  iterations=28, best rollout WR=0.043 at iter3, final WR=0.039
  random500=69.4%
  paired delta avg=-0.0133
  crustle3cd: 5.7% -> 5.3%, -0.3pp
  crustle96d: 5.7% -> 4.3%, -1.3pp
  crustleb141: 5.0% -> 2.7%, -2.3pp
  interpretation: failed; this remains a structural weakness.
```

Output files:

```text
logs/rl_v2_wave1_20260809/*.random.log
logs/rl_v2_wave1_20260809/*.delta.csv
logs/rl_v2_wave1_20260809/runner.log
```

Do not use Dragapult or Ogerpon wave1 checkpoints for submission. Marnie is the
only wave1 model with a measurable paired gain and random preserved. Lucario is
stable but effectively neutral.

GPU note after completion: PTCG wave1 left no matching `rl_finetune_vs_pool`,
`eval_bc.py`, or `eval_baseline_delta.py` processes. GPU 1-3 remained occupied
by `/home/byer/ARC/ttt_fast` jobs, unrelated to this repo.

## Active Remote Job: RL v3 Wave2 2026-08-09

Reason: wave1 proved the PPO infrastructure works but also showed destructive
drift for Dragapult/Ogerpon and only partial gain for Marnie. PPO v3 adds a
stronger trust region and matchup balancing:

- action-level reference-policy KL via `--ref-kl-coef`;
- PPO2-style `--value-clip-eps`;
- `--target-kl` early stop within an update;
- `--advantage-normalization opponent` and `--advantage-clip`;
- batch-level `--reward-weight-mode opponent_inverse_winrate`;
- `--save-policy both` so best rollout and final checkpoint are both kept.

Implementation file changed:

```text
tools/rl_finetune_vs_pool.py
```

Smoke test passed:

```text
script: /tmp/run_rl_v3_smoke_20260809.sh
log: logs/rl_v3_smoke_20260809/marnie_v3_smoke.log
result: 1 iter completed, ref KL/value clip/reward weighting/early stop worked
```

Active wave2:

```text
script: ks:/tmp/run_rl_v3_wave2_20260809.sh
runner PID: 428410
logs: logs/rl_v3_wave2_20260809
checkpoints: checkpoints/rl_v3_wave2_20260809
started: 2026-08-09 18:59 CST
GPU: only GPU0, because GPU1-3 are occupied by /home/byer/ARC/ttt_fast
```

Wave2 order:

```text
1. Marnie b8f hard PPO v3 vs Ogerpon 5899/697 only
   init: checkpoints/marnie_nightly_20260809/bc2_marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_b1536.npz
   best: checkpoints/rl_v3_wave2_20260809/marnie_b8f_vs_og5899_og697_rlv3_best.npz
   final: checkpoints/rl_v3_wave2_20260809/marnie_b8f_vs_og5899_og697_rlv3_final.npz

2. Mega Lucario 43d PPO v3 vs Marnie/Crustle/Ogerpon
   init: checkpoints/v12_0701_0807_history_baselines_20260808/bc2_mega_lucario_43d_v12_0701_0807_hist_cross_init.npz
   best: checkpoints/rl_v3_wave2_20260809/lucario_43d_vs_marnie_crustle_og_rlv3_best.npz
   final: checkpoints/rl_v3_wave2_20260809/lucario_43d_vs_marnie_crustle_og_rlv3_final.npz
```

Monitor:

```bash
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && for f in logs/rl_v3_wave2_20260809/*train.log; do echo ===$(basename $f); grep -E "iter [0-9]{4}|parallel rollout (32|64|128|192|256)/256|Traceback|RuntimeError|Killed|exit=" $f | tail -20; done'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && for f in logs/rl_v3_wave2_20260809/*.metrics.csv; do echo ===$(basename $f); tail -5 $f; done 2>/dev/null || true'
```

## Completed Remote Job: Marnie Big-Batch Restart 2026-08-09

The first Marnie nightly run from `2026-08-09 01:40 CST` loaded all 31 files
and then failed before epoch 1 with `RuntimeError: cuDNN error:
CUDNN_STATUS_NOT_INITIALIZED`. No usable checkpoint was produced by that run.

The user then requested a restart with batch size increased according to current
GPU resources. GPU0 was almost completely free; GPU1/2/3 still had other jobs.
The new runner was uploaded to `ks:/tmp` and started at about
`2026-08-09 09:42 CST`, bound to GPU0 only:

```text
runner script:
  /tmp/run_marnie_nightly_nocudnn_bigrun_20260809.sh
run id:
  marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809
runner PID file:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.pid
active process pattern:
  tools/bc2_train.py ... --batch-size 1536 ... --save checkpoints/marnie_nightly_20260809/bc2_marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_b1536.npz
checkpoint dir:
  checkpoints/marnie_nightly_20260809
runner log:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.runner.log
train log:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.b1536.train.log
heartbeat:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.heartbeat.log
status:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.status
```

Training recipe:

```text
corpus: data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12
archetype: Marnie Grimmsnarl
deck_sig: b8f251a476e7
date window: 2026-07-30..2026-08-07
score bands: 1200+ 1100-1199 1000-1099 900-999
arch: cross_attn, width=4, state_layers=2
history: history_k=32, log_history_k=128, board_history_k=12, board_history_feat_dim=32
init: checkpoints/v12_history_pilots_20260807/bc2_marnie_b8f_v12hist_cross_init.npz
batch fallback order: 1536 at 56GB, then 1024 at 48GB, then 768 at 40GB
epochs: 8
lr: 3e-5
device: CUDA_VISIBLE_DEVICES=0, --device cuda:0
PTCG_DISABLE_CUDNN=1
weights: win/loss/draw=1.5/0.4/0.8, first_action=1.5, option=0.15, set_loss=0
checkpoint_every: 1
```

Run completed successfully at `2026-08-09 12:57 CST`; no fallback was needed.
The actual run used `batch=1536`, and the best checkpoint was epoch 7:

```text
Best val=0.5242 ->
  checkpoints/marnie_nightly_20260809/bc2_marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_b1536.npz

epoch 1: train=0.5222 val=0.5303
epoch 5: train=0.4866 val=0.5250
epoch 6: train=0.4795 val=0.5245
epoch 7: train=0.4743 val=0.5242
epoch 8: train=0.4714 val=0.5245
```

Automatic post-eval:

```text
random 500:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.random_g500.log
  result: 499/500 = 99.8%

Marnie W4 baseline vs big-run delta against Ogerpon 5899/697, 300 games each:
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.vs_ogerpon_w4_delta_g300.csv
  logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.vs_ogerpon_w4_delta_g300.log
  vs og5899: W4 47/300 = 15.7%, big-run 48/300 = 16.0%, delta +0.3pp
  vs og697:  W4 40/300 = 13.3%, big-run 46/300 = 15.3%, delta +2.0pp
  summary: avg_delta=+1.2pp, candidate=15.7%, baseline=14.5%
```

Interpretation: the larger batch/no-cuDNN restart solved the training stability
issue and kept random performance high, but it did not materially solve Marnie
vs Ogerpon. Treat this as a stable baseline/ablation checkpoint rather than a
breakthrough. More epochs are unlikely to help much from this recipe: validation
was already flat and epoch 8 slightly regressed.

Reference monitor commands:

```bash
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "bc2_train.py.*nocudnn_bigrun|run_marnie_nightly_nocudnn_bigrun" && nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits'

ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.status && tail -n 120 logs/marnie_nightly_20260809/marnie_b8f_v12_0730_0807_histcross_w4_nocudnn_bigrun_20260809.b1536.train.log'
```

## 2026-08-09 Explicit Rule/Plan Pivot

User asked to first provide two models trained today that can be packaged for
submission, then pivot to explicit rule/plan methods using Kaggle community and
real PTCG strategy sources.

Two today-trained models that are technically packageable, but are only useful
as Kaggle ablation probes, not as strong local candidates:

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804

python3 tools/package_submission.py \
  --policy checkpoints/pair_teachers_v12_0701_0807_allquality_clean20/bc2_festival_lead_vs_mega_lopunny_clean_teacher_w4.npz \
  --deck logs/ladder_pool_0804_all/decks/41ffa7894f40_festival_lead_dominic_peel.csv \
  --out /home/jie/Do/0_PTCG/submission/ablation_20260809_festival_pair_teacher_vs_lopunny.tar.gz

python3 tools/package_submission.py \
  --policy checkpoints/pair_teachers_v12_0701_0807_allquality_clean20/bc2_festival_lead_vs_dragapult_clean_teacher_w4.npz \
  --deck logs/ladder_pool_0804_all/decks/41ffa7894f40_festival_lead_dominic_peel.csv \
  --out /home/jie/Do/0_PTCG/submission/ablation_20260809_festival_pair_teacher_vs_dragapult.tar.gz
```

Local quality caveat:

```text
Festival vs Lopunny pair-teacher random: 186/200 = 0.930
Festival vs Dragapult pair-teacher random: 180/200 = 0.900
Both target matchups worsened badly versus W4 baseline, so do not treat them as
score candidates unless the user explicitly wants Kaggle ablation.
```

Rule/plan code added:

- `ptcg_rl/rule_overlay.py`
  - new `strategy_plan`: broad exploratory explicit plan overlay.
  - new `strategy_pair`: narrow pair overlay. After 300-game checks it is still
    not submission-ready.
- `main.py`
  - packaged agent can read `rules.txt` or `PTCG_RULE_MODE` and apply the rule
    overlay after NumPy policy/MCTS selection.
- `tools/package_submission.py`
  - new `--rules <mode>` writes `rules.txt` into the tarball.
- `tools/eval_baseline_delta.py`
  - new `--rules-entry NAME=<mode>` for paired A/B.
- `tools/eval_rule_overlay_stats.py`
  - new sequential trigger tracer for counting rule reasons in concrete
    matchups.

Kaggle/community notes:

- Topic `724362` ("Top players’ methods, revealed by 30,000 games") suggests
  the top end likely combines trained models with bounded search/RL, but the
  comments also warn search only helps if state value is reliable. This matches
  our MCTS/PPO issue: without a trustworthy value head, search can amplify bad
  BC preferences.
- Topic `708810` confirms inference is CPU-only with a 600 second total game
  budget and roughly 1.6 vCPU/8GB RAM. Keep explicit rule/plan lightweight.
- Public PTCG data/guide sources checked as seeds, not proof:
  - Limitless decks overview: https://limitlesstcg.com/decks
  - Limitless Marnie's Grimmsnarl overview: https://limitlesstcg.com/decks/329
  - Limitless Crustle overview: https://limitlesstcg.com/decks/341
  - Limitless Mega Lucario deck list example: https://limitlesstcg.com/decks/list/27700
  - Limitless Crustle matchup stats: https://play.limitlesstcg.com/decks/crustle-dri/matchups?format=standard&rotation=2026&set=CRI
- Real PTCG strategic direction inferred from those sources and local card IDs:
  - Ogerpon: energy/draw engine; hard matchups may require partner/secondary
    routes, not just repeated Teal Dance.
  - Marnie: fast Grimmsnarl/Punk Up plus spread/disruption; simple "force
    evolution" alone hurt local Ogerpon matchup.
  - Lucario: route is Solrock/Lunatone/search/disruption into Mega Lucario or
    Hariyama payoff, not single-step attacker preference.

Rule probe results on ks:

```text
logs/rule_plan_20260809/strategy_plan_probe_summary.csv
  Ogerpon5899 strategy_plan vs Crustle3cd: +0.025 (0.020 -> 0.045), not enough.
  Marnie b8f strategy_plan vs Ogerpon5899: -0.060 (0.110 -> 0.050), bad.
  Crustle3cd strategy_plan vs Lopunny b0: +0.010, too small.
  Festival41 strategy_plan vs Lopunny: -0.085, bad.
  Festival41 strategy_plan vs Dragapult: +0.050 in first seed only.
  Lucario43d strategy_plan vs Marnie: +0.015, tiny.
  Lucario43d strategy_plan vs Crustle: -0.095, bad.

logs/rule_plan_20260809/strategy_pair_probe_summary.csv
  Ogerpon5899 strategy_pair vs Crustle3cd: +0.025 in 200g, but unstable.
  Crustle3cd strategy_pair vs Lopunny b0: +0.025 in 200g, but unstable.
  Festival41 strategy_pair vs Dragapult: -0.055, remove from trusted rules.

logs/rule_plan_20260809/ab_ogerpon5899_strategy_pair_narrow_vs_crustle3cd_g300.csv
  narrowed Ogerpon pair rule: -0.023 (0.040 -> 0.017), not trusted.

logs/rule_plan_20260809/ab_crustle3cd_strategy_pair_narrow_vs_lopunny_b0_g300.csv
  narrowed Crustle pair rule: -0.020 (0.547 -> 0.527), not trusted.

logs/rule_plan_20260809/triggers_ogerpon5899_strategy_pair_narrow_vs_crustle3cd_g100.log
  Ogerpon rules triggered heavily:
    pair:ogerpon_take_setup_over_blank_attack: 680
    pair:ogerpon_attach_before_teal_dance_vs_crustle: 620
  but WR stayed 5/100. This blocks bad attacks but does not construct a win
  route. Do not submit rule-overlay builds yet.
```

Current interpretation:

- The infrastructure for explicit rule/plan evaluation is now usable.
- Current hand-written rules are not enough and should not be used for Kaggle
  submission.
- The next useful step is trace-level route synthesis: compare complete clean
  win traces against normal losses, then convert the route into finite-state
  plan guards with trigger counters. Single-action preferences are too weak.

## 2026-08-09 Stateful Resource Planner

User challenged that the previous rule/plan layer still did not handle multi-step
resource planning. That was correct: `strategy_plan`/`strategy_pair` were mostly
single-decision gates over the current observation.

Implemented a first stateful explicit planner:

- New `ptcg_rl/resource_planner.py`
  - `ResourcePlanner` persists across a game.
  - Tracks route, phase, per-reason override limits, current turn, known own
    visible cards, and estimated unseen card counts from the actual 60-card deck.
  - Routes currently covered: Ogerpon vs Crustle, Marnie vs Ogerpon, Lucario
    engine-resource matchups.
- `main.py`
  - `PTCG_RULE_MODE=resource_plan` or packaged `rules.txt` now uses a persistent
    planner instance and resets it at game start.
- `tools/eval_bc.py`, `tools/eval_round_robin.py`, `tools/eval_rule_overlay_stats.py`
  - all support `resource_plan`, so local random/RR/baseline-delta can evaluate
    the stateful planner.

Key finding from the first probe:

```text
5899 Ogerpon deck:
  Ogerpon ex 96: 4
  Lillie's Clefairy ex 272: 0
  Mega Kangaskhan ex 756: 0
  Meowth ex 1071: 0
  Ultra Ball 1121: 0
  Boss's Orders 1182: 2
  Judge 1213: 4

697 Ogerpon deck:
  Ogerpon ex 96: 4
  272/756/1071/1121: all 0
  Boss's Orders 1182: 3
  Judge 1213: 4
```

So the earlier "secondary route vs Crustle" idea does not apply to these two
Ogerpon signatures. The stateful planner correctly enters `disrupt_fallback`,
not `find_secondary`, for 5899/697.

Probe logs:

```text
logs/resource_plan_20260809/random_ogerpon5899_resource_plan_g150.log
logs/resource_plan_20260809/random_marnie_b8f_resource_plan_g150.log
logs/resource_plan_20260809/random_crustle3cd_resource_plan_g150.log
logs/resource_plan_20260809/random_lucario43d_resource_plan_g150.log
  random stayed healthy:
    Ogerpon 150/150
    Marnie 150/150
    Crustle 149/150
    Lucario 147/150

logs/resource_plan_20260809/ab_ogerpon5899_resource_plan_vs_crustle3cd_g200.csv
  Ogerpon vs Crustle: 0.030 -> 0.030

logs/resource_plan_20260809/ab_marnie_b8f_resource_plan_vs_ogerpon5899_g200.csv
  Marnie vs Ogerpon: 0.180 -> 0.150

logs/resource_plan_20260809/ab_ogerpon5899_resource_plan_fallback2_vs_crustle3cd_g300.csv
  Ogerpon fallback2 vs Crustle: 0.027 -> 0.023

logs/resource_plan_20260809/triggers_ogerpon5899_resource_plan_fallback2_vs_crustle3cd_g100.log
  resource:ogerpon_attach_before_draw:route=ogerpon_secondary_vs_crustle:phase=disrupt_fallback: 476
  Boss/Judge fallback did not trigger in this sample.
```

Current interpretation:

- `resource_plan` is a real stateful/multi-step scaffold now, but the first
  rules are still not strong enough for submission.
- The most important immediate gain is diagnostic: it prevents planning around
  cards not present in the actual deck signature.
- For Ogerpon vs Crustle, 5899/697 likely need either a different deck sig with
  real non-Crustle route resources, or a much deeper plan involving target
  selection/prize race. Boss/Judge alone did not move the matchup.

## 2026-08-08 Aggressive Weak-Matchup Counter Work

User explicitly rejected small/filter-only changes after the Ogerpon vs Crustle
failure-filter pilot only reached about 2.5-4.2% local win rate. Current working
interpretation:

- Generic BC data is likely drowning the useful weak-matchup success signal.
  A win/loss filter across all opponents mostly preserves easy/general wins,
  not the specific counter-plan for a hard pair.
- `setup_success/tempo_success/strategy_success` are not enough unless they are
  matchup-conditioned. A successful generic Marnie or Ogerpon trajectory may
  teach the wrong plan into Ogerpon/Crustle.
- Structural weaknesses need visible-state rules or teacher/counter policies.
  Pure BC cannot learn a counter-plan when same-sig clean wins are sparse or
  the deck list lacks the needed line.

### Ogerpon vs Crustle Pure Success-Trajectory Test

Direct causal test requested by the user: train Ogerpon only on clean games
where Ogerpon beat Crustle, then fight the strong Crustle W4 model.

Remote files:

```text
runner: /tmp/run_ogerpon_crustle_pure_teacher_20260808.sh
log dir: logs/pure_teacher_ogerpon_vs_crustle_20260808
subset corpus: data/bc_pure_teacher_ogerpon_vs_crustle_20260808
checkpoints: checkpoints/pure_teacher_ogerpon_vs_crustle_20260808
baseline Ogerpon: checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_teal_mask_ogerpon_sig2_2a507219_v11all35_sigpure_top3_w4.npz
strong Crustle: checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_crustle_wall_sig1_3cd5039c_v11all35_sigpure_top3_w4.npz
```

Data:

```text
2a5072194fdf Ogerpon clean wins vs Crustle: 58 games, 4125 kept rows
all Ogerpon sig clean wins vs Crustle: 104 games, 7436 kept rows
```

Models trained:

```text
bc2_ogerpon2a_vs_crustle_clean_scratch_w4.npz
bc2_ogerpon2a_vs_crustle_clean_initpartial_w4.npz
bc2_ogerpon_all_vs_crustle_clean_scratch_w4.npz
```

Training intentionally used a high-capacity cross-attn + history + hierarchical
step-plan policy for 80 epochs to overfit the clean success trajectories. This
did not help. The models memorized the small clean subset: training loss went
near zero while validation loss exploded.

Official 300-game results:

```text
random:
  baseline Ogerpon 2a W4: 249/300 = 0.830
  2a clean scratch:       151/300 = 0.503
  2a clean initpartial:   183/300 = 0.610
  all-sig clean scratch:  162/300 = 0.540

vs strong Crustle W4:
  baseline Ogerpon 2a W4: 35/300 = 0.117
  2a clean scratch:       13/300 = 0.043, delta -0.073
  2a clean initpartial:    8/300 = 0.027, delta -0.090
  all-sig clean scratch:  11/300 = 0.037, delta -0.080
```

Interpretation:

- Directly imitating clean success trajectories is not enough. It made the
  weak matchup worse, not better.
- The clean wins likely contain a lot of state, luck, and opponent-error bias.
  Their per-action labels do not form a stable executable counter-plan under
  the current one-step BC policy.
- This is a strong negative result against "just mine weak-matchup wins and
  train BC harder." The next useful direction is to extract explicit
  trajectory-level invariants from these wins: setup milestones, cards held or
  preserved, Crustle engine disruption windows, prize-race timing, and only
  then use them as rules/options/reward shaping/plan labels.

### Follow-Up: Pair Teacher And Contrast Mining

`tools/mine_strategy_trajectories.py` now supports contrast labels:

```text
--label-csv PATH
--positive-condition clean_win=1
--negative-condition 'outcome=loss&opponent_normal=1'
--label-missing-policy drop
```

Use this to compare clean wins against normal losses, not generic wins against
generic losses. The tool writes `contrast_group` into `game_trajectories.csv`
and computes metric/event/ngram gaps only between positive and negative groups.

Remote Ogerpon vs Crustle contrast:

```text
out: logs/strategy_contrast_20260809/ogerpon_vs_crustle_clean_vs_normal_loss
games: 615 total, 104 clean-win positives, 451 normal-loss negatives
```

Initial Ogerpon/Crustle interpretation:

- Clean wins were not "Ogerpon active earlier and hit harder." Ogerpon active
  turns were much lower in positives.
- Clean wins overrepresent secondary route signals: Mega Kangaskhan ex,
  Meowth ex, Ultra Ball, and attack-to-hand windows.
- Losses overrepresent repeated Ogerpon ability/attach loops and Crustle being
  active/on-board for longer. The likely counter-plan is "route around Crustle
  with secondary attackers and close faster," not pure Ogerpon pressure.

Remote all-pair contrast runner:

```text
script: /tmp/run_strategy_contrast_all_clean20_20260809.sh
log: logs/strategy_contrast_20260809/all_clean20/runner.nohup.log
out root: logs/strategy_contrast_20260809/all_clean20
expected jobs: 17 weak pairs with clean_games >= 20
summary when complete: logs/strategy_contrast_20260809/all_clean20/strategy_seed_summary.csv
```

The all-pair contrast runner completed and wrote
`logs/strategy_contrast_20260809/all_clean20/strategy_seed_summary.csv`.
Top repeated signals are mostly "opponent key engine is already active/on
board" loss-overrepresented events: Crustle vs Ogerpon/Lucario, Lopunny vs
Festival/TRM, Dragapult vs Festival, and Mega Lucario engine pieces. This means
the next rule/planner work should focus on preventing, delaying, or routing
around key engine completion windows, not on replaying clean-win action labels.

Pair-teacher evaluation was also run:

```text
script: /tmp/run_pair_teacher_eval_20260809.sh
log: logs/pair_teacher_eval_20260809/runner.nohup.log
random: logs/pair_teacher_eval_20260809/pair_teacher_random_g200.csv
target summary: logs/pair_teacher_eval_20260809/pair_teacher_target_matchup_summary.csv
```

Completed pair-teacher results all got worse against their intended target:

```text
Marnie vs Ogerpon:       baseline 0.110 -> teacher 0.005, delta -0.105
Crustle vs Lopunny:      baseline 0.400 -> teacher 0.255, delta -0.145
Festival vs Lopunny:     baseline 0.260 -> teacher 0.090, delta -0.170
TR Mewtwo vs Lopunny:    baseline 0.350 -> teacher 0.150, delta -0.200
Festival vs Dragapult:   baseline 0.565 -> teacher 0.160, delta -0.405
```

This confirms the pure Ogerpon/Crustle result across several archetypes: clean
success trajectory imitation is not a viable standalone improvement method.
Do not spend more GPU time on pure pair-teacher BC unless it is only used as a
diagnostic or as input to explicit rule/plan extraction.

The counter-mixture runner ended with zero checkpoints:

```text
log: logs/counter_mixture_20260808/post_eval_watcher.nohup.log
result: checkpoints=0/8; post-eval aborted
```

Treat that path as obsolete for now. The next actionable direction is:

1. Read `strategy_seed_summary.csv` and per-pair `metric_gaps.csv` /
   `event_gaps.csv` / `ngram_gaps.csv`.
2. Convert repeated high-confidence signals into explicit plan states and
   guarded rules.
3. Evaluate rules/options on W4 baselines before any new BC/RL training.

Code added locally and synced to `ks`:

- `ptcg_rl/rule_overlay.py`
  - new `counter_plan` / `counter_plan_aggressive` modes.
  - Uses visible opponent active/bench/discard card IDs, not only opponent
    active.
  - `counter_plan_aggressive` is the broad failed experiment: it aggressively
    forces core evolution/setup for stage decks, Marnie setup vs visible
    Ogerpon, Ogerpon no-blank-ex-attack/attach discipline vs Crustle, Cynthia
    Spiritomb routing vs Crustle, Team Rocket board setup, Lucario engine
    setup, no-early-end, and late attack-window guard.
  - default `counter_plan` is now only the narrow high-confidence Ogerpon vs
    Crustle guard. Do not use `counter_plan_aggressive` for submission.
- `tools/audit_weak_pair_signal.py`
  - Scans v12 corpus and reports weak-pair decision share, weak-win decision
    share, and clean-teacher decision share for top weak pairs.
- `tools/build_counter_rule_eval_script.py`
  - Generates RR scripts comparing plain BC vs same BC with `counter_plan`
    across weak archetype pairs using existing candidate manifests.
- `tools/summarize_counter_rule_eval.py`
  - Summarizes base/rule win-rate delta per weak pair.
- `tools/build_pair_teacher_pipeline.py`
  - Generates clean teacher subsets and optional pair-specialist training
    commands for weak matchup pairs. This is the training-side answer to data
    drowning; do not use tiny aux weights here.
- `tools/build_counter_filter_csv.py`
  - Builds per-archetype trajectory filters with `counter_bad=1` for selected
    weak-matchup losses/dirty wins and `counter_clean=1` for clean teacher
    wins.
- `tools/build_counter_mixture_pipeline.py`
  - Generates submit-capable per-archetype counter-mixture training scripts:
    keep base corpus as anchor, drop selected weak failures, repeat clean
    teacher aux data heavily, and train a trajectory/step-plan conditioned
    cross-attention model from scratch.
- `tools/select_manifest_top_per_archetype.py`
  - Selects top rows per archetype from an existing policy/deck manifest. Used
    by the counter-mixture post-eval watcher to build a compact W4 opponent
    pool.

Active remote runner:

```text
remote script: /tmp/run_aggressive_counter_20260808.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
out dir: logs/aggressive_counter_20260808
corpus: data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12
weak pairs: logs/eval_next_v11_specialists_20260805/rr_weak_archetype_pairs.csv
teacher games: logs/v12_matchup_teachers_20260808_0701_0807/clean_teacher_selection_min10_brick010/selected_clean_teacher_games.csv
```

Runner stages:

1. `weak_pair_signal_v12_0701_0807_top60.csv`
   - confirms/denies whether clean teacher decisions are tiny relative to all
     BC decisions.
2. `rule_rr_top36_g100/` and `rule_rr_top36_g100_summary.csv`
   - evaluates plain vs `counter_plan` for top weak archetype pairs, not one
     hand-picked pair.
3. Builds `/tmp/pair_teacher_top24_clean_pipeline.sh`
   - does not auto-run training yet in this runner. Inspect the audit/RR first,
     then run it or regenerate with `--train` for specific pairs.

Completed results:

```text
signal: logs/aggressive_counter_20260808/weak_pair_signal_v12_0701_0807_top60.csv
aggressive rule RR: logs/aggressive_counter_20260808/rule_rr_top36_g100_summary.csv
guard rule RR: logs/aggressive_counter_20260808/rule_guard_rr_top36_g100_summary.csv
```

Interpretation:

- Clean-teacher signal is tiny in the current generic BC view:
  `clean_teacher_decision_share` mean `0.00309`, median `0.0`, max `0.0953`.
  Caveat: many zeros mean the quality audit has not mined that pair yet, not
  necessarily that no clean wins exist.
- Broad/aggressive `counter_plan` was a failure: `36` weak pairs, `7` improved,
  `29` worsened, average delta `-0.0661`, worst `-0.33`.
- This confirms the user's concern: small weighting/filtering and blunt rules
  do not solve structural weaknesses. Broad rules break the model's learned
  tempo.
- Default `counter_plan` was changed after the failed RR to a narrow high-
  confidence guard. The broad version remains available as
  `counter_plan_aggressive` only for reproduction.
- Guard RR is roughly neutral: `36` weak pairs, `17` improved, `18` worsened,
  average delta `-0.0094`, best `+0.13`, worst `-0.13`. Do not expect this to
  fix weak matchups by itself.
- Therefore the next real improvement path is pair-specific clean trajectory
  training / teacher rollout, then evaluate those specialists against the weak
  opponent and random. Use rules only as hard safety guards.

Obsolete/failed pair-teacher job:

```text
launcher: /tmp/pair_teacher_top60_clean10_train.sh
nohup log: logs/aggressive_counter_20260808/pair_teacher_top60_clean10_train.nohup.log
manifest: /tmp/pair_teacher_top60_clean10_train.manifest.csv
out corpus: data/bc_pair_teachers_v12_0701_0807_top60_clean10
checkpoints: checkpoints/pair_teachers_v12_0701_0807_top60_clean10
planned_train pairs: 12
skipped_low_clean_games pairs: 48
min clean games: 10
```

This job failed in the training phase, not because of model quality:

- generated `xargs bash -lc` commands split archetype names with spaces
  (`Teal Mask Ogerpon`, `Crustle Wall`, etc.);
- one Alakazam job also hit `CUDNN_STATUS_NOT_INITIALIZED` from GRU history.

The generator was fixed after this failure:

- each train task is now emitted as its own `/tmp/.../*.sh` job file;
- every history/GRU training job exports `PTCG_DISABLE_CUDNN=1`;
- new scripts should be regenerated from the fixed tool before running.

The 12 planned clean teacher pair specialists are:

```text
Teal Mask Ogerpon -> Crustle Wall, clean_games=68
Mega Lucario -> Crustle Wall, clean_games=32
Mega Lucario -> Marnie Grimmsnarl, clean_games=268
Marnie Grimmsnarl -> Teal Mask Ogerpon, clean_games=283
Team Rocket Mewtwo -> Teal Mask Ogerpon, clean_games=25
Festival Lead -> Mega Lopunny, clean_games=12
Alakazam -> Team Rocket Mewtwo, clean_games=861
Teal Mask Ogerpon -> Mega Lopunny, clean_games=99
Crustle Wall -> Mega Lopunny, clean_games=27
Mega Lucario -> Teal Mask Ogerpon, clean_games=38
Teal Mask Ogerpon -> Alakazam, clean_games=20
Crustle Wall -> Team Rocket Mewtwo, clean_games=405
```

Current aggressive jobs launched after the fix:

```text
counter mixture launcher: /tmp/counter_mixture_clean20_train.sh
counter mixture nohup log: logs/counter_mixture_20260808/counter_mixture_clean20_train.nohup.log
counter mixture manifest: /tmp/counter_mixture_clean20_train.manifest.csv
counter mixture checkpoints: checkpoints/counter_mixture_v12_0701_0807_clean20
counter mixture subset corpus: data/bc_counter_mix_teachers_v12_0701_0807_clean20

pair teacher launcher: /tmp/pair_teacher_allquality_clean20_train.sh
pair teacher nohup log: logs/aggressive_counter_20260808/pair_teacher_allquality_clean20_train.nohup.log
pair teacher manifest: /tmp/pair_teacher_allquality_clean20_train.manifest.csv
pair teacher checkpoints: checkpoints/pair_teachers_v12_0701_0807_allquality_clean20
pair teacher subset corpus: data/bc_pair_teachers_v12_0701_0807_allquality_clean20
teacher source: logs/v12_matchup_teachers_20260808_0701_0807/quality_audit/game_quality_all_pairs.csv
min clean games: 20

post-eval launcher: /tmp/run_counter_mixture_post_eval_20260808.sh
post-eval nohup log: logs/counter_mixture_20260808/post_eval_watcher.nohup.log
```

Counter-mixture planned archetypes:

```text
Alakazam -> Team Rocket Mewtwo
Crustle Wall -> Team Rocket Mewtwo, Mega Lopunny
Festival Lead -> Mega Lopunny, Dragapult
Marnie Grimmsnarl -> Teal Mask Ogerpon
Mega Lopunny -> Mega Lucario
Mega Lucario -> Marnie Grimmsnarl, Crustle Wall, Teal Mask Ogerpon
Teal Mask Ogerpon -> Alakazam, Mega Lopunny, Crustle Wall, Mega Lucario
Team Rocket Mewtwo -> Dragapult, Teal Mask Ogerpon, Mega Lopunny
```

`Mega Starmie`, `Dragapult`, and `Cynthia Garchomp` were not included in the
current counter-mixture run because this quality audit has no clean teacher
rows for their weak pairs. Do not treat that as proof they have no successful
games; it means teacher mining is incomplete for those archetypes.

Monitor active aggressive jobs:

```bash
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/counter_mixture_20260808/counter_mixture_clean20_train.nohup.log'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/aggressive_counter_20260808/pair_teacher_allquality_clean20_train.nohup.log'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/counter_mixture_20260808/post_eval_watcher.nohup.log'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd | grep -E "counter_mixture_clean20|pair_teacher_allquality|build_trajectory_targets|build_bc_subset|bc2_train.py" | grep -v grep'
```

Post-eval watcher behavior after counter-mixture training exits:

```text
1. build logs/counter_mixture_20260808/counter_mixture_clean20_eval_manifest.csv
   by mapping checkpoints to logs/ladder_pool_0805_all/pool_manifest.csv.
2. build logs/counter_mixture_20260808/w4_top1_by_arch_manifest.csv from
   logs/eval_deck_sig_specialists_v11all35_20260806/candidate_manifest_w4_random_ge097.csv.
3. run random 300:
   logs/counter_mixture_20260808/counter_mixture_clean20_random_g300.csv
4. run counter internal RR g80:
   logs/counter_mixture_20260808/counter_mixture_clean20_internal_rr_g80.csv
5. run counter + W4 top1 RR g60:
   logs/counter_mixture_20260808/counter_mixture_clean20_vs_w4top1_rr_g60.csv
```

Additional direct pure-teacher causality test launched after user asked for a
more extreme validation of successful trajectories:

```text
script: /tmp/run_ogerpon_crustle_pure_teacher_20260808.sh
nohup log: logs/pure_teacher_ogerpon_vs_crustle_20260808/runner.nohup.log
subset corpus: data/bc_pure_teacher_ogerpon_vs_crustle_20260808
checkpoints: checkpoints/pure_teacher_ogerpon_vs_crustle_20260808
target deck: logs/ladder_pool_0805_all/decks/2a5072194fdf_teal_mask_ogerpon_james_cox_henry_chao.csv
baseline policy: checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_teal_mask_ogerpon_sig2_2a507219_v11all35_sigpure_top3_w4.npz
opponent policy: checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_crustle_wall_sig1_3cd5039c_v11all35_sigpure_top3_w4.npz
opponent deck: logs/ladder_pool_0804_all/decks/3cd5039c59d2_crustle_wall_oshbocker.csv
```

Why `2a5072194fdf`: in `game_quality_all_pairs.csv`, Teal Mask Ogerpon clean
wins vs Crustle are concentrated at `2a5072194fdf / James Cox & Henry Chao`
with 58 clean games. The next largest sig has only 10, while `697a...` has 6
and `5899...` has 1. This test is therefore sig-specific, not a mixed Ogerpon
guess.

The pure-teacher runner trains three deliberately extreme models:

```text
bc2_ogerpon2a_vs_crustle_clean_scratch_w4.npz
bc2_ogerpon2a_vs_crustle_clean_initpartial_w4.npz
bc2_ogerpon_all_vs_crustle_clean_scratch_w4.npz
```

All train only on clean wins, 80 epochs, width 4 cross-attn, history/log/board
history, hierarchical step-plan. It then automatically runs:

```text
logs/pure_teacher_ogerpon_vs_crustle_20260808/pure_teacher_random_g300.csv
logs/pure_teacher_ogerpon_vs_crustle_20260808/pure_teacher_vs_crustle_w4_g300.csv
```

Monitor:

```bash
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/pure_teacher_ogerpon_vs_crustle_20260808/runner.nohup.log'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd | grep -E "pure_teacher_ogerpon|run_ogerpon_crustle|bc2_train.py|eval_baseline_delta|eval_manifest_random" | grep -v grep'
```

Do not use `counter_plan_aggressive` for submission. If testing rules, use
default `counter_plan` and only after RR confirms it is not harming the target
pool. If testing strategy improvement, evaluate the pair-teacher checkpoints
directly against their target opponent and random; expect overfit because these
are pure clean-win specialists.

Monitor:

```bash
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/aggressive_counter_20260808/audit_weak_pair_signal_top60.log'
ssh -F /home/jie/.ssh/config ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/aggressive_counter_20260808/counter_rule_rr_top36_g100.runner.log'
ssh -F /home/jie/.ssh/config ks 'pgrep -af "run_aggressive_counter_20260808|counter_rule_rr_top36"'
```

Local ssh note:

- Plain `ssh ks` currently failed on this machine because a system ssh config
  include had bad permissions. Use `ssh -F /home/jie/.ssh/config ks ...` and
  `scp -F /home/jie/.ssh/config ...`.

## 2026-08-08 BC Memory Reduction And Adaptive Date Windows

Follow-up completed at about 2026-08-08 20:35 Asia/Shanghai:

- Added `tools/plan_bc_date_windows.py`.
  - It scans extracted BC corpora by `archetype + deck_sig + date`.
  - Default `--count-mode kept` validates BC labels and is intended for a
    small candidate manifest.
  - `--count-mode raw` now has a fast path that reads only `deck_sig` arrays
    from each npz. Use this for broad/global planning; raw and kept counts have
    been very close in the v12 corpus.
  - The planner chooses the newest suffix of dates that reaches a target row
    count. Sparse sigs fall back to all available dates and get
    `low_data_all_dates` / `below_target_all_dates`.
- `tools/train_shadow_manifest.py` now accepts `--date-window-csv`; it matches
  rows by `archetype + deck_sig` and appends/replaces `--date-from/--date-to`
  in manifest training commands.
- `tools/plan_deck_specific_bc.py` now accepts `--date-window-csv`; generated
  pure sig specialist commands include the planned date window. If a mixed
  topK row contains multiple sigs, it conservatively uses the earliest
  `date_from` among those sigs.

Remote global raw plan:

```text
csv: logs/v12_strategy_conditioned_adaptive_20260808/bc_date_windows_top8_raw.csv
log: logs/v12_strategy_conditioned_adaptive_20260808/plan_bc_date_windows_top8_raw.log
command used count-mode raw over v12 0701-0807, bands 1200+/1100/1000/900, top8 per archetype.
```

Important rows from that plan:

```text
Alakazam 7f9a538936e3          date_from=2026-08-02 selected=377431 / total=2484992 status=ok
Crustle 3cd5039c59d2           date_from=2026-07-17 selected=182491 / total=548969  status=ok
Marnie b8f251a476e7            date_from=2026-08-06 selected=445740 / total=4250701 status=ok
Mega Lucario 43d6d8b0fce9      date_from=2026-08-02 selected=85815  / total=85815   status=below_target_all_dates
Ogerpon 697a82e582d5           date_from=2026-07-31 selected=173860 / total=180406  status=ok
Ogerpon 2a5072194fdf           date_from=2026-07-28 selected=98683  / total=98683   status=below_target_all_dates
Ogerpon 5899c772bace           date_from=2026-08-01 selected=78444  / total=78444   status=low_data_all_dates
Team Rocket Mewtwo 06f0...     date_from=2026-07-19 selected=212948 / total=233157  status=ok
Team Rocket Mewtwo f0bac...    date_from=2026-07-20 selected=80635  / total=80635   status=below_target_all_dates
Festival Lead e82dcbe62260     date_from=2026-07-27 selected=140891 / total=140891  status=below_target_all_dates
```

Interpretation:

- Full-date training is now memory-feasible for many sigs, but not always
  desirable. Huge sigs like `Marnie b8f` and `Alakazam 7f9` should keep recent
  suffixes; all-date would mostly add old meta and slow training.
- Sparse sigs such as `Mega Lucario 43d6`, `Ogerpon 5899`, and many Archaludon
  variants cannot be fixed by date-window selection. They need either all-date
  specialist training, adjacent-sig sharing, successful-trajectory mining, or
  constructed/teacher data.
- The active adaptive runner was launched before this planner existed, so its
  windows are close but not identical: it uses `Ogerpon 697 2026-08-01+` and
  `Crustle 3cd 2026-07-18+`; the raw planner recommends `2026-07-31+` and
  `2026-07-17+` respectively.

Code change committed after the strategy-conditioned pilot:

- `ptcg_rl/bc2/data.py`
  - Added `filter_npz_paths_by_date` and optional `date_from/date_to` to
    `discover_npz_paths`.
  - `BCCorpus` now skips files with zero kept rows.
  - `BCCorpus` now stores only kept rows for each loaded npz, remapping
    groups and history indices after filtering. This is the main memory fix.
- `tools/bc2_train.py`
  - Added `--date-from YYYY-MM-DD` and `--date-to YYYY-MM-DD`.
- `tools/build_trajectory_targets.py`
  - Added matching `--date-from` and `--date-to`.

Local smoke tests passed:

```text
python3 -m py_compile ptcg_rl/bc2/data.py tools/bc2_train.py tools/build_trajectory_targets.py
compact smoke ok: 6 synthetic rows, 3 kept rows, stored 3 rows, history/group collation passed
```

Reasoning:

- Old `BCCorpus` loaded whole npz arrays even when a `deck_sig` filter kept only
  a subset, and it also stored files with zero kept rows.
- For large sigs like `Alakazam 7f9`, all-date rows were `2,484,992`; this was
  the memory risk.
- Recommended default for low-memory strategy pilots is `--date-from
  2026-08-01`. It keeps:
  - `Ogerpon 697`: `154,339/180,406` rows, 85.6%.
  - `Ogerpon 5899`: `78,444/78,444` rows, 100%.
  - `Crustle 3cd`: `49,155/548,969` rows, 9.0%.
  - `Alakazam 7f9`: `398,860/2,484,992` rows, 16.1%.
- For Crustle, `2026-08-01+` was judged too small for this pilot, so the active
  adaptive runner uses `2026-07-18+`.

Old all-date strategy runner was stopped because the Alakazam process was still
using old full-memory code and had reached roughly `106GiB` host memory used.

The first uniform low-memory runner used `--date-from 2026-08-01` for every
deck:

```text
script: /tmp/run_v12_strategy_conditioned_recent0801_20260808.sh
runner log: logs/v12_strategy_conditioned_recent0801_20260808/runner.log
checkpoint dir: checkpoints/v12_strategy_conditioned_recent0801_20260808
date_from: 2026-08-01
PTCG_DISABLE_CUDNN=1 in training commands
```

It was then stopped deliberately before completion because one date window is
not appropriate for every deck. In particular, `Crustle 3cd` had only `49,155`
rows from `2026-08-01+`, which is likely too small for a wall/stall archetype.

Current adaptive runner:

```text
script: /tmp/run_v12_strategy_conditioned_adaptive_20260808.sh
runner log: logs/v12_strategy_conditioned_adaptive_20260808/runner.log
checkpoint dir: checkpoints/v12_strategy_conditioned_adaptive_20260808
PTCG_DISABLE_CUDNN=1 in training commands
runner pids at check: 2974502 wrapper, 2974504 bash runner
memory stayed around 54GiB host used at restart/check
```

Adaptive date windows:

```text
Ogerpon 697    date_from=2026-08-01, rows=154339
Ogerpon 5899   date_from=2026-08-01, rows=78444
Crustle 3cd    date_from=2026-07-18, expected rows about 160533
Alakazam 7f9   date_from=2026-08-01, rows=398860
```

Reason for the different windows:

```text
Crustle 3cd rows:
  >=2026-08-01:  49155 rows, likely too small.
  >=2026-07-18: ~160533 rows, better target size without returning to all 548969 rows.

Alakazam 7f9 rows:
  all dates:      2484992 rows, too large for this pilot.
  >=2026-08-01:   398860 rows, large enough and much safer.

Ogerpon 697:
  >=2026-08-01 keeps 85.6% of training-band rows.

Ogerpon 5899:
  data only exists from 2026-08-01 onward in selected bands, so no reduction.
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 160 logs/v12_strategy_conditioned_adaptive_20260808/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_v12_strategy_conditioned_adaptive|bc2_train.py|build_trajectory_targets.py"'
ssh ks 'free -h && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits'
```

## 2026-08-08 Aggressive Strategy-Conditioned BC

Code change committed locally:

```text
3e08ebb Condition BC actions on trajectory strategy targets
```

Changed behavior:

- `tools/bc2_train.py` now allows `--trajectory-target` and `--step-plan` in
  the same run.
- The model plan vector is now concatenated as:
  `trajectory targets first`, then `step-plan labels`.
- `ptcg_rl/bc2/losses.py` now teacher-forces any available trajectory/step
  plan labels into the hierarchical action scorer during training. This is the
  main behavioral change: game-level success/tempo/setup targets can directly
  condition legal-action scoring, instead of being an auxiliary head that does
  not affect action logits.

This is still submittable because inference uses predicted plan values; no
private opponent labels are required. The risk is exposure mismatch: at training
time the scorer can see teacher-forced strategy labels, while at inference it
must trust the predicted plan head.

Remote pilot runner launched:

```text
script: /tmp/run_v12_strategy_conditioned_pilots_20260808.sh
runner log: logs/v12_strategy_conditioned_20260808/runner.log
target/log dir: logs/v12_strategy_conditioned_20260808
checkpoint dir: checkpoints/v12_strategy_conditioned_20260808
runner pids at launch/check: 2949894 wrapper, 2949896 bash runner
```

Pilot archetypes/sigs:

```text
ogerpon_697    Teal Mask Ogerpon 697a82e582d5
ogerpon_5899   Teal Mask Ogerpon 5899c772bace
crustle_3cd    Crustle Wall      3cd5039c59d2
alakazam_7f9   Alakazam          7f9a538936e3
```

These are deliberately broader than the previous Lucario/Marnie/Ogerpon-only
focus. Alakazam and other untrained archetypes still need to be folded into the
same pipeline.

Trajectory target CSVs generated so far at 2026-08-08 19:59:

```text
ogerpon_697:  games=2761 wins=1465 losses=1293 draws=3 wr=0.531
ogerpon_5899: games=1102 wins=583 losses=519 draws=0 wr=0.529
crustle_3cd:  still building, scanned 40/108 files
alakazam_7f9: pending
```

The training command template uses:

```text
--hierarchical-plan
--trajectory-target outcome_win
--trajectory-target strategy_success
--trajectory-target setup_success
--trajectory-target tempo_success
--trajectory-target attack_by_6
--trajectory-target primary_board_by_4
--trajectory-target engine_board_by_4
--trajectory-target no_early_end
--trajectory-target pressing_main_rate
--trajectory-target attack_turn_norm
--trajectory-target primary_board_turn_norm
--trajectory-target primary_active_turn_norm
--trajectory-target-loss-weight 0.40
--step-plan --step-plan-loss-weight 0.20 --step-plan-teacher-forcing 0.75
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 160 logs/v12_strategy_conditioned_20260808/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_v12_strategy_conditioned_pilots|bc2_train.py|build_trajectory_targets.py"'
ssh ks 'free -h && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits'
```

Old Marnie seqplan note:

- `bc2_marnie_b8f_seqplan_scratch_w3_b512` did not finish. It was killed by
  the memory watchdog at `2026-08-08 19:47:20`, cgroup memory `246.4GiB`,
  limit `246.0GiB`, rc `143`.
- Last useful checkpoint is epoch 2:
  `checkpoints/v12_sequence_planner_20260808/bc2_marnie_b8f_seqplan_scratch_w3_b512.npz`.

## 2026-08-08 Strategy-Conditioned Adaptive Pilot Results

Remote result dir:

```text
logs/eval_v12_strategy_adaptive_20260808
```

Training completed for four pilots:

```text
checkpoints/v12_strategy_conditioned_adaptive_20260808/bc2_ogerpon_697_strategy_seqplan_2026-08-01_w3.npz
checkpoints/v12_strategy_conditioned_adaptive_20260808/bc2_ogerpon_5899_strategy_seqplan_2026-08-01_w3.npz
checkpoints/v12_strategy_conditioned_adaptive_20260808/bc2_crustle_3cd_strategy_seqplan_2026-07-18_w3.npz
checkpoints/v12_strategy_conditioned_adaptive_20260808/bc2_alakazam_7f9_strategy_seqplan_2026-08-01_w3.npz
```

Random g500:

```text
Alakazam 7f9 strategy:      98.0% (490/500)
Crustle 3cd strategy:       97.4% (487/500)
Ogerpon 697 strategy:       99.0% (495/500)
Ogerpon 5899 strategy:      99.4% (497/500)
```

This is not a basic-play collapse, but it is weaker than the strongest old w4
deck-sig specialists for Alakazam/Crustle/Ogerpon5899. Old w4 reference random:
Alakazam 7f9 `100%`, Crustle 3cd `99.5%`, Ogerpon5899 `100%`, Ogerpon697
`98.5%`.

New-four RR g80:

```text
new_crustle3cd    avg=0.854 worst=new_alakazam7f9:0.700
new_alakazam7f9   avg=0.571 worst=new_crustle3cd:0.300
new_ogerpon697    avg=0.296 worst=new_crustle3cd:0.075
new_ogerpon5899   avg=0.279 worst=new_crustle3cd:0.062
```

Against old w4 same/meta reference entries, candidate-only g80:

```text
new_ogerpon697:
  vs old_alakazam7f9 0.362, old_crustle3cd 0.037,
  old_ogerpon697 0.388, old_ogerpon5899 0.412

new_ogerpon5899:
  vs old_alakazam7f9 0.287, old_crustle3cd 0.037,
  old_ogerpon697 0.562, old_ogerpon5899 0.525

new_crustle3cd:
  vs old_alakazam7f9 0.637, old_crustle3cd 0.388,
  old_ogerpon697 0.975, old_ogerpon5899 0.963

new_alakazam7f9:
  vs old_alakazam7f9 0.362, old_crustle3cd 0.300,
  old_ogerpon697 0.613, old_ogerpon5899 0.750
```

Broader new Crustle vs old w4 top pool, candidate-only g60:

```text
weighted/avg=0.590, worst=mega_lopunny_sig2_276707c0:0.283
strong into:
  Ogerpon 697: 0.983
  Ogerpon 5899: 0.967
  Alakazam sigs: 0.650 / 0.733 / 0.817
  Cynthia 52f: 0.733
  Dragapult 0b7: 0.817
weak into:
  Team Rocket Mewtwo 06f: 0.333
  Marnie b8f: 0.350
  Festival e82: 0.383
  Lopunny 276: 0.283
  old Crustle 3cd mirror: 0.450
```

Interpretation:

- Do not treat the strategy-conditioned batch as a general upgrade.
- The two Ogerpon strategy checkpoints are regressions in local RR and should
  not be submitted as-is.
- `new_crustle3cd` is a useful local specialist for studying how Crustle
  crushes Ogerpon and many Alakazam/Dragapult/Cynthia variants, but it is not a
  universal ladder pick because it loses to several strong non-Ogerpon
  archetypes and to old Crustle mirror.
- `new_alakazam7f9` gained anti-Ogerpon behavior but regressed against old
  Alakazam/Crustle, so it is also not a broad upgrade.

Diagnosis added after checking the latest run:

- This pilot did not actually consume the previously mined clean-teacher or
  weak-matchup success subsets. It trained generic trajectory-conditioned BC
  from `data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12`, with
  cross-attention/history/board-history inputs, `--hierarchical-plan`, generic
  trajectory targets, and step-plan teacher forcing. It did not pass
  `--aux-corpus`, explicit clean-teacher subset paths, or opponent-specific
  training filters/weights.
- Ogerpon failed against Crustle because the supervised signal is almost absent
  for the target sigs. In the trajectory targets, Ogerpon 697 had `178` games
  vs Crustle with only `8` wins (`4.5%`), and Ogerpon 5899 had `77` games vs
  Crustle with only `2` wins (`2.6%`). These rows are also downweighted by the
  generic outcome/strategy weighting because they are mostly losses.
- Ogerpon's high-weight data is dominated by already-good matchups, especially
  Marnie (`~85%` win rate, roughly `39%` decision share, average strategy
  weight around `1.5`). The model therefore learns "how Ogerpon wins normal
  matchups", not "how Ogerpon should answer Crustle".
- Crustle 3cd improved in exactly the supported direction: it had `115` games
  vs Ogerpon with `100` wins (`87%`), so the new checkpoint strongly beats old
  Ogerpon candidates. This confirms that the evaluator is seeing the changed
  supervision; the issue is the target supervision, not a completely inert
  architecture.
- Earlier success-only fine-tunes were diagnostic and mostly not broad fixes.
  `ogerpon_xsig_vs_crustle_success_ft` was random-stable (`0.993`) but had
  worse focused delta (`avg_delta=-0.0167`, `candidate=0.0283`,
  `baseline=0.0450`). It should be treated as evidence that Ogerpon-vs-Crustle
  needs explicit teacher/rule/search construction, not as a finished solution.

Actionable follow-up:

- Make clean teachers first-class training data. Current clean-teacher outputs
  live under
  `logs/v12_matchup_teachers_20260808_0701_0807/clean_teacher_selection_min10_brick010/`
  and subset corpora under
  `data/bc_corpus_clean_teachers_v12_0701_0807_min10_brick010`.
- For Ogerpon-vs-Crustle, do not rely on same-sig 697/5899 successes. Use
  cross-sig teacher `2a5072194fdf` and/or generated teacher rollouts because
  same-sig positive examples are too sparse.
- Replace generic trajectory labels with matchup-specific plan labels such as
  anti-Crustle setup, resource preservation, pressure maintenance, or key
  target disruption. Generic `strategy_success/setup_success/tempo_success`
  is too coarse for structural weak matchups.
- Revisit `--step-plan-teacher-forcing 0.75`; the policy can see clean plan
  labels during training but must use predicted plan labels at inference, which
  creates exposure mismatch unless the plan head is itself reliable.

## 2026-08-08 Failure-Trajectory Filtering

Code change:

- `tools/bc2_train.py` now supports trajectory-level filtering:
  - `--trajectory-keep CONDITION`: keep only games satisfying all keep
    conditions from `--trajectory-csv`.
  - `--trajectory-drop CONDITION`: drop games satisfying any drop condition.
  - `--trajectory-filter-missing-policy keep|drop`: controls corpus games not
    present in the trajectory CSV.
  - Conditions now support simple boolean syntax: `A&B`, `A|B`, and `!A`.
    Example: `outcome_loss&setup_success==0&tempo_success==0`.
- `ptcg_rl/bc2/data.py` now applies that filter while indexing the corpus, so
  dropped trajectories do not occupy training batches or host memory. This is
  different from setting `--loss-weight 0`, which still indexes the rows.

Local and remote checks passed:

```text
python3 -m py_compile ptcg_rl/bc2/data.py tools/bc2_train.py
composite condition smoke ok
remote py_compile passed
remote `tools/bc2_train.py --help` shows --trajectory-keep/drop/filter-missing-policy
```

Active remote runner:

```text
script: /tmp/run_v12_failure_filtered_20260808.sh
runner log: logs/v12_failure_filtered_20260808/runner.log
checkpoint dir: checkpoints/v12_failure_filtered_20260808
eval dir: logs/v12_failure_filtered_20260808/eval
```

Purpose: test whether removing only hard-failure trajectories is better than
generic trajectory-conditioned BC. This is not winner-only. The current drop
condition is:

```text
outcome_loss&setup_success==0&tempo_success==0
```

It removes losses that have neither setup nor tempo success, while retaining
wins and retaining losses that still contain potentially useful setup/tempo
lines.

Two Ogerpon pilots were started from v12 corpus, using cross-sig teacher data
from `2a5072194fdf`:

```text
Ogerpon 697 + 2a507:
  targets: logs/v12_failure_filtered_20260808/ogerpon697_2a507.trajectory_targets.csv
  target games=4203 wins=2308 losses=1892 draws=3 wr=0.549
  trajectory filter allowed=2721 dropped=1482 games
  corpus kept=179734 decisions
  save: checkpoints/v12_failure_filtered_20260808/bc2_ogerpon697_2a507_drop_hardfail_seqplan_w3.npz

Ogerpon 5899 + 2a507:
  targets: logs/v12_failure_filtered_20260808/ogerpon5899_2a507.trajectory_targets.csv
  target games=2544 wins=1426 losses=1118 draws=0 wr=0.561
  trajectory filter allowed=1567 dropped=977 games
  corpus kept=111298 decisions
  save: checkpoints/v12_failure_filtered_20260808/bc2_ogerpon5899_2a507_drop_hardfail_seqplan_w3.npz
```

Training config:

```text
--arch cross_attn --width 3 --state-layers 2
--history-k 32 --log-history-k 128 --board-history-k 12
--hierarchical-plan --step-plan
--step-plan-teacher-forcing 0.35
--trajectory-target-loss-weight 0.35
--step-plan-loss-weight 0.18
--cuda-memory-gb 24 --batch-size 1024 --epochs 8 --lr 3e-5
```

The runner will automatically evaluate random g300 for both candidates and a
focused RR against old Crustle 3cd g120 after training. Monitor with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_v12_failure_filtered|bc2_train.py"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_failure_filtered_20260808/train_ogerpon697_2a507_drop_hardfail.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_failure_filtered_20260808/train_ogerpon5899_2a507_drop_hardfail.log'
```

Completed result:

```text
Train:
  697+2a best epoch 8, val=1.0222
  5899+2a best epoch 8, val=1.0254

Random g300:
  697 deck:  298/300 = 99.3%
  5899 deck: 297/300 = 99.0%

Focused vs old Crustle 3cd:
  697 drop-hardfail vs old Crustle: 3/120 = 2.5%
  5899 drop-hardfail vs old Crustle: 5/120 = 4.2%
```

Interpretation:

- Failure-trajectory filtering worked technically and did not cause a basic
  play collapse.
- It did not repair the structural Ogerpon-vs-Crustle weakness. This matches
  the earlier diagnosis: removing bad losses improves label hygiene, but does
  not create the missing anti-Crustle plan.
- Use this filter as a hygiene primitive for future clean-teacher/rollout
  training, not as the main fix. The next meaningful step is explicit
  matchup-teacher data: e.g. keep Ogerpon-vs-Crustle clean wins from
  `2a5072194fdf`, add matchup-specific plan labels, or generate teacher
  rollouts with an anti-Crustle policy.

## 2026-08-08 Packaged v12 Sequence Planner Candidates

The user asked to package the two most worthwhile completed non-Marnie
sequence-planner candidates. Selected:

- `Mega Lucario 43d6d8b0fce9`: clearest positive sequence-planner result.
  Random g500 `493/500`, same-sig RR row mean `0.536` vs w4 Lucario `0.452`,
  and baseline-delta vs w4 random-ge097 pool `+0.124`, lost `1/17`.
- `Festival Lead e82dcbe62260`: second-best current ablation candidate.
  Random g500 `500/500`, baseline-delta vs w4 random-ge097 pool `+0.046`.
  Same-sig RR is weaker than w4 Festival (`0.632` vs `0.679`), so treat this
  as a useful probe rather than a guaranteed ladder upgrade.

Do not submit current `Ogerpon 2a507` sequence-planner checkpoint; random g500
was only `382/500` and broad delta was negative. This may be partly a deck-sig
issue, but this exact packaged candidate was rejected.

Remote packaged tarballs:

```text
/home/jie/Do/0_PTCG/submission/v12_seqplan_candidates_20260808/v12_seqplan_lucario_43d6_scratch_w3.tar.gz
  sha256: 04d1e82fcb0f506e2f69fd64b7510d694be382675cbad5478a3f6ed082aff4e0

/home/jie/Do/0_PTCG/submission/v12_seqplan_candidates_20260808/v12_seqplan_festival_e82_scratch_w3.tar.gz
  sha256: d4bdb8487350bbbefe461ec7357d3f87772352b4de2caf7afb3557249d2348c9
```

Packaging script used:

```text
/tmp/package_v12_seqplan_candidates_20260808.sh
```

## 2026-08-08 Ogerpon v12 Sequence Planner Diagnosis

The current `Ogerpon 2a5072194fdf` sequence-planner checkpoint is weak, but
this should not be generalized to all Ogerpon.

Observed random results:

```text
v12 seqplan 2a507: 382/500 = 76.4%
v12 history-init 2a507: 426/500 = 85.2%
older cross-attn 697: 496/500 = 99.2%
older cross-attn 5899: 497/500 = 99.4%
w4 sigpure 697: 197/200 = 98.5%
w4 sigpure 2a507: 177/200 = 88.5%
w4 sigpure 5899: 200/200 = 100.0%
```

Training-band v12 corpus rows for Ogerpon sigs using bands
`900-999`, `1000-1099`, `1100-1199`, `1200+`:

```text
697a82e582d5: rows=180406, row_win_share=0.552,
  bands={1000-1099:93297, 1100-1199:33439, 1200+:41403, 900-999:12267}

2a5072194fdf: rows=98683, row_win_share=0.610,
  bands={1000-1099:98441, 1100-1199:242}

5899c772bace: rows=78444, row_win_share=0.568,
  bands={1000-1099:27456, 1100-1199:40535, 900-999:10453}
```

Important interpretation:

- The `2a507` sequence-planner model was not trained on low-band `600-699`
  rows because the training command filtered to `900+`, but it also has almost
  no `1100+` training signal and no `1200+` rows. It is effectively a
  1000-band imitation target, not a top-player Ogerpon target.
- `697` and `5899` remain much stronger Ogerpon candidates. If testing Ogerpon
  with v12/sequence planner, train/evaluate those sigs first.
- The v12 seqplan architecture is active: checkpoints contain history/log/board
  history and hierarchical `plan_*` weights. The conservative part is the
  objective: it is still mainly next-action BC. `trajectory_csv=none` and
  `trajectory_target_loss_weight=0.0` in the current seqplan run.
- Opponent action history is disabled (`opp_history_k=0`) for submittable
  models. Own action history at inference is self-generated, while training
  history is teacher/episode history; this exposure mismatch can hurt sequence
  models if early decisions go off-policy.

## 2026-08-07 v12 Multi-Stream History Extraction

The user asked for a more complete v12 extraction with aggressive history,
not another small v11 feature tweak. Current implementation introduces a
multi-stream history schema:

- Own previous labeled decisions: `own_hist_*`, default length 32.
- Opponent previous labeled decisions: `opp_hist_*`, default length 32. This is
  saved for offline diagnostics, trace, and ablation, but should not be enabled
  by default for Kaggle-submittable models because live inference cannot exactly
  reconstruct opponent label actions.
- Public observation logs: `log_hist_*`, default length 128. This is safe for
  Kaggle inference because it comes from `observation.logs`.
- Previous board snapshots from the same player perspective:
  `board_hist_cards`, `board_hist_feats`, `board_hist_mask`, default 12
  snapshots with 32 scalar features each.

Changed files:

```text
ptcg_rl/history_features.py
tools/bc_extract_v2.py
ptcg_rl/bc2/data.py
ptcg_rl/model.py
ptcg_rl/numpy_policy.py
tools/bc2_train.py
tools/bc2_accuracy.py
tools/bc2_failure_report.py
tools/train_bc_population.py
tools/build_shadow_pool.py
```

Smoke tests completed:

- Local `python3 -m py_compile` for all changed files passed.
- Local synthetic forward passed for both `pointer` and `cross_attn` using
  `history_k=32`, `opp_history_k=32`, `log_history_k=128`,
  `board_history_k=12`, `board_history_feat_dim=32`.
- Remote extraction smoke on `2026-08-05` with 200 episodes passed:
  `30968` decisions, `bad=0`, `err=0`.
- Remote smoke schema confirmed:
  `state=(80,)`, `option=(1,64)`, `own=(32,)`, `opp=(32,)`,
  `log=(128,)`, `board_cards=(12,12)`, `board_feats=(12,32)`.
- Remote random untrained v12 `pointer` and `cross_attn` checkpoints both ran
  through `tools/eval_bc.py` without NumPy inference crashes. Their losses/timeouts
  are expected for random weights and are not model-quality evidence.

Formal v12 extraction was started on `ks`:

```text
script: /tmp/run_extract_v12_0701_0805.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
raw episodes: /home/jie/Do/0_PTCG/workspace/episodes_raw
zip_count: 36, dates 2026-07-01 through 2026-08-05
output: data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12
log: logs/extract_v12_0701_0805_hist32_log128_board12.log
workers: 12
history: action=32, log=128, board=12, board_feat=32
main script pid at launch: 1411587
extract parent pid at launch: 1411593
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/extract_v12_0701_0805_hist32_log128_board12.log'
ssh ks 'pgrep -af "run_extract_v12_0701_0805|bc_extract_v2.py" | head -n 20'
```

Recommended first v12 training inputs after extraction completes:

- Use `--history-k 32 --log-history-k 128 --board-history-k 12
  --board-history-feat-dim 32`.
- Keep `--opp-history-k 0` for submittable models unless explicitly doing an
  offline-only ablation.
- Compare pointer vs cross-attn, and consider `--hierarchical-plan` only after
  base v12 history quality is measured.

Active status at 2026-08-07 23:43 Asia/Shanghai:

- v12 extraction is still running and healthy. It reached `24/36` zip files
  completed, with `bad=0` and `err=0` in the latest checked log.
- The next training runner has been uploaded and started:
  `/tmp/run_v12_history_pilots_20260807.sh`.
- Runner log:
  `logs/v12_history_pilots_20260807.runner.log`.
- Runner process observed:
  `bash /tmp/run_v12_history_pilots_20260807.sh`.
- The runner waits for both `/tmp/run_extract_v12_0701_0805.sh` and
  `tools/bc_extract_v2.py ... bc_corpus_banded_v12_0701_0805_hist32_log128_board12`
  to exit before training.
- It intentionally uses no `--opp-history-k`, so the resulting history models
  remain Kaggle-submittable if their evaluations are good.
- First v12 pilot wave after extraction:
  `marnie_b8f_v12hist_pointer_init`,
  `lucario_43d_v12hist_pointer_init`,
  `ogerpon_5899_v12hist_pointer_init`,
  `lucario_43d_v12nohist_pointer_refit`.
- Second wave:
  `marnie_b8f_v12nohist_pointer_refit`,
  `ogerpon_5899_v12nohist_pointer_refit`,
  `lucario_43d_v12hist_cross_init`,
  `marnie_b8f_v12hist_cross_init`.

Monitor the active v12 work with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_history_pilots_20260807.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_v12_history_pilots|bc2_train.py.*v12_history_pilots|bc2_accuracy.py.*v12_history_pilots|eval_bc.py.*v12_history_pilots" | head -n 40'
```

Ablation packages prepared on `ks`:

- `submissions/historyk_lucario_43d_hist8_init_w4.tar.gz`
- `submissions/historyk_marnie_b8f_hist8_init_w4.tar.gz`

Both are about `42M` and were validated to contain `cg/`, `deck.csv`,
`main.py`, `policy.npz`, and `ptcg_rl/`. The checkpoints use the previous
incomplete own-history-k implementation with `history_k=8`, so treat these as
ablation submissions, not final v12 candidates.

Update at 2026-08-08 03:30 Asia/Shanghai:

- v12 extraction completed and the pilot runner started wave 1.
- `lucario_43d_v12hist_pointer_init` trained, accuracy completed, and random
  audit was `288/300 = 96.0%`.
- `ogerpon_5899_v12hist_pointer_init` trained, accuracy completed, and random
  audit was `299/300 = 99.7%`.
- `lucario_43d_v12nohist_pointer_refit` appears to have trained and saved
  checkpoints, but accuracy/random were not reached because the runner later
  stopped.
- `marnie_b8f_v12hist_pointer_init` trained and saved
  `checkpoints/v12_history_pilots_20260807/bc2_marnie_b8f_v12hist_pointer_init.npz`,
  but `tools/bc2_accuracy.py` failed immediately afterward with
  `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`.
- Because `run_wave` treats any failed accuracy/random as wave failure, wave 2
  did not start. The missing next steps are to rerun Marnie accuracy, run
  Marnie random, evaluate the no-history Lucario checkpoint, then launch the
  remaining nohist/cross wave in a fresh script. Prefer CPU or a fresh CUDA
  process with lower memory pressure for the Marnie accuracy retry.

Update at 2026-08-08 10:40 Asia/Shanghai:

- The user requested parallel execution because all A800 GPUs had enough free
  memory. The sequential resume script was stopped at the wrapper level:
  `/tmp/run_v12_history_pilots_resume_20260808.sh`. Its existing Marnie CPU
  accuracy child was intentionally left running to preserve progress.
- Active parallel runner:
  `/tmp/run_v12_history_pilots_parallel_20260808.sh`, log
  `logs/v12_history_pilots_parallel_20260808.runner.log`.
- This runner evaluates existing Marnie hist and Lucario nohist checkpoints and
  runs pointer/no-history jobs in parallel:
  - `marnie_b8f_v12nohist_pointer_refit` on GPU0.
  - `ogerpon_5899_v12nohist_pointer_refit` on GPU1.
- `lucario_43d_v12hist_cross_init` failed again under regular cuDNN with
  `CUDNN_STATUS_NOT_INITIALIZED`. The old Marnie cross process was killed before
  it likely hit the same failure.
- Added `PTCG_DISABLE_CUDNN=1` support in `tools/bc2_train.py` and
  `tools/bc2_accuracy.py`. When this env var is set, `torch.backends.cudnn.enabled`
  is disabled before model construction. This is intended only for history/GRU
  CUDA runs that hit cuDNN initialization errors.
- Active cross-only retry:
  `/tmp/run_v12_cross_nocudnn_20260808.sh`, log
  `logs/v12_cross_nocudnn_20260808.runner.log`.
  It exports `PTCG_DISABLE_CUDNN=1` and runs:
  - `lucario_43d_v12hist_cross_init` on GPU2.
  - `marnie_b8f_v12hist_cross_init` on GPU3.
- The nocudnn Lucario cross retry passed the previous failure point and reached
  epoch 1, so the env-var workaround appears viable.

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 100 logs/v12_history_pilots_parallel_20260808.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 100 logs/v12_cross_nocudnn_20260808.runner.log'
ssh ks 'pgrep -af "run_v12_history_pilots_parallel|run_v12_cross_nocudnn|bc2_train.py.*v12_history_pilots|bc2_accuracy.py.*v12_history_pilots|eval_bc.py.*v12_history_pilots"'
```

Update at 2026-08-08 10:50 Asia/Shanghai:

- Do not schedule v12 work by GPU memory alone. The ks container has a cgroup
  memory limit of `274877906944` bytes, about `256GiB`, even though host
  `free -h` shows much more memory available.
- Parallel v12 did speed up small-corpus jobs, but Marnie b8f is a large corpus
  case: `3790712` kept samples across `115` files. When Marnie cross/no-history
  loaded alongside other v12 accuracy/random jobs, the kernel reported
  `Memory cgroup out of memory` and killed Python with status `137`.
- Completed/observed v12 results so far:
  - `lucario_43d_v12hist_pointer_init`: random `288/300 = 96.0%`.
  - `lucario_43d_v12nohist_pointer_refit`: random `286/300 = 95.3%`.
  - `ogerpon_5899_v12hist_pointer_init`: random `299/300 = 99.7%`.
  - `ogerpon_5899_v12nohist_pointer_refit`: random `296/300 = 98.7%`.
  - `marnie_b8f_v12hist_pointer_init`: random `300/300 = 100.0%`.
  - `lucario_43d_v12hist_cross_init` with `PTCG_DISABLE_CUDNN=1`: training
    completed, best `val=1.0184`; CPU accuracy exact/first/top3 was
    `0.672/0.687/0.924`; random was `296/300 = 98.7%`.
  - `marnie_b8f_v12hist_cross_init` with `PTCG_DISABLE_CUDNN=1`: killed by
    cgroup OOM during epoch 1, not a model/feature failure.
  - `marnie_b8f_v12nohist_pointer_refit`: killed by cgroup OOM before producing
    a checkpoint, not a model/feature failure.
- Active guard runner:
  `/tmp/run_v12_marnie_large_guarded_20260808.sh`, log
  `logs/v12_marnie_large_guarded_20260808.runner.log`.
  It waits until no `bc2_train.py`, `bc2_accuracy.py`, or `eval_bc.py` v12
  pilot jobs are active and cgroup `memory.current < 170GiB`, then runs Marnie
  no-history and Marnie cross sequentially. This is intentional: Marnie big
  jobs should not be parallelized until the data loader is made streaming or
  memory-mapped.
  At 2026-08-08 10:50, it started
  `marnie_b8f_v12nohist_pointer_refit` on GPU0 after Lucario cross evaluation
  finished.

Monitor the guard:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_marnie_large_guarded_20260808.runner.log'
ssh ks 'cat /sys/fs/cgroup/memory.current'
```

## 2026-08-08 Sequence Planner Route

The user explicitly asked to stop conservative BC fine-tuning and start a real
hierarchical/sequence planner. Treat this as a new route, not another
winner-only/filter/weight tweak.

Observed v12 history-cross result before this change:

- Non-Marnie v12 history-cross helped Lucario only modestly and did not rescue
  Ogerpon/TRM/Festival generally. Mini RR row means showed
  `v12_lucario=0.529` vs `w4_lucario=0.430`, but `v12_ogerpon=0.205` vs
  `w4_ogerpon=0.211`, `v12_trmewtwo=0.684` vs `w4_trmewtwo=0.671`, and
  `v12_festival=0.677` vs `w4_festival=0.593`.
- Baseline-delta against w4 random_ge097 was only `+0.055` for Lucario,
  `+0.009` for Festival, `-0.015` for TRM, and `-0.071` for Ogerpon.
- Interpretation: passive history features alone are not enough. The model
  needs an explicit plan mode tied to the current point in the game and the
  selected card/action/context sequence.

Implemented locally:

- New `ptcg_rl/plan_labels.py` derives per-decision multi-label plan targets:
  `setup`, `engine`, `power`, `attack`, `disrupt`, `preserve`, `stall`,
  `finish`.
- `BCCorpus(..., step_plan=True, archetype=...)` now groups by game and labels
  each kept row using row order, current board/hand/features, selected first
  action type/card/context, and `deck_plans.py` archetype tags.
- `sequence_nll` now supports `step_plan_weight` and
  `plan_teacher_forcing`. Teacher forcing feeds the step plan label into the
  hierarchical scorer for part of training while inference still uses the
  predicted plan head.
- `tools/bc2_train.py` exposes:
  `--step-plan`, `--step-plan-loss-weight`, and
  `--step-plan-teacher-forcing`.
- The checkpoint format stays submittable through existing NumPy inference
  because it still uses the existing sigmoid plan head and
  `plan_condition_fc`/`plan_score_fc*` keys.
- Local `py_compile` passed, and a synthetic CPU smoke test produced plan
  counts and a finite hierarchical loss.

First planner training should be from scratch. Do not pass `--init` for these
experiments unless the user explicitly asks for an ablation:

```bash
python3 tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12 \
  --archetype "Mega Lucario" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --deck-sig 43d6d8b0fce9 \
  --arch cross_attn --width 4 --state-layers 2 \
  --history-k 32 --log-history-k 128 --board-history-k 12 --board-history-feat-dim 32 \
  --hierarchical-plan --step-plan \
  --step-plan-loss-weight 0.30 --step-plan-teacher-forcing 0.50 \
  --epochs 10 --batch-size 2048 --lr 8e-5 \
  --first-action-weight 1.5 --win-weight 1.5 --loss-weight 0.4 --draw-weight 0.8 \
  --state-feat-dim 80 --opt-feat-dim 64 \
  --cuda-memory-gb 24 --device cuda:0 \
  --save checkpoints/v12_sequence_planner_20260808/bc2_lucario_43d_seqplan_scratch_w4.npz
```

Runtime update at 2026-08-08 17:05 Asia/Shanghai:

- The old `bc2_marnie_b8f_v12hist_cross_init.npz` checkpoint is usable even
  though its epoch 6 process was OOM-killed. Epoch 5 saved best
  `val=0.5404`, and a random audit completed:
  `495/500 = 99.0%`, log
  `logs/v12_history_pilots_20260807/marnie_b8f_v12hist_cross_init.random_g500.log`.
  Treat it as a v12 history-cross init ablation candidate, not as the new
  scratch sequence planner.
- First scratch sequence planner attempt with `width=4 batch=2048` failed
  immediately on Lucario/Festival with CUDA OOM around the hierarchical option
  scorer. Keep those logs; they show that this architecture's peak memory is
  driven by `batch * max_options * width`, not just model parameter count.
- Active scratch planner runner:
  `/tmp/run_sequence_planner_scratch_w3_20260808.sh`, log
  `logs/v12_sequence_planner_20260808/runner_w3.log`.
  It uses `width=3`, `batch=1024`, `PTCG_DISABLE_CUDNN=1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no `--init`, and
  `--step-plan --hierarchical-plan --step-plan-loss-weight 0.30
  --step-plan-teacher-forcing 0.50`.
- Active wave 1:
  - `lucario_43d` on GPU0, save
    `checkpoints/v12_sequence_planner_20260808/bc2_lucario_43d_seqplan_scratch_w3.npz`
  - `festival_e82` on GPU1, save
    `checkpoints/v12_sequence_planner_20260808/bc2_festival_e82_seqplan_scratch_w3.npz`
- If wave 1 succeeds, the runner automatically starts:
  - `trmewtwo_06f0` on GPU2
  - `ogerpon_2a507` on GPU3
- Marnie scratch planner is intentionally not in this runner. The container is
  usually already above `160GiB/256GiB` cgroup memory, and Marnie b8f corpus
  loads are large enough to cause process kills. Run it separately after memory
  drops or after making `BCCorpus` streaming/memmap.

Update at 2026-08-08 17:34 Asia/Shanghai:

- Wave 1 completed successfully.
  - `lucario_43d` best `val=0.9814`; accuracy on 30k samples:
    `exact=0.718`, `first=0.731`, `top3=0.945`; random:
    `300/300 = 100.0%`.
  - `festival_e82` best `val=0.9447`; accuracy on 30k samples:
    `exact=0.713`, `first=0.723`, `top3=0.935`; random:
    `300/300 = 100.0%`.
- Wave 2 started, but `trmewtwo_06f0` with `width=3 batch=1024` OOMed on GPU2
  in the hierarchical option scorer before producing a checkpoint.
- A separate retry was launched:
  `/tmp/run_trmewtwo_seqplan_w3_b512_retry_20260808.sh`, log
  `logs/v12_sequence_planner_20260808/trmewtwo_06f0_w3_b512.runner.log`,
  train log `logs/v12_sequence_planner_20260808/trmewtwo_06f0_w3_b512.train.log`,
  save path
  `checkpoints/v12_sequence_planner_20260808/bc2_trmewtwo_06f0_seqplan_scratch_w3_b512.npz`.
  It uses the same scratch sequence-planner recipe but `batch=512` and
  `--cuda-memory-gb 22`.
- `ogerpon_2a507` with `width=3 batch=1024` is still running normally. At the
  last check it had reached epoch 5/12 after saving bests through epoch 4
  (`val=0.9503` at epoch 4).
- Monitor active jobs:

```bash
ssh ks 'pgrep -af "bc2_train.py.*2a5072194fdf|bc2_train.py.*06f0b265154c|bc2_accuracy.py.*seqplan|eval_bc.py.*seqplan|run_trmewtwo_seqplan|run_sequence_planner_scratch_w3"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_sequence_planner_20260808/ogerpon_2a507_w3.train.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_sequence_planner_20260808/trmewtwo_06f0_w3_b512.train.log'
```

Update at 2026-08-08 17:51 Asia/Shanghai:

- Marnie scratch sequence-planner training was queued but not allowed to start
  immediately because cgroup memory was about `200GiB/256GiB`.
- Guard runner:
  `/tmp/run_marnie_seqplan_w3_b512_guard_20260808.sh`
  with log
  `logs/v12_sequence_planner_20260808/marnie_b8f_w3_b512.guard.log`
  and wrapper log
  `logs/v12_sequence_planner_20260808/marnie_b8f_w3_b512.runner.log`.
- Save path when it starts:
  `checkpoints/v12_sequence_planner_20260808/bc2_marnie_b8f_seqplan_scratch_w3_b512.npz`.
- It waits until no active Ogerpon/TRM seqplan train/eval jobs remain and
  cgroup memory is below `125GiB`. This is intentional. Previous Marnie b8f
  runs imply Marnie can need roughly `110-130GiB` CPU/cgroup memory by itself,
  so starting while the system baseline is around `160-200GiB` risks another
  status-137 cgroup OOM kill.
- Marnie recipe: scratch, no `--init`, `cross_attn width=3`,
  `batch=512`, `--cuda-memory-gb 22`, `--step-plan`,
  `--step-plan-loss-weight 0.30`, `--step-plan-teacher-forcing 0.50`,
  `checkpoint_every=1`, `epochs=12`.

Update at 2026-08-08 19:15 Asia/Shanghai:

- The `pkm_repo` orphan multiprocessing workers on `ks` were cleaned after the
  user confirmed they were stale. This removed 134 orphan worker processes and
  reduced cgroup usage from about `242GiB` to `137GiB`; RSS dropped from about
  `134GiB` to `32GiB`.
- `drop_caches` is read-only in this container, and cgroup
  `memory.force_empty` is unavailable. A non-destructive
  `posix_fadvise(DONTNEED)` file-cache evict scanned about `78.7GiB` and only
  reduced cgroup usage from `137.0GiB` to `132.5GiB`.
- The old waiting-only Marnie guard was stopped, and Marnie scratch sequence
  planner was started directly on `cuda:3` with a watchdog that kills Marnie
  first if cgroup usage exceeds `246GiB`.
- Active Marnie job:
  `checkpoints/v12_sequence_planner_20260808/bc2_marnie_b8f_seqplan_scratch_w3_b512.npz`.
  At the latest check it was around epoch `2/12`, step `3600/7610`, cgroup
  about `195GiB`, GPU3 about `7.8GiB`.
- TRM scratch sequence planner completed. Best was epoch 10:
  `val=0.9239`; epoch 11/12 slightly worsened. Random audit on the scratch
  checkpoint was `499/500 = 99.8%`.
- Non-Marnie scratch sequence-planner tests were completed in:
  `logs/eval_v12_sequence_planner_20260808_nonmarnie`.
  These are the current scratch sequence-planner checkpoints under
  `checkpoints/v12_sequence_planner_20260808`, not the older v12 history init
  checkpoints.
- Random g500:
  - `seqplan_festival_e82`: `500/500 = 100.0%`
  - `seqplan_lucario_43d`: `493/500 = 98.6%`
  - `seqplan_trmewtwo_06f0`: `499/500 = 99.8%`
  - `seqplan_ogerpon_2a507`: `382/500 = 76.4%`
- Same-sig mini RR row means (`g80`, row beats all other entries):
  - `seqplan_trmewtwo_06f0`: `0.670`
  - `seqplan_festival_e82`: `0.632`
  - `seqplan_lucario_43d`: `0.536`
  - `seqplan_ogerpon_2a507`: `0.175`
  - w4 baselines in same matrix: Festival `0.679`, TRM `0.639`,
    Lucario `0.452`, Ogerpon `0.214`.
- Baseline-delta against w4 random_ge097 pool (`17` opponents, `80` games):
  - `seqplan_lucario_43d`: `avg_delta=+0.124`, candidate `0.480`,
    baseline `0.357`, lost `1/17`; worst
    `mega_lopunny_sig2_276707c0:-0.037`, best
    `alakazam_sig2_5a5ea26c:+0.225`.
  - `seqplan_festival_e82`: `avg_delta=+0.046`, candidate `0.593`,
    baseline `0.547`, lost `3/17`; worst
    `mega_lopunny_sig1_b0cb21e2:-0.188`, best
    `marnie_grimmsnarl_sig2_2c22fa76:+0.188`.
  - `seqplan_trmewtwo_06f0`: `avg_delta=-0.024`, candidate `0.493`,
    baseline `0.517`, lost `11/17`; worst
    `team_rocket_mewtwo_sig1_06f0b265:-0.150`, best
    `teal_mask_ogerpon_sig1_697a82e5:+0.087`.
  - `seqplan_ogerpon_2a507`: `avg_delta=-0.074`, candidate `0.127`,
    baseline `0.201`, lost `13/17`; worst
    `mega_lopunny_sig1_b0cb21e2:-0.213`, best
    `festival_lead_sig1_e82dcbe6:+0.050`.
- Interpretation:
  - Scratch sequence planner is not globally better.
  - `seqplan_lucario_43d` is the clearest positive result and is a reasonable
    ablation submission candidate.
  - `seqplan_festival_e82` is mildly positive against the w4 pool but weaker
    than w4 Festival in same-sig RR, so it is a lower-priority candidate.
  - `seqplan_trmewtwo_06f0` passes random and same-sig RR, but loses to its w4
    baseline on the broader w4 pool; do not treat it as an improvement.
  - `seqplan_ogerpon_2a507` is a failed version. It fails random gate and
    broad matchup checks. Do not submit or continue training it by just adding
    epochs; the plan supervision likely distorts Ogerpon's core action
    distribution.

## 2026-08-08 Top-Player Strategy Mining

Added and synced `tools/mine_top_player_strategy.py`.

Purpose:

- Mine rule/teacher hypotheses from strong Kaggle team/deck episode behavior,
  for example a LiamK-like team, without hand-writing global rules from
  intuition.
- Compare target cohort rows such as `--target-team-name` or
  `--target-deck-sig` against same-matchup control rows.
- Output game-level trajectory metrics, selected-action/event gaps, legal
  opportunity choice gaps, 2/3-gram sequence gaps, ranked rule candidates, and
  `target_game_keys.csv`/`control_game_keys.csv` for downstream
  `build_bc_subset.py`.

Remote smoke test:

```text
corpus: data/bc_corpus_banded_v11_0804_only
archetype: Marnie Grimmsnarl
deck_sig: b8f251a476e7
target team: MissingNo.
target outcome: win
control outcome: loss
out: logs/top_player_strategy_smoke/marnie_missingno_vs_control_loss_v2
target games: 48
control games: 280
```

Smoke findings were sensible enough for tool validation, not a rule conclusion:
positive candidates included `MAIN turn=5-6 available=RETREAT`,
`MAIN turn=5-6 available=EVOLVE:648 Marnie's Grimmsnarl ex`, and sequence gaps
such as `MAIN:RETREAT > SWITCH:Marnie's Grimmsnarl ex`.

Recommended next usage:

```bash
python3 tools/build_team_deck_trajectories.py \
  --corpus data/bc_corpus_banded_v11_0701_0804 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --min-decisions 5000 \
  --min-episodes 30 \
  --top 40 \
  --out logs/top_player_strategy_20260808/marnie_team_trajectories.csv

python3 tools/mine_top_player_strategy.py \
  --corpus data/bc_corpus_banded_v11_0701_0804 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --deck-sig b8f251a476e7 \
  --target-team-name "<TOP_TEAM_FROM_TRAJECTORY_CSV>" \
  --target-outcome win \
  --control-outcome loss \
  --opponent-archetype "Teal Mask Ogerpon" \
  --min-games 10 \
  --min-rate-gap 0.08 \
  --min-choose-gap 0.10 \
  --top 40 \
  --out-dir logs/top_player_strategy_20260808/marnie_top_vs_ogerpon
```

Interpretation discipline:

- Positive `opportunity_gap` rows are narrow rerank/rule-probe candidates.
- `event_gap` n-grams and `metric_gap` rows are sequence/teacher hypotheses,
  not direct hard rules.
- Use `target_game_keys.csv` only after confirming the target cohort is
  genuinely strong in the specific matchup; otherwise it can encode a strong
  player's general style rather than a counter-plan.

## 2026-08-08 Matchup Teacher Quality / Clean Success Data

The user clarified that weak-matchup win mining must use all same-archetype
deck signatures/teams as potential teachers, not only the exact target
signature. We also must distinguish real counter-strategy wins from wins caused
by opponent brick, early end, or no setup.

Completed remote teacher quality audit:

```text
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
corpus: data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12
runner log: logs/v12_matchup_teachers_20260808/quality_audit.runner.log
quality summary: logs/v12_matchup_teachers_20260808/quality_audit/quality_all_pairs.csv
per-game labels: logs/v12_matchup_teachers_20260808/quality_audit/game_quality_all_pairs.csv
rows: 359 teacher rows, 15079 per-game rows
status: completed
```

Important conclusions:

- `Teal Mask Ogerpon => Crustle Wall` is not best explained by Ogerpon
  `5899...`; the clean teacher is `2a5072194fdf / James Cox & Henry Chao`.
  Full pair signal was `82/458 = 17.9%`, but this teacher was `61/150 = 40.7%`
  with `58/61` clean wins and only `4.9%` brick share.
- `Marnie Grimmsnarl => Teal Mask Ogerpon` has many clean teachers, mostly
  `b8f251a476e7` teams. Examples: Raihan Ramadistra `34/39` clean,
  `@kdcyberdude` `26/29`, Sixth Sense `27/34`, LiamK `20/35`, Dominic Peel
  `23/45`, plus some `2c22fa761816` teams. This is real learnable data, not
  just sparse lucky wins.
- `Mega Lucario => Teal Mask Ogerpon` and `Mega Lucario => Crustle Wall` both
  identify `43d6d8b0fce9 / Majkel1337` as the primary clean teacher. It had
  `33/34` clean wins vs Ogerpon and `15/16` clean wins vs Crustle.
- Other strong clean teacher pools found:
  `Alakazam => Team Rocket Mewtwo`,
  `Crustle Wall => Team Rocket Mewtwo`,
  `Festival Lead => Team Rocket Mewtwo`,
  `Team Rocket Mewtwo => Dragapult`.

Added tool:

```text
tools/select_clean_teacher_games.py
```

It reads `quality_all_pairs.csv` plus `game_quality_all_pairs.csv`, selects
teacher rows by clean-win/brick thresholds, and writes reusable game-key CSVs
by matchup and by archetype.

Remote selection already generated:

```text
out: logs/v12_matchup_teachers_20260808/clean_teacher_selection_min10_brick010
thresholds: min_clean_wins=10, min_clean_share=0.15, max_brick_share=0.10
selected: 65 teachers, 2228 clean games, 14 pair files, 7 archetype files
```

Largest selected clean pools:

```text
Alakazam vs Team Rocket Mewtwo:       861 games
Crustle Wall vs Team Rocket Mewtwo:   405 games
Marnie Grimmsnarl vs Teal Ogerpon:    267 games
Mega Lucario all selected matchups:   263 games
Teal Mask Ogerpon all selected:       163 games
Team Rocket Mewtwo all selected:      154 games
```

Active remote subset build:

```text
script: /tmp/build_v12_clean_teacher_subsets_20260808.sh
log: logs/v12_matchup_teachers_20260808/build_clean_teacher_subsets.runner.log
output root: data/bc_corpus_clean_teachers_v12_0701_0805_min10_brick010
pid at launch: 2464429
status at 13:50: running, scanning Alakazam first
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_matchup_teachers_20260808/build_clean_teacher_subsets.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "build_v12_clean_teacher_subsets|build_bc_subset.py"'
```

Recommended next step after subset build completes:

- Do not replace general BC with these subsets directly. Use them as
  matchup-conditioned auxiliary data, teacher policies, or representation/plan
  targets while keeping the broad BC anchor.
- First pilots should focus on:
  `Ogerpon 2a507 vs Crustle`, `Marnie b8f/2c22 vs Ogerpon`,
  `Lucario 43d vs Ogerpon/Crustle/Marnie`, and
  `Crustle 3cd/96d/b141 vs Team Rocket Mewtwo`.

### 2026-08-08 0806/0807 Episode Incremental Refresh

The user added Kaggle episode zips for `2026-08-06` and `2026-08-07`.

Remote raw zips verified:

```text
/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-06.zip
/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-07.zip
zipfile -t: both passed
```

Active incremental extraction:

```text
script: /tmp/run_extract_v12_0701_0807_incremental_20260808.sh
log: logs/extract_v12_0701_0807_incremental_20260808.log
source corpus: data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12
new corpus root: data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12
method: cp -al hardlink clone of 0701-0805, then extract only 0806/0807
workers: 1
pid at launch: 2476807
extract child observed: 2476816
status at 14:05: processing 2026-08-06, bad=0, err=0
```

Important correction: the first launch incorrectly extracted `EN_Card_Data.csv`
from `/home/jie/Do/0_PTCG/workspace/data/pokemon-tcg-ai-battle.zip` as if it
were a leaderboard. It was stopped immediately; no `2026-08-06` or `2026-08-07`
npz files had been written. The corrected script only accepts CSVs with
`TeamName` and `Score`. It downloaded and used:

```text
/tmp/lb_0808_v12_extract/pokemon-tcg-ai-battle.zip
pokemon-tcg-ai-battle-publicleaderboard-2026-08-08T05:57:06.csv
Leaderboard: 6547 teams
```

Active downstream refresh waiting on extraction:

```text
script: /tmp/run_v12_0701_0807_teacher_refresh_20260808.sh
runner log: logs/v12_matchup_teachers_20260808_0701_0807.runner.log
output dir: logs/v12_matchup_teachers_20260808_0701_0807
pid at launch: 2479153
status at 14:05: waiting for bc_extract_v2.py to finish
```

When extraction finishes, this runner will:

1. Validate `0806` and `0807` npz outputs and v12 schema.
2. Run `tools/find_matchup_teachers.py` on
   `data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12`.
3. Run `tools/run_matchup_quality_audits.py`.
4. Run `tools/select_clean_teacher_games.py`.
5. Build updated clean teacher subsets under
   `data/bc_corpus_clean_teachers_v12_0701_0807_min10_brick010`.

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/extract_v12_0701_0807_incremental_20260808.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/v12_matchup_teachers_20260808_0701_0807.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "bc_extract_v2.py.*ptcg_episodes_0806_0807|run_v12_0701_0807_teacher_refresh|find_matchup_teachers|run_matchup_quality_audits|select_clean_teacher_games"'
```

Update at 2026-08-08 14:25 Asia/Shanghai:

- Incremental extraction finished `2026-08-06`: `4631/4631 eps`,
  `760263` decisions, `bad=0`, `err=0`.
- It has started `2026-08-07`: `1000/4639 eps`, `159252` decisions,
  `bad=0`, `err=0` in the latest checked log.
- `2026-08-06` has written `39` npz files under
  `data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12`; `2026-08-07`
  had not written final npz files yet at the check.

Active waiting training runner:

```text
script: /tmp/run_v12_0701_0807_history_baselines_20260808.sh
runner log: logs/v12_0701_0807_history_baselines_20260808.runner.log
pid at launch: 2504196
status at 14:25: waiting for 0806/0807 extraction to finish
```

This runner intentionally starts only after extraction is complete and cgroup
memory is below `218GiB`. It runs two parallel waves of v12 `cross_attn`
history-only baselines, then accuracy and random g300:

```text
wave1:
  Mega Lucario 43d6d8b0fce9 on cuda:0
  Teal Mask Ogerpon 2a5072194fdf on cuda:1

wave2:
  Festival Lead e82dcbe62260 on cuda:0
  Team Rocket Mewtwo 06f0b265154c on cuda:1
```

Output:

```text
checkpoints/v12_0701_0807_history_baselines_20260808/
logs/v12_0701_0807_history_baselines_20260808/
```

Rationale: GPU memory is available, but cgroup memory was around
`214GiB / 256GiB` during extraction and Marnie cross training. Do not start
Alakazam, Crustle, or another Marnie large-corpus job in parallel until memory
pressure drops or the loader is made streaming/memmap.

Update at 2026-08-08 15:30 Asia/Shanghai:

- Incremental v12 extraction is complete. Final validation in
  `logs/extract_v12_0701_0807_incremental_20260808.log` reported
  `npz_total=1668`, `npz_0806=58`, `npz_0807=58`, sample state `(80,)`,
  option `(1,64)`, `feature_version=v12_multistream_history`, and history
  dimensions `32/128/12/32`.
- `logs/v12_0701_0807_history_baselines_20260808.runner.log` completed four
  v12 `cross_attn + history` baseline trainings from w4 init:
  - Mega Lucario `43d6d8b0fce9`: best val `0.9558`, accuracy
    exact/first/top3 `0.675/0.690/0.925`, random `298/300 = 99.3%`.
  - Teal Mask Ogerpon `2a5072194fdf`: best val `0.8366`, accuracy
    `0.723/0.740/0.929`, random `264/300 = 88.0%`. Treat this as failed for
    submission until random failure traces are inspected or another Ogerpon
    signature is retrained.
  - Festival Lead `e82dcbe62260`: best val `0.9207`, accuracy
    `0.658/0.669/0.908`, random `300/300 = 100.0%`.
  - Team Rocket Mewtwo `06f0b265154c`: best val `0.8858`, accuracy
    `0.705/0.718/0.939`, random `299/300 = 99.7%`.
- The 0701-0807 teacher refresh is active:
  `logs/v12_matchup_teachers_20260808_0701_0807.runner.log`. It has completed
  teacher scan, quality audit, and clean teacher selection. Current selected
  pools include Marnie vs Ogerpon `283` clean games, Lucario selected matchups
  `338`, Ogerpon selected matchups `240`, and Team Rocket Mewtwo selected
  matchups `154`.
- Clean subset build output root:
  `data/bc_corpus_clean_teachers_v12_0701_0807_min10_brick010`. Completed
  subset npz files at the status check: Alakazam, Crustle Wall, Festival Lead.
  It was actively building the Marnie Grimmsnarl subset.
- Marnie large guarded cross-attn history training is still active:
  `logs/v12_marnie_large_guarded_20260808.runner.log`, checkpoint
  `checkpoints/v12_history_pilots_20260807/bc2_marnie_b8f_v12hist_cross_init.npz`.
  At the check it was still in epoch 4/6 around step `2875/3331`.

Update at 2026-08-08 16:30 Asia/Shanghai:

- Non-Marnie v12 0701-0807 history-cross tests completed:
  `logs/eval_v12_0701_0807_nonmarnie_20260808.runner.log`.
- Random g500:
  - Mega Lucario `43d6d8b0fce9`: `496/500 = 99.2%`.
  - Festival Lead `e82dcbe62260`: `499/500 = 99.8%`.
  - Team Rocket Mewtwo `06f0b265154c`: `500/500 = 100.0%`.
  - Teal Mask Ogerpon `2a5072194fdf`: `426/500 = 85.2%`.
- Mini RR against same-signature w4 controls plus the other v12 models:
  `logs/eval_v12_0701_0807_nonmarnie_20260808/rr_v12_vs_w4_same_sig_g80.csv`.
  Row-average order was:
  `v12_trmewtwo=0.684`, `v12_festival=0.677`, `w4_trmewtwo=0.671`,
  `w4_festival=0.593`, `v12_lucario=0.529`, `w4_lucario=0.430`,
  `w4_ogerpon=0.211`, `v12_ogerpon=0.205`.
- Baseline-delta vs `candidate_manifest_w4_random_ge097.csv`, 80 games per
  opponent:
  - Lucario v12 vs w4: candidate `0.406`, baseline `0.351`,
    `avg_delta=+0.055`, `lost=3/17`; modest real improvement but still a low
    absolute strong-pool win rate.
  - Festival v12 vs w4: candidate `0.552`, baseline `0.543`,
    `avg_delta=+0.009`, `lost=7/17`; effectively tied with w4.
  - TR Mewtwo v12 vs w4: candidate `0.488`, baseline `0.503`,
    `avg_delta=-0.015`, `lost=7/17`; do not replace w4 with this checkpoint.
  - Ogerpon v12 2a507 vs w4: candidate `0.137`, baseline `0.208`,
    `avg_delta=-0.071`, `lost=15/17`; failed both random and strong-pool
    tests, do not submit.
- Interpretation: v12 history-cross did not deliver a broad upgrade over the
  w4 signature specialists. Random stability alone is insufficient; use
  baseline-delta/RR as the submission gate. The only non-Marnie v12 checkpoint
  worth deeper follow-up is Lucario `43d6d8b0fce9`, and even that is a modest
  ablation candidate rather than a strong submission recommendation.
- Marnie large guarded cross-attn history training was still active at the last
  check, epoch 5/6 around step `3175/3331`.

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
  - `tools/mine_strategy_trajectories.py`: new trajectory-level miner that groups BC corpus rows by game, compares winning vs losing whole-game trajectories, emits metric/event/ngram gaps, and writes `strategy_seeds.csv`.
  - `tools/build_bc_subset.py`: now supports whole-game filtering via `--game-key-csv` plus repeatable `--where` conditions, so strategy-conditioned subsets can be built from trajectory reports.
  - `ptcg_rl/deck_plans.py`: Mega Lucario plan now includes Riolu `677` and key 43d cards; Marnie plan now marks Spikemuth Gym `1259` as a search/stadium key. This fixes misleading Lucario reports that only tracked Riolu `333`.
  - `tools/eval_round_robin.py`: now supports `--manifest`, `--manifest-limit`, and `--manifest-random`. It reads CSVs with `eval_entry` or `checkpoint_path`/`deck_path`, skips exact duplicate entries, and suffixes duplicate names.
  - `tools/analyze_kaggle_replays.py`: `--known-decks-dir` is now repeatable, matching the README examples and allowing 0804/0802 deck pools to be loaded together for replay naming.
  - `tools/eval_round_robin.py`: now also supports `--mcts-entry NAME` for per-entry MCTS probes. Use this when checking whether search helps one candidate; the old global `--mcts` turns MCTS on for every policy entry and can pollute candidate-vs-opponent conclusions.
  - `ptcg_rl/numpy_policy.py`: now has `first_step_ranking()` for diagnostics only; submission selection is unchanged.
  - `tools/trace_matchup_decisions.py`: rich trace fields now include available action cards and first-step top option ranking.
  - `tools/trace_outcome_gap_report.py`: new report tool for loss-vs-win overrepresented decision patterns; includes a `miss_card_summary` view for available-card/actual-choice misses.
  - `tools/bc2_train.py` / `ptcg_rl/bc2/data.py`: `--card-weight ID=WEIGHT` multiplies samples whose true first selected option has that card id.
  - `tools/split_shadow_pools.py`: new tool to join shadow manifests with random audits and emit balanced environment/quality/stress/debug manifests.
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

## 2026-08-07 Trajectory BC / Mega Lucario 43d

The user explicitly pointed out that weak-matchup improvement is a continuous
strategy problem, not a single action-bias problem. Current conclusion supports
that: single-step Marnie weighting barely helped Lucario, but whole-game tempo
success filtering produced a measurable improvement.

Remote paths:

```text
logs/lucario43d_20260807/trace_marnie/
logs/strategy_trajectories_20260807/lucario43d_vs_marnie_v11_0701_0804_v2/
data/strategy_bc_subsets_20260807/Mega_Lucario/lucario43d_marnie_tempo_success/
data/strategy_bc_subsets_20260807/Mega_Lucario/lucario43d_marnie_fast_grimmsnarl_success/
checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_strategy_tempo_w4.npz
checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_strategy_fastgrim_w4.npz
logs/lucario43d_20260807/strategy_pilots/
```

Actual Kaggle episode corpus signal for `Mega Lucario` deck sig
`43d6d8b0fce9` vs `Marnie Grimmsnarl` from
`data/bc_corpus_banded_v11_0701_0804`:

```text
games=202 wins=104 losses=98 wr=0.515
decisions=13936
```

Important winning trajectory differences:

```text
attack_count                     win 5.375 vs loss 4.102
attack_by_4                      win 0.962 vs loss 0.786
attack_by_6                      win 1.000 vs loss 0.888
board_by_4_solrock               win 0.990 vs loss 0.847
board_by_4_lunatone              win 0.971 vs loss 0.847
board_by_4_makuhita              win 0.923 vs loss 0.745
board_by_4_mega_lucario_ex       win 0.875 vs loss 0.704
opp_board_by_4_grimmsnarl_ex     win 0.202 vs loss 0.388
opp_active_turns_grimmsnarl_ex   win 26.25 vs loss 37.15
```

Interpretation:

- Lucario 43d beats Marnie in real episodes when it plays a tempo plan:
  early Riolu/Mega Lucario, early Solrock/Lunatone/Makuhita, and attacks by
  turn 4-6.
- Losses are overrepresented when Marnie stabilizes Grimmsnarl ex early and
  keeps it active/bench for many turns. This is a choke-point signal: future
  work should look for earlier windows to pressure Impidimp/Morgrem/Grimmsnarl
  or prevent the game from reaching stable Grimmsnarl, not merely downweight one
  action.
- Spikemuth Gym appears as a loss-overrepresented candidate-side ability action
  in trace/corpus reports. Treat this as a sequence smell, not a direct ban:
  it often appears after the game has already entered a losing resource loop.

Strategy subsets built:

```text
tempo_success:              86 games, 6075 decisions
fast_grimmsnarl_success:    21 games, 1520 decisions
```

Pilot fine-tunes from existing 43d w4:

```text
strategy_tempo:
  random_g500: 475/500 = 95.0%
  vs Marnie b8f g300: 44/300 = 14.7%, baseline 35/300 = 11.7%, delta +3.0pp
  core g200 avg_delta +4.67pp
  core deltas: Marnie +4.5, Lopunny +5.0, Alakazam +4.5,
               Crustle +8.5, Dragapult +6.0, Ogerpon -0.5

strategy_fastgrim:
  random_g500: 472/500 = 94.4%
  vs Marnie b8f g300: 32/300 = 10.7%, baseline 11.7%, delta -1.0pp
```

Current recommendation:

- Keep `strategy_tempo` as the first useful proof that trajectory-conditioned BC
  is better than single-step weighting.
- Do not scale `fast_grimmsnarl_success` as-is; the 21-game subset is too sparse
  and did not improve the Marnie local matchup.
- Next improvement should generalize `mine_strategy_trajectories.py` across
  each archetype's weak matchups and auto-build tempo/choke/comeback subsets,
  then run small initialized pilots. Avoid returning to winner-only or
  filtered-only BC without whole-game conditions.

## 2026-08-07 Trajectory-Aware Initial BC Implementation

Trajectory information can now be introduced during the original BC training
run instead of only through post-hoc subset fine-tuning.

Changed files:

- `ptcg_rl/bc2/data.py`: `BCCorpus` accepts `trajectory_weights`,
  `trajectory_targets`, `trajectory_missing`, and `split_by_game`. When
  trajectory data is present, `bc2_train.py` defaults to splitting train/val by
  `episode_id:player_index` groups.
- `ptcg_rl/model.py`: optional `plan_dim` auxiliary head. Default is off, so
  old commands/checkpoints are unchanged.
- `ptcg_rl/bc2/losses.py`: optional BCE trajectory/plan auxiliary loss.
- `tools/bc2_train.py`: new CLI flags:
  - `--trajectory-csv PATH` repeatable, usually a `games.csv` from
    `tools/mine_strategy_trajectories.py`.
  - `--trajectory-weight CONDITION=WEIGHT`, e.g. `attack_by_4=1.25`,
    `attack_count>=5=1.15`, `outcome==win=1.10`.
  - `--trajectory-missing-policy default|drop`; `drop` trains only games found
    in the CSV.
  - `--trajectory-target COL_OR_CONDITION` plus
    `--trajectory-target-loss-weight W` enables the training-only plan head.
  - `--split-by-game` can be used without trajectory data for leakage-safe
    validation.
- `tools/bc2_accuracy.py` and `tools/bc2_failure_report.py` now ignore extra
  auxiliary plan tensors when loading policy checkpoints for diagnostics.
- `tools/bc2_train.py` strict init also tolerates extra `plan_*` tensors when
  loading a plan-head checkpoint into a normal no-plan model. `NumpyPolicy`
  already ignores extra checkpoint keys, so submission inference does not use
  the auxiliary head.

Example initial training command shape:

```bash
python3 tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0701_0804 \
  --archetype "Mega Lucario" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --deck-sig 43d6d8b0fce9 \
  --width 4 --device cuda:0 --cuda-memory-gb 8 \
  --epochs 24 --batch-size 4096 --lr 8e-5 \
  --win-weight 1.5 --loss-weight 0.4 --draw-weight 0.8 \
  --trajectory-csv logs/strategy_trajectories_20260807/lucario43d_vs_marnie_v11_0701_0804_v2/games.csv \
  --trajectory-weight attack_by_4=1.25 \
  --trajectory-weight primary_board_by_4=1.20 \
  --trajectory-weight outcome==win=1.10 \
  --trajectory-target attack_by_4 \
  --trajectory-target primary_board_by_4 \
  --trajectory-target outcome==win \
  --trajectory-target-loss-weight 0.05 \
  --save checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_traj_init_w4.npz
```

Important usage notes:

- Prefer binary or already normalized `--trajectory-target` columns. The plan
  auxiliary loss is BCE; do not feed raw counts such as `attack_count` unless
  using a comparison like `attack_count>=5`.
- Use modest `--trajectory-target-loss-weight` first (`0.03` to `0.10`). The
  plan head is a representation-shaping loss, not an inference-time planner.
- For targeted weak-matchup training, combine `--opponent-archetype` or
  `--opponent-deck-sig` with trajectory weights/targets. For general
  population training, use weights without filtering so the base distribution is
  not destroyed.
- `--trajectory-missing-policy drop` is closest to the previous strategy subset
  pilot; `default` is safer for initial full-corpus training.

### Trajectory BC Training Results So Far

Remote logs/checkpoints:

```text
logs/lucario43d_20260807/traj_init_eval/
logs/strategy_trajectories_20260807/lucario43d_vs_marnie_v11_0701_0804_traj_init/
checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_traj_init_w4_b3072.npz
checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_traj_weight_only_w4_b3072.npz
checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_traj_plan_only_w4_b3072.npz

logs/weak_matchup_traj_pilots_20260807/
checkpoints/weak_matchup_traj_pilots_20260807/
```

Lucario 43d vs Marnie trajectory-initial training:

```text
traj_combined random g500: 91.8%, vs Marnie shadow g300: 53/300 = 17.7%
traj_weight   random g500: 90.8%, vs Marnie shadow g300: 36/300 = 12.0%
traj_plan     random g500: 93.0%, vs Marnie shadow g300: 47/300 = 15.7%
strategy_tempo random g500: 94.2%, vs Marnie shadow g300: 55/300 = 18.3%
pure_refit random g500: 91.8%, vs Marnie shadow g300: 26/300 = 8.7%
```

Interpretation:

- The training-time plan head is safe mechanically, but did not beat the earlier
  `strategy_tempo` subset pilot in this matchup.
- Weight-only was the worst Lucario variant. Avoid using trajectory weights
  alone as the main technique.
- Combined trajectory weight + plan head was close to `strategy_tempo`, but
  still not better. It remains useful as a representation experiment, not a
  submit candidate.

Weak-matchup-only pilots were also run:

```text
marnie_b8f_vs_ogerpon     random 85.3%; vs og5899 6.3% vs base 8.7%; vs og697 10.7% vs base 20.0%
ogerpon697_vs_crustle     random 98.7%; vs crustle3 1.3% vs base 6.7%; vs crustleB 4.0% vs base 10.3%
dragapult_cc2_vs_crustle  random 58.7%; vs crustle3 3.7% vs base 5.3%; vs crustleB 2.3% vs base 5.0%
dragapult_cc2_vs_marnie   random 45.3%; vs marnie 7.7% vs base 9.0%
```

Conclusion:

- Do not continue pure weak-matchup-filtered BC as a standalone policy. It
  damages random stability and did not improve the focused matchup.
- The next direction should be mixed training: keep the full strong base
  distribution, then add weak-matchup successful trajectories as either
  downsampled aux corpus or small trajectory target loss. For very large decks
  like Marnie b8f, first build a balanced subset instead of scanning/training
  all 5M+ decisions every run.

## 2026-08-07 Architecture Experiments: Cross-Attn and Hierarchical Plan

Implemented and committed:

- `ptcg_rl/model.py`: pointer/cross-attn policies can now use
  `hierarchical_plan=True`. The plan path is a residual scorer conditioned on
  auxiliary trajectory targets, so old base scorer shapes remain loadable.
- `ptcg_rl/numpy_policy.py`: NumPy inference detects `plan_condition_fc.*` and
  applies the plan residual, so eval/submission uses hierarchical checkpoints.
- `tools/bc2_train.py`: added `--hierarchical-plan`, `--init-skip-prefix`, and
  `--reset-scorer`.
- `tools/build_trajectory_targets.py`: fast per-game trajectory-target builder.

Commits:

```text
df8f983 Add hierarchical plan-conditioned BC policy
0236922 Add lightweight trajectory target builder
6b0e460 Allow resetting BC scorer during init
8a9f8b4 Record hierarchical comparison runner
```

Remote validation:

```text
py_compile passed for model/numpy_policy/train/accuracy/failure.
NumPy smoke passed with:
PYTHONPATH=/home/jie/Do/0_PTCG/workspace:$PYTHONPATH
```

Important: non-interactive `ssh ks` often needs the `PYTHONPATH` above for
eval/submission scripts that instantiate `NumpyPolicy`.

Cross-attn full-date results:

```text
Ogerpon 5899 cross-full vs old w4 pool g80:
  avg_delta=-0.024, lost 11/17
Ogerpon 697 cross-full vs old w4 pool g80:
  avg_delta=-0.028, lost 12/17
Lucario 43d cross-full:
  random improved slightly, but strategy_tempo beat it 197-102-1.
  vs Marnie w4: 35/300 = 11.7%, worse than strategy_tempo 47/300 = 15.7%.
Marnie b8f cross-full:
  random 500/500, set F1 about 0.867.
  paired vs old Marnie w4 pool g80: avg_delta=-0.013, candidate=0.618,
  baseline=0.631, lost 8/17.
  It improved Ogerpon weak points only slightly:
  vs Ogerpon 5899 22.5% vs old 15.0%; vs Ogerpon 697 17.5% vs old 12.5%.
  The cost was broad regression, worst Cynthia -13.8pp, Marnie sig2 -8.8pp,
  Lopunny sig1 -8.8pp.
```

Conclusion: do not submit the cross-attn full-date checkpoints. The architecture
change is mechanically valid but did not improve broad ladder-pool strength.

Hierarchical plan comparison runner:

```text
script: /tmp/run_hier_plan_compare_20260807.sh
runner log: logs/hier_plan_compare_20260807.runner.log
checkpoints: checkpoints/hier_plan_compare_20260807/
targets: logs/hier_plan_fast_20260807/trajectory/
```

The first heavy trajectory miner attempt used `mine_strategy_trajectories.py`
and was stopped because Marnie b8f full-date mining consumed about 29GB RSS and
still had not written the trajectory CSV. Use `tools/build_trajectory_targets.py`
for these broad full-date target builds.

The first fast hierarchical run with `batch-size=4096` and
`--cuda-memory-gb 24` OOMed for all three pilot jobs. The active comparison
runner uses `batch-size=2048`, `--cuda-memory-gb 32`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Completed hierarchical init results:

```text
Ogerpon 5899 hier-init:
  accuracy exact about 0.915, set F1 0.985
  random g300: 299/300 = 99.7%
  paired vs old 5899 w4 pool g80:
    avg_delta=-0.002, weighted_delta=-0.002,
    candidate=0.597, baseline=0.599, lost 8/17
    worst: Festival sig3 -22.5pp, Alakazam sig1 -13.8pp
    gains: Lopunny sig3 +12.5pp, Festival sig1 +10.0pp, TR Mewtwo +7.5pp

Lucario 43d hier-init:
  accuracy exact about 0.704, set F1 0.792
  random g300: 286/300 = 95.3%
  RR g300:
    hier-init vs old strategy_tempo: 154-145-1 = 51.3%
    hier-init vs Marnie w4: 35/300 = 11.7%
    old strategy_tempo vs Marnie w4: 47/300 = 15.7%
```

Conclusion so far: hierarchical residual is also not a submit candidate yet.
Ogerpon is basically flat; Lucario wins the same-deck comparison by a small
amount but worsens the actual weak matchup. Keep the runner active to finish
Marnie init plus reset/scratch controls, because those controls answer whether
the old scorer is anchoring too hard.

Current active remote job at 2026-08-07 19:11:

```text
This older hierarchical-plan run has been superseded by the history-k and v12
experiments below. Re-check logs before reusing any hier-plan checkpoint.
```

Monitor with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -80 logs/hier_plan_compare_20260807.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_hier_plan_compare|bc2_train.py.*hierarchical-plan|bc2_accuracy.py.*hier_plan_compare|eval_bc.py.*hier_plan_compare"'
```

### True Past-K History Policy Implementation

Implemented after the hierarchical-plan result showed that game-level target
conditioning was not real sequential memory.

New behavior:

- `tools/bc2_train.py --history-k K` enables a real past-K own-decision history
  encoder. Default `K=0` keeps old behavior/checkpoints unchanged.
- Training history is built inside `ptcg_rl/bc2/data.py` from the previous K
  kept decisions in the same `episode_id:player_index` game. Rows are
  chronological within each game and history is right-aligned old-to-new, so
  current/future labels are not leaked.
- Each history event contains the first selected option's type/card/card2/attack,
  option context/select type, selected-count ratio, and a mask.
- `ptcg_rl/model.py` adds a small GRU history encoder for both pointer and
  cross-attn policies. Its output is fused back to the normal state embedding
  at fixed `hd`, preserving the scorer dimensions.
- `ptcg_rl/numpy_policy.py` now detects `history_pos_emb.weight` in checkpoints,
  keeps a rolling history buffer during inference, and applies the same GRU
  math in NumPy.
- `main.py`, `tools/eval_bc.py`, `tools/eval_round_robin.py`,
  `tools/trace_matchup_decisions.py`, and `tools/generate_rollout_bc.py` reset
  policy history at each new game to avoid cross-game leakage.
- `tools/bc2_accuracy.py` and `tools/bc2_failure_report.py` infer `history_k`
  from checkpoints and build matching corpus history.

Limitations of this first version:

- It records our own past decisions only. Opponent history is not included yet,
  because the current BC corpus is target-player rows by archetype; opponent
  action rows are not reliably available in the same sample stream.
- It stores one compact event per decision, using the first selected option for
  multi-select contexts. This is enough to test sequential conditioning, but a
  later version can add selected-set sketches.
- Rule overlays and random exploration can still make the rolling history differ
  from the model's pre-overlay action. Submission path has no rule overlay, so
  this is mainly a rollout-generation caveat.

Remote smoke validation:

```text
py_compile passed on local and ks for model/numpy_policy/bc2/train/eval files.
Torch-vs-NumPy history encode smoke passed on ks:
  PolicyValueNet max abs diff 4.8e-07
  CrossAttentionPolicyValueNet max abs diff 8.9e-07
Real corpus CPU smoke:
  command used Mega Lucario 43d, pointer width 0.25, history_k=4, 1 epoch.
  checkpoint saved to /tmp/bc2_history_smoke.npz
  bc2_accuracy smoke over 2048 samples completed.
  eval_bc smoke completed 2 games without inference errors.
Old-checkpoint init smoke:
  old Mega Lucario w4 -> --history-k 2 loaded 24 old tensors successfully.
  CPU training was manually stopped after init/first batch because width4 CPU
  was too slow; use GPU for real runs.
```

Example history training command shape:

```bash
python3 tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0701_0804 \
  --archetype "Mega Lucario" \
  --score-bands 1200+ 1100-1199 1000-1099 900-999 \
  --deck-sig 43d6d8b0fce9 \
  --arch pointer --width 4 --history-k 8 \
  --batch-size 2048 --epochs 8 --lr 5e-5 \
  --cuda-memory-gb 24 --device cuda:1 \
  --win-weight 1.5 --loss-weight 0.4 --draw-weight 0.8 \
  --first-action-weight 1.5 --option-weight 0.15 \
  --init checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_mega_lucario_sig2_43d6d8b0_v11all35_sigpure_top3_w4.npz \
  --save checkpoints/history_k_20260807/bc2_mega_lucario_43d6_hist8_w4init.npz
```

Evaluation commands do not need a new flag; they infer history automatically
from checkpoint weights. Always keep the `PYTHONPATH` prefix in non-interactive
remote runs.

### History-K Compare Results

The first incomplete sequential-memory experiment finished on `ks`:

```text
runner: logs/history_k_20260807.runner.log
train logs: logs/history_k_20260807/train/
accuracy logs: logs/history_k_20260807/accuracy/
random logs: logs/history_k_20260807/random/
checkpoints: checkpoints/history_k_20260807/
script: /tmp/run_history_k_compare_20260807.sh
```

Training setup:

```text
corpus=data/bc_corpus_banded_v11_0701_0804
score bands=1200+ 1100-1199 1000-1099 900-999
width=4, pointer, history_k=8, epochs=8, batch=2048, lr=5e-5
win/loss/draw=1.5/0.4/0.8
```

Results:

```text
lucario_43d_hist8_init:
  init=checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_strategy_tempo_w4.npz
  best_val=0.9656, exact=0.686, first=0.700, top3=0.929
  random_g300=285/300 = 95.0%

lucario_43d_hist8_scratch:
  best_val=1.1238, exact=0.614, first=0.629, top3=0.891
  random_g300=268/300 = 89.3%

marnie_b8f_hist8_init:
  init=checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_marnie_grimmsnarl_sig1_b8f251a4_v11all35_sigpure_top3_w4.npz
  best_val=0.5471, exact=0.870, first=0.874, top3=0.983
  random_g300=297/300 = 99.0%

marnie_b8f_hist8_scratch:
  best_val=0.5933, exact=0.858, first=0.863, top3=0.983
  random_g300=299/300 = 99.7%

ogerpon_5899_hist8_init:
  init=checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_teal_mask_ogerpon_sig3_5899c772_v11all35_sigpure_top3_w4.npz
  best_val=0.4087, exact=0.893, first=0.895, top3=0.989
  random_g300=300/300 = 100.0%

ogerpon_5899_hist8_scratch:
  best_val=0.5465, exact=0.843, first=0.846, top3=0.978
  random_g300=299/300 = 99.7%
```

Interpretation:

- Init clearly helps history-k convergence versus scratch for all three
  archetypes. Scratch is not a good default for this architecture.
- The old specialist w4 random references were already high:
  Marnie b8f `200/200`, Ogerpon 5899 `200/200`, Mega Lucario 43d sigpure
  `185/200`; Lucario strategy_tempo was `475/500`.
- History-k init did not clearly improve random quality over its init sources.
  Lucario is roughly tied with strategy_tempo, while Marnie/Ogerpon are slightly
  below their old perfect random audits.
- No RR/shadow baseline-delta was run for `history_k_20260807`. Do not treat
  these as submission candidates without a paired RR against current candidates
  and weakness pools.
- This result supports the v12 direction: single own-decision history is
  mechanically safe but too weak/incomplete. Use public log history and board
  deltas in v12 before drawing conclusions about sequence modeling.

## 2026-08-05 Search/Rules/Community Diagnostics

Local community logs are under `logs/kaggle_topics_20260805/`. The most relevant Kaggle discussion signals collected so far:

- "How consistent is imitation learning in this setting?" reports imitation learning leaderboard instability; one participant observed validation accuracy can improve while leaderboard score remains poorly correlated.
- "RL Beginner Question..." includes a report of policy accuracy rising from about 79% to 95%, again with leaderboard score not tightly correlated to accuracy.
- "Kiyota RL/MCTS sample rerun..." says the starter RL/MCTS approach was weak around 250-320, while another participant reports a silver-zone RL-only agent using top replays for value-function training.
- "Are top agents mostly heuristic..." includes a direct claim that expert BC plus rule guards can reach about 1000 points.
- "Submission Strategy" emphasizes collecting score-band-specific agent replays and building better local validation instead of overfitting live LB.
- "Differences Between the Official Pokemon TCG Rules and the Simulator Behavior" confirms the simulator exposes legal options and the agent only chooses indices; rule overlays should be strategic guards, not legality filters.

Non-training diagnostics were run on `ks` with `/tmp/run_v11_next_phase_nontrain_diag.sh`.

Outputs:

- Remote/local logs: `logs/eval_next_v11_search_rules_20260805/`
- Pulled locally after completion.

Key results:

- Current `main.py` keeps `USE_MCTS = False`. BC checkpoints only train the policy head; MCTS depends on a meaningful value head and is not currently a submission-path improvement.
- Conservative rule overlay hurt random stability:
  - Marnie pop v11all b8f251a4: `496/500 = 0.992`
  - Ogerpon pop v11all 5899c772: `490/500 = 0.980`
- Marnie vs Ogerpon focused RR:
  - Plain g120: vs `og5899` `0.075`, vs `og697` `0.250`
  - Conservative rules g120: vs `og5899` `0.075`, vs `og697` `0.158`
  - Per-entry MCTS g40: vs both Ogerpon entries `0.000`
- Ogerpon 5899 vs Crustle focused RR:
  - Plain g120: vs `crustle_pop` `0.0417`, vs `crustle_safe` `0.0417`
  - Conservative rules g120: both `0.025`
  - Per-entry MCTS g40: vs `crustle_pop` `0.050`, vs `crustle_safe` `0.000`
- Trace outcomes:
  - `trace_marnie_vs_og5899_g200`: `34-166`
  - `trace_marnie_vs_og697_g200`: `54-146`
  - `trace_ogerpon_vs_crustle_pop_g200`: `5-195`
  - `trace_ogerpon_vs_crustle_safe_g200`: `6-194`
- Top loss-vs-win trace priorities:
  - Ogerpon vs Crustle MAIN: loss states show higher `miss_attach_rate` and `miss_play_rate`.
  - Marnie vs Ogerpon 697 MAIN: loss states show higher `miss_ability_rate`.
  - Marnie vs Ogerpon 5899 MAIN: loss states show higher `miss_evolve_rate`.

Interpretation:

- Do not enable MCTS for pure BC submissions yet.
- Do not submit current conservative/aggressive rule overlay as-is.
- Rule support is still promising, but must be trace-driven and narrow: guard specific high-confidence strategic mistakes, not globally override all early END/ABILITY/EVOLVE/ATTACK cases.
- Next complex-scene improvement should be outcome-aware: compare loss-vs-win traces, identify exact card/option contexts, then create targeted data filters or narrow rule/rerank guards. The previous global complex weighting improved supervised metrics but did not improve matchup win rate.

Follow-up rich trace and card-weight experiments:

- Rich trace outputs:
  - `logs/eval_next_v11_rich_trace_20260805/`
  - `rich_outcome_gap_report.csv`
  - `rich_outcome_gap_report_v2.csv`
  - `failure_trace_priority.csv`
- Rich trace outcomes:
  - Marnie vs Ogerpon 5899: `17-183`
  - Marnie vs Ogerpon 697: `42-158`
  - Ogerpon 5899 vs Crustle pop: `10-190`
  - Ogerpon 5899 vs Crustle safe: `9-191`
- Marnie signals from `miss_card_summary`:
  - vs 5899: missed `Marnie's Morgrem` while choosing Basic Darkness Energy, Munkidori ability, Spikemuth Gym ability, Munkidori play, Rare Candy, etc.
  - vs 697: strongest aggregate miss was Basic Darkness Energy attach being skipped for Marnie basics/Spikemuth/Munkidori; missed Morgrem was present but less concentrated.
- Ogerpon vs Crustle rich trace is dominated by long losing loops: Teal Dance, repeated grass attach, draw/search, and attacks into Crustle. This looks more like structural matchup/wall handling than a simple single-action card-weight issue.

Targeted rules tested:

- Code modes added for local experiments only: `marnie_setup`, `ogerpon_attach`, `targeted`.
- Output path: `logs/eval_next_v11_targeted_rules_20260805/`
- Random:
  - Marnie `marnie_setup`: `499/500 = 0.998`
  - Ogerpon `ogerpon_attach`: `499/500 = 0.998`
- Focused RR g200:
  - Marnie plain vs `og5899`: `0.090`; `marnie_setup`: `0.140`
  - Marnie plain vs `og697`: `0.200`; `marnie_setup`: `0.165`
  - Ogerpon plain vs `crustle_pop`: `0.045`; `ogerpon_attach`: `0.030`
  - Ogerpon plain vs `crustle_safe`: `0.045`; `ogerpon_attach`: `0.050`
- Interpretation: these rule modes are not submission candidates. Keep only as diagnostic scaffolding; do not broaden them without a positive focused + broad validation.

Card-weight BC tested:

- Training logs: `logs/v11_complex_cardw_20260805/`
- Checkpoints: `checkpoints/complex_v11_cardw_20260805/` on `ks`
- Marnie/Ogerpon filtered corpus smoke test kept `111,658` decisions; Ogerpon 5899 vs Crustle kept only `1,553` decisions and split to `train=165`, so narrow Ogerpon-vs-Crustle BC was not trained.
- Marnie variants:
  - `bc2_marnie_b8f_vs_ogerpon_evolve_cardw_w2.npz`: MAIN 1.15, EVOLVE 1.7, card 647=2.3, 648=1.8. Best val `0.6668`.
  - `bc2_marnie_b8f_vs_ogerpon_evolve_attach_cardw_w2.npz`: MAIN 1.15, EVOLVE 1.5, ATTACH 1.25, card 647=2.0, 648=1.6, Basic Darkness Energy 7=1.5. Best val `0.7029`.
- Eval logs: `logs/eval_next_v11_cardw_20260805/`
- Random g500:
  - baseline: `500/500 = 1.000`
  - evolve: `496/500 = 0.992`
  - evolve_attach: `495/500 = 0.990`
- Baseline-delta vs two Ogerpons, g300:
  - evolve vs `og5899`: `+0.0167`; vs `og697`: `-0.0067`; avg `+0.005`
  - evolve_attach vs `og5899`: `-0.0100`; vs `og697`: `+0.0033`; avg `-0.003`
- Interpretation: simple card weighting is not enough. Do not scale this recipe or submit these checkpoints. The useful output is the tooling and negative result.

## 2026-08-06 Human Matchup Strategy Ingestion

User clarified the intended direction: when a weakness looks structural, the next step is not simply copying sparse winning replays. We should search human PTCG strategy sources, matchup stats, card text, tournament/deck guides, and community articles for how strong players navigate specific matchups, then translate those ideas into simulator-observable rules, teacher rollouts, or matchup-conditioned training data.

New files:

- `docs/11_human_matchup_strategy.md`
- `data/matchup_strategy_seeds_v1.csv`
- `data/matchup_strategy_seed_cards_v1.csv`
- `tools/plan_strategy_seed_jobs.py`
- `tools/summarize_strategy_seed_jobs.py`

Current seed table has 9 strategy hypotheses:

- Marnie Grimmsnarl vs Teal Mask Ogerpon setup.
- Marnie core Froslass/Munkidori/Grimmsnarl engine.
- Conditional caution around Froslass into Munkidori-style opponents.
- Crustle anti-ex wall plan.
- Crustle wall plus resource-pressure plan.
- Ogerpon vs Crustle early Dwebble punish.
- Ogerpon vs Crustle avoid futile ex attacks into established Crustle.
- Cynthia vs Crustle Spiritomb active/counter plan.
- Dragapult setup coherence into Marnie/Crustle.

Use these only as hypotheses. Required path for each seed:

1. Confirm the card/deck exists in the exact deck CSV and local `data/EN_Card_Data.csv`.
2. Run rich trace and count where the desired card/action was available but not selected.
3. Implement the narrowest possible `--rules-entry`/rerank probe or teacher policy.
4. Validate random first, then focused weakness delta, then broad balanced-shadow delta.
5. Only after focused and broad validation pass, generate teacher-rollout success data and distill into mixed BC.

Do not submit or scale a human-strategy rule solely because the explanation sounds correct. Previous rule/card-weight experiments showed that plausible global nudges can improve narrow metrics while hurting broad play.

## 2026-08-06 Strategy Optimization Run

Conclusion before this run:

- The previous `weakup_v11_20260805` broad matchup weighting was not enough. It gave small broad gains for Alakazam and Cynthia, almost no useful gain for Marnie, and hurt Dragapult/Ogerpon.
- Seed rule probes were also negative or too weak:
  - Ogerpon no-futile vs Crustle: rule delta `-0.010` at g200.
  - Cynthia Spiritomb vs Crustle: rule delta `-0.030` at g200.
  - Marnie setup vs Ogerpon: only small positive in narrow probe and not broad-safe.
- Therefore the next test is not another global rule. It is narrow matchup-conditioned BC with stronger complex-scene weighting plus a `winner-only` branch where the corpus has enough success games.

Remote run started on `ks` at 2026-08-06 09:17 Asia/Shanghai:

```text
script: /tmp/run_strategy_opt_20260806.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
runner log: logs/strategy_opt_20260806/runner.log
train logs: logs/strategy_opt_20260806/train_*.log
checkpoints: checkpoints/strategy_opt_20260806/
candidate manifest: logs/strategy_opt_20260806/candidate_manifest_strategy_opt.csv
random eval: logs/strategy_opt_20260806/strategy_opt_random_g500.csv
delta summary: logs/strategy_opt_20260806/strategy_opt_delta_summary.csv
```

The script launches up to 4 training jobs concurrently, one per GPU, with
`MEM_GB=24`, `TRAIN_EPOCHS=8`, `BATCH_SIZE=1024`, `WORKERS=32`,
`RANDOM_GAMES=500`, and `EVAL_GAMES=160`.

Experiments in the run:

- `marnie_b8f_vs_ogerpon_filter_complex`
- `marnie_b8f_vs_ogerpon_winonly_complex`
- `dragapult_cc2_vs_marnie_filter_complex`
- `dragapult_cc2_vs_marnie_winonly_complex`
- `dragapult_cc2_vs_crustle_filter_complex`
- `dragapult_cc2_vs_crustle_winonly_complex`
- `cynthia_52f_vs_crustle_filter_complex`
- `cynthia_52f_vs_crustle_winonly_complex`
- `ogerpon_2a507_vs_crustle_winonly_complex`
- `ogerpon_697_vs_crustle_crosssig_winonly_complex`
- `alakazam_7f_vs_marnie_winonly_complex`
- `alakazam_7f_vs_trm_winonly_complex`

Common training knobs:

- Init from v11 population checkpoint, partial load, `state=80`, `option=64`.
- `first_action_weight=1.8`, `option_weight=0.35`.
- `multi_select_weight=2.0`, `set_loss_weight=0.25`, `set_loss_negative_weight=0.15`.
- Context upweights: `MAIN=1.15`, `ATTACH_FROM=1.8`, `ATTACH_TO=1.4`,
  `DAMAGE=1.5`, `SKILL_ORDER=2.0`, `ATTACK=1.35`.
- `filter_complex` keeps the weak matchup and uses strong win/loss weights.
- `winonly_complex` keeps only winning games for that weak matchup.

Monitor with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/strategy_opt_20260806/runner.log'
ssh ks 'pgrep -af "run_strategy_opt_20260806|bc2_train.py|eval_baseline_delta.py|eval_manifest_random.py"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/strategy_opt_20260806/strategy_opt_random_g500.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && cat logs/strategy_opt_20260806/strategy_opt_delta_summary.csv'
```

If a job fails because of memory pressure, do not restart the whole run. Rerun
only the failed train command from its `train_*.log` with `MEM_GB=16` or a
smaller `BATCH_SIZE`. If a `winner-only` job fails from too few samples, record
that as evidence that the matchup needs teacher-generated success data rather
than filtered BC.

Results of the first run:

- All 12 training jobs completed and wrote checkpoints under
  `checkpoints/strategy_opt_20260806/`.
- Random g500:
  - Stable: Marnie filter/winonly `0.998/1.000`; Ogerpon 697 crosssig-name
    `0.986`; Alakazam Marnie/TRM winonly `0.974/0.972`; Cynthia filter/winonly
    `0.958/0.948`.
  - Unstable: Dragapult Marnie filter/winonly `0.800/0.788`; Ogerpon 2a
    winonly `0.780`; Dragapult Crustle `0.900/0.908` is borderline.
- Focused baseline-delta g160:
  - `alakazam_marnie_winonly`: avg delta `+0.029`, but hurts TRM by `-0.031`.
  - `ogerpon_2a_winonly`: avg delta `+0.025` vs Crustle, but random collapsed
    to `0.780`, so not deployable.
  - `drag_crustle_filter`: avg delta `+0.0025`, not meaningful and lost 3/5.
  - Marnie, Dragapult-Marnie, Cynthia, Ogerpon-697 variants were negative.
- Interpretation: narrow filtered/winner-only BC can move a local target a few
  points, but often breaks random or nearby matchups. It is not enough for the
  structural weak matchups. Use it as a diagnostic, not a submission recipe.

New reusable slice audit tool:

```bash
python3 tools/audit_matchup_slices.py \
  --corpus data/bc_corpus_banded_v11_0724_0804 \
  --archetype "Marnie Grimmsnarl" \
  --opponent-archetype "Teal Mask Ogerpon" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --group-by archetype deck_sig team_name opponent_archetype opponent_deck_sig opponent_team_name \
  --min-games 20 \
  --min-wins 5 \
  --top 80 \
  --out-csv logs/strategy_opt_20260806/slices/marnie_vs_ogerpon.csv
```

Slice audit outputs:

- `logs/strategy_opt_20260806/slices/*.csv`
- Marnie vs Ogerpon: the top successful slices are all `b8f251a476e7` into
  Ogerpon `2a5072194fdf`, e.g. Dominic Peel `18/30=0.600` and flg
  `28/50=0.560`. There is no comparable successful slice for 5899/697.
- Ogerpon vs Crustle: only two usable slices, both `2a5072194fdf` James Cox &
  Henry Chao, against `47756cdfd20f` and `7ee600c6f769`.
- Dragapult vs Crustle and Cynthia vs Crustle had no slice meeting
  `min_games=20,min_wins=5`.
- Dragapult vs Marnie had two `cc2e995b5ad0` flg slices with `18/28=0.643`
  and `18/31=0.581`.
- Alakazam vs Marnie had several `7f9a538936e3` slices above 0.5; Alakazam vs
  Team Rocket Mewtwo had only one modest slice `9/22=0.409`.

Wave2 team-slice low-LR run:

```text
script: /tmp/run_strategy_opt_wave2_20260806.sh
logs: logs/strategy_opt_wave2_20260806/
checkpoints: checkpoints/strategy_opt_wave2_20260806/
```

Wave2 used high-success team/opponent slices, lower LR `6e-6`, 4 epochs, and
lighter complex weights. All 8 jobs completed.

Wave2 results:

- Random g500:
  - Marnie Dominic/flg vs Ogerpon 2a slice: `0.998/0.994`.
  - Alakazam Majkel filter/winonly: both `0.986`.
  - Dragapult flg filter/winonly: `0.886/0.902`.
  - Ogerpon 2a filter/winonly: `0.762/0.794`.
- Focused delta g200:
  - `marnie_dominic`: avg `+0.0117`; improves Ogerpon 2a `+0.020` and 697
    `+0.040`, but hurts 5899 `-0.025`.
  - `marnie_flg`: avg `-0.0067`; not useful.
  - `alak_majkel_*`: improves pop Marnie by `+0.060` to `+0.070`, but hurts
    Marnie shadow and TRM, so only a narrow diagnostic.
  - `drag_flg_*` and `ogerpon_2a_*` are negative; do not continue this recipe.

Updated interpretation:

- Existing replay data contains local successful lines for some slices, but
  they do not transfer reliably across nearby opponent signatures.
- The Crustle-wall structural weakness is not solved by filtered BC; Ogerpon
  and Dragapult either collapse random or lose focused delta. Move these to
  teacher/rule construction.
- For Marnie vs Ogerpon, data exists for beating `2a507...`, but not for live
  hard Ogerpon variants like 5899. Need trace-guided teacher construction or
  strategy/rule seeds for 5899/697 specifically.
- For Alakazam vs Marnie, matchup-conditioned specialist training has some
  promise, but needs a deployment mechanism or broader mixture so it does not
  hurt Team Rocket Mewtwo.

The seed-card mapping table is deliberately separate from the strategy table.
Use it to validate that a human guide's named cards are present in the exact
Kaggle deck before turning the idea into a hard rule or generated-data teacher.
Basic Energy names may come from the simulator runtime rather than
`data/EN_Card_Data.csv`, so validate those from traces/deck IDs instead of using
card DB presence alone.

Seed-driven planner:

```bash
python3 tools/plan_strategy_seed_jobs.py \
  --candidate-manifest logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv \
  --opponent-manifest logs/eval_v11_0724_0804/shadow_pools_20260805/mixed_shadow_popfallback_environment_balanced.csv \
  --out-dir logs/strategy_seed_jobs_20260806 \
  --limit-candidates 1 \
  --limit-opponents 1 \
  --games 120 \
  --rule-games 80 \
  --workers 16 \
  --progress-every 20
```

It writes `strategy_seed_tasks.csv`, `strategy_seed_skipped.csv`,
`teacher_specs.jsonl`, and `run_strategy_seed_traces.sh`. The generated shell
script runs trace jobs, per-task gap reports, aggregate gap report, and available
rule probes. It does not implement new teacher policies; `teacher_specs.jsonl`
is the handoff contract for that next layer.

Local smoke at 2026-08-06:

- `python3 -m py_compile tools/plan_strategy_seed_jobs.py`: passed.
- Single Marnie-vs-Ogerpon seed dry-run wrote 1 task with
  `teacher_status=rule_probe_available`.
- Full seed dry-run against local v11 candidate/environment manifests wrote 24
  tasks and 1 skipped seed (`marnie_core_engine`, because `ALL` is not expanded
  unless `--expand-all` is supplied): `rule_probe_available=1`,
  `needs_teacher_policy=9`, `needs_rule_implementation=12`,
  `trace_then_matchup_bc=2`.
- `data/matchup_strategy_seeds_v1.csv` had one malformed Dragapult row because
  a natural-language field contained commas without CSV quoting. This was fixed,
  and the planner now hard-fails malformed CSV rows.

Remote job started on `ks` at 2026-08-06 02:10 Asia/Shanghai:

```text
PID parent: 2795955
PID runner: 2795956
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
out_dir: logs/strategy_seed_jobs_20260806
log: logs/strategy_seed_jobs_20260806/run_strategy_seed_traces.log
script: logs/strategy_seed_jobs_20260806/run_strategy_seed_traces.sh
```

Remote plan generation wrote 24 tasks and 1 skipped seed; all 24 tasks had
`card_check=ok` against remote deck CSVs. The skipped seed is
`marnie_core_engine` because `opponent_archetype=ALL` is not expanded unless
`--expand-all` is supplied.

Initial live result from the first generated task:

- Marnie vs Ogerpon 5899 trace completed.
- `marnie_setup` rule probe over 80 games: plain Marnie vs Ogerpon `6/80 =
  0.075`; rule Marnie vs Ogerpon `8/80 = 0.100`. This is only a tiny focused
  improvement and still indicates a structural weakness.

The 24-task run completed. Summary files on `ks`:

```text
logs/strategy_seed_jobs_20260806/strategy_seed_summary_matchups.csv
logs/strategy_seed_jobs_20260806/strategy_seed_summary_seeds.csv
logs/strategy_seed_jobs_20260806/strategy_seed_summary.md
logs/strategy_seed_jobs_20260806/strategy_seed_gap_report.csv
```

Seed-level summary:

- `ogerpon_vs_crustle_dwebble_punish`: `4-116`, WR `0.033`; recommendation
  `prioritize_teacher_rollout_success_data`.
- `ogerpon_vs_crustle_no_futile_attack`: `5-115`, WR `0.042`; initial narrow
  rule idea needs proof and is not enough.
- `cynthia_vs_crustle_spiritomb`: `8-112`, WR `0.067`; Spiritomb idea needs
  deeper teacher/success construction or much narrower triggers.
- `dragapult_setup_vs_marnie_crustle`: vs Crustle `10-110`, vs Marnie `15-105`;
  recommendation `prioritize_matchup_conditioned_bc`.
- `marnie_vs_ogerpon_setup`: `11-109`, WR `0.092`; `marnie_setup` rule delta
  was only `+0.025`, so do not scale the current rule.
- `crustle_wall_plan` / `crustle_resource_pressure`: average WR around
  `0.71-0.73`, but Crustle is weak into Alakazam/Marnie/Lopunny in this probe.

Two additional experimental rule modes were added for probing only:

- `ogerpon_no_futile_crustle`
- `cynthia_spiritomb_crustle`

Rule-probe-only run on `ks`:

```text
logs/strategy_seed_rule_probes_20260806/
```

Results:

- Ogerpon no-futile vs Crustle, 200 games: plain `10/200 = 0.050`; rule
  `8/200 = 0.040`; delta `-0.010`.
- Cynthia Spiritomb vs Crustle, 200 games: plain `30/200 = 0.150`; rule
  `24/200 = 0.120`; delta `-0.030`.

Interpretation: do not scale these rules. They are useful negative probes. The
Crustle-wall weaknesses need generated success data / teacher rollout or
matchup-conditioned BC, not broader single-action guards.

Trace fields now include `my_bench_cards`, `my_bench_card_names`,
`opp_bench_cards`, and `opp_bench_card_names`. Use this for the next
Dwebble-punish analysis, because the earlier trace only had active cards and
could not reliably detect whether Dwebble was still punishable on the bench.

## Top-K Deck-Signature Training Axis

User raised an important training-selection issue: many archetypes may be better
served by exact `top1`, `top2`, or `top3` deck-signature training rather than
archetype-level mixed training, even when local random or broad RR looks close.
Treat this as an explicit ablation axis for every serious archetype, not just
as an Ogerpon special case.

Evidence already in the project:

- Ogerpon `v10_fixed_top2` recovered the v7-level Kaggle result, with final
  around `951.8` and user-observed peak around `1040`.
- Ogerpon `v10_fixed_top3` dropped to around `720.5`; adding one more high-sample
  signature diluted the policy rather than improving it.
- README already warns that all-mixed Ogerpon can have good offline accuracy
  while losing game-plan sharpness.

Technical interpretation:

- `bc2_train.py` can filter by `--deck-sig`, but the submitted policy does not
  receive an explicit own `deck_sig` feature at inference time. If two signatures
  under the same archetype require different sequencing in visually similar
  states, mixed BC averages incompatible teachers.
- Top-k training reduces teacher entropy and preserves a sharper game plan.
  It can beat broader mix on Kaggle even if local random/RR differences are
  small, because Kaggle opponents punish plan dilution more than legal random.
- `mix` should be used only when the archetype's top signatures share a coherent
  action policy, or as a robustness/shadow-opponent recipe rather than a
  submission recipe.

Required future comparison per archetype:

1. Train/evaluate `top1`, `top2`, `top3`, optionally `top5`, and controlled
   win-weighted `mix`.
2. Pair each checkpoint with a compatible submission deck; do not let registry
   silently map a top-k checkpoint to an unrelated top1 deck.
3. Compare random, focused weakness delta, balanced-shadow RR, and first-step
   ranking entropy/top1-vs-top3 gaps.
4. Use Kaggle live probes sparingly to calibrate which local metric predicts
   top-k sharpness for that archetype.

## 2026-08-06 Aggressive Search / Teacher Pivot

User explicitly requested stopping filtered BC / winner-only BC as the main
optimization path. Treat the 2026-08-06 strategy-opt runs as negative evidence:
small replay filtering, card weights, and broad rule nudges are not enough for
the structural weak matchups. The new direction is to use compute to generate
or discover successful behavior, then distill or fine-tune from it.

New tooling:

- `tools/generate_rollout_bc.py`: plays local candidate-vs-opponent games with
  mixed actor modes and writes candidate-side decisions from selected outcomes
  as normal BC corpus `.npz` files. Output layout is
  `<out_root>/<Archetype>/<band>/*.npz`, so `tools/bc2_train.py` can read it.
  Actor modes include `greedy`, `sample@T`, `random`, `mcts`, and
  `+rules:<RULE_MODE>`.
- `tools/bc2_train.py`: added `--aux-corpus`, `--aux-score-bands`,
  `--aux-archetype`, and `--aux-repeat`. This mixes generated rollout corpus
  into normal BC training without using `--winner-only`.
- `tools/summarize_rollout_bc.py`: summarizes rollout worker CSVs and generated
  corpus `.npz` files, including rows, win rate, actor mode distribution,
  opponent distribution, and first action type distribution.
- `tools/generate_rollout_bc.py` now also supports
  `--worker-progress-every` and `--flush-every-games`. Use these for large
  rollout jobs so progress appears before a worker finishes and partial `.npz`
  chunks survive interruptions.

Remote smoke tests passed:

- `generate_rollout_bc.py` Marnie vs Ogerpon, `--keep-outcomes win`, 8 games:
  no wins, no rows, no engine errors. This confirms the known weakness is hard.
- `generate_rollout_bc.py` Marnie vs Ogerpon, `--keep-outcomes all`, 4 games:
  wrote 289 decisions under `data/generated_rollout_bc_smoke/...`.
- Existing `BCCorpus` loaded the smoke corpus successfully:
  `raw=289 kept=289`, state feature `(80,)`, option feature `(N,64)`.
- `bc2_train.py --aux-corpus` smoke trained 1 CPU epoch from the generated
  smoke corpus and wrote `/tmp/bc2_smoke_aux_marnie.npz`.

First remote rollout-search batch started at 2026-08-06 10:00 Asia/Shanghai:

```text
runner pid reported: 2915802
script: /tmp/run_rollout_search_20260806.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
logs: logs/rollout_search_20260806/
output corpus: data/generated_rollout_bc_rollout_search_20260806/
games/job: 1600
workers/job: 8
jobs: 6, total rollout workers about 48
```

This first batch used an `mcts` actor. It was stopped at about 11:55 because
after nearly two hours it had produced zero `.npz` files and no worker
completion. Treat this as a negative infrastructure result: current
`NumpyPolicy.select_mcts()` is too slow/sticky for large rollout-data
generation. Do not include `mcts` in rollout generation batches unless a small
isolated MCTS smoke test proves completion speed first.

Replacement rollout batch started at 2026-08-06 12:00 Asia/Shanghai:

```text
runner pid reported: 3051748
script: /tmp/run_rollout_search_nomcts_20260806.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
logs: logs/rollout_search_nomcts_20260806/
output corpus: data/generated_rollout_bc_rollout_search_nomcts_20260806/
games/job: 1200
workers/job: 8
max_turns: 500
progress: --worker-progress-every 25
flush: --flush-every-games 50
jobs: same six matchup pools, with high-temperature/sample/random/rule actors and no MCTS
```

This 1200-game no-MCTS batch was also stopped quickly because early worker
progress showed slow policy-vs-policy games and no output yet. It was replaced
by a quick signal batch.

Active quick no-MCTS rollout batch started at 2026-08-06 12:04 Asia/Shanghai:

```text
runner pid reported: 3059297
script: /tmp/run_rollout_search_nomcts_quick_20260806.sh
repo: /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804
logs: logs/rollout_search_nomcts_quick_20260806/
output corpus: data/generated_rollout_bc_rollout_search_nomcts_quick_20260806/
games/job: 240
workers/job: 8
max_turns: 350
progress: --worker-progress-every 5
flush: --flush-every-games 10
jobs: same six matchup pools, with high-temperature/sample/random/rule actors and no MCTS
```

Initial quick progress showed many workers at `1/30` games with no wins yet and
rates around `0.01-0.03 games/s` including initialization overhead. Let it reach
at least `10/30` per worker before judging; the first partial `.npz` can only
appear after a worker has both a kept win and hits the 10-game flush boundary.

Observed quick partial result at about 12:36:

- `npz=8`, then watcher saw `npz=11`.
- Marnie vs Ogerpon produced several wins by `5/30` worker progress, but all
  observed Marnie winning episodes in the first flushed chunks were against
  `ogerpon2a`, not 5899/697. This means generated data may help Ogerpon 2a
  first, but still may not solve the harder live-style 5899/697 variants.
- Ogerpon 2a vs Crustle produced wins against at least `crustle_b141` and
  `crustle477`.
- Cynthia 52f vs Crustle produced wins against `crustle_b141`, `crustle477`,
  and `crustle3cd`.
- Dragapult and Alakazam were still zero-win at `5/30` progress in the checked
  logs.

Updated partial corpus summary at about 12:43:

- Marnie vs Ogerpon: `1802` generated decision rows from `17` winning episodes,
  all against `ogerpon2a`. Actor mix was mostly
  `sample@1.7+rules:marnie_setup`, `sample@2.2`, `sample@1.5`, and
  `sample@3.2`.
- Cynthia 52f vs Crustle: `659` rows from `6` wins across `crustle477`,
  `crustle3cd`, and `crustle_b141`.
- Ogerpon 2a vs Crustle: `205` rows from `2` wins against `crustle_b141` and
  `crustle477`.
- Alakazam 7f vs Marnie/TRM: `142` rows from `1` win against `trmf1`.
- Dragapult still had no generated wins in the checked partial corpus.

Post-rollout watcher started at 2026-08-06 12:37 Asia/Shanghai:

```text
runner pid reported: 3098447
script: /tmp/run_rollout_distill_after_quick_20260806.sh
logs: logs/rollout_distill_20260806/
checkpoints: checkpoints/rollout_distill_20260806/
behavior: waits for quick rollout to finish, summarizes generated corpus, then trains/evaluates aux-corpus distill models for Marnie, Ogerpon 2a, and Cynthia if their generated .npz files exist
```

The watcher intentionally waits before training so it does not read partially
written `.npz` files. Monitor with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/rollout_distill_20260806/runner.log'
```

Jobs:

- `marnie_b8f_vs_ogerpon_pool`
- `ogerpon2a_vs_crustle_pool`
- `dragapult_cc2_vs_crustle_pool`
- `dragapult_cc2_vs_marnie_pool`
- `alakazam7f_vs_marnie_trm_pool`
- `cynthia52_vs_crustle_pool`

Monitor with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/rollout_search_20260806/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "generate_rollout_bc.py"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && python3 tools/summarize_rollout_bc.py --summary-glob "logs/rollout_search_20260806/*_pool.csv" --corpus data/generated_rollout_bc_rollout_search_20260806 --out-csv logs/rollout_search_20260806/rollout_summary.csv'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/rollout_search_nomcts_20260806/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && python3 tools/summarize_rollout_bc.py --summary-glob "logs/rollout_search_nomcts_20260806/*_pool.csv" --corpus data/generated_rollout_bc_rollout_search_nomcts_20260806 --out-csv logs/rollout_search_nomcts_20260806/rollout_summary.csv'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/rollout_search_nomcts_quick_20260806/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && python3 tools/summarize_rollout_bc.py --summary-glob "logs/rollout_search_nomcts_quick_20260806/*_pool.csv" --corpus data/generated_rollout_bc_rollout_search_nomcts_quick_20260806 --out-csv logs/rollout_search_nomcts_quick_20260806/rollout_summary.csv'
```

First PPO pilot batch started at 2026-08-06 10:05 Asia/Shanghai:

```text
runner pid reported: 2924167
script: /tmp/run_ppo_pilots_20260806.sh
logs: logs/rl_pilot_20260806/
checkpoints: checkpoints/rl_pilot_20260806/
jobs: 4, one per GPU, CUDA memory cap 8GB
anchor corpus: data/bc_corpus_banded_v11_0724_0804
anchor bands: 1200+ 1100-1199 1000-1099 900-999
```

PPO jobs:

- `marnie_b8f_vs_ogerpon_ppo`
- `ogerpon2a_vs_crustle_ppo`
- `dragapult_cc2_vs_crustle_marnie_ppo`
- `alakazam7f_vs_marnie_trm_ppo`

This full-anchor PPO batch was stopped at about 11:55. All four processes had
run for nearly two hours with zero-byte logs, likely while indexing the full
`data/bc_corpus_banded_v11_0724_0804` anchor in four concurrent processes.
Do not rerun this exact configuration. For future PPO pilots, either:

- run without anchor for a very short reward-signal smoke test;
- use a much lighter anchor band subset and set progress logging; or
- wait for generated rollout success data and use that as a compact anchor /
  distillation source.

Important interpretation:

- Generated rollout success data is the first-class path now. If a weak matchup
  produces wins, train with `--aux-corpus data/generated_rollout_bc_rollout_search_20260806 --aux-score-bands win_search --aux-repeat N` from the strong v11
  init, then validate random, focused delta, and balanced RR.
- If a weak matchup still produces zero wins after high-temperature/rule/MCTS
  rollout, the existing BC policy manifold probably lacks the counterplay. Move
  to a more explicit teacher/search policy or simulator-specific strategic rule
  construction rather than more replay filtering.
- Do not submit generated-data or PPO checkpoints without the usual random and
  RR gates. The purpose of these jobs is to build a better local improvement
  loop, not immediate leaderboard probing.

## 2026-08-05 Specialist BC Wave 1

Started on `ks` at `2026-08-05 22:25 Asia/Shanghai`.

Runner:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/v11_specialists_20260805/wave1_runner.log'
```

Remote PID observed after startup:

- parent command: `2041992`
- runner shell: `2041993`
- first training jobs:
  - `marnie_vs_ogerpon_wl`: PID `2042003`, GPU `0`
  - `marnie_vs_ogerpon_winner_only`: PID `2042009`, GPU `2`

The initial `nohup` launch failed once because `logs/v11_specialists_20260805/` did not exist before shell redirection. Directories were created and the second launch succeeded. No old training process was left behind.

Script:

- local construction path: `/tmp/run_v11_specialist_bc_wave1.sh`
- remote execution path: `/tmp/run_v11_specialist_bc_wave1.sh`
- validation helper: `/tmp/validate_runner_paths.py`
- all referenced baseline checkpoints and deck CSVs passed path validation (`checked=36 missing=0`).

Resource choices:

- Only GPUs `0` and `2` are used by this runner.
- `MAX_PARALLEL=2`
- `CUDA_MEMORY_GB=24`
- `WORKERS=32`
- random audit games: `300`
- focused baseline-delta games: `200`
- max turns: `700`

Wave 1 intent:

- Test whether matchup-conditioned BC with outcome reweighting or winner-only filtering repairs weak RR matchups better than simple card weighting.
- Do not treat these checkpoints as submission candidates until their random and focused delta audits complete.

Training outputs:

```text
checkpoints/specialist_v11_20260805/
logs/v11_specialists_20260805/
```

Evaluation outputs after training finishes:

```text
logs/eval_next_v11_specialists_20260805/specialist_wave1_manifest.csv
logs/eval_next_v11_specialists_20260805/specialist_wave1_random_g300.csv
logs/eval_next_v11_specialists_20260805/delta_*_g200.csv
logs/eval_next_v11_specialists_20260805/specialist_wave1_summary.txt
```

Wave 1 jobs:

- `marnie_vs_ogerpon_wl`: Marnie Grimmsnarl vs Teal Mask Ogerpon, init `pop_v11all_marnie_grimmsnarl_b8f251a4_1`, `win/loss/draw=2.2/0.12/0.8`.
- `marnie_vs_ogerpon_winner_only`: same matchup, winner-only.
- `ogerpon_vs_crustle_wl`: Teal Mask Ogerpon vs Crustle Wall, init `pop_v11all_teal_mask_ogerpon_5899c772_2`, `win/loss/draw=3.0/0.05/0.8`.
- `ogerpon_vs_crustle_winner_only`: same matchup, winner-only.
- `dragapult_vs_marnie_wl`: Dragapult vs Marnie Grimmsnarl, init `pop_v11all_dragapult_cc2e995b_2`, `win/loss/draw=2.0/0.15/0.8`.
- `dragapult_vs_crustle_wl`: Dragapult vs Crustle Wall, init `pop_v11all_dragapult_cc2e995b_2`, `win/loss/draw=2.4/0.10/0.8`.
- `alakazam_vs_trm_wl`: Alakazam vs Team Rocket Mewtwo, init `pop_v11all_alakazam_7f9a5389_1`, `win/loss/draw=2.4/0.10/0.8`.
- `lopunny_vs_cynthia_wl`: Mega Lopunny vs Cynthia Garchomp, init `pop_v11all_mega_lopunny_f1445356_1`, `win/loss/draw=3.0/0.08/0.8`.
- `cynthia_vs_crustle_wl`: Cynthia Garchomp vs Crustle Wall, init `pop_v11all_cynthia_garchomp_52f46739_1`, `win/loss/draw=2.5/0.10/0.8`.
- `trm_vs_ogerpon_wl`: Team Rocket Mewtwo vs Teal Mask Ogerpon, init `pop_v11all_team_rocket_mewtwo_f0bac971_1`, `win/loss/draw=2.2/0.12/0.8`.

Weak-pair/corpus planning files:

```text
logs/eval_next_v11_specialists_20260805/rr_weak_archetype_pairs.csv
logs/eval_next_v11_specialists_20260805/weak_pair_corpus_plan.csv
logs/eval_next_v11_specialists_20260805/corpus_matchup_counts_by_deck.csv
```

Key sampled weak-pair counts from `weak_pair_corpus_plan.csv`:

- Teal Mask Ogerpon vs Crustle Wall: RR mean `0.0451`, corpus decisions `45,409`, win decisions `6,427`, loss decisions `38,982`.
- Marnie Grimmsnarl vs Teal Mask Ogerpon: RR mean `0.1992`, corpus decisions `246,069`, win decisions `75,030`, loss decisions `170,949`.
- Dragapult vs Marnie Grimmsnarl: RR mean `0.1021`, corpus decisions `208,622`, win decisions `116,480`, loss decisions `92,032`.
- Dragapult vs Crustle Wall: RR mean `0.0780`, corpus decisions `35,905`, win decisions `12,554`, loss decisions `23,351`.
- Alakazam vs Team Rocket Mewtwo: RR mean `0.3093`, corpus decisions `99,223`, win decisions `26,137`, loss decisions `73,086`.

Monitor commands:

```bash
ssh ks 'pgrep -af "run_v11_specialist_bc_wave1|bc2_train.py|eval_manifest_random.py|eval_baseline_delta.py" || true'
ssh ks 'tail -80 /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/v11_specialists_20260805/wave1_runner.log'
ssh ks 'ls -lh /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/checkpoints/specialist_v11_20260805'
```

## 2026-08-06 Rule/Success FT Wave 2

Small code updates:

- `tools/compare_bc_subsets.py` now loads card names from
  `data/EN_Card_Data.csv` by default, with the old small mapping as fallback.
  This fixed unknown ids in success/loss reports, e.g. Cynthia `387` now maps to
  `Cynthia's Spiritomb`.
- `ptcg_rl/rule_overlay.py` has a new diagnostic-only mode
  `primary_active`. It only fires in single-select `SWITCH` / `TO_ACTIVE`
  contexts and chooses a deck-plan primary attacker when available. It is not
  enabled by default and is not submission-ready.

Crustle submission command recommended from previously live-probed models:

```bash
mkdir -p submissions
python3 tools/package_submission.py \
  --policy checkpoints/pop/bc2_crustle_wall_v10pop_all0803_set_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/b141ae295739_crustle_wall_where_is_my_orbit.csv \
  --out submissions/crustle_wall_v10pop_b141ae29.tar.gz
```

Reason: this is the v10pop Crustle checkpoint already used in a Kaggle probe
(`bc: crustle_wall_v10pop`, observed around 880 and earlier around 920). The
v11 Crustle candidates are useful for internal testing but have weaker or less
live-validated evidence.

Remote success/loss compare outputs with full card names:

```text
logs/rule_success_20260806/
```

Key compare signals:

- Marnie `b8f` vs Ogerpon: wins switch/select `Marnie's Grimmsnarl ex` more
  often and are less dominated by Ogerpon target contexts.
- Ogerpon vs Crustle: successful rows strongly prefer Mega Kangaskhan ex as
  active over Ogerpon in some setup/active contexts. This points to structural
  anti-wall play, not just more Teal Dance/attach.
- Dragapult vs Marnie: wins choose Dragapult ex for `TO_ACTIVE`/`SWITCH` more
  often and use Drakloak in `ATTACH_FROM`.
- Dragapult vs Crustle: success uses more Dreepy bench/setup and less
  Crustle-target fixation.
- Cynthia vs Crustle: wins switch into `Cynthia's Spiritomb` more and avoid
  over-switching into `Cynthia's Garchomp ex`.
- Lucario vs Marnie: wins put Mega Lucario ex active more often.
- TR Mewtwo vs Ogerpon: wins put Team Rocket's Mewtwo ex and Team Rocket's
  Mimikyu active/bench more often.

Remote runner:

```text
/tmp/run_success_ft_wave2.py
logs/rule_success_train_20260806/wave2_runner.log
```

The first launch had a runner bookkeeping bug: it started the first four jobs
but did not append them to `running`. The bug was fixed in `/tmp` and the runner
was restarted with resume/skip-existing logic. The stuck old runner was killed;
no `bc2_train.py` process remained afterward.

Wave 2 checkpoints:

```text
checkpoints/success_ft_v11_20260806/
```

All eight success-only fine-tunes completed:

- `marnie_b8f_vs_ogerpon_success_ft`
- `ogerpon_xsig_vs_crustle_success_ft`
- `dragapult_cc2_vs_marnie_success_ft`
- `dragapult_cc2_vs_crustle_success_ft`
- `cynthia_52f_vs_crustle_success_ft`
- `lucario_43d_vs_marnie_success_ft`
- `trm_f0b_vs_ogerpon_success_ft`
- `alakazam_7f_vs_trm_marnie_success_ft`

Evaluation outputs:

```text
logs/eval_rule_success_20260806/success_ft_wave2_manifest.csv
logs/eval_rule_success_20260806/success_ft_wave2_random_g300.csv
logs/eval_rule_success_20260806/delta_*_g200.csv
logs/eval_rule_success_20260806/success_ft_wave2_summary.txt
logs/eval_rule_success_20260806/success_ft_wave2_delta_summary.csv
```

Random g300:

- Dragapult vs Marnie success FT: `0.803`
- Dragapult vs Crustle success FT: `0.890`
- Lucario vs Marnie success FT: `0.903`
- Cynthia vs Crustle success FT: `0.923`
- Alakazam vs TRM/Marnie success FT: `0.990`
- Ogerpon vs Crustle cross-sig success FT: `0.993`
- Marnie vs Ogerpon success FT: `0.997`
- TR Mewtwo vs Ogerpon success FT: `1.000`

Focused baseline-delta g200:

- Marnie vs Ogerpon: `avg_delta=+0.0167`; slight local gain, still weak
  absolute WR around `0.395`.
- TR Mewtwo vs Ogerpon: `avg_delta=+0.0100`; slight local gain.
- Cynthia vs Crustle: `avg_delta=+0.0050`; much smaller than the earlier wave1
  win/loss specialist.
- Dragapult vs Crustle: `avg_delta=-0.0050`.
- Alakazam vs TRM/Marnie: `avg_delta=-0.0067`.
- Lucario vs Marnie: `avg_delta=-0.0067`.
- Dragapult vs Marnie: `avg_delta=-0.0083` and random only `0.803`.
- Ogerpon vs Crustle: `avg_delta=-0.0167`; success-only BC did not repair this
  structural weakness.

Interpretation:

- Success-only FT is useful as a diagnostic teacher/data source, not as a
  submission recipe.
- Marnie/Ogerpon and TRM/Ogerpon have small positive signal worth tracing, but
  the absolute weakness remains.
- Ogerpon/Crustle likely needs explicit anti-wall strategy construction or a
  narrow rule/rerank guard around attacker selection and win condition; simply
  cloning scarce winning Ogerpon rows made the focused matchup worse.
- Dragapult success-only variants degraded random stability; do not scale this
  recipe for Dragapult without mixing back broad population data.

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

Important Kaggle replay constraint:

- Kaggle ladder currently only uses the latest two submissions per account for new matches. When analyzing live replay results, prioritize only the latest two `jie` submissions plus the latest two `by` submissions; older replay pulls are useful only as historical/ablation evidence.
- Current active set checked at 2026-08-05 14:59 Asia/Shanghai:
  - `jie`: `55252321` (`bc: v10shadow_festival_iliamna`, score 930.6), `55241767` (`bc: submission_alakazam_v10pop`, score 902.7).
  - `by`: `55254351` (`bc: v10shadow_marnie_dries_tufa_labs`, score 955.2), `55252351` (`bc: v10shadow_alakazam_majkel1337`, score 841.2).
- Focused latest-two replay analysis is saved locally and remotely at `logs/kaggle_replay_latest2_20260805/`. It includes replay JSONs, row CSVs, decision summaries, opponent deck CSVs, `submission_summary.csv`, `loss_opponents.csv`, and `summary.txt`. Local copy size was about 478 MB.
- A partial broader historical pull was also left on `ks` at `logs/kaggle_replay_dual_20260805/`; it includes `jie` old Ogerpon/v8/v9/v7 comparisons, but it is not the current live ladder set.

## Matchup Relations

New reusable tool:

```bash
python3 tools/analyze_episode_matchups.py --help
```

It aggregates directional matchup results from Kaggle episode ZIPs and can
overlay pulled replay row CSVs from `tools/analyze_kaggle_replays.py`.

Generated outputs are local and remote at:

```text
logs/matchup_notes_20260805/0804_score900/
logs/matchup_notes_20260805/0724_0804_score900/
```

Committed record:

```text
docs/09_matchup_relations.md
```

Data windows:

- `0804_score900`: 4,811 episode files, 4,382 games used, current 900+ known
  deck signatures only, plus 212 latest replay rows.
- `0724_0804_score900`: 54,105 episode files, 46,566 games used, current 900+
  known deck signatures only, plus the same 212 latest replay rows.

Stable high-level priors from both windows:

- Mega Lucario strongly beats Mega Lopunny.
- Mega Lopunny beats Crustle Wall, Festival Lead, Teal Mask Ogerpon, and has a
  smaller edge into Marnie Grimmsnarl.
- Crustle Wall beats Teal Mask Ogerpon and Dragapult.
- Teal Mask Ogerpon beats Marnie Grimmsnarl and Cynthia Garchomp.
- Alakazam beats Crustle Wall, Teal Mask Ogerpon, and Cynthia Garchomp.
- Festival Lead beats Marnie Grimmsnarl.
- Marnie Grimmsnarl has a smaller long-window edge into Alakazam.
- Dragapult has a smaller long-window edge into Alakazam.

12-day-only priors that did not pass the stricter 0804 single-day threshold:

- Team Rocket Mewtwo beats Alakazam.
- Teal Mask Ogerpon beats Team Rocket Mewtwo.
- Crustle Wall beats Festival Lead.
- Mega Lucario beats Alakazam.
- Alakazam slightly beats Mega Lopunny.
- Cynthia Garchomp beats Team Rocket Mewtwo.
- Festival Lead slightly beats Teal Mask Ogerpon.
- Crustle Wall slightly beats Cynthia Garchomp.

Important live replay caveat:

- Latest replay rows are biased to the newest two submissions per account.
  Replay evidence should be treated as live probes, not as a complete metagame
  matrix.
- Current replay pull: Marnie shadow `b8f251a476e7` was strong into Alakazam
  `7f9a538936e3` (6/7 from the Marnie-side pull; combined Alakazam-side rows
  were 12/26). Festival shadow `e82dcbe62260` was 9/15 into Marnie but 0/3 into
  Dragapult. Alakazam vs Crustle replay rows were poor despite episode priors
  favoring Alakazam, so inspect exact Crustle signatures before training.

For weakness-pool construction, use `docs/09_matchup_relations.md` first, then
select concrete opponent policies from `deck_sig_counter_edges.csv` and
intersect them with the audited shadow pools.

V11 live submission check at 2026-08-05 18:05 Asia/Shanghai:

- `jie` submitted `55264182` (`bc: pop_v11all_marnie_grimmsnarl_b8f251a4_1`,
  observed score 889.2) and `55264151`
  (`bc: pop_v11all_teal_mask_ogerpon_5899c772_2`, observed score 751.1).
- Replay analysis was saved locally and remotely at
  `logs/kaggle_replay_v11_submit_20260805/`.
- After using `--team-name "Jie Orkarin"` to include most same-deck cases:
  - Marnie `55264182`: 23 attributed games, 15/8, WR 0.652.
  - Ogerpon `55264151`: 24 attributed games, 15/9, WR 0.625.
- Interpretation: Marnie did not really contradict local weighted RR; its local
  weighted RR was also about 0.652. The score is probably early rating variance
  and opponent-rating effects.
- Ogerpon did contradict the local aggregate ranking. The replay sample had
  Ogerpon 0/4 into Crustle Wall, matching the known hard weakness, plus losses
  to live signatures not represented in the candidate RR pool, especially Mega
  Lucario `ab089ccfad1a`.
- Local candidate RR is therefore a useful internal candidate filter, not a
  Kaggle score predictor until live replay opponent signatures are added to the
  environment pool.

Remote RL pilot status from 2026-08-05 18:09 Asia/Shanghai:

- Both pilot jobs completed under `logs/rl/`.
- Marnie `b8f251` vs Ogerpon pool:
  `logs/rl/pilot_marnie_b8f_vs_ogerpon_metrics.csv`, 8 iterations x 64 games,
  per-iteration WR ranged 0.203-0.344.
- Ogerpon `5899` vs Crustle pool:
  `logs/rl/pilot_ogerpon_5899_vs_crustle_metrics.csv`, 8 iterations x 64 games,
  per-iteration WR ranged 0.016-0.047.
- Do not treat the saved RL checkpoints as candidates yet. They need random,
  baseline-delta, and broad RR validation. The pilot mainly confirms these are
  hard weakness pools, especially Ogerpon into Crustle.

## Complex Scene Diagnostics

User ran four v11 diagnostic commands for:

- Marnie Grimmsnarl `b8f251a476e7` vs Ogerpon.
- Teal Mask Ogerpon `5899c772bace` vs Crustle Wall.

Local copies are in:

```text
logs/complex_v11/
```

Marnie vs Ogerpon findings:

- This is not primarily a data-scarcity issue: `MAIN` has 52,842 examples and
  exact/top3 are 0.724/0.958.
- Accuracy falls with option count: exact 0.895 for 2 options, 0.637 for 6-10,
  and 0.586 for 11+.
- Main weak contexts are `DISCARD` exact 0.153 on 222 samples, `ATTACH_TO`
  exact 0.408 on 2,468 samples, `TO_BENCH` exact 0.642, `SWITCH` exact 0.674,
  and `TO_HAND` exact 0.684.
- `ATTACH_TO` has a cardinality/over-selection problem:
  true length 2.95 vs predicted 3.39, under 0.109, over 0.342.
- Type confusions are mostly `PLAY <-> ABILITY`, `ATTACH -> PLAY`, and
  `ATTACK -> PLAY`; `RETREAT` exact is only 0.494 and often becomes early
  `END` or `ATTACK`.
- Interpretation: the right action is often in top3, but ranking, multi-target
  cardinality, and stage-specific target selection are weak.

Ogerpon `5899` vs Crustle findings:

- This is data-scarce after filters: only 997 `MAIN` examples and 1,553 total
  decisions in the diagnostic table.
- `MAIN` exact/top3 are 0.684/0.932. `ATTACH_TO` is fine at 0.978; set-style
  targets are mostly fine except tiny `DISCARD` n=19.
- The main strategic miss is tempo: true `ATTACK` exact is 0.688 and
  miss-attack is 0.312; confusion has `ATTACK -> PLAY` 46 times, the largest
  non-correct class.
- Do not train a pure Ogerpon-vs-Crustle BC specialist from this narrow slice
  alone; it is likely to overfit. Prefer mixed Ogerpon training with
  archetype-level Crustle overweight plus attack/attach/retreat/context weights,
  then validate against random, Crustle shadow pool, and broad RR.

Immediate training direction from these diagnostics:

- For Marnie-like decks, prioritize complex-weighted BC: higher weights for
  option-count >= 6, `TO_HAND`, `ATTACH_TO`, `DISCARD`, `TO_BENCH`, `SWITCH`,
  `ATTACH`, `RETREAT`, and `ATTACK`, plus stronger set/cardinality loss.
- For Ogerpon-like decks, prioritize tempo diagnostics and attack-vs-play
  ranking in weakness matchups; RL can be used after trace, but current PPO
  pilot checkpoints are not candidate quality yet.

Complex-weighted v11 training check at 2026-08-05 20:47 Asia/Shanghai:

- Checkpoints exist on `ks` under `checkpoints/complex_v11/`:
  `bc2_marnie_b8f_v11all_complex_w2.npz`,
  `bc2_marnie_b8f_vs_ogerpon_complex_w2.npz`,
  `bc2_ogerpon_5899_v11all_tempo_complex_w2.npz`, and
  `bc2_ogerpon_5899_vs_crustle_tempo_probe_w2.npz`.
- Post diagnostics are on `ks` under `logs/complex_v11/post/`.
- Marnie b8f vs Ogerpon overall exact/top3:
  pre `0.7663/0.9633`, global complex `0.7665/0.9639`, conditioned probe
  `0.7693/0.9643`.
- Marnie did improve in targeted areas, but only modestly:
  `DISCARD` exact `0.153 -> 0.189 -> 0.216`,
  `ATTACH_TO` exact `0.408 -> 0.412 -> 0.423`,
  `RETREAT` exact `0.494 -> 0.592 -> 0.617`,
  `ATTACK` miss rate `0.197 -> 0.159 -> 0.146`.
- Marnie tradeoff: `PLAY` exact dropped from `0.717` to `0.679/0.692`, so
  the current hand-tuned weights are too blunt for submission without random
  and RR validation.
- Ogerpon 5899 vs Crustle overall exact/top3:
  pre `0.7489/0.9491`, global tempo-complex `0.7733/0.9530`, conditioned
  probe `0.7675/0.9491`.
- Ogerpon global model is the better BC checkpoint from this run. It improved
  `MAIN` exact `0.684 -> 0.724`, `ATTACK` exact `0.688 -> 0.822`, attack miss
  rate `0.312 -> 0.178`, and `ATTACH` exact `0.603 -> 0.756`.
- Ogerpon conditioned probe overcorrected attack timing: `ATTACK` exact reached
  `0.879`, but `PLAY` exact dropped to `0.581` and `PLAY -> ATTACK` confusion
  increased, so do not treat it as a submission candidate.

Complex-v11 validation run completed on `ks` at 2026-08-05 21:08 Asia/Shanghai:

- Remote outputs are under `logs/eval_complex_v11/`; summary is
  `logs/eval_complex_v11/summary.txt`. CSVs and summary were pulled locally to
  the same path.
- Random g500 was stable for all four complex checkpoints:
  Marnie global/conditioned both `0.998`, Ogerpon global `0.990`, Ogerpon
  conditioned `0.994`.
- Marnie vs Ogerpon pool g200:
  - `marnie_global`: `avg_delta=-0.0088`, `weighted_delta=-0.0088`, lost `3/4`.
  - `marnie_cond_ogerpon`: `avg_delta=+0.0037`, `weighted_delta=+0.0037`, lost
    `3/4`; only improved vs the new complex Ogerpon, not robustly vs original
    Ogerpon pool.
- Ogerpon 5899 vs Crustle pool g200:
  - `og5899_global_tempo`: `avg_delta=-0.0138`, lost `3/4`.
  - `og5899_cond_crustle`: `avg_delta=-0.0213`, lost `3/4`.
  - This invalidates the apparent supervised attack-timing gain as a Crustle
    matchup fix.
- Ogerpon broad environment g80:
  - `og5899_global_tempo`: `avg_delta=-0.0283`, `weighted_delta=-0.0399`, lost
    `32/49`; worst was `shadow_alakazam_7f9a5389_miya:-0.3125`.
  - `og5899_cond_crustle`: `avg_delta=-0.0316`, `weighted_delta=-0.0445`, lost
    `36/49`.
  - Do not submit or scale these Ogerpon complex-weighted recipes.
- Marnie broad environment g80:
  - `marnie_global`: `avg_delta=+0.0094`, `weighted_delta=+0.0031`, lost
    `20/49`.
  - `marnie_cond_ogerpon`: `avg_delta=+0.0115`, `weighted_delta=+0.0161`, lost
    `20/49`; it improved some Alakazam/TR Mewtwo/Marnie rows but had large
    regressions, including `shadow_team_rocket_mewtwo_2c3f6873:-0.1500`,
    `shadow_marnie_grimmsnarl_2c22fa76_dries_tufa_labs:-0.1125`, and
    `shadow_marnie_grimmsnarl_2c22fa76_jz:-0.1000`.
- Interpretation: random is too weak as a gate, and current hand-tuned complex
  sample weights can move supervised metrics without improving matchup win
  rate. Do not scale this recipe across population yet. Next work should make
  complex metrics outcome-aware: compare loss-vs-win trace states and train on
  the specific decision contexts that differ in won games, rather than globally
  increasing `ATTACK`/`ATTACH`/multi-select weights.

## Specialist And Weak-Upweight BC

Pure matchup-conditioned specialist wave 1 completed on `ks`.

Important wave 1 repair:

- The original runner printed `training complete failed=0`, but
  `cynthia_vs_crustle_wl` actually OOMed under set-loss. It was retried with
  batch `512`, CUDA cap `16GB`, and `--set-loss-weight 0.0`; the repaired
  checkpoint is
  `checkpoints/specialist_v11_20260805/bc2_cynthia_vs_crustle_wl_w2.npz`.

Wave 1 repaired random g300:

- Marnie vs Ogerpon WL: `1.000`; winner-only: `0.9967`.
- Ogerpon vs Crustle WL: `1.000`; winner-only: `0.9933`.
- Dragapult vs Marnie WL: `0.8467`.
- Dragapult vs Crustle WL: `0.9233`.
- Alakazam vs TR Mewtwo WL: `0.9867`.
- Lopunny vs Cynthia WL: `0.9800`.
- Cynthia vs Crustle WL: `0.9567`.
- TR Mewtwo vs Ogerpon WL: `0.9900`.

Wave 1 focused g200 deltas:

- `cynthia_vs_crustle_wl`: `avg_delta=+0.090`.
- `lopunny_vs_cynthia_wl`: `+0.0283`.
- `alakazam_vs_trm_wl`: `+0.0267`.
- `dragapult_vs_crustle_wl`: `+0.0100`.
- `ogerpon_vs_crustle_wl`: `+0.0067`.
- `marnie_vs_ogerpon_wl`: `+0.0033`.
- `marnie_vs_ogerpon_winner_only`: `-0.0033`.
- `dragapult_vs_marnie_wl`: `-0.0083`.
- `trm_vs_ogerpon_wl`: `-0.0200`.

Wave 1 broad guardrail against the balanced 49-entry environment failed for
almost every pure specialist:

- `ogerpon_vs_crustle_wl`: `avg_delta=+0.0015`, `weighted_delta=+0.0086`,
  `lost=20/49`; effectively flat/noisy, not a clear upgrade.
- `ogerpon_vs_crustle_winner_only`: `avg_delta=-0.0028`,
  `weighted_delta=+0.0018`, `lost=18/49`.
- `alakazam_vs_trm_wl`: `avg_delta=-0.0079`,
  `weighted_delta=-0.0116`, `lost=24/49`.
- `lopunny_vs_cynthia_wl`: `avg_delta=-0.0202`,
  `weighted_delta=-0.0423`, `lost=30/49`.
- `cynthia_vs_crustle_wl`: `avg_delta=-0.0513`,
  `weighted_delta=-0.0469`, `lost=34/49`.

Interpretation: target-only filtering is causing narrow gains and broad
regression. Do not use wave 1 checkpoints as general replacements. Keep them as
diagnostic specialist policies only.

Code update made locally and synced to `ks`:

- `ptcg_rl/bc2/data.py` and `tools/bc2_train.py` now support
  `--opponent-archetype-weight "NAME=WEIGHT"` and
  `--opponent-deck-sig-weight SIG=WEIGHT`. These multiply sample weights without
  filtering out non-target games.
- Local smoke test verified that opponent archetype weights combine with
  win/loss/draw weights as expected.

Weak-upweight wave 2 started on `ks` at `2026-08-05 23:12 Asia/Shanghai`.

Runner:

```bash
ssh ks 'tail -f /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804/logs/v11_weakup_20260805/wave2_runner.log'
```

Script:

```text
/tmp/run_v11_weakup_wave2.sh
```

Outputs:

```text
checkpoints/weakup_v11_20260805/
logs/v11_weakup_20260805/
logs/eval_next_v11_weakup_20260805/
```

Wave 2 recipe:

- Keep each candidate on full `data/bc_corpus_banded_v11_0724_0804` for its
  `--deck-sig`; do not filter to only weak-opponent games.
- Initialize from the corresponding population checkpoint under
  `checkpoints/pop_v11_0724_0804/`.
- Six fine-tunes, six epochs, `lr=5e-5`, `batch=1024`, `set_loss=0.10`,
  `win/loss/draw=1.5/0.4/0.8`, `multi_select_weight=1.1`, feature dims `80/64`.
- Weak-opponent weights:
  - Marnie `b8f251a476e7`: Ogerpon `2.5`.
  - Ogerpon `5899c772bace`: Crustle `4.0`.
  - Alakazam `7f9a538936e3`: TR Mewtwo `2.5`, Marnie `2.0`.
  - Lopunny `f1445356c3a7`: Cynthia `2.5`, Alakazam `2.0`.
  - Cynthia `52f467394857`: Crustle `2.5`, Ogerpon `2.0`.
  - Dragapult `cc2e995b5ad0`: Marnie `2.5`, Crustle `2.5`, Ogerpon `2.0`.

Status at `2026-08-05 23:16 Asia/Shanghai`:

- Training finished at `2026-08-05 23:38 Asia/Shanghai`, `failed=0`.
- Saved best checkpoints exist for all six weak-upweight candidates.
- Random g500 completed and wrote
  `logs/eval_next_v11_weakup_20260805/weakup_wave2_random_g500.csv`.
- Focused delta did not run correctly because the runner's target-manifest
  generation snippet used `f.fieldnames` instead of `reader.fieldnames`.
  The focused logs currently show `FileNotFoundError` for missing
  `target_*.csv`. This needs a small repair script; broad eval is unaffected.
- Broad eval started with `broad_delta_marnie_env80_g80.csv` and was running at
  the 23:41 check.

Weak matchup game-count audit from `data/bc_corpus_banded_v11_0724_0804`:

```text
marnie_b8f_vs_ogerpon: games=2664 W/L/D=738/1925/1 wr=0.277 decisions=197329 win_decision_share=0.304
ogerpon_5899_vs_crustle: games=57 W/L/D=1/56/0 wr=0.018 decisions=5144 win_decision_share=0.015
ogerpon_697_vs_crustle: games=161 W/L/D=5/156/0 wr=0.031 decisions=15998 win_decision_share=0.009
ogerpon_2a507_vs_crustle: games=204 W/L/D=81/123/0 wr=0.397 decisions=16862 win_decision_share=0.329
alakazam_7f_vs_trm_marnie: games=6124 W/L/D=2477/3646/1 wr=0.404 decisions=465135 win_decision_share=0.402
lopunny_f144_vs_cynthia_alakazam: games=199 W/L/D=104/95/0 wr=0.523 decisions=15850 win_decision_share=0.554
cynthia_52f_vs_crustle_ogerpon: games=524 W/L/D=192/332/0 wr=0.366 decisions=39811 win_decision_share=0.354
dragapult_cc2_vs_marnie_crustle_ogerpon: games=800 W/L/D=442/358/0 wr=0.552 decisions=94659 win_decision_share=0.581
```

Interpretation: some weak matchups have enough winning demonstrations for
outcome-weighted BC, but Ogerpon `5899c772bace` and `697a82e582d5` into
Crustle have almost no successful same-sig data. BC cannot invent a counter-plan
from mostly losing labels. For Ogerpon vs Crustle, mine/transfer the successful
`2a5072194fdf` games or generate new successful trajectories with rules/search/RL
before expecting fine-tuning to improve the matchup.

Rule/success-data exploration started:

- New tool: `tools/build_bc_subset.py`. It builds a BCCorpus-compatible filtered
  subset under `OUT/Archetype/out_band/name.npz`, preserving the original npz
  schema and metadata. Use it for winner-only success pools, loss pools, and
  trace/BC distillation probes.
- New tool: `tools/audit_matchup_success_data.py`. It scans a BC corpus and
  counts available wins/losses/decisions by `archetype`, `deck_sig`, and
  `opponent_archetype`. With `--weak-plan`, it emits a success-data plan with
  recommendations:
  `same_sig_success_bc_ok`, `same_sig_sparse_use_cross_sig_or_generate`,
  `same_sig_sparse_generate_more`, `cross_sig_teacher_needed`, and
  `generate_success_needed`.
- New tool: `tools/compare_bc_subsets.py`. It compares two subset `.npz` files
  by chosen first action type/card over context, context+turn, and
  context+option-count buckets. Current implementation is intentionally light
  and does not import the full project card registry to keep remote runs fast.
- New doc: `docs/10_rule_success_data.md`, covering rule-overlay discipline,
  external source usage, and success-data classes.
- Remote script used: `/tmp/build_success_subsets_v1.sh`.
- Remote log: `logs/rule_success_20260805/build_success_subsets_v1.log`.
- Local copies pulled:
  `logs/rule_success_20260805/ogerpon_deck_compare.txt`,
  `logs/rule_success_20260805/success_subset_shapes.txt`, and
  `logs/eval_next_v11_weakup_20260805/weak_matchup_game_counts.csv`.

Constructed remote success/loss subset corpora:

```text
data/bc_success_subsets_v11_20260805/Teal_Mask_Ogerpon/ogerpon_crustle_success_xsig/ogerpon_2a507_5899_697_vs_crustle_wins.npz
  rows=4456 games=66 state=(80,) option=(1,64)
data/bc_success_subsets_v11_20260805/Teal_Mask_Ogerpon/ogerpon_crustle_target_losses/ogerpon_5899_697_vs_crustle_losses.npz
  rows=10950 games=105 state=(80,) option=(1,64)
data/bc_success_subsets_v11_20260805/Marnie_Grimmsnarl/marnie_b8f_ogerpon_success/marnie_b8f_vs_ogerpon_wins.npz
  rows=35715 games=446 state=(80,) option=(2,64)
data/bc_success_subsets_v11_20260805/Alakazam/alakazam_7f_trm_marnie_success_sample/alakazam_7f_vs_trm_marnie_win_games1000.npz
  rows=99024 games=1323 state=(80,) option=(2,64)
```

Important Ogerpon caveat:

- `2a5072194fdf` is not a small edit of `5899c772bace` or `697a82e582d5`; it is
  a materially different Ogerpon box. It has fewer Teal Mask Ogerpon, plus Mega
  Kangaskhan ex, Meowth ex, Lillie's Clefairy ex, and many different trainer
  ids. Do not directly train `5899/697` to imitate all `2a507` decisions unless
  the chosen cards/actions also exist in the target deck or the aim is only a
  diagnostic teacher policy.
- Better next step for Ogerpon vs Crustle: compare `2a507` wins against
  `5899/697` losses by action type/card and by turn window, then extract only
  shared actionable rules such as attack-window timing, END/ABILITY loop guards,
  or target/retreat heuristics.

Latest rule/success audit outputs:

```text
logs/rule_success_20260805/matchup_success_counts_top24.csv
logs/rule_success_20260805/success_data_plan_top24.csv
logs/rule_success_20260805/success_data_plan_top24_summary.txt
logs/rule_success_20260805/matchup_success_counts_top24_1000p.csv
logs/rule_success_20260805/success_data_plan_top24_1000p.csv
logs/rule_success_20260805/success_data_plan_top24_1000p_summary.txt
```

All-score-band top24 audit, deck-sig rows:

- `same_sig_success_bc_ok`: 13.
- `same_sig_sparse_use_cross_sig_or_generate`: 37.
- `same_sig_sparse_generate_more`: 26.
- `cross_sig_teacher_needed`: 5.
- `generate_success_needed`: 8.

High-score `1000+` top24 audit, deck-sig rows:

- `same_sig_success_bc_ok`: 13.
- `same_sig_sparse_use_cross_sig_or_generate`: 20.
- `same_sig_sparse_generate_more`: 17.
- `cross_sig_teacher_needed`: 10.
- `generate_success_needed`: 29.

Important high-score `1000+` target counts:

```text
Marnie b8f vs Ogerpon: games=1517 W/L=446/1071 win_decisions=35715 same_sig_success_bc_ok
Dragapult cc2 vs Marnie: games=532 W/L=326/206 win_decisions=41289 same_sig_success_bc_ok
Dragapult cc2 vs Crustle: games=73 W/L=25/48 win_decisions=3294 sparse_use_cross_sig_or_generate
Cynthia 52f vs Crustle: games=135 W/L=52/83 win_decisions=4025 same_sig_success_bc_ok
Mega Lucario 43d vs Marnie: games=200 W/L=103/97 win_decisions=7199 same_sig_success_bc_ok
TR Mewtwo f0b vs Ogerpon: games=47 W/L=20/27 win_decisions=1442 sparse_use_cross_sig_or_generate
Mega Lopunny f144 vs Cynthia: games=5 W/L=1/4 win_decisions=65 sparse_generate_more
Ogerpon 5899 vs Crustle: games=17 W/L=1/16 win_decisions=77 sparse_use_cross_sig_or_generate
Ogerpon 697 vs Crustle: games=93 W/L=4/89 win_decisions=103 sparse_use_cross_sig_or_generate
Ogerpon 2a507 vs Crustle: games=149 W/L=61/88 win_decisions=4276 same_sig_success_bc_ok
```

Batch success-vs-loss subset/compare run:

- Script: `/tmp/build_compare_success_pairs_v1.sh`.
- Status: completed successfully.
- Remote success/loss subsets were written under
  `data/bc_success_subsets_v11_20260805/`; large `.npz` files were not pulled
  locally.
- Local/remote compare CSVs live under
  `logs/rule_success_20260805/*_success_vs_loss/`.

Batch compare highlights:

- `marnie_b8f_vs_ogerpon`: success rows `35715`, loss rows `76353`. Success
  states choose/switch to `Marnie's Grimmsnarl ex` far more often and are less
  dominated by Ogerpon target/card choices. This supports a narrow "complete and
  keep main Grimmsnarl line vs Ogerpon" hypothesis, not a global evolve rule.
- `dragapult_cc2_vs_marnie`: success rows `41289`, loss rows `21146`. Success
  states choose `Dragapult ex` for `TO_ACTIVE`/`SWITCH` more often and choose
  opponent Marnie utility cards less in damage-counter contexts. This points to
  active/target selection rather than simple attack frequency.
- `dragapult_cc2_vs_crustle`: success rows `3294`, loss rows `7627`. Data is
  thinner, but wins show more Dreepy bench setup and Drakloak attach-from,
  while losses overselect Crustle-related contexts. Treat as a trace target
  before rules.
- `cynthia_52f_vs_crustle`: success rows `4025`, loss rows `8396`. Success
  states switch into non-Garchomp card `387` and choose ACTIVATE YES more often;
  loss states switch to Garchomp ex more often. Needs card-name mapping before
  a rule is attempted.
- `lucario_43d_vs_marnie`: success rows `7199`, loss rows `6605`. Success
  states choose `Mega Lucario ex` for `TO_ACTIVE`/`SWITCH` more often and choose
  Spikemuth Gym ability less. Candidate rule: prefer main attacker in active
  windows and avoid utility-stadium loops if attack windows exist.
- `trm_f0b_vs_ogerpon`: success rows `1442`, loss rows `1639`. Sparse but
  readable: wins select Team Rocket's Mewtwo ex as active more often; losses
  choose Mimikyu/Spidops active/switch lines more often.
- `lopunny_f144_vs_cynthia`: high-score success set has only 65 decisions from
  one winning game; do not train or write rules from this same-sig success set.

External source note:

- Public PTCG sources checked include Limitless deck pages/API docs,
  TrainerHill meta analysis, and PokemonMeta win rates. Use them only as
  matchup and card-package hypotheses. Limitless Ogerpon Box style pages support
  the local finding that Kangaskhan/Meowth/Clefairy tech packages are real
  Ogerpon Box plans, but Kaggle deck signatures still require local validation.

Rule strategy direction:

- Keep BC as the default policy.
- Add rules as narrow rerank/veto/bonus layers, not global action replacement.
- Every rule should log a reason and be validated by random, focused weak-pool
  delta, and broad environment delta.
- If a rule produces wins in a weak matchup, use those rollouts as generated
  successful trajectories and distill back into BC/RL-anchor training.

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -80 logs/v11_weakup_20260805/wave2_runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && for f in logs/v11_weakup_20260805/*.log; do b=$(basename "$f"); printf "%s | " "$b"; tail -1 "$f"; done'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && ps -eo pid,ppid,stat,etime,cmd | grep -E "bc2_train.py|eval_manifest_random.py|eval_baseline_delta.py|run_v11_weakup_wave2" | grep -v grep'
```

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

Focused Kaggle latest-two replay analysis, 2026-08-05:

```text
logs/kaggle_replay_latest2_20260805/summary.txt
logs/kaggle_replay_latest2_20260805/submission_summary.csv
logs/kaggle_replay_latest2_20260805/loss_opponents.csv
```

Current active-submission replay outcomes:

| Account | Submission | Archetype | Score | Replay W/L/D | Replay WR | Main observed losses |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `jie` | `55252321` | Festival Lead shadow | 930.6 | 31/24/0 | 56.4% | Marnie `b8f251...` 9/15, Alakazam `7f9...` 5/7, Lopunny `2767...` 1/2 |
| `jie` | `55241767` | Alakazam v10pop | 902.7 | 42/22/0 | 65.6% | Marnie `b8f251...` 7/19, Crustle `96d...` 0/2 |
| `by` | `55254351` | Marnie shadow Dries | 955.2 | 34/14/0 | 70.8% | mostly dispersed; Alakazam `7f9...` was strong at 6/7 |
| `by` | `55252351` | Alakazam shadow Majkel | 841.2 | 27/18/0 | 60.0% | dispersed unknown sigs; Marnie `b8f251...` was 5/7, not the main problem |

Interpretation:

- The best current live replay signal is `by` Marnie shadow: strong non-mirror replay WR and good result into `7f9...` Alakazam, despite many skipped same-deck mirrors.
- `jie` Alakazam v10pop has a real observed weakness into Marnie `b8f251...` (7/19), matching local pressure concerns.
- `by` Alakazam shadow does not show the same Marnie collapse in this sample, but its losses are spread across unknown signatures and it has higher early-end rate (`0.0258`, loss-side `0.0440`), suggesting narrower/generalization issues.
- `jie` Festival shadow has poor second-player replay WR (41.7%) and only 56.4% overall in the current active replay sample; treat it as a probe, not a confirmed strong candidate.
- Many same-deck mirrors are skipped because replay agent identification currently uses deck matching and both agents can have the same 60-card list. Next replay-tool improvement should use team/submission metadata as a fallback for same-deck matches.

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
- Random analysis was written on `ks` to `logs/eval_v11_0724_0804/v11_random_analysis.txt`.
- Population top3 random g500 (`logs/eval_v11_0724_0804/pop_top3_0804_random_g500.csv`): `n=32 mean=0.937 weighted=0.965 median=0.977 min=0.638 max=1.000`; `>=0.97=19`, `>=0.95=20`, `<0.90=7`, timeouts `0`.
- Weak population archetypes vs random are Mega Starmie (`mean=0.682`), Dragapult (`0.866`), Mega Lucario (`0.889`), and one Ogerpon deck (`2a5072194fdf`, `0.796`). Strong population archetypes are Marnie (`0.999`), Festival Lead (`0.995`), Team Rocket Mewtwo (`0.994`), Mega Lopunny (`0.983`), Alakazam (`0.979`), Cynthia (`0.970`).
- Shadow pop-init random g500 (`logs/eval_v11_0724_0804/shadow_v11_0724_0804_popinit_random_g500.csv`): `n=93 mean=0.943 weighted=0.948 median=0.976 min=0.388 max=1.000`; `>=0.97=58`, `>=0.95=65`, `<0.90=15`.
- Shadow timeouts are concentrated in exactly one row: Teal Mask Ogerpon `cc3f2796d570` team `tonakaiiii`, WR `0.388`, timeouts `273`. Excluding this row, shadow summary is `n=92 mean=0.950 weighted=0.954 median=0.976 min=0.700`, `>=0.97=58`, `<0.90=14`.
- Low shadow WR is not mainly explained by data volume or ladder score: Pearson correlation with random WR was low (`decisions=0.108`, `episodes=0.107`, `trajectory_score=0.086`, `date_count=0.115`; `decision_win_rate=-0.110`, `avg_score=-0.008`).
- Main random failure clusters: Dragapult (`mean=0.858`, 11/13 below 0.95), Crustle (`mean=0.937`, 9/14 below 0.95), and a few Ogerpon specialist shadows (`cc3f2796`, `050cfefe`, `1784e485`, `0e532395`, `2a507219`). Marnie, Team Rocket Mewtwo, Alakazam, Mega Lopunny, and Cynthia shadows are random-stable.
- Candidate manifest `logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv` has 90 rows = all 32 population aliases plus 58 shadow rows with random WR `>=0.97`. For submission selection, treat population rows below `0.95` as diagnostic unless RR proves a compelling matchup reason; for local ladder diversity they can still be included deliberately.
- Current shadow pools were split at 2026-08-05 15:20 Asia/Shanghai:

```text
logs/eval_v11_0724_0804/shadow_pools_20260805/
```

Generated by:

```bash
python3 tools/split_shadow_pools.py \
  --manifest logs/shadow_pool_manifest_v11_0724_0804_popinit_set.csv \
  --random logs/eval_v11_0724_0804/shadow_v11_0724_0804_popinit_random_g500.csv \
  --fallback-random logs/eval_v11_0724_0804/pop_top3_0804_random_g500.csv \
  --out-dir logs/eval_v11_0724_0804/shadow_pools_20260805 \
  --env-per-archetype 6 \
  --quality-per-archetype 5 \
  --stress-per-archetype 4 \
  --debug-per-archetype 5 \
  --per-deck-sig 2 \
  --fallback-per-missing-archetype 2
```

Pool files and intended use:

| File | Rows | Archetypes | Use |
| --- | ---: | ---: | --- |
| `shadow_all_enriched.csv` | 93 | 10 | Full audited shadow inventory with random WR/timeouts; use for inspection, not as a balanced ladder proxy. |
| `shadow_pool_quality_strict_ge097.csv` | 58 | 9 | High-trust shadow-only pool; excludes low-random/timeouts but misses Mega Starmie and Mega Lucario. |
| `shadow_pool_quality_balanced_inclusive.csv` | 34 | 10 | Balanced high-quality shadow pool with forced best fallback for thin archetypes; good default for candidate sanity. |
| `shadow_pool_environment_balanced.csv` | 47 | 10 | Balanced shadow environment pool; cap prevents Marnie/Mega Lopunny dominance while preserving weaker archetypes. |
| `shadow_pool_stress_balanced.csv` | 27 | 10 | Stress/debug pool containing high-trajectory plus low-random/timeout rows; use for weakness discovery, not score prediction. |
| `shadow_pool_debug_low_random.csv` | 18 | 5 | Low-random/timeout diagnostics for trace and feature/BC failure analysis. |
| `mixed_shadow_popfallback_quality_balanced_inclusive.csv` | 36 | 11 | Same as quality balanced, but adds Mega Starmie population fallback rows because audited shadow lacks Mega Starmie. |
| `mixed_shadow_popfallback_environment_balanced.csv` | 49 | 11 | Recommended broad local ladder proxy for now when all archetypes must be represented. |
| `mixed_shadow_popfallback_stress_balanced.csv` | 29 | 11 | Stress pool with Mega Starmie population fallback. |

Important limits:

- Audited current shadow covers only 10 archetypes; Mega Starmie has no shadow row in the random-audited set. The mixed manifests add two Mega Starmie population rows (`random WR 0.638/0.726`) only for coverage, so they are diagnostic/fallback rows, not high-trust shadows.
- `mixed_shadow_popfallback_environment_balanced.csv` is the best current environment proxy if every archetype must be present.
- `shadow_pool_quality_balanced_inclusive.csv` is cleaner for pass/fail candidate gating when pure shadow-only evaluation is desired.
- `shadow_pool_stress_balanced.csv` and `shadow_pool_debug_low_random.csv` deliberately contain unreliable rows such as Ogerpon timeout/low-random shadows; use them to find bad decisions and failure modes, not to estimate Kaggle score.
- Remote smoke test passed: `eval_round_robin.py --manifest logs/eval_v11_0724_0804/shadow_pools_20260805/mixed_shadow_popfallback_environment_balanced.csv --manifest-limit 2 --games 1` wrote `/tmp/shadow_pool_split_manifest_smoke.csv`.
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

## 2026-08-06 Aggressive Teacher Rollout Pivot

User explicitly redirected away from filtered BC, winner-only BC, and small data/parameter tweaks as the main path. Treat those as negative controls. The current direction is to use compute for generated success data and teacher policies:

- Generate weak-matchup success trajectories from policy portfolios.
- Make exploration controlled by the model's top-ranked actions rather than full random sampling.
- Use narrow, trace-auditable rules as teacher scaffolding only.
- Distill generated wins only after the generated corpus is summarized by matchup, opponent, and actor mode.

New code added locally and synced to `ks`:

- `tools/generate_rollout_bc.py`
  - Adds actor mode `topk@K`.
  - Actor examples: `topk@2+rules:targeted=2`, `topk@3+rules:stage2_setup=2`.
  - Records rule intervention reason in `actor_mode`, e.g. `topk@3+rules:targeted|stage2_setup_evolve`.
- `ptcg_rl/numpy_policy.py`
  - `NumpyPolicy.select(..., top_k=K)` masks sampling to the top K currently legal rows at each sequential pick.
- `ptcg_rl/rule_overlay.py`
  - Adds `stage2_setup`.
  - `targeted` now includes a generic early Stage-2 setup guard: prefer available evolution-chain actions through turn 10 and setup/primary active choices through turn 4.
  - This is for teacher rollout generation and probes, not a submission default.
- `tools/plan_rollout_teacher_jobs.py`
  - Reads RR or baseline-delta weakness CSVs plus candidate manifests.
  - Emits a balanced weak-matchup rollout plan CSV and a runnable shell script.
  - Default actor portfolio is `greedy/topk/sample/random + rules:<rule_mode>` with game-level actor scope.
- `tools/train_bc_population.py`
  - Adds `--aux-corpus`, `--aux-score-bands`, and `--aux-repeat` pass-through to `bc2_train.py`.
  - Adds `--init-manifest` to initialize each archetype from the first matching manifest checkpoint.
  - `--min-decisions 0` now skips slow `.npz` decision counting and is suitable for large-corpus dry-runs.
- `tools/filter_rollout_corpus.py`
  - Filters generated rollout `.npz` corpora by `actor_mode`, `opponent_name`, `final_status`, and `won`.
  - Use this before aux-distill so random/fallback-generated wins do not pollute BC.

Local smoke command used:

```bash
python3 tools/plan_rollout_teacher_jobs.py \
  --weakness-csv logs/eval_v11_0724_0804/rr_candidates_pop_top3_shadow_ge097_g100.csv \
  --candidate-manifest logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv \
  --max-win-rate 0.45 \
  --max-per-archetype 3 \
  --max-per-candidate 2 \
  --max-jobs 30 \
  --games 480 \
  --workers 12 \
  --parallel-jobs 5 \
  --max-turns 420 \
  --rule-mode targeted \
  --out-root data/generated_rollout_bc_teacher_v11all_rr \
  --out-band weak_win_search \
  --log-dir logs/rollout_teacher_v11all_rr \
  --out-csv logs/rollout_teacher_v11all_rr/plan.csv \
  --skipped-csv logs/rollout_teacher_v11all_rr/skipped.csv \
  --out-sh logs/rollout_teacher_v11all_rr/run_rollout_teacher_jobs.sh
```

The local plan generated 30 jobs with 3 weak matchups each for these archetypes: Alakazam, Cynthia Garchomp, Dragapult, Festival Lead, Marnie Grimmsnarl, Mega Lopunny, Mega Lucario, Mega Starmie, Teal Mask Ogerpon, Team Rocket Mewtwo.

Remote active jobs at 2026-08-06 13:18:

- Quick no-MCTS rollout still running:
  - Script: `/tmp/run_rollout_search_nomcts_quick_20260806.sh`
  - Corpus: `data/generated_rollout_bc_rollout_search_nomcts_quick_20260806`
  - Logs: `logs/rollout_search_nomcts_quick_20260806/`
  - At last check: about 27 `.npz` chunks and about 54 matching rollout processes.
- Distill watcher still waiting for quick rollout:
  - Script: `/tmp/run_rollout_distill_after_quick_20260806.sh`
  - Log: `logs/rollout_distill_20260806/runner.log`
  - It will train/evaluate Marnie/Ogerpon2a/Cynthia aux-corpus distill only after quick rollout ends.
- New teacher-rollout waiter started:
  - PID observed: `3138676`
  - Script: `/tmp/run_rollout_teacher_v11all_rr_topk_20260806.sh`
  - Log: `logs/rollout_teacher_v11all_rr_topk_20260806/runner.log`
  - Output corpus: `data/generated_rollout_bc_rollout_teacher_v11all_rr_topk_20260806`
  - It waits for quick rollout, distill watcher, and related train/eval processes to finish before launching 30 topk teacher jobs.

Check status with:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 40 logs/rollout_distill_20260806/runner.log && tail -n 40 logs/rollout_teacher_v11all_rr_topk_20260806/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && echo quick=$(pgrep -af "generated_rollout_bc_rollout_search_nomcts_quick_20260806" | wc -l) teacher_waiter=$(pgrep -af "run_rollout_teacher_v11all_rr_topk_20260806" | wc -l) npz=$(find data/generated_rollout_bc_rollout_search_nomcts_quick_20260806 -name "*.npz" 2>/dev/null | wc -l)'
```

Interpretation as of the pivot:

- Pure `sample@T` rollout can discover some Marnie vs Ogerpon success games, but Dragapult vs Marnie/Crustle remains nearly barren.
- If the topk+targeted teacher batch still cannot generate wins for a matchup, that matchup likely needs a stronger hand-authored teacher or simulator-search primitive rather than more replay filtering.
- Do not train from the generated corpus until `tools/summarize_rollout_bc.py` shows adequate rows/episodes per target and actor modes are not dominated by fallback/random.

Remote dry-run confirmed `train_bc_population.py` can generate aux-distill commands, for example:

```bash
python3 tools/train_bc_population.py \
  --corpus data/bc_corpus_banded_v11_0724_0804 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --min-decisions 0 \
  --aux-corpus data/generated_rollout_bc_rollout_teacher_v11all_rr_topk_20260806 \
  --aux-score-bands weak_win_search \
  --aux-repeat 5 \
  --init-manifest logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv \
  --init-partial \
  --dry-run
```

Update at 2026-08-06 16:24:

- The earlier topk waiter did not launch the teacher batch. It had no active process and produced `0` `.npz` under `data/generated_rollout_bc_rollout_teacher_v11all_rr_topk_20260806`.
- A new immediate aggressive rollout batch was started:
  - Script: `/tmp/run_rollout_teacher_v11all_rr_aggressive_20260806.sh`
  - Runner log: `logs/rollout_teacher_v11all_rr_aggressive_20260806/runner.log`
  - Plan: `logs/rollout_teacher_v11all_rr_aggressive_20260806/plan.csv`
  - Output corpus: `data/generated_rollout_bc_rollout_teacher_v11all_rr_aggressive_20260806`
  - Config: 48 weak matchups, 720 games each, 12 workers per job, 4 parallel jobs, `max-turns=420`.
  - Actor portfolio: `greedy/topk@2/topk@3/topk@5/topk@8/sample/random` with `targeted` and `stage2_setup` rules.
  - At first check: about 52 `generate_rollout_bc` processes, first four jobs launched, `npz=0` because flush had not happened yet.
- Quick rollout-distill result was mixed/mostly negative:
  - Marnie quick distill worsened vs Ogerpon 5899/697 and only improved vs Ogerpon 2a.
  - Ogerpon2a quick distill had random WR only `83.4%`; do not submit or scale.
  - Cynthia quick distill improved vs Crustle in focused delta and kept random `97.6%`, but needs broad RR before any submission.

When aggressive rollout finishes, first run:

```bash
python3 tools/summarize_rollout_bc.py \
  --summary-glob 'logs/rollout_teacher_v11all_rr_aggressive_20260806/*.csv' \
  --corpus data/generated_rollout_bc_rollout_teacher_v11all_rr_aggressive_20260806 \
  --out-csv logs/rollout_teacher_v11all_rr_aggressive_20260806/rollout_teacher_summary.csv

python3 tools/filter_rollout_corpus.py \
  --in-root data/generated_rollout_bc_rollout_teacher_v11all_rr_aggressive_20260806 \
  --out-root data/generated_rollout_bc_rollout_teacher_v11all_rr_aggressive_20260806_filtered \
  --score-bands weak_win_search \
  --actor-exclude-regex 'fallback_random|epsilon_random|^random' \
  --min-rows 20
```

Update at 2026-08-06 16:47:

- User asked to pause aggressive rollout and keep enough notes to restart later.
- Aggressive rollout is stopped:
  - Parent process count: `0`
  - `generate_rollout_bc.py` child process count for this run: `0`
  - Output corpus `.npz` count: `0`
- It was paused because the first wave of very hard matchups had produced no wins and was slow enough to imply an 18-24 hour run for the full batch.
- Restart point if needed:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && setsid -f /tmp/run_rollout_teacher_v11all_rr_aggressive_20260806.sh >/tmp/run_rollout_teacher_v11all_rr_aggressive_20260806.nohup.log 2>&1 < /dev/null'
```

- Existing remote paths remain useful for later restart/analysis:
  - Script: `/tmp/run_rollout_teacher_v11all_rr_aggressive_20260806.sh`
  - Plan: `logs/rollout_teacher_v11all_rr_aggressive_20260806/plan.csv`
  - Generated job script: `logs/rollout_teacher_v11all_rr_aggressive_20260806/run_jobs.sh`
  - Runner log: `logs/rollout_teacher_v11all_rr_aggressive_20260806/runner.log`
  - Output corpus: `data/generated_rollout_bc_rollout_teacher_v11all_rr_aggressive_20260806`

## 2026-08-06 Deck-Sig Pure Specialists

Current pivot:

- Stop filtered BC / winner-only BC as the main line for now. Recent experiments showed these are too conservative and can overfit narrow slices without improving broad play.
- Next main line is pure deck-signature specialist BC:
  - For each archetype, rank deck signatures by available high-score decisions.
  - Train separate `sig1`, `sig2`, `sig3` specialists instead of mixing topK signatures into one policy.
  - Run model-size comparison with width `2`, `3`, and `4`.
  - Keep labels as clean as possible: filter by `--deck-sig`; do not add `--team-name` unless deliberately testing trajectory specialists.

Code update:

- `tools/plan_deck_specific_bc.py` now supports:
  - `--top-n-per-archetype N`: emit separate pure deck-signature jobs for top N signatures.
  - `--interleave-archetypes`: reorder the generated plan one signature per archetype per pass, avoiding several large signatures from the same archetype loading at once.
  - `--cuda-memory-gb`, `--cuda-memory-fraction`, `--init`, `--init-partial`, `--include-empty`, `--load-progress-every`.
  - `--set-loss-weight`, `--set-loss-min-count`, `--set-loss-negative-weight`.
  - `--card-weight`, `--opponent-deck-sig-weight`, and `--opponent-archetype-weight`.
  - Extra archetype slug mappings for `Archaludon`, `Iono Bellibolt`, and `N's Zoroark`.

Full 35-day v11 extraction:

- Remote raw episodes now contain `2026-07-01` through `2026-08-04`, 35 zip files total.
- New full corpus target: `data/bc_corpus_banded_v11_0701_0804`.
- Old comparison corpora must be kept:
  - `data/bc_corpus_banded_v11_0804_only`
  - `data/bc_corpus_banded_v11_0803_0804`
  - `data/bc_corpus_banded_v11_0724_0804`
- Remote extraction/planning job started at 2026-08-06 16:55:
  - Script: `/tmp/run_v11all35_extract_and_sig_specialist_plan_20260806.sh`
  - Runner log: `logs/deck_sig_specialists_v11all35_20260806/runner.log`
  - Extract log: `logs/deck_sig_specialists_v11all35_20260806/extract_v11_0701_0804.log`
  - Stats dir: `logs/deck_sig_specialists_v11all35_20260806/stats/`
  - Plans: `logs/deck_sig_specialists_v11all35_20260806/plan_w2.csv`, `plan_w3.csv`, `plan_w4.csv`
  - Train scripts: `logs/deck_sig_specialists_v11all35_20260806/train_w2.sh`, `train_w3.sh`, `train_w4.sh`
  - Checkpoints: `checkpoints/deck_sig_specialists_v11all35_20260806/w2`, `/w3`, `/w4`
  - Started with `TRAIN=0 WORKERS=12`, so extraction/stats/plans run first and training is not launched automatically.
- Leaderboard download in the script failed once and fell back to `logs/lb_snapshots/leaderboard_20260803_1530.csv` with `6174` teams. This is acceptable for score bands, but note that it is an older snapshot.
- At 16:56, extraction had started 12 workers and no `.npz` had been written yet.
- A width-2 training watcher was started at 2026-08-06 16:58:
  - Script: `/tmp/start_v11all35_sig_w2_after_plan_mem8_20260806.sh`
  - Log: `logs/deck_sig_specialists_v11all35_20260806/w2_mem8_watcher.log`
  - It waits for `.extract_done` and `train_w2.sh`, patches generated training commands from `--cuda-memory-gb 18`/`18.0` to `--cuda-memory-gb 8`, then launches `train_w2.sh`.
  - Width 3/4 are intentionally not auto-started.
  - At 16:58 it was still waiting: `extract_done=no train_script=no`.

Update at 2026-08-06 17:53:

- Full v11 all-data extraction completed:
  - `extract_done=Thu Aug  6 05:32:17 PM CST 2026`
  - Files: `1495` `.npz`
  - Size: `2.7G`
  - Shape check passed: state feature `(80,)`, option feature `(2, 64)`, `feature_version=v11_matchup_mechanic`, and `opponent_*` metadata present.
- Stats and plans completed:
  - `stats_done=Thu Aug  6 05:43:41 PM CST 2026`
  - `plan_done=Thu Aug  6 05:43:41 PM CST 2026`
  - Main plans: `plan_w2.csv`, `plan_w3.csv`, `plan_w4.csv`
  - Main w2 plan has `35` jobs across `12` archetypes. `Iono Bellibolt` and `N's Zoroark` are absent from 900+ because their total/high-score sample counts are too small.
- Main w2 was first launched by the watcher with `batch-size=1024` and `--cuda-memory-gb 8`.
  - This was too tight: Alakazam sig2/sig3 failed with PyTorch CUDA OOM at about `8.3-8.5GiB` process use while the allocator cap was `8GiB`.
  - This was a cap/batch issue, not physical A800 capacity. Physical free memory was still tens of GiB.
- The batch=1024 main w2 and the low-data watcher were stopped to avoid accumulating more OOM failed jobs.
- `tools/plan_deck_specific_bc.py` was updated again:
  - `--skip-existing`: generated scripts skip jobs whose final checkpoint already exists.
  - `--torch-cuda-alloc-conf`: generated scripts can export `PYTORCH_CUDA_ALLOC_CONF`.
- Resume job started at 2026-08-06 17:51:
  - Script: `/tmp/run_v11all35_w2_resume_bs512_mem7_20260806.sh`
  - Log dir: `logs/deck_sig_specialists_v11all35_20260806/w2_resume_bs512_mem7/`
  - Resume runner: `logs/deck_sig_specialists_v11all35_20260806/w2_resume_bs512_mem7/train_w2_resume_bs512_mem7.runner.log`
  - Checkpoint dir remains: `checkpoints/deck_sig_specialists_v11all35_20260806/w2`
  - Config: `batch-size=512`, `--cuda-memory-gb 7`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--skip-existing`.
  - At first check it skipped existing Archaludon and Crustle final checkpoints and was running Alakazam sig1/sig2/sig3 plus Cynthia sig1.
  - Stability check after about 7 minutes: no actual OOM/Traceback/FAILED in log files, checkpoint count `10`, and GPU memory stayed around `21-27GB` used per card.
- Low-data follow-up for missing archetypes was restarted:
  - Script: `/tmp/run_v11all35_lowdata_sig_w2_after_main_20260806.sh`
  - Log dir: `logs/deck_sig_specialists_v11all35_20260806/lowdata_w2/`
  - It waits for no active `v11all35_sigpure_top3_w2` process, then trains 4 low-data jobs:
    - Iono Bellibolt `142204e32bde`, `4073` decisions.
    - N's Zoroark `115babb20c85`, `7413` decisions.
    - N's Zoroark `085ceaad1b1d`, `7257` decisions.
    - N's Zoroark `6b9dd8aa6a69`, `3894` decisions.

Update at 2026-08-06 19:35:

- Width 2 completed:
  - Main checkpoints: `35/35` final checkpoints under `checkpoints/deck_sig_specialists_v11all35_20260806/w2`.
  - Low-data checkpoints: `4/4` final checkpoints under `checkpoints/deck_sig_specialists_v11all35_20260806/w2_lowdata`.
  - Old OOM errors remain in the first failed `train_w2.runner.log`/`w2/train_alakazam_*.log`; the `w2_resume_bs512_mem7` run completed.
- Width 3 was started with the safer runner:
  - Script: `/tmp/run_v11all35_sig_width_train_20260806.sh 3 384 8`
  - Log dir: `logs/deck_sig_specialists_v11all35_20260806/w3_bs384_mem8/`
  - Checkpoint dir: `checkpoints/deck_sig_specialists_v11all35_20260806/w3`
  - Config: `batch-size=384`, `--cuda-memory-gb 8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--skip-existing`.
- Width 4 was initially started concurrently with width 3 using:
  - Script: `/tmp/run_v11all35_sig_width_train_20260806.sh 4 256 8`
  - Log dir: `logs/deck_sig_specialists_v11all35_20260806/w4_bs256_mem8/`
  - Checkpoint dir: `checkpoints/deck_sig_specialists_v11all35_20260806/w4`
  - Config: `batch-size=256`, `--cuda-memory-gb 8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--skip-existing`.
- Running width 3 and width 4 together created 8 concurrent large corpus loaders. GPU memory was fine, but one width-3 Alakazam sig3 job was system-`Killed`, likely CPU RAM/process memory pressure rather than CUDA OOM.
- A later check showed width-3 Alakazam sig2 was also killed during the same initial burst. Treat these as initial concurrency failures, not as proof that width 3 cannot train.
- Width 4 main and width-4 lowdata were stopped. The width-4 main worker process group was `3605371`; it was killed with `kill -TERM -- -3605371` followed by `kill -KILL -- -3605371`.
- Width-3 lowdata watcher was also stopped to avoid racing the width-3 repair.
- The normal width runner was updated to pass `--interleave-archetypes`; future width-4 main training should not schedule Alakazam/Marnie signatures in a burst.
- New serial repair runner:
  - Script: `/tmp/run_v11all35_sig_width_repair_serial_20260806.sh`
  - Current repair command inside the controller: `/tmp/run_v11all35_sig_width_repair_serial_20260806.sh 3 256 8 0`
  - It uses one GPU worker, same checkpoint names, and `--skip-existing`, so it should only fill missing final checkpoints.
- Current staged controller:
  - Script: `/tmp/watch_repair_w3_then_w4_20260806.sh`
  - Controller log: `logs/deck_sig_specialists_v11all35_20260806/controller_w3_then_w4_20260806/controller.log`
  - Sequence:
    1. Wait until current width-3 main quiesces.
    2. Rerun width-3 main serial repair via `/tmp/run_v11all35_sig_width_repair_serial_20260806.sh 3 256 8 0`; `--skip-existing` should repair only missing/failed jobs.
    3. Run width-3 lowdata via `/tmp/run_v11all35_sig_lowdata_width_train_20260806.sh 3 384 8`.
    4. Run width-4 main via `/tmp/run_v11all35_sig_width_train_20260806.sh 4 256 8`; this now uses interleaved archetype scheduling and existing partial final checkpoints will be skipped.
    5. Run width-4 lowdata via `/tmp/run_v11all35_sig_lowdata_width_train_20260806.sh 4 256 8`.
- Do not use the old generated `logs/deck_sig_specialists_v11all35_20260806/train_w3.sh` or `train_w4.sh` directly; those still contain the earlier `batch-size=1024`/`--cuda-memory-gb 18` style settings.

Current deck-sig specialist monitor commands:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/deck_sig_specialists_v11all35_20260806/controller_w3_then_w4_20260806/controller.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && echo w3_final=$(find checkpoints/deck_sig_specialists_v11all35_20260806/w3 -maxdepth 1 -name "bc2_*_v11all35_sigpure_top3_w3.npz" 2>/dev/null | grep -v "_ep[0-9]" | wc -l) w4_final=$(find checkpoints/deck_sig_specialists_v11all35_20260806/w4 -maxdepth 1 -name "bc2_*_v11all35_sigpure_top3_w4.npz" 2>/dev/null | grep -v "_ep[0-9]" | wc -l) w3_low=$(find checkpoints/deck_sig_specialists_v11all35_20260806/w3_lowdata -maxdepth 1 -name "bc2_*.npz" 2>/dev/null | grep -v "_ep[0-9]" | wc -l) w4_low=$(find checkpoints/deck_sig_specialists_v11all35_20260806/w4_lowdata -maxdepth 1 -name "bc2_*.npz" 2>/dev/null | grep -v "_ep[0-9]" | wc -l)'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && find logs/deck_sig_specialists_v11all35_20260806/w3_bs384_mem8 logs/deck_sig_specialists_v11all35_20260806/w4_bs256_mem8 logs/deck_sig_specialists_v11all35_20260806/controller_w3_then_w4_20260806 -type f -name "*.log" -print0 | xargs -0 grep -Hn "FAILED\|OutOfMemory\|Traceback\|Killed" | sed -n "1,160p"'
```

Width-2 evaluation update at 2026-08-06 20:18:

- Manifest/eval files:
  - `logs/eval_deck_sig_specialists_v11all35_20260806/manifest_w2_all39.csv`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/random_w2_all39_g200.csv`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/random_w2_all39_g200_joined.csv`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/random_w2_all39_summary.txt`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/candidate_manifest_w2_random_ge097.csv`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/rr_w2_random_ge097_g80.csv`
  - `logs/eval_deck_sig_specialists_v11all35_20260806/rr_w2_random_ge097_g80_summary.txt`
- Random g200:
  - Evaluated `26/39` checkpoints. The other `13` currently lack deck CSVs in the known 0802/0803/0804 ladder pool deck dirs, so they cannot be safely evaluated or packaged yet.
  - Mean `0.913`, median `0.965`, min `0.545`, max `1.000`.
  - `12` checkpoints are `>=0.970`; `7` are `<0.900`.
  - Strong random candidates: Alakazam sig1/sig2/sig3, Cynthia sig1, Festival sig1/sig2, Marnie sig1/sig2, Mega Lopunny sig3, Ogerpon sig1/sig3, Crustle sig1.
  - Weak random rows: Archaludon sig3 `0.545`, Cynthia sig2 `0.680`, Dragapult sig2/sig3 `0.785/0.750`, Crustle sig2/sig3 `0.860/0.805`, Ogerpon sig2 `0.825`.
- RR g80 among the 12 random-stable candidates:
  - Best weighted/average rows:
    - Crustle sig1 `3cd5039c`: avg `0.652`, worst `0.263` vs Marnie sig1.
    - Ogerpon sig3 `5899c772`: avg `0.590`, worst `0.013` vs Crustle sig1.
    - Marnie sig1 `b8f251a4`: avg `0.583`, worst `0.125` vs Ogerpon sig3.
    - Alakazam sig1 `7f9a5389`: avg `0.582`, worst `0.300` vs Marnie sig1.
    - Ogerpon sig1 `697a82e5`: avg `0.562`, worst `0.050` vs Crustle sig1.
  - Festival sig2 has perfect random but poor RR avg `0.312`; do not submit based on random alone.
  - Marnie still has the known structural Ogerpon weakness: sig1/sig2 are only about `0.075-0.138` into Ogerpon sig1/sig3 in this small RR.
  - Ogerpon still has the known Crustle weakness: Ogerpon sig1/sig3 are about `0.050/0.013` into Crustle sig1.
- Submission interpretation:
  - If spending a Kaggle slot purely as a probe, the only width-2 model that looks locally strongest is Crustle sig1 `3cd5039c`.
  - Secondary probes: Ogerpon sig3 `5899c772`, Marnie sig1 `b8f251a4`, Alakazam sig1 `7f9a5389`. These have clear known bad matchups and should not be treated as ladder-safe.
  - Do not submit unchecked missing-deck rows, Festival sig2, weak Dragapult/Crustle/Ogerpon sig2 rows, or any row with random `<0.95`.

Check progress:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 80 logs/deck_sig_specialists_v11all35_20260806/extract_v11_0701_0804.log && tail -n 40 logs/deck_sig_specialists_v11all35_20260806/runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && echo done=$(test -f data/bc_corpus_banded_v11_0701_0804/.extract_done && echo yes || echo no) npz=$(find data/bc_corpus_banded_v11_0701_0804 -name "*.npz" 2>/dev/null | wc -l)'
```

After the plans are generated, reduce training memory from the current generated default of 18GB to 8GB while LLaMA Factory is active:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && python3 - <<'"'"'PY'"'"'
from pathlib import Path
root = Path("logs/deck_sig_specialists_v11all35_20260806")
for path in [root / "train_w2.sh", root / "train_w3.sh", root / "train_w4.sh"]:
    if not path.exists():
        print("missing", path)
        continue
    text = path.read_text()
    text = text.replace("--cuda-memory-gb 18.0", "--cuda-memory-gb 8")
    text = text.replace("--cuda-memory-gb 18", "--cuda-memory-gb 8")
    path.write_text(text)
    print("patched", path)
PY'
```

Then start width 2 first:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && setsid -f bash logs/deck_sig_specialists_v11all35_20260806/train_w2.sh > logs/deck_sig_specialists_v11all35_20260806/train_w2.runner.log 2>&1 < /dev/null'
```

The watcher above should start width 2 automatically, so run this manually only if the watcher is stopped. Only start width 3/4 after width 2 has no failed jobs and GPU memory remains stable.

## 2026-08-06 Deck-Sig Width 3/4 Status

Remote controller:

```text
script: /tmp/watch_repair_w3_then_w4_20260806.sh
log: logs/deck_sig_specialists_v11all35_20260806/controller_w3_then_w4_20260806/controller.log
checkpoints:
  checkpoints/deck_sig_specialists_v11all35_20260806/w3/
  checkpoints/deck_sig_specialists_v11all35_20260806/w3_lowdata/
  checkpoints/deck_sig_specialists_v11all35_20260806/w4/
  checkpoints/deck_sig_specialists_v11all35_20260806/w4_lowdata/
```

Status checked at `2026-08-06 23:20 Asia/Shanghai`:

- Width 3 main completed: `35/35`.
- Width 3 lowdata completed: `4/4`, saved under `w3_lowdata/`.
- Width 4 main completed: `35/35`.
- Width 4 lowdata completed: `4/4`, saved under `w4_lowdata/`.
- No BC training jobs from this runner remained active.
- There were earlier `Killed` events for Alakazam width 3 repairs, but final
  width 3 checkpoints exist and load; treat the killed logs as process/memory
  pressure during repair, not as corrupt completed artifacts.

Width 3 random g200 outputs:

```text
logs/eval_deck_sig_specialists_v11all35_20260806/manifest_w3_all39.csv
logs/eval_deck_sig_specialists_v11all35_20260806/random_w3_all39_g200.csv
logs/eval_deck_sig_specialists_v11all35_20260806/random_w3_all39_g200_joined.csv
logs/eval_deck_sig_specialists_v11all35_20260806/random_w3_all39_summary.txt
```

Width 3 random summary:

- Evaluated `26/39`; same 13 rows are blocked by missing deck CSVs:
  Archaludon sig1/2, Marnie sig3, Mega Lucario sig1/3, Mega Starmie sig1/2/3,
  Team Rocket Mewtwo sig2, Iono Bellibolt sig1, and all three N's Zoroark
  lowdata signatures.
- Mean `0.932`, median `0.980`, min `0.595`, max `1.000`.
- `>=0.97`: 15 entries; `>=0.95`: 18 entries; `<0.90`: 7 entries.
- Compared to width 2 random, width 3 is clearly better on random:
  mean `0.913 -> 0.932`, median `0.965 -> 0.980`, and `>=0.97` count
  `12 -> 15`.

Width 3 best random rows by archetype:

```text
Alakazam sig1 7f9a5389: 1.000
Archaludon sig3 22aed761: 0.595
Crustle Wall sig1 3cd5039c: 0.975
Cynthia Garchomp sig1 52f46739: 1.000
Dragapult sig1 0b7b5a14: 0.965
Festival Lead sig2 41ffa789: 1.000
Marnie Grimmsnarl sig1 b8f251a4: 1.000
Mega Lopunny sig3 f1445356: 0.995
Mega Lucario sig2 43d6d8b0: 0.920
Teal Mask Ogerpon sig3 5899c772: 0.995
Team Rocket Mewtwo sig1 06f0b265: 1.000
```

Width 3 random-stable RR g80 outputs:

```text
logs/eval_deck_sig_specialists_v11all35_20260806/candidate_manifest_w3_random_ge097.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w3_random_ge097_g80.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w3_random_ge097_g80_summary.txt
```

Width 3 RR top rows among 15 random-stable candidates:

```text
Crustle sig1 3cd5039c: avg=0.622 worst=0.263 losses=6
Marnie sig1 b8f251a4: avg=0.604 worst=0.150 losses=2
Alakazam sig1 7f9a5389: avg=0.596 worst=0.062 losses=4
Ogerpon sig3 5899c772: avg=0.557 worst=0.037 losses=7
Festival sig1 e82dcbe6: avg=0.546 worst=0.375 losses=6
Ogerpon sig1 697a82e5: avg=0.542 worst=0.025 losses=7
```

Interpretation:

- Width 3 is a useful random-stability improvement over width 2, but not a
  clean RR improvement. Its RR pool is larger and harder (`15` entries vs
  width 2's `12`), so compare trends rather than raw avg alone.
- Crustle sig1 remains the strongest local deck-sig candidate and is still the
  first Kaggle probe candidate from this axis.
- Marnie sig1 improved as a local RR candidate (`0.583` width 2 to `0.604`
  width 3), but the Ogerpon weakness remains severe: `0.200` vs Ogerpon 5899
  and `0.150` vs Ogerpon 697 in the width 3 RR.
- Alakazam sig1 has excellent random, but the RR worst row is Team Rocket
  Mewtwo sig1 at `0.062`; do not treat it as ladder-safe without a live probe.
- Team Rocket Mewtwo sig1 is polarizing: it crushes Alakazam in this RR but
  loses badly to Ogerpon and Mega Lopunny; average only `0.475`.

Width 4 random/RR outputs were saved locally and remotely:

```text
logs/eval_deck_sig_specialists_v11all35_20260806/random_w4_all39_g200.csv
logs/eval_deck_sig_specialists_v11all35_20260806/random_w4_all39_g200_joined.csv
logs/eval_deck_sig_specialists_v11all35_20260806/random_w4_all39_summary.txt
logs/eval_deck_sig_specialists_v11all35_20260806/candidate_manifest_w4_random_ge097.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w4_random_ge097_g80.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w4_random_ge097_g80_summary.txt
logs/eval_deck_sig_specialists_v11all35_20260806/submit_analysis_w4_0805.txt
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w4_recovered_mini_g100.csv
logs/eval_deck_sig_specialists_v11all35_20260806/rr_w4_recovered_mini_g100_summary.txt
```

Width 4 random g200:

- Evaluated `26/39`; same 13 rows are blocked by missing deck CSVs.
- Mean `0.950`, median `0.988`, min `0.620`, max `1.000`.
- `>=0.97`: 17 entries; `>=0.95`: 18 entries; `<0.90`: 5 entries.
- It is the best random-stability width so far, but still weak for
  Archaludon, Dragapult sig2/sig3, Cynthia sig2, and Ogerpon sig2.

Width 4 RR g80 among 17 random-stable candidates:

```text
Crustle sig1 3cd5039c: avg=0.630 worst=0.325 losses=6
Marnie sig1 b8f251a4: avg=0.630 worst=0.175 losses=3
Alakazam sig1 7f9a5389: avg=0.618 worst=0.113 losses=3
Ogerpon sig3 5899c772: avg=0.598 worst=0.025 losses=6
Ogerpon sig1 697a82e5: avg=0.580 worst=0.000 losses=6
Marnie sig2 2c22fa76: avg=0.567 worst=0.087 losses=3
Mega Lopunny sig2 276707c0: avg=0.545 worst=0.075 losses=6
Mega Lopunny sig3 f1445356: avg=0.544 worst=0.113 losses=6
```

Interpretation:

- Width 4 improves random and keeps the same top local shape: Crustle sig1,
  Marnie sig1, and Alakazam sig1 are the strongest internal RR rows.
- Width 4 does not solve structural weaknesses:
  Ogerpon still loses almost completely to Crustle (`0.025` for 5899 and
  `0.000` for 697), Marnie still loses hard to Ogerpon (`0.175/0.188`),
  Alakazam still loses hard to Team Rocket Mewtwo (`0.113`), and Dragapult sig1
  collapses in RR despite random `0.970`.
- Submission priority from this width axis:
  1. `w4 Crustle sig1 3cd5039c` as the strongest local RR probe.
  2. `w4 Marnie sig1 b8f251a4` as the broad/local co-top probe.
  3. `w4 Alakazam sig1 7f9a5389` only as a controlled probe; the current 0805
     environment has many Marnie games and Alakazam is not well positioned.
  4. `w4 Mega Lopunny sig3/f144` only as a meta-prior probe; local RR is
     mediocre, but 0805 ladder archetype prior favors Mega Lopunny.
  Avoid w4 Ogerpon for today's first probe because Crustle remains common
  enough and the local Crustle matchup is catastrophic.

After building `ladder_pool_0805_all`, two previously missing w4 signatures
became evaluable:

- `Mega Starmie sig1 e2f9eb4c`: random g200 `1.000`, but recovered mini RR
  avg `0.423`, worst `0.130` vs Marnie. Do not submit.
- `Team Rocket Mewtwo sig2 206a1cf0`: random g200 `1.000`, but recovered mini
  RR avg `0.230`, worst `0.010` vs Ogerpon 5899. Do not submit.

Recovered mini RR g100 ranking:

```text
Crustle sig1 3cd5039c: avg=0.634 worst=0.300
Alakazam sig1 7f9a5389: avg=0.587 worst=0.310
Marnie sig1 b8f251a4: avg=0.569 worst=0.140
Ogerpon sig3 5899c772: avg=0.517 worst=0.000
Mega Lopunny sig3 f1445356: avg=0.516 worst=0.290
Ogerpon sig1 697a82e5: avg=0.513 worst=0.030
Mega Starmie sig1 e2f9eb4c: avg=0.423 worst=0.130
TR Mewtwo sig2 206a1cf0: avg=0.230 worst=0.010
```

Submission tarballs generated on `ks`:

```text
submissions/w4_0805_candidates/w4_crustle_sig1_3cd5039c.tar.gz
submissions/w4_0805_candidates/w4_marnie_sig1_b8f251a4.tar.gz
submissions/w4_0805_candidates/w4_alakazam_sig1_7f9a5389.tar.gz
submissions/w4_0805_candidates/w4_mega_lopunny_sig3_f1445356.tar.gz
```

Kaggle w4 submissions:

```text
55302986 bc: w4_0805_candidates/w4_marnie_sig1_b8f251a4.tar.gz score=976.8
55303028 bc: w4_0805_candidates/w4_crustle_sig1_3cd5039c score=915.0
```

Replay analysis saved locally and remotely:

```text
logs/kaggle_replay_w4_submit_20260807/
logs/kaggle_replay_w4_submit_20260807/w4_kaggle_vs_local_summary.txt
```

Only `60` Marnie and `57` Crustle replay episodes were exposed by Kaggle at
the 2026-08-07 morning pull; one same-team episode per submission was skipped by
team-name matching. Deck-only matching loses same-deck mirrors, so use the
team-name rows as the main result.

Replay vs local RR:

- Marnie `55302986`: `36/59 = 0.610` in replay, while w4 local RR avg was
  `0.630`. Overall direction is close, matching the 976.8 score.
- Marnie first/second split: first `20/29 = 0.690`, second `16/30 = 0.533`.
- Marnie online mismatch:
  - vs Alakazam overall `8/16 = 0.500`; vs 7f sig `4/9 = 0.444`.
    Local RR expected Marnie vs Alakazam 7f around `0.700`; 0805 episode prior
    was `0.632` for b8f vs 7f. Local validation overestimated this matchup.
  - vs Mega Lopunny `1/5 = 0.200`; vs f144 sig `0/3`.
    Local RR expected Marnie vs f144 around `0.625`; 0805 episode prior also
    showed Mega Lopunny beating Marnie. This is the main reason Marnie did not
    go higher despite a good score.
  - vs Crustle `6/7 = 0.857`, better than local and 0805 prior.
- Crustle `55303028`: `37/56 = 0.661` in replay, while w4 local RR avg was
  `0.630`. The online win rate is consistent or slightly better than local, but
  the score is lower because opponent/rating mix matters.
- Crustle first/second split: first `19/30 = 0.633`, second `18/26 = 0.692`.
- Crustle online matchup:
  - vs Marnie overall `13/20 = 0.650`; vs b8f sig `8/15 = 0.533`.
    Local RR expected Crustle vs Marnie b8f only `0.325`, so local RR
    underestimated Crustle here.
  - vs Alakazam `6/8 = 0.750`; vs 7f sig `4/6 = 0.667`.
    Local RR expected `0.713`, consistent.
  - vs Ogerpon `2/2 = 1.000`, consistent with the known Crustle > Ogerpon edge.
  - vs Mega Lucario `2/6 = 0.333`; this is the important live weakness not
    covered by the 17-entry w4 RR pool.

Interpretation: local RR correctly ranked Marnie/Crustle as the best w4 probes,
but it is not matchup-calibrated enough. It overestimated Marnie into
Alakazam/Mega Lopunny, underestimated Crustle into Marnie, and missed live Mega
Lucario pressure. Add Mega Lucario signatures such as `ab089ccfad1a` to the
local environment pool before using RR as a Kaggle score predictor.

## 2026-08-07 High-Score Disadvantage Replay Audit

User asked whether 1000+/1100+/1200+ Kaggle players can win their bad
matchups. Remote latest episode data currently only goes through 0805:

```text
/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-05.zip
```

Ran high-score replay analysis on `ks` and pulled outputs locally:

```text
logs/high_score_disadvantage_20260807/0805/
```

Generated files:

```text
threshold_summary.csv
exact_band_summary.csv
archetype_priors_score900.csv
disadvantage_matchups_by_threshold.csv
teams_high_score_disadvantage.csv
high_score_player_rows.csv
teams_by_archetype_high_score_disadvantage.csv
teams_by_decksig_high_score_disadvantage.csv
summary.md
```

Definitions used:

- Player-side rows from 0805 daily Kaggle replay episodes; one game contributes
  two rows.
- Only rows where both teams had current leaderboard score >=900 were used for
  matchup priors.
- Hard disadvantage = archetype prior games >=30, non-mirror, and prior WR
  <=0.45.
- A separate soft check with WR <=0.50 was computed from
  `high_score_player_rows.csv` to handle borderline cases such as Mega Lucario
  vs Marnie.

Hard disadvantage summary:

```text
1000+: overall 1635/3150 = 0.519; bad-matchup 347/930 = 0.373; share 0.295
1100+: overall  628/1110 = 0.566; bad-matchup 120/305 = 0.393; share 0.275
1200+: overall    77/118 = 0.653; bad-matchup   3/13  = 0.231; share 0.110
```

Exact bands:

```text
1000-1099: overall 1007/2040 = 0.494; bad-matchup 227/625 = 0.363
1100-1199: overall  551/992  = 0.555; bad-matchup 117/292 = 0.401
1200+:     overall   77/118  = 0.653; bad-matchup   3/13  = 0.231
```

Important interpretation:

- High score does not mean the agent turns bad matchups into good matchups.
  1000+/1100+ players improve bad-matchup win rate only modestly versus the
  global hard-prior baseline, and still remain below 50% overall in those rows.
- 1100+ players show real lift in some specific bad matchups:
  - Marnie vs Mega Lopunny: `37/71 = 0.521`, prior `0.376`.
  - Marnie vs Festival Lead: `10/14 = 0.714`, prior `0.322` but small sample.
  - Dragapult vs Crustle: `9/16 = 0.563`, prior `0.364`.
  - Alakazam vs Marnie: `11/21 = 0.524`, prior `0.401`.
- Some hard weaknesses remain hard even at high score:
  - Marnie vs Ogerpon at 1100+: `8/30 = 0.267`, prior `0.237`.
  - Mega Lopunny vs Mega Lucario at 1100+: `1/29 = 0.034`, prior `0.087`.
  - Crustle vs Alakazam at 1100+: `3/18 = 0.167`, prior `0.333`.

The 1200+ bucket is only `Majkel1337` in this snapshot, so treat it as a
single-player case study, not a population estimate:

```text
Majkel1337 Mega Lucario sig 43d6d8b0fce9:
  overall 73/103 = 0.709
  hard-disadvantage rows: 0
  main opponents: Marnie 40, Mega Lopunny 33, Alakazam 15

Majkel1337 Ogerpon sig 697a82e582d5:
  overall 4/15 = 0.267
  hard-disadvantage 3/13 = 0.231
```

With soft disadvantage WR <=0.50, 1200+ changes to:

```text
1200+: overall 77/118 = 0.653; soft bad-matchup 20/53 = 0.377; share 0.449
Mega Lucario vs Marnie: 17/40 = 0.425, prior 0.453
Ogerpon bad matchups: 3/13 = 0.231
```

This says the 1200+ score came mostly from strong Mega Lucario global matchup
coverage and opponent mix, not from reliably overcoming hard counters.

Useful successful bad-matchup replay sources for imitation/trace mining:

```text
Raihan Ramadistra Marnie b8f251a476e7:
  overall 100/168 = 0.595; bad-matchup 34/67 = 0.508
  pairs: Marnie<=Mega Lopunny 38, <=Ogerpon 14, <=Dragapult 12, <=Festival 3

flg Dragapult cc2e995b5ad0:
  overall 52/80 = 0.650; bad-matchup 11/21 = 0.524
  pairs: Dragapult<=Mega Lopunny 17, <=Crustle 4

flg Alakazam 791e3c4c20f4:
  overall 28/49 = 0.571; bad-matchup 11/22 = 0.500
  pairs: Alakazam<=Marnie 21, <=Dragapult 1

James Cox & Henry Chao Ogerpon 2bd9da52c43a:
  overall 71/113 = 0.628; bad-matchup 31/56 = 0.554
  pairs: Ogerpon<=Alakazam 26, <=Mega Lopunny 13, <=Dragapult 10, <=Crustle 7

MissingNo. Marnie b8f251a476e7:
  overall 15/25 = 0.600; bad-matchup 7/10 = 0.700
  small sample, but useful for trace mining.
```

Conclusion for pipeline:

- Use `high_score_player_rows.csv` to extract successful bad-matchup replay
  seeds for trace/teacher mining.
- Do not assume high score alone implies bad-matchup mastery. For training,
  filter by `(won == 1 and is_disadvantage == 1)` and then group by
  `team/archetype/deck_sig/pair`.
- Add a soft-disadvantage option (`prior_wr <= 0.50`) for near-even but
  practically difficult pairs, especially Mega Lucario vs Marnie.

Additional seed files generated locally and synced back to `ks`:

```text
logs/high_score_disadvantage_20260807/0805/success_disadvantage_seeds.csv
logs/high_score_disadvantage_20260807/0805/success_disadvantage_summary_1000plus.csv
logs/high_score_disadvantage_20260807/0805/success_disadvantage_summary_1100plus.csv
logs/high_score_disadvantage_20260807/0805/success_disadvantage_summary_1200plus.csv
```

Counts:

```text
success_disadvantage_seeds.csv rows=470
success_disadvantage_summary_1000plus.csv rows=123
success_disadvantage_summary_1100plus.csv rows=31
success_disadvantage_summary_1200plus.csv rows=2
```

Recommended first trace/teacher-mining queue from 1100+ successful hard
disadvantage seeds:

```text
Raihan Ramadistra Marnie b8f251a476e7 vs Mega Lopunny: 20/38 wins
flg Dragapult cc2e995b5ad0 vs Mega Lopunny: 11/17 wins
flg Alakazam 791e3c4c20f4 vs Marnie: 11/21 wins
RtoABC Dragapult 46ceec8cc5ae vs Crustle: 7/9 wins
Raihan Ramadistra Marnie b8f251a476e7 vs Dragapult: 7/12 wins
Sixth Sense Marnie b8f251a476e7 vs Mega Lopunny: 7/12 wins
Raihan Ramadistra Marnie b8f251a476e7 vs Ogerpon: 6/14 wins
```

The 1200+ success seed file only has two useful Ogerpon successes
(`Ogerpon<=Alakazam` and `Ogerpon<=Dragapult`), so it should be used as
reference material, not as a training pool.

## 2026-08-06 Episode 0805 Ladder Read

The 0805 episode zip exists on `ks`:

```text
/home/jie/Do/0_PTCG/workspace/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-05.zip
```

Built and pulled locally:

```text
logs/ladder_pool_0805_all/
logs/matchup_notes_20260806/0805_score900/
```

0805 pool generation processed `4740` episodes and selected `132` deck
signatures. Current leaderboard scores were available (`leaderboard_teams=6174`).
The high-score matchup rerun used `4053` games after filtering.

Score>=900 archetype weight shares:

```text
Marnie Grimmsnarl 30.5%
Alakazam          15.7%
Mega Lopunny      13.3%
Crustle Wall       9.2%
Teal Mask Ogerpon  8.5%
Dragapult          7.8%
Mega Lucario       5.8%
Festival Lead      2.8%
Team Rocket Mewtwo 1.6%
Cynthia Garchomp   1.5%
```

Important 0805 archetype edges from actual ladder episodes, games>=30:

```text
Mega Lucario > Mega Lopunny      0.920 over 75 games
Crustle Wall > Teal Mask Ogerpon 0.883 over 60 games
Ogerpon > Marnie                 0.825 over 252 games
Mega Lopunny > Crustle           0.822 over 90 games
Festival Lead > Marnie           0.689 over 74 games
Alakazam > Crustle               0.682 over 154 games
Crustle > Dragapult              0.667 over 57 games
Mega Lopunny > Ogerpon           0.648 over 108 games
Alakazam > Ogerpon               0.647 over 153 games
Mega Lopunny > Marnie            0.634 over 309 games
Dragapult > Marnie               0.596 over 161 games
Marnie > Alakazam                0.596 over 527 games
Mega Lucario > Alakazam          0.586 over 58 games
```

Current Kaggle score refresh:

- `jie` account refreshed successfully at `2026-08-06T15:16:22+00:00`.
- Latest two `jie` submissions then were
  `55294036 shadow_crustle_wall_96d57241_liamk` score `827.6` and
  `55294007 shadow_mega_lopunny_f1445356_ntumlnoob_w2` score `626.5`.
  These active slots are weak and worth replacing if submit budget permits.
- The remote `by` account currently fails Kaggle CLI auth with
  `Authentication required`; `/root/.kaggle/by` needs refreshed Kaggle auth or
  an accepted token format before current scores can be checked or monitored.

## 2026-08-07 Cross-Attention Architecture Wave 1

User asked for a more aggressive model-architecture step because trajectory
auxiliary heads and weak-matchup-filtered BC did not teach continuous plans.
Implemented a new stateless architecture first, before full game-history
models, because submission inference is currently pure NumPy and Kaggle agent
resources are CPU-only.

Code changes:

- `ptcg_rl/model.py`: added `CrossAttentionPolicyValueNet`, `build_policy_model`,
  `checkpoint_arch`, and `checkpoint_feature_dims`.
- `ptcg_rl/numpy_policy.py`: `NumpyPolicy.load()` now auto-detects cross-attn
  checkpoints and runs matching NumPy state-token and option cross-attention
  inference.
- `tools/bc2_train.py`: added `--arch pointer|cross_attn` and `--state-layers`.
- `tools/bc2_accuracy.py` and `tools/bc2_failure_report.py`: auto-load pointer
  or cross-attn checkpoints.
- `tools/train_bc_population.py` and `tools/build_shadow_pool.py`: pass through
  `--arch` and `--state-layers`; shadow manifest records `arch`.

Architecture:

- Tokens: one scalar feature token, 12 board-card tokens, 25 hand-card tokens.
- Board/hand tokens use card embedding plus state area/index embeddings.
- State tokens pass through small self-attention blocks.
- Legal option embeddings cross-attend to state tokens before the existing
  autoregressive pointer scorer.
- Public model interface stays compatible with `sequence_nll`,
  `greedy_decode`, PPO utilities, and diagnostics.

Submission/resource notes:

- Kaggle official FAQ shows submission size limit as 197.7 MiB after page
  rendering; current w4 cross-attn submission package is only 33.3 MiB.
- Remote parameter counts:

```text
cross_attn w1: 0.67M params
cross_attn w2: 2.38M params
cross_attn w3: 5.11M params
cross_attn w4: 8.88M params
```

Smoke checks completed:

```text
python3 -m py_compile ptcg_rl/model.py ptcg_rl/numpy_policy.py tools/bc2_train.py ...
PYTHONPATH=.:.. python3 /tmp/ptcg_cross_attn_remote_smoke.py
numpy_cross_attn_smoke_ok steps=30
```

Real corpus smoke:

- Script: `/tmp/run_cross_attn_smoke_train.sh`
- Trained `Mega Lucario` 43d6 on `data/bc_corpus_banded_v11_0803_0804`
  for one epoch, width 1.
- Saved:
  `checkpoints/arch_smoke/bc2_mega_lucario_43d_cross_attn_w1_smoke.npz`
- Accuracy smoke loaded successfully and reported exact `0.398`, first `0.411`,
  top3 `0.760` on 4096 samples.

Lucario 43d wave1:

- Script: `/tmp/run_lucario43d_cross_attn_wave1.sh`
- Corpus: `data/bc_corpus_banded_v11_0701_0804`
- Target: `Mega Lucario`, `deck_sig=43d6d8b0fce9`, bands
  `1200+ 1100-1199 1000-1099 900-999`
- Output directory:
  `checkpoints/arch_cross_attn_20260807/`
- Logs:
  `logs/arch_cross_attn_20260807/`

Trained two w4 models, 16 epochs, batch 2048, 24 GiB cap:

```text
bc2_mega_lucario_43d6_cross_w4_scratch.npz
  best val=1.0669
  accuracy exact=0.669 first=0.685 top3=0.923 set_f1=0.783

bc2_mega_lucario_43d6_cross_w4_init_tempo.npz
  init from checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_strategy_tempo_w4.npz
  partial init loaded 20 tensors, skipped old state_fc1/state_fc2 tensors
  best val=1.0378
  accuracy exact=0.661 first=0.676 top3=0.918 set_f1=0.788
```

Evaluation script:

```text
/tmp/eval_lucario43d_cross_attn_wave1.sh
logs/arch_cross_attn_20260807/eval/
```

Local eval results:

```text
baseline strategy_tempo_w4:
  random g300 = 278/300 = 92.7%
  vs Marnie b8f w4 g300 = 29/300 = 9.7%

cross_w4_scratch:
  random g300 = 297/300 = 99.0%
  vs Marnie b8f w4 g300 = 39/300 = 13.0%

cross_w4_init_tempo:
  random g300 = 299/300 = 99.7%
  vs Marnie b8f w4 g300 = 33/300 = 11.0%
```

Interpretation:

- Cross-attn is a real positive architecture signal for Lucario 43d: random
  stability improved sharply and focused Marnie matchup improved from `9.7%` to
  `13.0%` for scratch.
- It does not solve structural weak matchups by itself. The gain is not enough
  to claim continuous-plan learning, but it is a stronger base than the pointer
  policy for the next wave.
- Scratch beat partial-init on focused Marnie despite worse val NLL; do not rely
  on validation loss alone for architecture selection.
- Cross-attn NumPy inference is slower than pointer but still practical in
  local eval (`~26 games/s` vs random with 16 workers; focused RR around
  `11-12 games/s`).

Recommended next architecture work:

1. Do not replace the current w4 pointer specialists with the first cross-attn
   wave. The architecture works in code and improved Lucario random stability,
   but the first multi-deck wave lost to old w4 specialists in paired delta.
2. If continuing cross-attn, train it on the full `v11_0701_0804` corpus or
   with teacher/distillation from the current w4 specialists. The first wave2
   used only `v11_0803_0804`, which is a likely reason it lost global quality.
3. Keep `--set-loss-weight 0` as the default for cross-attn until there is a
   stronger counterexample. `0.05` did not improve Marnie, Ogerpon5899,
   Crustle b141, Crustle3cd, or Dragapult in this wave.
4. Only after cross-attn has a baseline that does not regress against w4, add
   real history input from observation game logs or agent-maintained recent
   action tokens. Otherwise history-vs-attention effects will be mixed.

Cross-attn wave2 core results:

- Script: `/tmp/run_cross_attn_wave2_core.sh`
- Corpus: `data/bc_corpus_banded_v11_0803_0804`
- Output: `checkpoints/arch_cross_attn_wave2_20260807/`
- Logs: `logs/arch_cross_attn_wave2_20260807/`
- Trained w4 `cross_attn` set000 and set050 for:
  Marnie b8f, Ogerpon697, Ogerpon5899, Crustle b141, Crustle3cd, Dragapult cc2.

Offline accuracy summary:

```text
name                                      exact first top3 set_f1
crustle_3cd_cross_w4_set000              0.564 0.573 0.889 0.683
crustle_3cd_cross_w4_set050              0.553 0.560 0.885 0.686
crustle_b141_cross_w4_set000             0.651 0.658 0.914 0.794
crustle_b141_cross_w4_set050             0.640 0.647 0.911 0.792
dragapult_cc2_cross_w4_set000            0.665 0.675 0.919 0.759
dragapult_cc2_cross_w4_set050            0.646 0.656 0.914 0.740
marnie_b8f_cross_w4_set000               0.746 0.760 0.957 0.777
marnie_b8f_cross_w4_set050               0.744 0.758 0.957 0.775
ogerpon_5899_cross_w4_set000             0.874 0.879 0.985 0.958
ogerpon_5899_cross_w4_set050             0.873 0.877 0.986 0.956
ogerpon_697_cross_w4_set000              0.674 0.682 0.903 0.824
ogerpon_697_cross_w4_set050              0.675 0.684 0.906 0.822
```

Random g500 for set000:

```text
marnie_b8f_cross_w4_set000     99.6% 498/500
ogerpon_697_cross_w4_set000    98.2% 491/500
ogerpon_5899_cross_w4_set000   99.8% 499/500
crustle_b141_cross_w4_set000   75.4% 377/500
crustle_3cd_cross_w4_set000    79.2% 396/500
dragapult_cc2_cross_w4_set000  65.6% 328/500
```

Paired baseline-delta versus the existing w4 specialist for the same deck,
using `candidate_manifest_w4_random_ge097.csv`, 17 opponents, 80 games each:

```text
marnie_cross:
  avg_delta=-0.076 candidate=0.535 baseline=0.611
  worst=marnie_grimmsnarl_sig1_b8f251a4:-0.163 lost=14/17
ogerpon5899_cross:
  avg_delta=-0.027 candidate=0.596 baseline=0.624
  worst=alakazam_sig3_91f48d8e:-0.275 lost=11/17
ogerpon697_cross:
  avg_delta=-0.039 candidate=0.520 baseline=0.559
  worst=alakazam_sig3_91f48d8e:-0.225 lost=10/17
```

Interpretation:

- Cross-attn code path is usable, and Lucario showed a real local improvement.
- The 0803-0804 wave2 cross-attn models are not submission candidates.
- Cross-attn gave some local gains, especially Ogerpon vs Marnie and some
  Festival/Lopunny cases, but lost too much global quality against the w4 pool.
- Crustle and Dragapult cross-attn set000 failed random stability and should
  not be used as shadow or submission candidates from this wave.

Cross-attn full-date retrain started:

- Active script on `ks`: `/tmp/run_cross_attn_full_dates_core.sh`
- Active runner PID observed at start: `951318`
- Corpus: `data/bc_corpus_banded_v11_0701_0804`
- Checkpoints: `checkpoints/arch_cross_attn_full32_20260807/`
- Logs: `logs/arch_cross_attn_full32_20260807/`
- Runner log: `logs/arch_cross_attn_full32_20260807/full_runner.log`
- Targets:
  - `Mega Lucario`, `deck_sig=43d6d8b0fce9`, 18 epochs
  - `Marnie Grimmsnarl`, `deck_sig=b8f251a476e7`, 8 epochs
  - `Teal Mask Ogerpon`, `deck_sig=697a82e582d5`, 18 epochs
  - `Teal Mask Ogerpon`, `deck_sig=5899c772bace`, 20 epochs
- Config:
  - `--arch cross_attn --width 4 --state-layers 2`
  - `--batch-size 4096 --lr 8e-5`
  - `--cuda-memory-gb 32`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `--set-loss-weight 0 --split-by-game`
  - win/loss/draw weights `1.5/0.4/0.8`
- The first attempt used `--cuda-memory-gb 24` under
  `logs/arch_cross_attn_full_20260807/` and failed on Lucario with CUDA OOM
  after the first batch. It was stopped and superseded by the `full32` run.
- The full32 attempt showed that `batch-size 4096` is still unsafe for
  cross-attn on high-option batches:
  - Lucario failed after saving early epochs, with a 32 GiB allocator cap.
  - Marnie failed after loading all full-date data and starting epoch 1, also
    with a 32 GiB allocator cap.
  - Do not use the failed unsuffixed Lucario/Marnie checkpoints as final
    candidates from this run.
- Batch-2048 recovery jobs were started:
  - `/tmp/run_lucario_full_cross_b2048.sh`, PID observed at start `958536`,
    checkpoint `bc2_mega_lucario_43d6_cross_full_w4_b2048.npz`, GPU0,
    `--batch-size 2048 --cuda-memory-gb 24`.
  - `/tmp/run_marnie_full_cross_b2048.sh`, PID observed at start `962439`,
    checkpoint `bc2_marnie_b8f_cross_full_w4_b2048.npz`, GPU1,
    `--batch-size 2048 --cuda-memory-gb 24`.
- Status at 2026-08-07 17:05 Asia/Shanghai:
  - Lucario b2048 was training and had saved at least epoch 3.
  - Marnie b2048 had started and was loading the full-date corpus.
  - Ogerpon5899 full32 finished training; best val observed `0.4208`.
  - Ogerpon697 full32 was still training and had reached at least epoch 8.
- Status at 2026-08-07 17:33 Asia/Shanghai:
  - Only Marnie b2048 was still running, around epoch 4/8. Each epoch is about
    12 minutes because this deck sig has about 3.6M decisions.
  - Ogerpon5899, Ogerpon697, and Lucario b2048 finished training and random g500.
  - `bc2_accuracy.py` had a bug for w4 checkpoints: it defaulted to width 2 and
    silently loaded only matching tensors, producing false low accuracy around
    0.25-0.27. This was fixed by adding `checkpoint_width()` and auto-inferring
    width in `bc2_accuracy.py` and `bc2_failure_report.py`; remote files were
    synced while Marnie was still training, so Marnie's automatic post-train
    accuracy should be valid.

Fixed accuracy/random for completed full-date cross-attn models:

```text
Ogerpon5899 full32:
  best val 0.4208
  fixed accuracy exact=0.918 first=0.920 top3=0.994 set_f1=0.992
  random g500 = 497/500 = 99.4%

Ogerpon697 full32:
  best val 0.6908
  fixed accuracy exact=0.719 first=0.730 top3=0.938 set_f1=0.857
  random g500 = 496/500 = 99.2%

Mega Lucario 43d b2048:
  best val 1.0600
  fixed accuracy exact=0.674 first=0.691 top3=0.925 set_f1=0.776
  random g500 = 496/500 = 99.2%
```

Completed full-date cross-attn quality conclusion:

- Ogerpon5899 full32 is better than the two-day cross-attn wave on offline
  imitation metrics, but worse than the old w4 specialist in paired local
  evaluation. Against the 17-entry w4 candidate pool at 80 games each:
  `avg_delta=-0.024`, candidate mean `0.574`, baseline mean `0.599`,
  lost `11/17`; worst gap was Alakazam sig3 at `-0.162`.
- Ogerpon697 full32 is also better than the two-day cross-attn wave on offline
  imitation/random, but worse than the old w4 specialist in paired local
  evaluation. Against the same pool:
  `avg_delta=-0.028`, candidate mean `0.553`, baseline mean `0.581`,
  lost `12/17`; worst gap was Alakazam sig3 at `-0.175`.
- Mega Lucario 43d cross-full b2048 is only slightly better than earlier
  cross-attn scratch on offline/random, but worse than the existing
  `strategy_tempo` candidate in the local objective. `strategy_tempo` beat
  cross-full `197-102-1` over 300 games, and cross-full vs Marnie w4 was only
  `35-265-0`, win rate `11.7%`. The previous `strategy_tempo` vs Marnie w4
  was `42-258-0`, win rate `14.0%`.
- Therefore completed cross-attn full-date models should not be treated as
  submission candidates yet. They prove the architecture can fit BC labels and
  beat random, but they do not fix the local strong-pool/weak-matchup objective.

## 2026-08-07 Hierarchical Plan Policy

Implemented and committed:

- `df8f983 Add hierarchical plan-conditioned BC policy`
  - `ptcg_rl/model.py`: pointer and cross-attn models support
    `hierarchical_plan=True`. The design preserves the old base scorer shape
    and adds a zero-initialized plan residual scorer, so w4 checkpoints can
    keep their learned `score_fc1/score_fc2` weights when used as init.
  - `ptcg_rl/numpy_policy.py`: submission/eval inference auto-detects
    `plan_condition_fc.*` and applies the plan residual scorer.
  - `tools/bc2_train.py`: new `--hierarchical-plan`; requires
    `--trajectory-target`.
  - `tools/bc2_accuracy.py` and `tools/bc2_failure_report.py`: infer
    `plan_dim` and hierarchical mode from checkpoint before loading.
- `0236922 Add lightweight trajectory target builder`
  - `tools/build_trajectory_targets.py`: streaming per-game target CSV builder
    for hierarchical BC. It avoids the heavy event/ngram gap work in
    `mine_strategy_trajectories.py` and writes columns such as `attack_by_4`,
    `primary_board_by_4`, `evolve_count`, `ability_count`, and
    `early_end_count`.
- `6b0e460 Allow resetting BC scorer during init`
  - `tools/bc2_train.py`: adds `--init-skip-prefix` and `--reset-scorer`.
    `--reset-scorer` loads encoder/state/option weights from `--init` but skips
    `score_fc*` and `stop_vec`, allowing a "good representation, fresh action
    head" baseline.

Remote validation:

- Remote py_compile passed for model, NumPy policy, train, accuracy, and failure
  scripts.
- Remote NumPy smoke passed with explicit
  `PYTHONPATH=/home/jie/Do/0_PTCG/workspace:$PYTHONPATH`:
  `ok pointer True 4 (4,)`.
- Non-interactive ssh does not automatically include the `cg` engine path; eval
  scripts that instantiate `NumpyPolicy` should use that `PYTHONPATH` prefix
  unless running from an environment that already provides `cg`.

Remote hierarchical pilot history:

- Heavy first attempt:
  `/tmp/run_hier_plan_pilots_20260807.sh`, output
  `logs/hier_plan_20260807.runner.log`.
  It was stopped because `mine_strategy_trajectories.py` was too expensive for
  Marnie b8f full-date data: after scanning 112 files it held about 29GB RSS
  and still had not written `game_trajectories.csv`.
- Replacement runner:
  `/tmp/run_hier_plan_pilots_fast_20260807.sh`, output
  `logs/hier_plan_fast_20260807.runner.log`.
  It successfully built lightweight target CSVs but the training wave used
  `batch-size 4096 --cuda-memory-gb 24` and all three jobs OOMed. Do not use
  its partial training outputs as candidates.

Active remote hierarchical comparison:

- Runner:
  `/tmp/run_hier_plan_compare_20260807.sh`, output
  `logs/hier_plan_compare_20260807.runner.log`.
- It reuses target CSVs from `logs/hier_plan_fast_20260807/trajectory/` and
  runs three waves:
  - `init`: existing w4/tempo init + hierarchical plan residual.
  - `reset`: same init but `--reset-scorer`, keeping encoders while re-learning
    action scorer.
  - `scratch`: no init, full random start.
- It uses `batch-size 2048`, `--cuda-memory-gb 32`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Wave layout:
  - `marnie_b8f`: init
    `checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_marnie_grimmsnarl_sig1_b8f251a4_v11all35_sigpure_top3_w4.npz`,
    GPU0, 8 epochs.
  - `ogerpon_5899`: init
    `checkpoints/deck_sig_specialists_v11all35_20260806/w4/bc2_teal_mask_ogerpon_sig3_5899c772_v11all35_sigpure_top3_w4.npz`,
    GPU2, 8 epochs.
  - `lucario_43d`: init
    `checkpoints/lucario43d_20260807/bc2_mega_lucario_43d6_strategy_tempo_w4.npz`,
    GPU3, 8 epochs.
- Shared trajectory targets:
  `attack_by_4`, `attack_by_6`, `primary_board_by_2`,
  `primary_board_by_4`, `primary_board_by_6`, `primary_active_by_4`,
  `evolve_count>=1`, `ability_count>=2`, `attach_count>=2`,
  `early_end_count==0`.
- Training output:
  `checkpoints/hier_plan_compare_20260807/`
- Logs:
  - `logs/hier_plan_compare_20260807/train/*.train.log`
  - `logs/hier_plan_compare_20260807/accuracy/*.accuracy.log`
  - `logs/hier_plan_compare_20260807/random/*.random_g300.log`
- Status at 2026-08-07 18:27 Asia/Shanghai:
  - `init` wave is running.
  - `ogerpon_5899_init` and `lucario_43d_init` reached epoch 1 and saved best
    checkpoints, so the 2048/32GB config is viable.
  - `marnie_b8f_init` was still loading its much larger 112-file corpus.

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/hier_plan_compare_20260807.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_hier_plan_compare|bc2_train.py.*hierarchical-plan|tools/eval_bc.py.*hier_plan_compare"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && for f in logs/hier_plan_compare_20260807/train/*.train.log; do echo ===$f===; tail -20 "$f"; done'
```

Evaluation rule:

- Random/accuracy alone is not enough. After the three pilots finish, compare
  each against its init/baseline with paired RR or baseline-delta.
- Treat a pilot as "better" only if paired local evaluation improves. If offline
  accuracy improves but paired pool delta is negative, record it as worse, same
  as the completed full-date cross-attn wave.

Monitor:

```bash
ssh ks 'pgrep -af "run_cross_attn_full_dates_core|checkpoints/arch_cross_attn_full32_20260807"'
ssh ks 'pgrep -af "run_lucario_full_cross_b2048|run_marnie_full_cross_b2048"'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/arch_cross_attn_full32_20260807/full_runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && for f in logs/arch_cross_attn_full32_20260807/*.train.log; do echo ===$f===; tail -25 "$f"; done'
```

When this finishes, first inspect `*.accuracy.log` and `*.random_g500.log`.
Only if Marnie/Ogerpon/Lucario random is stable should the next step be paired
baseline-delta versus the existing w4 specialist pool.

## 2026-08-08 Matchup Teacher Mining

Important user correction:

- For weak-matchup success mining, do not restrict demonstrations to the exact
  target deck signature. Example: for Ogerpon into Crustle, wins from all Teal
  Mask Ogerpon signatures are relevant because the deck plans are similar.
- First find which same-archetype signature/team best solves the weak matchup:
  which Ogerpon beats Crustle, which Marnie beats Ogerpon, which Lucario beats
  its weak target, and so on.
- Also separate real strategy wins from lucky wins. A weak-matchup win should
  not automatically become a high-weight teacher if the opponent bricked, failed
  to attack, failed to set up its primary line, or made repeated early-end
  decisions.

New tools added locally and synced to `ks`:

```text
tools/find_matchup_teachers.py
tools/audit_teacher_win_quality.py
tools/run_matchup_quality_audits.py
tools/build_trajectory_targets.py  # enhanced strategy target columns
```

`find_matchup_teachers.py` scans a BC corpus and ranks same-archetype teacher
cohorts by `(archetype, opponent_archetype, deck_sig, team_name)`. It writes
pair-level success coverage and teacher-level rankings by support, win rate,
share of pair wins, and quality score.

`audit_teacher_win_quality.py` takes two `build_trajectory_targets.py` CSVs for
both sides of a matchup and pairs the two players by episode. It reports:

- `clean_wins`: candidate win, candidate setup/tempo is coherent, and opponent
  also had a plausible setup/tempo.
- `opponent_brick_wins`: opponent failed attack/setup or showed low pressing
  rate/early-end behavior.
- `strategy_wins`: candidate win with candidate setup or tempo success.

Use `clean_teacher` rows as high-weight strategy data. Treat
`mostly_opponent_brick` rows as low-weight evidence or exclude them from success
BC subsets.

Remote teacher scan completed:

```text
script: /tmp/run_v12_matchup_teacher_scan_20260808.sh
runner log: logs/v12_matchup_teachers_20260808.runner.log
outputs:
  logs/v12_matchup_teachers_20260808/pairs_major_900plus.csv
  logs/v12_matchup_teachers_20260808/teachers_major_900plus_top30.csv
corpus: data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12
score bands: 1200+, 1100-1199, 1000-1099, 900-999
archetypes:
  Alakazam, Crustle Wall, Dragapult, Festival Lead, Marnie Grimmsnarl,
  Mega Lopunny, Mega Lucario, Teal Mask Ogerpon, Team Rocket Mewtwo
status:
  completed, pairs=81, teacher rows=1756
```

Key first finding:

- `Teal Mask Ogerpon=>Crustle Wall` is weak overall in the 900+ v12 corpus:
  `82/458 = 17.9%`.
- The best same-archetype teacher is not `5899c772bace`. It is
  `2a5072194fdf / James Cox & Henry Chao`: `61/150 = 40.7%`, about 74% of all
  Ogerpon wins into Crustle.
- Initial strict quality logic wrongly marked these wins as mostly not clean
  because it required primary Ogerpon/no-early-end and treated Crustle no-attack
  as brick. This was fixed by tracking secondary/setup/engine routes and making
  no-attack/early-end diagnostic rather than hard failure for wall/control
  matchups.
- With the revised quality gate, `2a5072194fdf / James Cox & Henry Chao` has
  `58/61` clean wins and `brick_share=0.049` into Crustle. This is a strong
  teacher seed for Ogerpon-vs-Crustle strategy data.
- `Marnie Grimmsnarl=>Teal Mask Ogerpon` also has many clean teachers, mostly
  `b8f251a476e7` teams. Top clean rows included:
  `Raihan Ramadistra 34/39`, `@kdcyberdude 26/29`,
  `Sixth Sense 27/34`, `やる気元気ミワハルキ 21/23`,
  `Dries @ Tufa Labs 26/49`, `LiamK 20/35`, `Dominic Peel 23/45`.
  This points to b8f team-conditioned clean subsets rather than winner-only BC.
- `Mega Lucario=>Teal Mask Ogerpon` has a very strong clean teacher:
  `43d6d8b0fce9 / Majkel1337`, `33/34` clean wins and `34/42 = 80.95%` game
  win rate in the teacher scan.

New commits after the first teacher-tool commit:

```text
54251d5 Emit per-game teacher quality labels
5c486f0 Track alternate strategy routes in teacher quality
bf4cb5c Relax wall matchup quality gates
```

Active remote paired quality audit:

```text
script: /tmp/run_v12_matchup_quality_audit_20260808.sh
runner log: logs/v12_matchup_teachers_20260808/quality_audit.runner.log
out dir: logs/v12_matchup_teachers_20260808/quality_audit
status at last update:
  pair 1/18 Teal Mask Ogerpon=>Crustle Wall completed
  pair 2/18 Marnie Grimmsnarl=>Teal Mask Ogerpon completed
  pair 3/18 Mega Lucario=>Teal Mask Ogerpon completed
  pair 4/18 Mega Lucario=>Crustle Wall running
```

Monitor:

```bash
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -n 120 logs/v12_matchup_teachers_20260808/quality_audit.runner.log'
ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && pgrep -af "run_v12_matchup_quality_audit|run_matchup_quality_audits|build_trajectory_targets|audit_teacher_win_quality"'
```

The running quality audit command is:

```bash
python3 tools/run_matchup_quality_audits.py \
  --corpus data/bc_corpus_banded_v12_0701_0805_hist32_log128_board12 \
  --pairs-csv logs/v12_matchup_teachers_20260808/pairs_major_900plus.csv \
  --weak-pair "Teal Mask Ogerpon=>Crustle Wall" \
  --weak-pair "Marnie Grimmsnarl=>Teal Mask Ogerpon" \
  --weak-pair "Mega Lucario=>Teal Mask Ogerpon" \
  --weak-pair "Mega Lucario=>Crustle Wall" \
  --weak-pair "Mega Lucario=>Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 900-999 \
  --max-pair-wr 0.45 \
  --min-pair-games 80 \
  --min-pair-wins 1 \
  --min-clean-wins 3 \
  --min-clean-share 0.15 \
  --max-brick-share 0.55 \
  --limit 18 \
  --top 30 \
  --progress-every 12 \
  --force \
  --out-dir logs/v12_matchup_teachers_20260808/quality_audit
```

Run long commands via `/tmp` scripts on `ks`, not large inline heredocs.

## Next Work

## 2026-08-09 Aggressive RL Direction

Current conclusion after the failed success-trajectory/filtered-BC waves:

- Do not keep treating weak-matchup wins as reliable teacher data by default.
  The direct overfit test on successful weak-matchup trajectories did not repair
  the weak matchup, so many of those wins are probably opponent bricks, one-off
  tactical mistakes, or non-causal correlations.
- The next serious attempt is RL self-exploration, not another winner-only or
  filtered BC pass.
- Kaggle discussion signals match this direction: strong RL reports emphasize
  representation, value quality, refined curriculum, and large rollout scale;
  search is useful only with a trustworthy value head; inference remains
  CPU-only with 1.6 vCPU, 8GB RAM, and 600s total per team/game.

Local code change:

- `tools/rl_finetune_vs_pool.py` now supports aggressive league PPO:
  - adaptive opponent sampling via `--opponent-weight-mode adaptive_lossrate`
    or `adaptive_inverse_winrate`;
  - per-opponent rollout logging via `--opponent-stats-csv`;
  - previous-policy league snapshots via `--league-bootstrap-current`,
    `--league-snapshot-every`, and `--league-max-snapshots`;
  - linear schedules for rollout temperature, rollout top-k, entropy, ref-KL,
  and BC-anchor weight.
- This is designed to let PPO leave the BC local optimum while preserving enough
  logging to catch collapse. It does not change the submission-time policy
  format.
- User explicitly requested no more fine-tuning: BC is locked. Do not run
  weak-matchup BC/PPO fine-tunes as the main path. Existing BC checkpoints are
  only baselines, fixed opponents, submission references, and architecture
  templates. For future structural-weakness work, use from-scratch / major
  retraining. The new wrapper `tools/rl_train_league.py` should be used with
  `--init-mode random`; `--policy-init` is only a dimension/template source in
  that mode. Use `--init-mode resume` only for checkpoints generated by the
  scratch RL run itself. Keep `--bc-anchor-weight 0` and `--ref-kl-coef 0`
  unless the user reverses this.

Current remote note:

- The older v3 Lucario fine-tune and its follow-up baseline-delta evaluation
  were stopped after the user locked BC and rejected further fine-tuning.
- Scratch league RL wave started on all four GPUs:
  - runner: `/tmp/run_scratch_league_rl_20260809.sh`
  - runner log: `logs/rl_scratch_league_20260809/runner.log`
  - checkpoints: `checkpoints/rl_scratch_league_20260809/`
  - logs: `logs/rl_scratch_league_20260809/`
  - decks: Marnie b8f, Ogerpon 2a507, Mega Lucario 43d, Dragapult cc2.
  - phase 1: `--init-mode random`, random/curriculum opponents, no BC anchor,
    no ref-KL.
  - phase 2: `--init-mode resume` from phase-1 scratch checkpoint, hard pool +
    adaptive league, no BC anchor, no ref-KL.
- Initial runtime check: Marnie/Dragapult pointer templates roll out about
  9-11 games/s on 10 workers. Ogerpon/Lucario history/cross-attn templates roll
  out about 1.4-1.5 games/s, much slower but still progressing.

Recommended first aggressive run:

- Use large rollout batches and many iterations, not a tiny smoke test.
- Start with Lucario, Marnie, and Ogerpon because they have known weak pools and
  existing v3/w4 baselines.
- Validate every checkpoint with both random and focused baseline-delta. A
  weak-pool win-rate rise without random/RR stability is an exploit of the local
  shadow pool, not a production candidate.

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

2026-08-10 Kaggle replay/environment update:

- Remote 0809 episode zip was parsed with a temporary fast `orjson`
  analyzer because `tools/build_ladder_pool.py` standard-json parsing was too
  slow under concurrent rollout load.
- Fast environment outputs on `ks`:
  - `logs/kaggle_env_fast_20260810/band_archetype_0809.csv`
  - `logs/kaggle_env_fast_20260810/band_archetype_0805_0809.csv`
  - `logs/kaggle_env_fast_20260810/archetype_matchups_0809_600p.csv`
  - `logs/kaggle_env_fast_20260810/deck_sig_matchups_0809_900p.csv`
  - `logs/ladder_pool_0809_fast_all/pool_manifest.csv`
  - `logs/ladder_pool_0805_0809_fast_all/pool_manifest.csv`
- 0809 environment, by weight:
  - 1100-1199: Marnie is dominant, then Mega Lopunny.
  - 1000-1099: Alakazam is dominant, then Ogerpon/Dragapult/Crustle.
  - 1200+: Dragapult and Mega Lucario dominate.
  - 900-999: Crustle/Festival/Ogerpon/Marnie are the main climb gate.
  - 700-799: Ogerpon is the largest archetype, so a submission starting from
    600 must not fold to Ogerpon even if it is aimed at high-rank Marnie/Lopunny.
- Recent jie replay rows on `ks`:
  - `logs/kaggle_replay_recent_20260810_fast/*.csv`
  - `55370364` Marnie v12hist bigbatch: 60 games, 34-26, WR 0.567.
  - `55364746` w4 Marnie resubmit: 74 games, 44-30, WR 0.595.
  - `55356700` w4 Marnie resubmit: 59 games, 34-25, WR 0.576.
  - `55356873` Ogerpon v7sig rerere: 40 games, 22-18, WR 0.550.
  - `55328252` history-k Marnie: 77 games, 43-34, WR 0.558.
  - `55349290` seqplan Lucario: 33 games, 14-19, WR 0.424.
  - `55349364` seqplan Festival: 42 games, 21-21, WR 0.500.
  - `55294007` shadow Lopunny f144: 35 games, 17-18, WR 0.486.
- Main replay conclusion:
  - Current strong Marnie submissions only run about 56-60% in pulled replay
    samples, but this is enough to stay around 950+ because they avoid broad
    collapse.
  - Marnie is consistently weak to f144 Mega Lopunny in live replay; local RR
    underestimates that specific danger.
  - Ogerpon still looks overvalued by local weighted RR. Recent live samples
    are only about 55%, not stable enough for a 600-start climb.
  - Our Lucario/history/seqplan submissions did not replicate real 1200+
    Lucario quality; do not select Lucario just because the archetype is good.
- Packaged, not submitted, on `ks`:
  - Primary: `submissions/env0809_20260810/w4_marnie_sig2_2c22fa76_env0809.tar.gz`
  - Primary/exploratory: `submissions/env0809_20260810/w4_cynthia_sig1_52f46739_env0809.tar.gz`
  - Backup: `submissions/env0809_20260810/w4_alakazam_sig1_7f9a5389_env0809_backup.tar.gz`
  - Backup/riskier: `submissions/env0809_20260810/w4_lopunny_sig3_f1445356_env0809_backup.tar.gz`
- Reasoning:
  - Marnie 2c22 is an unsubmitted exact deck/model and is more balanced than
    b8f in local RR against mirror/Lopunny/Alakazam/Crustle.
  - Cynthia 52f is unsubmitted and actual 0809 ladder matrix has it doing well
    into b8f Marnie, f144 Lopunny, and 7f9 Alakazam, but it may be exposed to
    Ogerpon/Crustle during the 600-to-900 climb.
  - Alakazam 7f9 is a stable backup because the archetype/signature is common
    and previously Alakazam reached about 900, but it is capped by current
    Marnie/Dragapult pressure.
  - Lopunny f144 matches the high-rank environment but shadow f144 already
    failed online; only submit the w4 specialist if explicitly testing whether
    the pure specialist fixes the shadow quality issue.
- During this analysis, the long rule-guided rollout workers were temporarily
  `SIGSTOP`ed to free CPU and then resumed with `SIGCONT`. They were not
  killed. Check `logs/rule_guided_20260810b/stage1b.runner.log` if continuing
  that run.

2026-08-10 RR pool filtering update:

- User pointed out that weak local RR opponents inflate candidate win rates.
  Do not use the full w4/all39 pool for headline candidate scoring.
- New tool: `tools/filter_rr_pool.py`.
  - Inputs: candidate manifest, random audit, optional RR CSV, optional live
    ladder manifests.
  - Outputs manifest-compatible filtered pools with `eval_entry`,
    `checkpoint_path`, and `deck_path` preserved.
  - Primary gate used now: random WR `>=0.97`, timeouts `0`, and either
    RR mean `>=0.38`, live score `>=900`, or live environment weight `>=500`.
- Current remote outputs:
  - `logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_primary.csv`
  - `logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_balanced.csv`
  - `logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_environment.csv`
  - `logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_diagnostic_low_quality.csv`
  - `logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_filter_summary.txt`
- Interpretation:
  - `rr_pool_primary.csv`: use for headline RR/submission gating. It has 15
    entries across 8 archetypes: Marnie, Mega Lopunny, Alakazam, Crustle,
    Ogerpon, Cynthia, Festival, and Team Rocket Mewtwo.
  - `rr_pool_balanced.csv`: slightly looser but still no weak fallback. It has
    16 entries across the same 8 archetypes.
  - `rr_pool_environment.csv`: live-ladder coverage only. Use it to detect
    environment gaps, not as a headline score, because it includes currently
    weak local Mega Lucario.
  - `rr_pool_diagnostic_low_quality.csv`: do not use for candidate scoring.
    It identifies training gaps: current local Archaludon, Dragapult, Mega
    Lucario, Mega Starmie, N's Zoroark, and some secondary signatures are too
    weak or missing audits/deck paths to be trusted as strong RR opponents.
- Example headline RR command:

```bash
python3 tools/eval_round_robin.py \
  --manifest logs/rr_pool_filters_20260810/w4_env0809_strict/rr_pool_primary.csv \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --skip-bad-entries \
  --out-csv logs/rr_pool_filters_20260810/w4_env0809_strict/rr_primary_g100.csv
```

Remote operations:

- Avoid killing LLaMA Factory or `/home/byer/PTCG/...` jobs unless the user explicitly asks.
- Keep CUDA memory caps around `--cuda-memory-gb 8` while shared GPU jobs are active.
- If starting more long-running jobs, write logs under a distinct `logs/...` directory and record command/status here.

2026-08-10 public strategy source update:

- New source catalog: `data/public_strategy_sources_v1.csv`.
  - High-priority strategy sources: Limitless Play/Labs for matchup priors and
    lists, TCGplayer/PokeBeach/Japanese note.com/YouTube VODs for sequencing,
    official Pokemon card database for card text.
  - Medium-priority discussion sources: PokeBeach forums, Reddit r/pkmntcg and
    r/PTCGL, public Discord/manual search, X/Twitter public search.
  - Paid/private sources such as Metafy or Discord private channels must not be
    scraped or bypassed; use only user-provided notes or public/manual content.
- New helper: `tools/plan_public_strategy_search.py`.
  - Example smoke-test outputs:
    - `logs/public_strategy_search/ogerpon_vs_crustle_tasks.csv` rows=264.
    - `logs/public_strategy_search/marnie_vs_ogerpon_tasks.csv` rows=264.
  - Use it to generate source-specific query tasks before writing new rules.
- Important Ogerpon-vs-Crustle caveat:
  - Public Limitless/TCG Ogerpon wins vs Crustle may rely on Cornerstone Mask
    Ogerpon ex / Demolish, which can attack through effects on the opponent
    Active.
  - Local Kaggle Ogerpon `5899c772bace` and `697a82e582d5` do not contain
    Cornerstone Mask Ogerpon ex, so those lists cannot directly execute the
    main public anti-Crustle route.
  - `2f538fcfa698` has Cornerstone, while `2a5072194fdf` has broader box pieces
    such as Wellspring/Mega Kangaskhan/Meowth/Lillie's Clefairy. Treat those as
    separate route families, not as small signature variations.
- Rule/planner infrastructure:
  - Added `opportunity_plan` to `RULE_MODES`.
  - Added stateful `OpportunityPlanner` and unified
    `apply_rule_decision`/`make_rule_planner` helpers.
  - Updated `main.py`, `tools/eval_bc.py`, `tools/eval_round_robin.py`,
    `tools/eval_rule_overlay_stats.py`, `tools/generate_rollout_bc.py`, and
    `tools/build_weakness_state_bank.py` to use the unified rule path.
  - `opportunity_plan` is experimental. It is meant for short TTL opportunity
    windows, not broad matchup route forcing. Validate with trigger stats and
    focused+broad RR before any submission packaging.

2026-08-10 CPU cleanup and Ogerpon-vs-Crustle rule probe:

- Remote slowness was caused by old PTCG rollout/eval jobs, not current GPU
  training. `uptime` initially showed a very high 15-minute load; top CPU was
  from orphaned/long `tools/generate_rollout_bc.py` and old
  `eval_rule_overlay_stats.py` work under:
  - `data/generated_rollout_bc_resource_plan_20260810b`
  - `logs/rule_guided_20260810b`
  - `logs/cornerstone_deck_ablation_20260810`
- Cleaned only these PTCG-related stale processes. Do not kill `/home/byer/ARC`
  or other byer jobs unless explicitly requested. After cleanup, remote load
  fell to roughly single digits on the 1-minute window.
- Added/fixed rule diagnostics:
  - `tools/eval_rule_overlay_stats_fast.py` is a bounded parallel trigger
    audit tool. Use it instead of the old serial stats script for matchup rule
    probes.
  - Fixed `ptcg_rl/rule_overlay.py::_option_card` for PLAY options shaped like
    `{"type": 7, "index": n}`. These index the current player's hand. Without
    this fix, rule code parsed many played cards as card id 0 and could not see
    Ultra Ball, Cyrano, Area Zero, Passimian, etc.
- Ogerpon 2a vs Crustle analysis:
  - Historical corpus showed `2a5072194fdf` has real Kaggle games around the
    30% decision/game-level range into Crustle, unlike `697a82e582d5` and
    `5899c772bace`, but local BC still only scores single digits.
  - Mining established-Crustle wins showed the real 2a route is likely
    Passimian + wide Basic board + early attacks, not Cornerstone. Do not cut
    `978 Passimian` from 2a deck variants.
  - Added Passimian route cards to `DeckPlan` and added
    `ogerpon_passimian_break_wall` in `OpportunityPlanner`.
  - Focused paired eval:
    `logs/cornerstone_deck_ablation_20260810/passimian_rules_2a_vs_crustle_g240.csv`
    showed only a small gain:
    - vs `3cd5039c`: base 0.071, rules 0.096, delta +0.025
    - vs `b141ae29`: base 0.087, rules 0.117, delta +0.029
    - average rule WR 0.106, avg delta +0.027
  - Conclusion: current Passimian overlay is directionally positive but far
    below the 30% target. Do not package it for submission. It is useful as a
    diagnostic and parser fix, but not strong enough as a teacher unless the
    plan becomes a true multi-turn controller/search policy.

2026-08-10 rollout status/effect audit:

- Current remote CPU status after cleanup: no `generate_rollout_bc`,
  `rl_finetune_vs_pool`, PPO, `train_bc_population`, or `bc2_train.py`
  processes are running. Load has returned to normal for a 54-core box.
- `logs/rule_guided_20260810`:
  - Planned 28 rollout teacher jobs, but the first wave was killed by the
    earlier CPU overload. No npz payload exists under
    `data/generated_rollout_bc_resource_plan_20260810`.
- `logs/rule_guided_20260810b`:
  - Planned 18 rollout teacher jobs.
  - Only 5 jobs wrote complete summary CSVs; jobs 006/007 have partial logs and
    partial npz, and the remaining jobs were not launched before cleanup.
  - Complete CSV totals: 2000 games, 23 wins, WR 0.0115, 1479 written rows.
  - All npz payloads including partial jobs: 31 files, 2238 rows.
  - Row coverage is narrow:
    - Dragapult: 1012 rows
    - Festival Lead: 519 rows
    - Teal Mask Ogerpon: 508 rows
    - Mega Starmie: 199 rows
    - No useful Marnie, Mega Lucario, Team Rocket Mewtwo, Alakazam, etc.
  - Actor quality is poor:
    - `greedy+rules:resource_plan`: 9/335 WR 0.0269
    - `topk@3+rules:resource_plan`: 7/575 WR 0.0122
    - `topk@2+rules:resource_plan`: 3/550 WR 0.0055
  - These jobs were generated before the PLAY option card-id parser fix, and
    used old broad `resource_plan`, not the newer Passimian `opportunity_plan`.
  - Conclusion: do not train from `data/generated_rollout_bc_resource_plan_20260810b`.
    It is too small, too low quality, incomplete, and generated with a known
    parser bug.
- Older `logs/rollout_distill_20260806` quick distill:
  - Marnie rollout kept random WR 0.996 but did not improve the weak Ogerpon
    rows: vs `5899` delta -0.025, vs `697` delta -0.01875, only vs easier `2a`
    delta +0.04375.
  - Ogerpon 2a rollout random WR collapsed to 0.834. Versus Crustle, average
    delta was only +0.008, with one opponent negative.
  - Cynthia rollout random WR 0.976 and vs Crustle average delta +0.046, but
    absolute candidate WR was only 0.179, still below the 0.30 weak-matchup
    target.
  - Conclusion: 08-06 rollout distill is not a strong path either; at best it
    produced a small Cynthia diagnostic, not a generally useful pipeline.

2026-08-10 full workspace audit:

- Remote audit command scanned
  `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804` for files
  modified on `2026-08-10` or paths containing `0810`, `20260810`, `08-10`,
  or `2026-08-10`.
- Audit outputs:
  - `logs/workspace_audit_20260810/all_0810_or_mtime.csv`
  - `logs/workspace_audit_20260810/mtime_20260810.csv`
  - `logs/workspace_audit_20260810/name_0810.csv`
- Totals: 1666 matching files, about 6.2GB.
  - `checkpoints`: 77 files, about 4.0GB.
  - `logs`: 1521 files, about 2.1GB.
  - `submissions`: 4 files, about 114MB.
  - `data`: 34 files, about 544KB.
- Important directories found by audit:
  - `checkpoints/rl_scratch_league_20260809`: RL scratch checkpoints modified
    during 2026-08-10 00:00-03:28.
  - `logs/rl_scratch_league_20260809`: corresponding train/random/delta logs.
  - `logs/search_teacher_20260810`: state-bank, root-search teacher, motif, and
    online guided tests.
  - `logs/rule_guided_20260810` and `logs/rule_guided_20260810b`: planned and
    partial rollout-teacher jobs.
  - `logs/cornerstone_deck_ablation_20260810`: deck ablation, rule trigger
    stats, and Passimian paired eval.
  - `logs/kaggle_env_fast_20260810`, `logs/kaggle_replay_recent_20260810_fast`,
    `logs/rr_pool_filters_20260810`: 0809 environment/replay/pool filtering.
  - `submissions/env0809_20260810`: four packaged submission candidates:
    Cynthia 52f, Marnie 2c22, Alakazam 7f9 backup, Lopunny f144 backup.
- RL scratch effect from `logs/rl_scratch_league_20260809`:
  - Random WR remained high:
    - Marnie phase2 best: 298/300, WR 0.993.
    - Dragapult phase2 best: 299/300, WR 0.997.
    - Ogerpon 2a mem48 phase2 best: 293/300, WR 0.977.
    - Lucario 43d mem48 phase2 best: 294/300, WR 0.980.
  - Core paired deltas were bad:
    - Marnie scratch vs Ogerpon: deltas -0.1375, -0.1750, -0.4750.
    - Dragapult scratch vs core: deltas -0.1063, -0.0875, -0.0750.
    - Ogerpon 2a mem48 vs Crustle: +0.0063, +0.0063, -0.0438.
    - Lucario 43d mem48 vs core: -0.1063, -0.1125, -0.3313.
  - Conclusion: scratch RL learned enough basic legality/random play but did
    not beat BC on the weak/core matchups. Do not submit these RL scratch
    checkpoints without further broad RR showing a reversal.
- Search-teacher effect from `logs/search_teacher_20260810`:
  - Offline root-search looked promising on 24 sampled loss states per pair:
    - Dragapult vs Marnie: mean best WR 0.156 vs baseline 0.052.
    - Lucario vs Marnie: 0.458 vs 0.240.
    - Marnie vs Ogerpon 5899: 0.406 vs 0.240.
    - Ogerpon 2a vs Crustle: 0.328 vs 0.156.
  - However online search-guided Ogerpon 2a vs Crustle failed:
    - ungated g4: 0/4.
    - gated g8: 0/8.
  - Conclusion: current root-search labels are useful diagnostics, but the
    action-level heuristic does not compose into a game-winning multi-turn
    policy yet.
- Remote workspace hygiene note:
  - Remote git status is very dirty and includes many untracked top-level copies
    such as `resource_planner.py`, `rule_overlay.py`, `eval_bc.py`,
    `generate_rollout_bc.py`, etc. These appear to be accidental scp copies to
    the repo root. Do not delete without user approval, but do not treat them as
    active package files. The active package files are under `ptcg_rl/` and
    `tools/`.

2026-08-11 current RR pool rebuild:

- User requested rebuilding a current RR pool using all current/new decks and
  all archetypes, with date windows adjusted for enough data, then random and
  RR testing.
- Remote runner uploaded to:
  - `/tmp/current_rr_pool_runner_20260811.py`
- Active remote background command:
  - PID observed: `3396773`
  - Log: `logs/current_rr_pool_20260811/runner.log`
  - Command started from remote repo:
    `python3 -u /tmp/current_rr_pool_runner_20260811.py --stage all --top-per-arch 2 --cap-per-arch 6 --target-rows 120000 --min-train-rows 15000 --min-launch-rows 2500 --gpus 0,3 --max-parallel 2 --cuda-memory-gb 12 --batch-size 512 --random-workers 32 --rr-workers 32 --rr-games 100`
- Selection source:
  - `logs/ladder_distribution_0810_20260811/deck_sig_summary_0810.csv`
  - Existing deck dirs plus recovered 0810 deck CSVs under
    `logs/current_rr_pool_20260811/decks_extra`.
- Corpus:
  - `data/bc_corpus_banded_v12_0801_0810_hist32_log128_board12_20260811`
  - v12 dims observed earlier: state 80, option 64.
- Plan output:
  - `logs/current_rr_pool_20260811/current_deck_plan.csv`
  - 56 selected current signatures, 15 archetypes, 44 trainable with
    `selected_rows >= 2500`, no missing deck paths.
  - Per-archetype selected counts: Alakazam 6, Archaludon 1, Crustle Wall 6,
    Cynthia Garchomp 3, Dragapult 6, Festival Lead 6, Iono Bellibolt 1,
    Marnie Grimmsnarl 6, Mega Lopunny 2, Mega Lucario 2, Mega Starmie 2,
    N's Zoroark 2, Other 4, Teal Mask Ogerpon 6, Team Rocket Mewtwo 3.
  - Important current Ogerpon sigs included: `90abbfb0eee0`,
    `2bd9da52c43a`, `d17573abc0e3`, `cc81a7ad2d24`, `85d941c82b94`,
    `6205fc379380`.
- Pipeline outputs when complete:
  - Checkpoints: `checkpoints/current_rr_pool_20260811/w4/*.npz`
  - Manifest: `logs/current_rr_pool_20260811/candidate_manifest.csv`
  - Random audit: `logs/current_rr_pool_20260811/random_g500.csv`
  - Primary RR manifest: `logs/current_rr_pool_20260811/rr_manifest_primary.csv`
  - Primary RR: `logs/current_rr_pool_20260811/rr_primary_g100.csv`
  - Summary: `logs/current_rr_pool_20260811/rr_primary_g100_summary.csv`
- Quick status commands:
  - `ssh ks 'cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804 && tail -80 logs/current_rr_pool_20260811/runner.log'`
  - `ssh ks 'pgrep -af "current_rr_pool_runner_20260811|bc2_train.py.*current_rr_pool"'`
  - `ssh ks 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'`

2026-08-11 high-purity 1100+ all-date specialist cohort:

- User asked whether exact deck-sigs with 1100+ data from July through 0810 can
  be trained and whether this can transfer to climbing from 600+. Current
  working assumption: yes, as a separate high-purity specialist cohort, but it
  must pass random/RR because pure 1100+ labels may under-cover lower-ladder
  weird states.
- Remote runner uploaded to:
  - `/tmp/high1100_specialist_runner_20260811.py`
- It builds a non-duplicated symlink corpus:
  - Early source: `data/bc_corpus_banded_v12_0701_0807_hist32_log128_board12`
  - Late source: `data/bc_corpus_banded_v12_0801_0810_hist32_log128_board12_20260811`
  - Merged output:
    `data/bc_corpus_banded_v12_0701_0810_merged_hist32_log128_board12_20260811`
  - Feature dims checked: early and late are both state 80 / option 64,
    feature_version `v12_multistream_history`.
- Plan output:
  - `logs/high1100_specialists_20260811/plan.csv`
  - 8 entries:
    - Stable July-to-0810 exact sigs:
      - Alakazam `7f9a538936e3`, 1,324,654 1100+ rows,
        2026-07-10..2026-08-10.
      - Dragapult `cc2e995b5ad0`, 91,955 1100+ rows,
        2026-07-26..2026-08-10.
      - Marnie `b8f251a476e7`, 1,070,137 1100+ rows,
        2026-07-20..2026-08-10.
      - Marnie `2c22fa761816`, 201,719 1100+ rows,
        2026-07-05..2026-08-10.
    - Current high 1100+ exact sigs with enough rows but not July-stable:
      - Mega Lopunny `f1445356c3a7`, 136,114 rows,
        2026-08-02..2026-08-10.
      - Mega Lucario `43d6d8b0fce9`, 121,781 rows,
        2026-08-02..2026-08-10.
      - Ogerpon `90abbfb0eee0`, 22,719 rows,
        2026-08-06..2026-08-10.
      - Ogerpon `2bd9da52c43a`, 28,444 rows,
        2026-08-06..2026-08-10.
- Active remote background command:
  - PID observed: `3407914`
  - Log: `logs/high1100_specialists_20260811/runner.log`
  - Command:
    `python3 -u /tmp/high1100_specialist_runner_20260811.py --stage all --include-current-high --min-1100-rows 15000 --min-current-high-rows 15000 --cap-per-arch 2 --wait-for-current-training --gpus 0,3 --max-parallel 2 --cuda-memory-gb 12 --batch-size 512 --random-workers 32 --rr-workers 32`
  - It waits until no `bc2_train.py.*checkpoints/current_rr_pool_20260811`
    processes are running, then trains on GPU 0/3.
- Pipeline outputs when complete:
  - Checkpoints: `checkpoints/high1100_specialists_20260811/w4/*.npz`
  - Manifest: `logs/high1100_specialists_20260811/manifest.csv`
  - Random audit: `logs/high1100_specialists_20260811/random_g500.csv`
  - RR: `logs/high1100_specialists_20260811/rr_g100.csv`
  - Summary: `logs/high1100_specialists_20260811/rr_g100_summary.csv`
- A stale old high1100 `--stage plan` process from the first inefficient
  recovery attempt was killed (`kill 3404369`). Do not restart that old form.

2026-08-11 Marnie b8f w4legacy 0801-0810 score-drop diagnosis:

- User reported packaged
  `marnie_b8f_w4legacy_0801_0810_b256` only stayed around 700+ Kaggle score,
  far below the previous 960+ Marnie result, and asked whether samples below
  900 were accidentally mixed in.
- Checked remote log:
  `logs/bc_retrain_0801_0810_20260811/train_marnie_b8f_w4legacy_0801_0810_b256.log`
- It did not train on bands below 900. The actual `BC2:` line is:
  `Marnie Grimmsnarl ['1200+', '1100-1199', '1000-1099', '900-999']`
  with `date_from=2026-08-01`, `date_to=2026-08-10`, and
  `deck_sigs=['b8f251a476e7']`.
- The 30 loaded files are 10 days times the available 1100/1000/900 bands.
  No 800/700/600 files were loaded by that training command, even though those
  files exist in the corpus directory.
- Exact row/outcome count for b8f in v12 0801-0810:
  - 1100-1199: 400,476 rows, WR 0.5538.
  - 1000-1099: 768,668 rows, WR 0.5187.
  - 900-999: 919,495 rows, WR 0.4704.
  - 900+ total: 2,088,639 rows, 1,052,961 win rows, 1,034,613 loss rows,
    1,065 draw rows.
  - Below 900 rows for the same sig exist in the corpus: 450,833 rows, but
    were not used by this run.
- Current interpretation: this score drop is not caused by accidental <900
  samples. More likely, "0801-0810 900+ non-winner-only" is a much noisier
  label mix for b8f in the changed 0810 environment; the 900-999 band alone is
  huge and losing-biased. Do not use this checkpoint as a strong submission
  candidate unless later RR/random/Kaggle evidence reverses this.
- The already-started high1100 cohort directly tests the cleaner alternative:
  Marnie b8f and 2c22 trained on 1100+ only over the merged 0701-0810 v12
  corpus.

2026-08-11 current RR pool and high1100 completion:

- `logs/current_rr_pool_20260811` completed:
  - Training: 44/44 done, no failures.
  - Random: `random_g500.csv`, 44/44 entries complete.
  - Primary RR: `rr_primary_g100.csv` and
    `rr_primary_g100_summary.csv`, 9 primary entries.
  - `rr_manifest_coverage.csv` was generated with 13 archetype-covering
    entries, but coverage RR has not been run yet.
- Current primary RR top candidates:
  - `cur0810_marnie_grimmsnarl_b8f251a4`: random 0.980, RR
    avg 0.588, weighted avg 0.587, min 0.480.
  - `cur0810_mega_lucario_43d6d8b0`: random 0.984, RR avg 0.604,
    weighted avg 0.476, min 0.400.
  - `cur0810_festival_lead_e82dcbe6`: random 0.994, RR avg 0.552,
    weighted avg 0.492, min 0.340.
  - `cur0810_cynthia_garchomp_52f46739`: random 0.990, weighted avg
    0.502, but min only 0.130 into Festival, so it is riskier.
- `logs/high1100_specialists_20260811` completed:
  - Training: 8/8 done, no failures.
  - Random: `random_g500.csv`.
  - RR: `rr_g100.csv` and `rr_g100_summary.csv`.
- High1100 results:
  - `high1100_marnie_grimmsnarl_b8f251a4`: random 0.998, RR
    avg 0.730, weighted avg 0.639, min 0.550, losses 0. This is the
    current strongest submission candidate.
  - `high1100_alakazam_7f9a5389`: random 0.992, RR avg 0.690,
    weighted avg 0.495, min 0.420. Strong second choice, but weak-ish
    into high Marnie.
  - `high1100_mega_lucario_43d6d8b0`: random 0.990, RR avg 0.573,
    but weighted avg only 0.341 because it loses badly to high Marnie.
  - Do not submit high1100 Ogerpon or high1100 Dragapult from this cohort:
    random/RR are weak (`Ogerpon` random around 0.84-0.85, `Dragapult`
    random 0.762 and RR avg 0.247).
- Recommended submission order if choosing from completed 2026-08-11 jobs:
  1. `high1100_marnie_grimmsnarl_b8f251a4`
  2. `high1100_alakazam_7f9a5389`
  3. `cur0810_mega_lucario_43d6d8b0`
  4. `cur0810_festival_lead_e82dcbe6`
  5. `cur0810_marnie_grimmsnarl_b8f251a4` only as an ablation against the
     high1100 Marnie, not together with it on the same account unless testing
     same-archetype training-window effects.

2026-08-11 submission selection refined with coverage matrix and 0810 ladder
band weighting:

- Added helper tools:
  - `tools/rr_archetype_matrix.py`: converts `eval_round_robin.py` pairwise
    CSV into deck-vs-opponent-archetype matrices.
  - `tools/rr_ladder_weighted_score.py`: weights an archetype matrix by a
    score-band ladder distribution and a climb-sensitive multiplier.
  - `tools/baseline_delta_archetype_matrix.py`: converts
    `eval_baseline_delta.py` output into candidate-vs-opponent-archetype
    matrices.
- 0810 ladder distribution source:
  - `logs/ladder_distribution_0810_20260811/archetype_summary_0810.csv`
  - `logs/ladder_distribution_0810_20260811/band_archetype_players_0810.csv`
  - `logs/ladder_distribution_0810_20260811/band_archetype_rows_0810.csv`
- Important 0810 distribution facts:
  - Overall player entries: Marnie 29.4%, Alakazam 19.4%, Ogerpon 11.8%,
    Lopunny 11.3%, Dragapult 9.2%, Crustle 5.8%, Lucario 4.0%.
  - 1100-1199 player entries: Lucario 23.8%, Marnie 20.9%, Ogerpon 19.2%,
    Dragapult 12.3%, Lopunny 11.0%.
  - 900-999 player entries: Marnie 38.2%, Alakazam 21.6%, Lopunny 11.5%,
    Crustle 11.1%, Ogerpon 7.7%.
  - 800-899 player entries: Ogerpon 27.4%, Marnie 22.4%, Alakazam 17.6%,
    Other 10.8%, Dragapult 8.4%.
  - 600/700 bands are small and noisy, but early-score impact is large, so
    use them in a climb-sensitive view without letting them fully dominate.
- Current coverage RR:
  - Manifest: `logs/current_rr_pool_20260811/rr_manifest_coverage.csv`
  - RR: `logs/current_rr_pool_20260811/rr_coverage_g100.csv`
  - Archetype matrix:
    `logs/current_rr_pool_20260811/rr_coverage_g100_arch_matrix.csv`
  - 0810 weighted outputs:
    `logs/current_rr_pool_20260811/rr_coverage_g100_ladder0810_players_weighted.csv`
    and `..._rows_weighted.csv`
- Coverage matrix showed that current Alakazam's high broad average is partly
  misleading: it is 0.040 vs current TR Mewtwo. Because TR Mewtwo is rare, it
  still ranks high by 0810 climb weight, but this is a real hard-risk matchup.
- High-vs-coverage audit:
  - Log/CSV:
    `logs/current_rr_pool_20260811/high_vs_coverage/high_vs_current_marnie_coverage_g100.log`
    and `.csv`
  - Matrix:
    `logs/current_rr_pool_20260811/high_vs_coverage/high_vs_coverage_arch_matrix.csv`
  - Weighted:
    `logs/current_rr_pool_20260811/high_vs_coverage/high_vs_coverage_ladder0810_players_weighted.csv`
    and `..._rows_weighted.csv`
  - `high_marnie`: 13/13 matchup wins vs coverage, macro 0.760,
    0810 climb-weighted 0.757 (players) / 0.763 (rows), worst archetype
    Festival 0.550. This is the strongest current submission candidate.
  - `high_alakazam`: macro 0.732, climb-weighted 0.684 (players) /
    0.676 (rows), but worst TR Mewtwo 0.380. It is still second-best, but
    has a clear rare hard-risk matchup.
  - `high_lucario`: macro 0.593 and weighted delta negative vs current
    Marnie; not recommended for submission from this wave.
  - `high_lopunny`: macro 0.587 and weighted delta negative vs current
    Marnie; not recommended for submission from this wave.
- Updated recommended submission order from 2026-08-11 trained models:
  1. `high1100_marnie_grimmsnarl_b8f251a4`
  2. `high1100_alakazam_7f9a5389`
  3. `cur0810_festival_lead_e82dcbe6` as a diversified probe, not because it
     beats high Marnie.
  4. `cur0810_marnie_grimmsnarl_b8f251a4` only if explicitly comparing
     current-window Marnie vs high1100 Marnie.

2026-08-11 correction after actual Kaggle replay for
`high1100_marnie_grimmsnarl_b8f251a4`:

- Submission:
  - `55424557`, message `bc: high1100_marnie_b8f251a4_0811`, public score
    885.2 at inspection time.
  - The first replay is Kaggle self-validation/self-play and should not be
    counted as ladder performance.
- Replay analysis outputs on ks:
  - `logs/kaggle_replay_0811_high1100_marnie/jie_55424557_high1100_marnie_rows_teamname.csv`
  - `logs/kaggle_replay_0811_high1100_marnie/jie_55424557_high1100_marnie_by_deck_teamname.csv`
  - `logs/kaggle_replay_0811_high1100_marnie/jie_55424557_high1100_marnie_path_teamname.chronological.csv`
  - Cached replay JSONs under `logs/kaggle_replays_recent_cache_fast/55424557/`.
- Effective ladder games after excluding the self-validation ambiguity:
  46 known games, 29-17, 63.0% WR. By rough chronological climb phase:
  early 6-3, mid 10-5, late 13-9. This is far below the local
  high-vs-coverage expectation (~0.76 macro / ~0.757 climb-weighted).
- Enriched loss attribution:
  - Mirror/Marnie: 4-5, including same `b8f251a4` mirrors at 3-4.
  - Alakazam: 6-4 after reclassifying unnamed Alakazam signatures.
  - Mega Starmie: 1-2 after reclassifying unnamed Starmie signatures.
  - Teal Mask Ogerpon: 1-2.
  - Archaludon: 3-2, with two losses; Archaludon is not yet in the main
    `deck_plans.py` label set, so it can leak into `Other`.
  - Crustle: 1-1, Lucario: 2-1.
- Interpretation:
  - The earlier local coverage pool underrepresented live same-sig Marnie
    mirrors, unnamed Alakazam/Starmie variants, and Archaludon. Local RR
    therefore overestimated `high1100_marnie_b8f251a4`.
  - The issue is not simply "missing 900-1100 samples". The current-window
    `cur0810_marnie_b8f251a4` used 2026-08-10 900/1000/1100 bands but was
    not enough to recover Kaggle strength. Future Marnie retraining should
    be a new large recipe: all-date 1100+ plus recent 900-1100 winner-only
    and explicit mirror/Alakazam/Starmie/Archaludon coverage, with those
    opponents included in validation.
- Downgrade `high1100_marnie_b8f251a4` from "strongest current submission
  candidate" to "local-pool false positive until live-opponent coverage is
  rebuilt".

2026-08-12 high+climb winner retraining implementation:

- Objective:
  - Test whether `high1100` specialists underperforming early Kaggle ladder is
    caused by missing 600-1100 climb opponents/decks.
  - Train from scratch, not fine-tune: all-date 1100+/1200+ data remains the
    main corpus, and recent 600-1099 winner-only games from stable high-score
    teams on the same deck sig are added through `--aux-corpus`.
- Remote workspace:
  `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`.
- Builder script uploaded to:
  `/tmp/build_high_climb_aux.py`.
- Source corpus:
  `data/bc_corpus_banded_v12_0701_0810_merged_hist32_log128_board12_20260811`.
- High target manifest:
  `logs/high1100_specialists_20260811/manifest.csv`.
- Generated auxiliary corpora:
  - raw:
    `data/bc_corpus_high_climb_winners_20260812_raw`
  - balanced:
    `data/bc_corpus_high_climb_winners_20260812_balanced`
- Generated logs/manifests:
  - `logs/high_climbwin_20260812/build_high_climb_aux.log`
  - `logs/high_climbwin_20260812/stable_high_teams.csv`
  - `logs/high_climbwin_20260812/aux_file_summary.csv`
  - `logs/high_climbwin_20260812/aux_deck_sig_summary.csv`
  - `logs/high_climbwin_20260812/aux_opponent_coverage_balanced.csv`
  - `logs/high_climbwin_20260812/train_manifest_balanced.csv`
- Aux construction result:
  - 8 target deck sigs, 49 stable high teams.
  - 58 raw/balanced source files.
  - 283 balanced opponent coverage rows.
  - Balanced aux rows by sig:
    - Alakazam `7f9a538936e3`: 6,049 rows / 80 game-file chunks,
      11 opponent archetypes.
    - Dragapult `cc2e995b5ad0`: 25,687 rows / 209 chunks,
      13 opponent archetypes.
    - Marnie `b8f251a476e7`: 28,663 rows / 350 chunks,
      13 opponent archetypes.
    - Marnie `2c22fa761816`: 12,664 rows / 317 chunks,
      11 opponent archetypes.
    - Mega Lopunny `f1445356c3a7`: 11,297 rows / 150 chunks,
      8 opponent archetypes.
    - Mega Lucario `43d6d8b0fce9`: 4,407 rows / 66 chunks,
      8 opponent archetypes.
    - Ogerpon `2bd9da52c43a`: 6,694 rows / 95 chunks,
      9 opponent archetypes.
    - Ogerpon `90abbfb0eee0`: no recent same-sig climb-winner aux rows
      found; training still runs on all-date high main data.
- Training runner started on ks:
  - PID family includes `train_shadow_manifest.py
    logs/high_climbwin_20260812/train_manifest_balanced.csv`.
  - Runner log: `logs/high_climbwin_20260812/runner.log`.
  - Per-job logs:
    `logs/high_climbwin_20260812/train_runner_logs/`.
  - Checkpoints:
    `checkpoints/high_climbwin_20260812/w4/`.
  - GPUs: `0,1,2,3`, one job per GPU, `--cuda-memory-gb 24`,
    `--batch-size 1024`, width 4 pointer architecture, no `--init`.
  - First batch launched: Alakazam, Dragapult, Marnie b8f, Marnie 2c22.
    Dragapult finished quickly and Mega Lopunny was launched next.
- Post-training evaluation watcher:
  - Script: `/tmp/run_high_climbwin_eval.sh`.
  - Log: `logs/high_climbwin_20260812/post_eval.log`.
  - It waits for the training runner to exit, skips eval if failed jobs are
    reported, then runs:
    - `eval_manifest_random.py` 500 games / 32 workers to
      `logs/high_climbwin_20260812/random_g500.csv`.
    - `eval_baseline_delta.py` with all highclimb candidates vs
      `logs/current_rr_pool_20260811/rr_manifest_coverage.csv`, using
      existing `high1100_marnie_grimmsnarl_b8f251a4` as shared baseline,
      100 games / 32 workers, output
      `logs/high_climbwin_20260812/highclimb_vs_coverage_g100.csv`.
    - `baseline_delta_archetype_matrix.py` to
      `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_arch_matrix.csv`.
    - `rr_ladder_weighted_score.py` with 0810 player and decision-row
      distributions to:
      `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_ladder0810_players_weighted.csv`
      and
      `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_ladder0810_rows_weighted.csv`.
- DVH comparison watcher:
  - User noted DVH should remain a reference baseline after this experiment.
  - Script uploaded to `/tmp/run_high_climbwin_dvh_compare.sh`.
  - Log: `logs/high_climbwin_20260812/dvh_compare.log`.
  - It waits for
    `logs/high_climbwin_20260812/highclimb_vs_coverage_g100.csv`, then runs
    `dvh_marnie_b8f` from
    `logs/dvh_vs_high_20260811/unpack/policy.npz` +
    `logs/dvh_vs_high_20260811/unpack/deck.csv` against the same
    `logs/current_rr_pool_20260811/rr_manifest_coverage.csv` opponent pool
    using `eval_round_robin.py --candidate-only`, 100 games / 32 workers.
  - It then joins DVH WR with the highclimb candidate CSV and writes:
    - `logs/high_climbwin_20260812/dvh_vs_coverage_g100.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_dvh_g100_join.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_dvh_g100_summary.csv`
- Notes:
  - This is still BC from scratch, not micro-finetuning. It is a data-window
    repair experiment for the specific high1100 early-ladder failure mode.
  - `scp` once failed due local system ssh config permissions; explicit
    `scp -F /home/jie/.ssh/config ... ks:...` worked.

2026-08-12 high+climb winner retraining results:

- Training completed successfully:
  - `logs/high_climbwin_20260812/runner.log`
  - 8/8 jobs done, 0 failed.
  - Final checkpoints:
    `checkpoints/high_climbwin_20260812/w4/bc2_highclimb_*_w4.npz`.
- Post-eval and DVH comparison completed:
  - `logs/high_climbwin_20260812/post_eval.log` ended with
    `post-eval done 2026-08-12T08:15:52+08:00`.
  - `logs/high_climbwin_20260812/dvh_compare.log` ended with
    `dvh-compare done 2026-08-12T08:17:12+08:00`.
  - Main outputs:
    - `logs/high_climbwin_20260812/random_g500.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_coverage_g100.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_arch_matrix.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_ladder0810_players_weighted.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_coverage_g100_ladder0810_rows_weighted.csv`
    - `logs/high_climbwin_20260812/dvh_vs_coverage_g100.csv`
    - `logs/high_climbwin_20260812/highclimb_vs_dvh_g100_summary.csv`
- Random g500:
  - `highclimb_marnie_grimmsnarl_b8f251a4`: 1.000.
  - `highclimb_alakazam_7f9a5389`: 0.990.
  - `highclimb_mega_lopunny_f1445356`: 0.982.
  - `highclimb_mega_lucario_43d6d8b0`: 0.972.
  - `highclimb_marnie_grimmsnarl_2c22fa76`: 0.926.
  - `highclimb_teal_mask_ogerpon_90abbfb0`: 0.788.
  - `highclimb_teal_mask_ogerpon_2bd9da52`: 0.764.
  - `highclimb_dragapult_cc2e995b`: 0.748.
- Coverage pool, 0810 player-distribution climb-weighted:
  - Existing baseline `high1100_marnie_grimmsnarl_b8f251a4`: 0.7547.
  - Best new model `highclimb_marnie_grimmsnarl_b8f251a4`: 0.7181.
  - `highclimb_alakazam_7f9a5389`: 0.6429.
  - `highclimb_mega_lopunny_f1445356`: 0.5404.
  - `highclimb_mega_lucario_43d6d8b0`: 0.4647.
  - `highclimb_marnie_grimmsnarl_2c22fa76`: 0.4574.
  - `highclimb_teal_mask_ogerpon_90abbfb0`: 0.3912.
  - `highclimb_dragapult_cc2e995b`: 0.2923.
  - `highclimb_teal_mask_ogerpon_2bd9da52`: 0.1892.
- Relative to `high1100_marnie_grimmsnarl_b8f251a4` shared baseline:
  - `highclimb_marnie_grimmsnarl_b8f251a4`: weighted_delta -0.050,
    candidate avg 0.745 vs baseline avg 0.765; lost 8/13; worst Crustle
    -0.160.
  - `highclimb_alakazam_7f9a5389`: weighted_delta -0.194; worst TR Mewtwo
    -0.470.
  - All other highclimb candidates are much worse: weighted_delta from -0.279
    to -0.586.
- Relative to DVH (`logs/dvh_vs_high_20260811/unpack/policy.npz`):
  - DVH macro over this coverage pool: 0.772.
  - `highclimb_marnie_grimmsnarl_b8f251a4`: weighted_delta_vs_dvh -0.063,
    avg_delta -0.027; lost 9/13; worst Crustle -0.100.
  - `highclimb_alakazam_7f9a5389`: weighted_delta_vs_dvh -0.207; worst TR
    Mewtwo -0.440.
  - Other new models are not close to DVH.
- Important matchup matrix points:
  - Existing `high1100_marnie_b8f`: macro 0.765; worst Ogerpon 0.530.
  - DVH Marnie b8f: macro 0.772; Ogerpon 0.540, Crustle 0.750, Alakazam
    0.840, Marnie mirror 0.780.
  - New `highclimb_marnie_b8f`: macro 0.745; improves Ogerpon slightly to
    0.570 but drops Crustle to 0.650 and Marnie mirror to 0.680.
  - New `highclimb_alakazam`: macro 0.694 but hard-loses TR Mewtwo at 0.400.
- Interpretation:
  - The high+recent-climb-winner recipe did not repair the early-ladder
    failure. It slightly shifts Marnie b8f toward Ogerpon but damages other
    important matchups, especially Crustle and mirror.
  - Do not prioritize this cohort for Kaggle submission. If a submission is
    needed from this cohort for ablation, only `highclimb_marnie_b8f` is
    defensible, and it is still below both existing high1100 Marnie and DVH
    on local coverage.
  - The result argues against "just add recent 600-1100 winner-only climb data"
    as the main fix. The missing ingredient is likely policy/trajectory/rule
    quality or better team-specific clean behavior, not this coarse aux-corpus
    mix.

2026-08-12 high+climb degradation diagnosis:

Question addressed: why did high+climb degrade, were weights/settings at
fault, and does the older RR pool tell a different story?

Important submission policy update:
- DVH artifacts, including `/tmp/submission_marnie_dvh.tar.gz` and
  `logs/dvh_vs_high_20260811/unpack/policy.npz`, are local comparison
  baselines only.
- Do not recommend DVH for future Kaggle submissions unless the user
  explicitly reverses this. Use DVH only as a reference line for RR/replay
  comparisons.

- Old high1100 pool re-evaluation:
  - Script used: `/tmp/run_highclimb_oldpool_eval.py`.
  - Output:
    `logs/high_climbwin_20260812/highclimb_vs_high1100_oldpool_g100.csv`,
    `logs/high_climbwin_20260812/highclimb_vs_high1100_oldpool_g100_summary.csv`,
    `logs/high_climbwin_20260812/highclimb_vs_high1100_oldpool_g100_arch_matrix.csv`.
  - Against the old high1100 pool, `highclimb_marnie_grimmsnarl_b8f251a4`
    was only slightly worse than `high1100_marnie_grimmsnarl_b8f251a4`:
    avg_delta -0.010, weighted_delta -0.012, candidate WR 0.700 vs baseline
    0.710.
  - Other high+climb models were still poor on the old pool:
    `highclimb_alakazam_7f9a5389` weighted_delta -0.214,
    `highclimb_dragapult_cc2e995b` -0.502, `highclimb_ogerpon_90abbfb0`
    -0.413, `highclimb_ogerpon_2bd9da52` -0.510.
  - This means the Marnie b8f degradation is not a total collapse, but the
    high+climb recipe should not be generalized across all deck sigs.

- Training/optimization confound:
  - Old `high1100_marnie_b8f` was trained with `batch_size=512`, 6 epochs,
    best val 0.7314.
  - New high+climb Marnie b8f used `batch_size=1024`, 6 epochs, same LR scale,
    best val 0.7491.
  - Because epoch count stayed fixed while batch doubled, optimizer update
    steps roughly halved. This is a real confound, not just score noise.

- Continuation experiment for Marnie b8f:
  - Runner scripts:
    `/tmp/run_highclimb_marnie_continue.py`,
    `/tmp/run_highclimb_marnie_continue_seed25.py`,
    `/tmp/run_highclimb_marnie_continue_eval.py`.
  - Checkpoints:
    `checkpoints/high_climbwin_20260812/continue/bc2_highclimb_marnie_b8f_cont_b1024_lr4e5_ep4.npz`,
    `.../bc2_highclimb_marnie_b8f_cont_b512_lr3e5_ep4.npz`,
    `.../bc2_highclimb_marnie_b8f_cont_b1024_lr4e5_ep4_seed25.npz`,
    `.../bc2_highclimb_marnie_b8f_cont_b512_lr3e5_ep4_seed25.npz`.
  - Same-split seed25 continuation reduced val from the original high+climb
    0.7491 to:
    - batch1024/lr4e-5/4ep: best val 0.7346.
    - batch512/lr3e-5/4ep: best val 0.7333.
  - Therefore, insufficient update steps was a confirmed cause of part of
    the degradation. Bigger batches must be paired with more epochs/update
    steps or schedule changes.

- Continuation RR results:
  - Outputs:
    `logs/high_climbwin_20260812/continue_marnie_b8f_eval/marnie_continue_vs_oldhigh1100_g100_summary.csv`,
    `logs/high_climbwin_20260812/continue_marnie_b8f_eval/marnie_continue_vs_coverage_g100_summary.csv`,
    plus corresponding `*_arch_matrix.csv`.
  - Old high1100 pool:
    - Original highclimb Marnie b8f: weighted_delta -0.0157.
    - cont_b1024_lr4e5_ep4: +0.0274.
    - cont_b512_lr3e5_ep4: +0.0493.
    - cont_b1024_lr4e5_ep4_seed25: +0.0305, no negative matchup in this
      8-opponent pool.
    - cont_b512_lr3e5_ep4_seed25: -0.0077.
  - Current coverage pool:
    - Original highclimb Marnie b8f: weighted_delta -0.0582, avg WR 0.728.
    - cont_b1024_lr4e5_ep4: weighted_delta -0.0122, avg WR 0.748.
    - cont_b512_lr3e5_ep4: weighted_delta -0.0359, avg WR 0.749.
    - cont_b1024_lr4e5_ep4_seed25: weighted_delta +0.0076, avg WR 0.775.
    - cont_b512_lr3e5_ep4_seed25: weighted_delta -0.0093, avg WR 0.759.
  - Coverage matrix highlights:
    - `cont_b1024_lr4e5_ep4_seed25` macro 0.775 vs baseline 0.772, but worst
      archetype is Festival Lead 0.490.
    - It improves broad metrics, but still has matchup tradeoffs; it is not a
      general proof that high+climb winner-only is correct.

- Practical interpretation:
  - The original high+climb degradation had at least two components:
    1. Undertraining from larger batch with fixed epoch count.
    2. Data/weight-induced matchup tradeoff from adding climb winner-only aux
       at win/loss/draw weights 1.5/0.4/0.8.
  - The aux corpus did not globally swamp the data; for Marnie b8f it was only
    about 28.7k rows out of ~1.10M kept rows. Its impact is more likely from
    biased conditional winner-only behavior and rare-action/matchup shifts.
  - Do not conclude "more recent winner-only climb data fixes early ladder".
    Do conclude that future large-batch BC training should be specified by
    update steps, not only epochs.
  - If a high+climb Marnie ablation is needed, the best local candidate from
    this diagnostic is
    `checkpoints/high_climbwin_20260812/continue/bc2_highclimb_marnie_b8f_cont_b1024_lr4e5_ep4_seed25.npz`,
    but it is still an ablation, not a primary strategy replacement.

## Update Checklist

## 2026-08-12 Marnie b8f Live Replay Drop Analysis

Context:

- Submitted model: `submission_high900win_old_marnie_b8f_20260812`,
  Kaggle submission id `55455876`, score fluctuated around 920-970 after an
  early climb.
- Submitted checkpoint/deck:
  `checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_marnie_grimmsnarl_b8f251a4_w4.npz`
  with
  `logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv`.

Latest replay pull on ks:

```bash
python3 tools/analyze_kaggle_replays.py 55455876 \
  --team-name "Jie Orkarin" \
  --deck logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --known-decks-dir logs/ladder_pool_0805_0809_fast_all/decks \
  --known-decks-dir logs/ladder_pool_0809_fast_all/decks \
  --known-decks-dir logs/ladder_pool_0805_all/decks \
  --known-decks-dir logs/ladder_pool_0804_all/decks \
  --cache-dir logs/kaggle_replay_high900win_marnie_b8f_20260812/cache_latest \
  --out logs/kaggle_replay_high900win_marnie_b8f_20260812/episodes_latest_teamdetect.csv \
  --summary-out logs/kaggle_replay_high900win_marnie_b8f_20260812/summary_latest_teamdetect_by_deck_sig.csv \
  --group-by opponent_deck_sig \
  --write-opponent-decks \
  --opponent-decks-dir logs/kaggle_replay_high900win_marnie_b8f_20260812/opponent_decks_latest_teamdetect \
  --progress-every 20 --timeout 180
```

Replay result:

- 44 episodes fetched, one self-validation mirror with both teams
  `Jie Orkarin` skipped.
- 43 analyzable games: 26W / 17L, WR 0.605.
- Main loss clusters:
  - `b8f251a476e7` Marnie mirror: 6/10.
  - `f1445356c3a7` Mega Lopunny: 1/4.
  - `7f9a538936e3` Alakazam: 4/6.
  - `cc2e995b5ad0` Dragapult: 1/2.
  - Single losses: `65c1997e6846` Crustle Wall,
    `ab7e4b818773` Teal Mask Ogerpon, `ca082d1c1eab` Alakazam,
    `e82dcbe62260` Festival Lead, `ecb67fcd9c0b` Alakazam,
    `140f7d8b2f09` Dragapult.
- By archetype from latest replay:
  - Marnie Grimmsnarl 7/11.
  - Alakazam 6/10.
  - Mega Lopunny 2/6.
  - Dragapult 1/3.
  - Crustle Wall 2/3.
  - Festival Lead 1/2.
  - Teal Mask Ogerpon 1/2.

Training coverage audit:

- Script used: `/tmp/audit_marnie_b8f_replay_coverage_fast_20260812.py`.
- Latest output:
  `logs/kaggle_replay_high900win_marnie_b8f_20260812/coverage_latest/replay_loss_training_coverage.csv`.
- Key finding: the drop is not mostly because the opponent archetypes were
  unseen.
  - mirror `b8f251a476e7`: 6894 train games, train WR 0.7485.
  - `f1445356c3a7` Mega Lopunny: 893 train games, train WR 0.7122.
  - `7f9a538936e3` Alakazam: 2386 train games, train WR 0.8424.
  - `cc2e995b5ad0` Dragapult: 402 train games, train WR 0.7637.
  - exact `65c1997e6846` Crustle: 0 train games.
  - exact `ab7e4b818773` Ogerpon: 1 train game, loss.
- Interpretation: exact-sig gaps exist for a few single losses, but the major
  live loss mass is against well-covered arch/sigs. The larger issue is likely
  live opponent strength / strategy quality and imperfect local shadow, not
  just deck signature missingness.

Live opponent local validation:

- Built manifest:
  `/tmp/build_live_opponent_eval_manifest_20260812.py`
  ->
  `logs/kaggle_replay_high900win_marnie_b8f_20260812/live_opponent_eval_manifest.csv`.
- 24 unique replay opponent decks, 14 matched to existing local policies.
- Local baseline-delta:
  `logs/kaggle_replay_high900win_marnie_b8f_20260812/live_opponent_eval/old_vs_cur0810_b8f_g100.csv`.
- Old submitted b8f local WR against matched live opponents:
  - mirror 0.720.
  - f144 Lopunny 0.690.
  - 7f9 Alakazam 0.710.
  - cc2e Dragapult 0.870.
  - e82 Festival 0.640.
  - 5899 Ogerpon 0.130.
- Current RR b8f was worse than old b8f against all 14 matched live opponents:
  avg delta -0.188, candidate avg 0.545 vs old avg 0.733.
- Interpretation:
  - Existing local shadow/current_rr opponents still under-represent true live
    opponent quality. For example local old b8f predicts high WR vs f144/7f9,
    but live samples had severe f144 pressure.
  - The `cur0810` b8f variant should not replace the old high900win method.

Targeted supplemental data:

- Built live-hard winner aux from b8f winning games against replay-loss
  sigs/archetypes:
  `/tmp/build_marnie_b8f_livehard_aux_20260812.py`.
- Output corpus:
  `data/bc_corpus_livehard_marnie_b8f_20260812/Marnie_Grimmsnarl/livehard_win/*.npz`.
- Summary:
  - candidate_games 19350, selected_games 5250, files 56, rows 515386.
  - by arch rows:
    - Marnie 109524.
    - Mega Lopunny 93446.
    - Alakazam 87420.
    - Crustle 66583.
    - Dragapult 61784.
    - Ogerpon 53030.
    - Festival 43599.
  - key exact rows: b8f 98612, f144 70215, 7f9 60324,
    cc2e 34201, e82 23220, 5899 5698.
- Caveat: builder currently used `episodes_teamdetect.csv` from the earlier
  42-game pull. The latest 43-game pull only added one more f144 loss; f144 is
  already heavily represented in live-hard aux. Update the script constants to
  `episodes_latest_teamdetect.csv` before future reuse.

Active targeted training:

- Runner: `/tmp/run_marnie_b8f_livehard_oldmethod_20260812.sh`.
- Logs:
  `logs/marnie_b8f_livehard_oldmethod_20260812/runner.log`,
  `logs/marnie_b8f_livehard_oldmethod_20260812/train/marnie_b8f_high900_livehard_w1_seed91.log`,
  `.../marnie_b8f_high900_livehard_w2_seed92.log`.
- Checkpoints:
  `checkpoints/marnie_b8f_livehard_oldmethod_20260812/bc2_marnie_b8f_high900_livehard_w1_seed91.npz`,
  `.../bc2_marnie_b8f_high900_livehard_w2_seed92.npz`.
- Recipe:
  - old pointer width4, scratch training, no history.
  - base: v12 all-date 1100+ and 1200+ b8f.
  - aux: high900 winners plus livehard wins.
  - W1: one livehard aux.
  - W2: livehard aux duplicated as coarse x2 weight.
  - epochs 6, batch 512, lr 8e-5.
- W1 completed at 2026-08-12 21:45:
  - best/final val loss 0.6510, val first_action_acc 0.798.
  - early random audit:
    `logs/marnie_b8f_livehard_oldmethod_20260812/eval_w1_early/random_w1_g500.log`
    = 500/500, 100.0%.
  - early live-opponent baseline-delta:
    `logs/marnie_b8f_livehard_oldmethod_20260812/eval_w1_early/livehard_w1_vs_old_liveopponents_g100.csv`.
    Summary: avg_delta +0.008, weighted_delta +0.008,
    livehard_w1 0.737 vs old_b8f 0.729, lost 5/14 opponents.
  - W1 gains: 7f9 Alakazam +0.090, ca082 Alakazam +0.140,
    f144 Lopunny +0.050, 43d6 Lucario +0.040, e82 Festival +0.030.
  - W1 regressions: 5899 Ogerpon -0.100, two Crustle opponents -0.060
    and -0.130, cc2e Dragapult -0.040. This is not yet a submit-worthy
    improvement because it hurts important hard matchups.
- W2 completed at 2026-08-12 22:30; runner done at 22:34.
  - random audit: 500/500, 100.0%.
  - coverage/current_rr matrix improved over old b8f:
    macro 0.788 vs old 0.767; players-weighted all 0.7593 vs old 0.7440;
    worst archetype Teal Mask Ogerpon 0.580 vs old worst Festival Lead 0.520.
  - coverage gains: Crustle 0.870 vs old 0.830, Festival 0.680 vs old
    0.520, Mega Lucario 0.790 vs old 0.760.
  - coverage regressions: Alakazam 0.750 vs old 0.720 is still a gain but
    much lower than W1's 0.810; Mega Lopunny 0.780 vs old 0.820.
- Additional live-opponent replay manifest evaluation completed after runner:
  `logs/marnie_b8f_livehard_oldmethod_20260812/eval/livehard_w2_vs_old_liveopponents_g100.csv`.
  Summary: avg_delta -0.014, weighted_delta -0.014, livehard_w2 0.738
  vs old_b8f 0.752, lost 9/14 opponents. Key issue: f144 Mega Lopunny
  regressed 0.770 -> 0.650 and 5899 Ogerpon stayed extremely weak
  0.110 -> 0.090.
- Interpretation: W2 looks better on the generic coverage/current_rr pool but
  worse on the actual recent live replay opponent pool. Treat W2 as not
  submit-worthy unless later ladder environment differs sharply from this live
  opponent set.

Offline action diagnostics:

- First attempt had a shell quoting bug because `--archetype "Marnie Grimmsnarl"`
  was expanded from a plain string. It was killed.
- Corrected runner:
  `/tmp/run_marnie_b8f_offline_accuracy_20260812.sh`.
- Output:
  `logs/kaggle_replay_high900win_marnie_b8f_20260812/offline_accuracy_fixed/`.
- It evaluates old submitted b8f action accuracy on main 1100+/1200+ and
  high900 winner aux by opponent sig:
  b8f, f144, 7f9, cc2e, e82, 5899.
- Completed at 2026-08-12 21:58.
- Outputs:
  `accuracy_main_by_sig.csv` and `accuracy_auxwin_by_sig.csv` in
  `logs/kaggle_replay_high900win_marnie_b8f_20260812/offline_accuracy_fixed/`.

Immediate next steps:

- When live-hard W2 finishes, compare:
  - `logs/marnie_b8f_livehard_oldmethod_20260812/eval/random_g500.csv`
  - `.../vs_coverage_g100.csv`
  - `.../vs_coverage_g100_arch_matrix.csv`
  - `.../vs_coverage_g100_ladder0810_players_weighted.csv`
  against old submitted b8f.
- Also run W1/W2 against live opponent manifest:

```bash
python3 tools/eval_baseline_delta.py \
  --baseline old_b8f=checkpoints/high900win_oldmethod_20260812/w4/bc2_high900win_old_marnie_grimmsnarl_b8f251a4_w4.npz:logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --candidate livehard_w1=checkpoints/marnie_b8f_livehard_oldmethod_20260812/bc2_marnie_b8f_high900_livehard_w1_seed91.npz:logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --candidate livehard_w2=checkpoints/marnie_b8f_livehard_oldmethod_20260812/bc2_marnie_b8f_high900_livehard_w2_seed92.npz:logs/ladder_pool_0805_0809_fast_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --opponent-manifest logs/kaggle_replay_high900win_marnie_b8f_20260812/live_opponent_eval_manifest.csv \
  --skip-bad-entries --games 100 --workers 24 --max-turns 700 --progress-every 25 \
  --out-csv logs/kaggle_replay_high900win_marnie_b8f_20260812/live_opponent_eval/livehard_vs_old_b8f_g100.csv
```

Decision rule:

- Do not submit live-hard only because it improves val loss.
- Consider it only if it improves f144/Lopunny, 7f9/Alakazam, mirror, and
  e82/Festival locally without worsening 5899/Ogerpon and broad coverage
  matrix. Otherwise keep submitted old b8f as the current Marnie baseline.

## Update Checklist

Whenever changing active state, update:

- Date/time at top of this file.
- Active remote jobs and output paths.
- Kaggle account/score monitor status.
- Latest submission IDs and observed scores.
- New corpus/checkpoint feature dimensions.
- New conclusions from round-robin, Kaggle replay, or failure traces.
- Any commands that the next agent should continue or avoid.

## Dragapult Pipeline v13 Work 2026-08-13

User request: completed Dragapult specialists should be evaluated first, then
move to bottom-level BC pipeline/encoder changes because prior Dragapult runs
remain weak.

Completed non-traj/non-history Dragapult eval on ks:

- Random output:
  `logs/dragapult_specialists_20260813/completed_nontraj_random_g500.csv`
- RR/baseline-delta output:
  `logs/dragapult_specialists_20260813/completed_nontraj_vs_coverage_g100.csv`
- Matrix:
  `logs/dragapult_specialists_20260813/completed_nontraj_vs_coverage_g100_arch_matrix_with_random.csv`
- 0812 ladder weighted:
  `logs/dragapult_specialists_20260813/completed_nontraj_vs_coverage_g100_ladder0812_appearances_weighted.csv`

Results:

- `dragapult_cc2e_all900_old_w4_ep16`: random 0.944, coverage macro 0.546,
  avg delta +0.079 vs old high900 Dragapult, 0812 climb weighted 0.4969.
  Worst: Crustle 0.150; Mega Lucario 0.300.
- `dragapult_cc2e_all900_rawmetric_w4_ep16`: random 0.924, macro 0.528,
  avg delta +0.062. Better Crustle 0.220 but worse Ogerpon 0.360.
- `dragapult_cc2e_recent0805_0811_w4_ep14`: random 0.834, macro 0.496.
- `dragapult_cc2e_high900aux_old_w4_ep14`: random 0.894, macro 0.483.
- `dragapult_cc2e_kh0a_all900_w4_ep14`: random 0.732, macro 0.321,
  lost 13/13 vs old baseline.
- `dragapult_7ac_all900_w4_ep14`: random 0.734, macro 0.234.

Conclusion:

- Current Dragapult remains non-submit-worthy. Wide all900 data is best, but
  still far below target. Narrow recent/high900/team-specific selection does
  not fix it and often collapses.
- This is likely a bottom-level BC pipeline/representation problem rather than
  a simple date/weight/signature problem.

New local branch:

- `dragapult-pipeline-v13`

Implemented local v13 bottom-level changes:

- `ptcg_rl/encoder.py`
  - Added `STATE_TOKEN_FEAT_DIM = 24`.
  - `EncodedDecision` now carries `state_token_feats`.
  - `FastEncoder` emits per board/hand token scalar features:
    owner, active/bench, hp, damage, energy, stage, ex/mega, retreat, tools,
    attack-ready, skill, primary/setup/evolution/engine flags, attack damage
    metadata, and Dragapult-line flag.
- `tools/bc_extract_v2.py`
  - Feature version now `v13_state_token_multistream_history`.
  - Saves `state_token_feats` and `state_token_feat_dim`.
- `ptcg_rl/bc2/data.py`
  - `BCCorpus` accepts `state_token_feat_dim`.
  - Collates `state_token_feats`, zero-filling for older corpora.
- `ptcg_rl/model.py`
  - `CrossAttentionPolicyValueNet` can use `state_token_feat_dim`.
  - New `state_token_feat_fc` projects per-token scalar features into state
    tokens. Pointer models remain compatible.
  - Added `checkpoint_state_token_feat_dim`.
- `ptcg_rl/numpy_policy.py`
  - NumPy cross-attn inference loads and uses `state_token_feat_fc`.
  - Live `FastEncoder` state-token features are passed into `_encode_state_cross`.
- `ptcg_rl/bc2/losses.py`, `tools/bc2_train.py`
  - Added `--multi-select-sequence-weight` to reduce ordered sequence NLL for
    multi-select labels while using stronger order-free set loss.
  - `sequence_loss_parts` now passes `state_token_feats` into model state
    encoding.
- `ptcg_rl/bc2/decode.py`, `tools/bc2_accuracy.py`
  - Updated diagnostics to pass `state_token_feats`.

Local verification:

- `python3 -m py_compile` passed for modified files.
- Torch forward smoke passed:
  cross-attn width1 with state token feat dim 24 produced
  `h=[3,256]`, `opts=[3,5,128]`, `logits=[3,6]`.
- Local NumPy load smoke cannot run because local environment lacks `cg`; run
  ks smoke after syncing.

Immediate next steps:

1. Sync branch/code to ks.
2. On ks run smoke tests:
   - import/py_compile.
   - build/load `NumpyPolicy` with v13 cross-attn dummy checkpoint.
   - run tiny `bc_extract_v2.py --max-episodes` and inspect `state_token_feats`
     shape.
3. Re-extract a Dragapult-focused v13 corpus, preferably enough dates for
   0801-0812 or full available dates depending time.
4. Train AB:
   - v13 cross-attn state-token features, no history/traj.
   - old pointer/cross-attn baseline on same v13 corpus with
     `--state-token-feat-dim 0` if needed.
   - Use set-heavy multi-select config, e.g.
     `--set-loss-weight 0.8 --set-loss-min-count 2 --set-loss-negative-weight 0.15
      --multi-select-sequence-weight 0.25`.
5. Evaluate with the exact same Dragapult random and coverage RR used above.

Update 2026-08-13 12:55 CST:

- Local branch `dragapult-pipeline-v13` is committed at `2bd6c4a`
  (`Add v13 state token BC pipeline`).
- Code has been synced to ks workspace:
  `/home/jie/Do/0_PTCG/workspace/ptcg_rl_git_v7_baseline_20260804`.
- Remote smoke passed:
  `NumpyPolicy` loaded a dummy cross-attn checkpoint with
  `_state_token_feat_dim=24`.
- Tiny v13 extraction smoke passed:
  `state_token_feats` shape is `(37, 24)`, feature version is
  `v13_state_token_multistream_history`.
- Full v13 0801-0812 extraction is running on ks:
  `data/bc_corpus_banded_v13_0801_0812_state_token_20260813`.
  Logs:
  `logs/dragapult_v13_pipeline_20260813/extract_v13_0801_0812.log`.
- The original runner shell was accidentally killed while checking a false
  Dragapult ID alarm; the extract worker process survived. A watcher was
  started to continue automatically after extraction:
  `/tmp/watch_and_train_dragapult_v13_20260813.sh`,
  log `logs/dragapult_v13_pipeline_20260813/watcher.log`.
- Important: Dragapult card IDs were rechecked by `cardId`, not list index:
  `119 Dreepy`, `120 Drakloak`, `121 Dragapult ex`. The v13 constant
  `{119, 120, 121}` is correct.
- Watcher will train/evaluate three AB models after extraction:
  `cross_w3_state_token`, `cross_w4_state_token`, and
  `cross_w4_no_state_token`, all against
  `cc2e995b5ad0_dragapult_kh0a.csv`, with the old all900 w4 Dragapult model
  as baseline.

Update 2026-08-13 20:15 CST:

- ks active workspace after server recovery is:
  `/data/jie/ptcg_rl_git_v7_baseline_20260804`.
- v13 0801-0812 extraction completed:
  `data/bc_corpus_banded_v13_0801_0812_state_token_20260813`
  with 729 `.npz` files. Smoke sample:
  `state=(80,)`, `option=(3,64)`, `state_token=(37,24)`,
  `feature_version=v13_state_token_multistream_history`.
- Dragapult v13 training completed successfully, 4/4 jobs:
  `checkpoints/dragapult_v13_0801_0812_20260813/`
  final models:
  `bc2_dragapult_cc2e_v13_cross_w3_state_token.npz`,
  `bc2_dragapult_cc2e_v13_cross_w4_state_token.npz`,
  `bc2_dragapult_cc2e_v13_cross_w4_no_state_token.npz`,
  `bc2_dragapult_cc2e_v13_recent0805_cross_w4_state_token.npz`.
- 0812 score-labeled shadow training completed successfully, 81/81 jobs:
  `checkpoints/shadow_0812_v13_cross_w3_20260813/`.
  Enriched manifest:
  `logs/restore_0812_dragapult_v13_20260813/shadow_manifest_0812_v13_cross_w3_enriched.csv`.
  Names include source/climb score labels like `shadow_s1117_...`.
- Final models and key logs/manifests/decks have been pulled back locally,
  excluding epoch checkpoints:
  `remote_artifacts/restore_0812_dragapult_v13_20260813/`.
  Local verification found 4 Dragapult final `.npz` and 81 shadow final `.npz`.
  Do not commit this directory; it is a local artifact backup.
- ks random evaluation is running in background:
  pid file `logs/eval_restore_0812_v13_20260813/rr_runner.pid` is for RR,
  random process was started from `/tmp/run_random_restore_0812_v13_20260813.sh`.
  Random outputs:
  `logs/eval_restore_0812_v13_20260813/dragapult_v13_random_g500.csv`,
  `logs/eval_restore_0812_v13_20260813/shadow_0812_v13_random_g500.csv`.
  Dragapult random g500 completed:
  `w3_state_token=0.978`, `w4_state_token=0.984`,
  `w4_no_state_token=0.958`, `recent0805_w4_state_token=0.956`.
- ks RR / baseline-delta evaluation is running in background:
  `logs/eval_restore_0812_v13_20260813/rr_runner.pid`.
  Outputs:
  `logs/eval_restore_0812_v13_20260813/rr_dragapult_v13_candidates_g300.csv`
  and
  `logs/eval_restore_0812_v13_20260813/dragapult_v13_vs_shadow81_g80.csv`.
  First phase is 4 Dragapult candidate RR, then it compares w4/no-state/recent
  against w3 over all 81 enriched shadow opponents.

Update 2026-08-13 20:35 CST:

- Dragapult v13 random g500 was checked in detail. All four candidate rows had
  `timeouts=0`, so the gap to 100% is not an engine timeout or illegal-action
  fallback issue.
  Results:
  `w3_state_token=0.978`, `w4_state_token=0.984`,
  `w4_no_state_token=0.958`, `recent0805_w4_state_token=0.956`.
- Interpretation: Dragapult random weakness is partly deck/style variance
  because same-deck random can occasionally attack/prize-race if our setup
  stalls, but `0.956-0.984` is still below the reliability expected from a
  submit-ready specialist. Treat it as a real BC/setup-coherence problem.
- A low-CPU trace job was started to inspect `w4_state_token` losses versus
  same-deck random:
  `logs/eval_restore_0812_v13_20260813/trace_random/drag_w4_vs_random_g800.*`.
  It records win/loss decision summaries for attack/evolve/attach/play misses,
  early end, and top policy choices.
- Partial shadow random audit after 50 completed rows:
  mean `0.871`, median `0.910`, min `0.470`, max `0.996`;
  `>=0.97`: 13/50, `>=0.95`: 17/50, `>=0.90`: 27/50.
  Current poor clusters are Dragapult, Crustle, several Ogerpon signatures, and
  some Mega Lopunny/Marnie variants. Do not use raw 81-shadow manifest as a
  trusted ladder proxy. Use the random audit with `tools/split_shadow_pools.py`
  to create trusted/environment/debug pools after random finishes.

Update 2026-08-13 21:18 CST:

- ks random and RR recovery evals completed.
- Full 81-shadow random g500:
  mean `0.855`, median `0.884`, min `0.460`, max `0.996`;
  `>=0.99`: 6/81, `>=0.97`: 16/81, `>=0.95`: 24/81,
  `>=0.90`: 38/81. By archetype, Dragapult shadows are very poor
  (`n=8`, mean `0.615`, max `0.788`), Crustle is also weak
  (`n=8`, mean `0.812`, max `0.906`). Strong clusters include selected
  Marnie/Ogerpon/Festival/Lucario/Starmie shadows. Raw 0812 shadow81 is not a
  trusted ladder pool. Build filtered pools before using it for decisions.
- Dragapult v13 local candidate RR phase:
  `w4_state_token` beats `w3_state_token` head-to-head `0.567`, beats
  `w4_no_state_token` `0.513`, and beats `recent0805` `0.597`.
  `recent0805` is the weakest in this local mirror/candidate set.
- Dragapult vs 81 shadows with `w3_state_token` as baseline:
  `drag_w4` is best: average delta `+0.069`, candidate mean `0.717` vs
  baseline `0.649`, lost only `12/81`. The improvement also holds on
  `opp_random_wr>=0.90` (`+0.074`) and `>=0.97` (`+0.080`) subsets.
  `drag_w4_no_state` is positive but weaker: average delta `+0.039`,
  candidate mean `0.687`, lost `21/81`.
  `drag_recent0805` regresses: average delta `-0.030`, lost `53/81`.
- Dragapult `w4_state_token` random loss trace g800:
  `791-9-0`, WR `0.989`, no timeouts. Loss games are much longer
  (`119.7` decisions/game) than wins (`74.8`), and have worse MAIN miss rates:
  attack `0.813` vs `0.800`, attach `0.748` vs `0.709`, evolve `0.645` vs
  `0.608`, early END `0.033` vs `0.016`. Losses show more digging/play/ability
  activity, which looks like stalled setup or poor finish/prize planning rather
  than a pure opening-randomness issue.
- Dragapult corpus audit for target sig `cc2e995b5ad0` over v13 0801-0812:
  about `438634` decisions, `3894` games, game WR `0.568`.
  Score-band rows are mostly `1000-1099` (`235962`) and `900-999`
  (`124427`), with less `1100-1199` (`41957`) and `1200+` (`36288`).
  Top team styles differ substantially: `flg` `315` games WR `0.667`,
  `213tubo` `477` games WR `0.614`, `Kh0a` `714` games WR `0.580`,
  `Pokemon Siuuuu` `380` games WR `0.529`, `Pokopia` `95` games WR `0.453`,
  `Budew` `90` games WR `0.400`.
- Interpretation for Dragapult:
  data volume is not the main shortage for `cc2e995b5ad0`; the problem is
  mixed/noisy strategy labels plus a training target that still mostly imitates
  one-step actions. The current v13 runs enabled `--state-token-feat-dim 24`
  but did not enable `--history-k`, `--history-summary`, `--step-plan`, or
  `--hierarchical-plan`. Public Dragapult strategy and local trace both say
  the deck needs Dreepy/Drakloak/Dragapult setup coherence, Crispin/energy
  routing, and multi-prize Phantom Dive planning. Treat future Dragapult work
  as a pipeline/strategy-target problem, not as "add more epoch" or "take only
  newer dates".
- Existing strategy seed already records the useful Dragapult hypothesis:
  `data/matchup_strategy_seeds_v1.csv` has
  `dragapult_setup_vs_marnie_crustle`, saying successful rows choose
  Dragapult ex active more often, use Drakloak attach-from, and set up Dreepy
  earlier. This seed has not yet become a strong training constraint; do not
  count it as already solved.
- Recommended next Dragapult direction:
  build a setup-coherent filtered/teacher corpus from scratch rather than
  micro-finetuning. Prefer high-quality `cc2e995b5ad0` teams (`flg`,
  `213tubo`, `Kh0a` with caution, `やる気元気ミワハルキ`, `Petit Canard`) and
  downweight or exclude low-WR teams when training a specialist. Add explicit
  auxiliary targets for early Dreepy count, Drakloak/Dragapult online timing,
  first Phantom Dive turn, correct red/psychic energy route, attack gaps, and
  late prize-map/bench-counter selection. Also split shadow pools by random
  quality before trusting RR results.

Update 2026-08-13 23:10 CST:

- Important history pipeline bugs were found and fixed. These likely explain
  why earlier history/summary experiments looked almost unchanged or worse.
- Bug 1: `BCCorpus.collate(..., history_augment=True)` recomputed
  `history_summary` from the currently enabled raw streams after augmentation.
  For summary-only runs (`history_k=0`, `log_history_k=0`,
  `board_history_k=0`, `history_summary_dim=48`) this silently replaced the
  corpus summary with all zeros. This affected prior summary-only Marnie
  history repair attempts.
- Bug 2: extracted `history_summary` previously included offline
  `opp_hist_*` labeled opponent decisions, but Kaggle live inference can only
  reproduce own decisions, public logs, and board snapshots. Training and live
  inference therefore had mismatched history features. The default training
  path is now live-compatible: `offline_opp_history=False`, and
  `--offline-opp-history` must be explicitly passed for offline diagnostics.
- Fixes applied locally and synced to
  `ks:/data/jie/ptcg_rl_git_v7_baseline_20260804`:
  - `ptcg_rl/bc2/data.py`: summary is rebuilt from raw own/log/board streams
    when available; opponent labeled history is excluded by default; old saved
    summary fallback zeros opponent-label fields; history augmentation no
    longer erases summary.
  - `tools/bc2_train.py`: added `--offline-opp-history`, and logs
    `offline_opp_history=...`.
  - `tools/bc_extract_v2.py`: future extracted summaries are live-compatible
    and do not include offline opponent labeled action history.
  - `ptcg_rl/numpy_policy.py`: added `remember_decision(obs, picks)` and
    `select_mcts(..., update_history=...)`.
  - `main.py`, `tools/eval_bc.py`, `tools/eval_round_robin.py`,
    `tools/eval_rule_overlay_stats*.py`, `tools/build_weakness_state_bank.py`,
    `tools/generate_rollout_bc.py`, `tools/search_action_teacher.py`: rule
    overlay/search/fallback paths now record the final legal action actually
    sent to the engine, not the raw pre-rule model action.
  - `tools/bc2_accuracy.py` and `tools/bc2_failure_report.py`: now load
    `history_summary_dim`; `bc2_failure_report.py` also loads
    `state_token_feat_dim`. Earlier diagnostics for history/state-token models
    may have been incomplete.
  - Added `tools/check_history_summary_integrity.py` for preflight checks.
- Remote smoke tests on the real v13 0801-0812 corpus passed:
  - Dragapult `cc2e995b5ad0`, `--max-files 2`: summary abs mean `0.20885`,
    nonzero `0.407`, opponent-label slice `0.00000000` for noaug, summary-aug,
    and raw+summary-aug.
  - Marnie `b8f251a476e7`, `--max-files 2`: summary abs mean `0.19336`,
    nonzero `0.401`, opponent-label slice `0.00000000` for all three cases.
- A checked overnight runner was started on ks:
  `/tmp/run_overnight_strategy_checked_20260813.sh`.
  Main remote PID tree starts at `663042`; logs are under
  `logs/overnight_strategy_checked_20260813/`, checkpoints under
  `checkpoints/overnight_strategy_checked_20260813/`.
  It first ran the summary integrity smoke tests, copied the existing Marnie
  trajectory from `logs/overnight_strategy_safe_20260813/`, then launched four
  GPU queues:
  - GPU0: `drag_plan_full` then `marnie_summary`
  - GPU1: `drag_lowloss` then `marnie_rawhist`
  - GPU2: `drag_success_aux`
  - GPU3: `drag_teamclean`
  After all training queues finish, the runner will evaluate random g500 and
  baseline-delta g60 versus the 81-score-labeled 0812 shadow manifest.
- Current run sanity at launch:
  four Dragapult jobs entered epoch 1 successfully; GPU memory was about
  16-20 GiB per card; `offline_opp_history=False` appeared in training logs.
  `drag_teamclean` retained `201407` decisions after team filtering and saved
  an epoch-1 checkpoint, so the UTF-8 team-name filter did not collapse to zero
  even though `ps` may display one Japanese team name as question marks.
- Monitor commands:
  `ssh ks 'cd /data/jie/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/overnight_strategy_checked_20260813/runner.log'`
  `ssh ks 'cd /data/jie/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/overnight_strategy_checked_20260813/drag_plan_full_checked_w4_seed61.log'`
  `ssh ks 'cd /data/jie/ptcg_rl_git_v7_baseline_20260804 && cat logs/overnight_strategy_checked_20260813/status.csv'`
  `ssh ks 'nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits'`
- When this checked run finishes, first inspect:
  `logs/overnight_strategy_checked_20260813/eval/random_g500.csv`,
  `logs/overnight_strategy_checked_20260813/eval/drag_vs_shadow81_g60.csv`,
  and `logs/overnight_strategy_checked_20260813/eval/marnie_vs_shadow81_g60.csv`.
  Pull key logs/checkpoints back locally if results look useful; ks storage is
  ephemeral.

Update 2026-08-14 06:20 CST:

- After the live-history fix, an additional audit found that the old PPO/RL
  script was still structurally incomplete for v13/history checkpoints.
  `tools/rl_finetune_vs_pool.py` now infers and logs both
  `history_summary_dim` and `state_token_feat_dim` from `--policy-init`, passes
  them into candidate/ref models and BC anchor corpora, and passes
  `state_token_feats` into `model.encode_state()` during trainable rollout
  sampling.
- PPO live summary construction is now aligned with submission inference:
  summary uses own action history, public log history, and board snapshots; it
  does not use offline opponent labeled decisions. Summary-only PPO rollouts
  keep a 32-action / 12-board buffer instead of effectively shrinking to a few
  events.
- `ptcg_rl/model.py` now drops `state_token_feat_dim` when building legacy
  pointer models, so RL can still use old checkpoints as templates without
  immediately failing on an unexpected constructor argument.
- Important: this does not affect the already-running
  `overnight_strategy_checked_20260813` jobs, because those Python processes
  loaded code before this patch. Use this fix for the next RL/league run.

Update 2026-08-14 08:20 CST:

- Local commits from this repair block:
  - `08b68c9 Fix live-compatible history strategy pipeline`
  - `1f213ec Fix RL history state-token compatibility`
  - `f65deb1 Fix legacy MCTS live feature inputs`
- The checked overnight training completed all 6 train jobs with `rc=0`.
  `random_g500.csv`:
  - Dragapult `drag_plan_full`: `94.4%`
  - Dragapult `drag_lowloss`: `95.6%`
  - Dragapult `drag_success_aux`: `94.4%`
  - Dragapult `drag_teamclean`: `86.2%`
  - Marnie `marnie_summary`: `99.8%`
  - Marnie `marnie_rawhist`: `99.8%`
- Dragapult baseline-delta versus 81 score-labeled 0812 shadows is complete
  and negative for every new Dragapult candidate:
  - `drag_plan_full`: avg delta `-0.078`, candidate mean `0.640`,
    baseline `0.718`, lost `63/81`
  - `drag_lowloss`: avg delta `-0.104`, candidate mean `0.615`,
    baseline `0.718`, lost `74/81`
  - `drag_success_aux`: avg delta `-0.122`, candidate mean `0.596`,
    baseline `0.718`, lost `72/81`
  - `drag_teamclean`: avg delta `-0.214`, candidate mean `0.505`,
    baseline `0.718`, lost `81/81`
  Do not submit these Dragapult checked strategy candidates. This result
  suggests the current strategy-success labels/filtering are actively harmful
  or still too weak/noisy, despite the history bug fix. The next Dragapult step
  should be a deeper encoder/target redesign or RL exploration, not another
  small reweighting of these same labels.
- Marnie baseline-delta was still running at this checkpoint:
  `logs/overnight_strategy_checked_20260813/eval/marnie_vs_shadow81_g60.log`.
  It had completed random g500 and was in the baseline phase around opponent
  `21/81`. Check the final CSV before judging the two Marnie repaired-history
  candidates.
- A remote accidental directory exists:
  `/data/jie/ptcg_rl_git_v7_baseline_20260804/tmp_sync_placeholder`.
  It only contains duplicate source files from a mistyped `scp` target and is
  not on the Python execution path. It can be removed after user confirmation.

Update 2026-08-14 12:10 CST:

- User noted the current Kaggle environment is very bad for Marnie and asked to
  focus on Dragapult and Alakazam. Remote `ks` has
  `/data/jie/episodes_raw/pokemon-tcg-ai-battle-episodes-2026-08-13.zip`.
- Analyzed 0812/0813 episode distributions on `ks` with
  `/tmp/analyze_env_0812_0813_20260814.py`; outputs were pulled locally under
  `logs/ladder_distribution_0812_0813_20260814/`.
  Important approximate leaderboard-band conclusions:
  - 0813 overall rows: Dragapult `1817`, Marnie `1778`, Ogerpon `1588`,
    Alakazam `1332`, Lopunny `1053`.
  - Marnie is clearly receding: 0812 max score `1230.3`, 0813 max score
    `1127.1`; 1200+ Marnie rows disappeared in 0813.
  - Dragapult is the most important current archetype: 0813 `1100-1199` has
    `611` Dragapult rows, led by `cc2e995b5ad0`.
  - Alakazam remains relevant but is not as dominant. 0813 `1100-1199` has
    `148` Alakazam rows; top sigs include `75f1d900d851`,
    `ecb67fcd9c0b`, and `7f9a538936e3`.
  - Episode matchup summary says Alakazam's biggest broad weakness is
    Dragapult: 0812 Alakazam vs Dragapult `0.249`, 0813 `0.379` over large
    samples. Dragapult is strong into Alakazam: 0812 Dragapult vs Alakazam
    `0.751`.
- Pulled latest jie Kaggle submissions/replays locally only, using the local
  jie credentials. Do not use the by account. Outputs are under
  `logs/kaggle_live_20260814/`.
  Latest live replay summaries:
  - `55485247` old Marnie b8f repeat: `53` analyzed games, `34-18-1`,
    WR `0.642`; losses concentrated in Archaludon and Mega Lucario, with some
    same-sig Marnie losses.
  - `55484585` v13 shadow Marnie b8f: `46` games, `28-18`, WR `0.609`;
    weak into Archaludon (`1/5`) and Ogerpon (`0/2`), same archetype only
    `7/13`.
  - `55478159` Alakazam lossopt 7f9: `45` games, `30-15`, WR `0.667`;
    main live weakness in this sample is Marnie b8f (`7/16` against sig
    `b8f251a476e7`), not Dragapult because only one Dragapult appeared in the
    pulled replay sample.
  - `55484570` v13 shadow Lucario had only `9` games and `4-5`, not useful.
- Current v14 sequence population training remains active on `ks`:
  `logs/v14_sequence_0808_0812/pop_parallel3.nohup.log`.
  It uses `data/seq_corpus_v14_0808_0812`, not 0813. Status around 12:10 CST:
  `failed=0`; major running jobs include top-sig Alakazam `7f9a5389`,
  Dragapult `cc2e995b`, Marnie `b8f251a4`, Lopunny `f1445356`, and Lucario
  `43d6d8b0`.
  Training metrics at this checkpoint:
  - Alakazam `7f9a5389`: epoch 3, top1 about `0.86`, setF1 about `0.86`.
  - Dragapult `cc2e995b`: epoch 4, top1 about `0.67`, setF1 about `0.67`.
    This confirms Dragapult is intrinsically harder for the current sequence
    target/model than Alakazam.
- Started 0813-only v14 extraction on `ks` so current environment can be added
  to the next Dragapult/Alakazam training/eval:
  - runner: `/tmp/run_v14_extract_0813_20260814.sh`
  - nohup log: `logs/v14_sequence_0813_extract_20260814.nohup.log`
  - output corpus: `data/seq_corpus_v14_0813_only`
  - manifest target:
    `logs/v14_sequence_0813_20260814/train_manifest_0813_top3_allbands.csv`
  The extractor reports "Processing 1 zips with 1 workers" because work is
  sharded per zip, not per episode. Progress is printed every 250 episodes.

Update 2026-08-14 13:35 CST:

- Inspected the C++ engine under `/home/jie/Do/0_PTCG/ptcg_engine/ptcgProgram 22`.
  Important action semantics for the v14 sequence branch:
  - `ApiSelect()` accepts a vector of option indices and `State::checkPlayerSelect()`
    enforces `selectMin/selectMax`, so the engine truly supports same-step
    multi-select.
  - `ToJson` exports `minCount`, `maxCount`, `remainDamageCounter`, and
    `remainEnergyCost`; `FastEncoder` already stores these in state feature
    indices 19-22.
  - Dragapult ex Phantom Dive is implemented as `DamageCounterAny` with
    `eVal(6).targetBench()`. The engine then sets `remainDamageCounter=6`,
    pushes `SelectDamageCounterAny`, and repeats a single-card selection six
    times (`selectMin=selectMax=1`) via `setLastFunctionCallCount(count)`.
    Therefore Dragapult's key bench-damage allocation is not represented by
    `target_k>1`; it is a sequence of six single-target resolution decisions.
- Consequence: v14 currently covers continuous decisions only at the raw engine
  select-step level. It has causal sequence attention, prior-action fields,
  ledger features, current state-token attention, and future-plan auxiliary
  targets, and the live `TorchSequencePolicy` keeps a rolling sequence buffer.
  However, it does not yet group resolution steps such as `DamageCounterAny`
  into one tactical sub-plan. For Dragapult, the next redesign should add a
  grouped damage-counter allocation audit/target and evaluate whether the six
  choices create KO/setup breakpoints, not only next-option top1/setF1.
- Added local/remote v14 multi-select diagnostics:
  - `SequenceLossConfig.multi_target_weight`
  - train progress prints `m2n/m2r/cap2/m2F1/m2Ord`
  - audit prints valid `target_k`, raw action entry counts, min/max-count
    distributions, and action types/opponents responsible for true same-step
    multi-select.
  A CPU smoke train on `ks` confirmed the new metrics are wired.
- The 0813-only extraction completed with `759683` rows. The old runner failed
  only after extraction because it invoked `v14_audit_sequence_corpus.py` with
  the obsolete positional CLI. A manifest was built manually:
  `logs/v14_sequence_0813_20260814/train_manifest_0813_top3_allbands.csv`
  with 31 entries. Top relevant entries include Dragapult `cc2e995b5ad0`
  (`120734` rows, Petit Canard, score `1218.2`) and Alakazam `7f9a538936e3`
  (`49806` rows, fishcat, score `1127.1`).
- Remote audits were started for 0813 Dragapult and Alakazam:
  `logs/v14_sequence_0813_20260814/audit_dragapult_0813_focus.log` and
  `logs/v14_sequence_0813_20260814/audit_alakazam_0813_focus.log`. If they are
  still slow, the next tool should add a faster context-specific audit for
  `act_context=14` (`DamageCounterAny`) rather than scanning everything through
  the generic audit path.

Update 2026-08-14 14:00 CST:

- Implemented the first explicit engine-aligned Dragapult/DamageCounterAny
  redesign on branch `sequence-decision-pipeline-v14`.
  Engine-confirmed basis: `DamageCounterAny` is `act_context=14` in non-web
  JSON and is resolved as repeated single-target selections with
  `remainDamageCounter` decreasing each step.
- Code changes:
  - `ptcg_rl/encoder.py`: `OPT_FEAT_DIM` is now `72`; option features 64-71
    are DamageCounterAny-specific signals derived from current target HP,
    remaining counters, KO-now/full-remaining-KO, opponent bench target, and
    overkill. Old checkpoints remain loadable because model forward truncates
    or pads feature dims from checkpoint config.
  - `ptcg_rl/seq/constants.py`: feature version is
    `v14_sequence_decision_dca_v2`; `LEDGER_FEAT_DIM` is now `80`; added
    `DAMAGE_COUNTER_ANY_CONTEXT = 14`.
  - `ptcg_rl/seq/features.py`: `SequenceLedger` now tracks a live-compatible
    DamageCounterAny prefix (steps already allocated, unique targets, focus
    fraction proxy, remain/progress) in ledger features 64-79.
  - `tools/v14_extract_sequences.py`: extracted rows now include
    `dca_group_index`, `dca_pos`, `dca_len`, `dca_remaining`,
    `dca_selected_slot`, `dca_prior_*`, `dca_group_unique_slots`, and
    `dca_group_focus_frac` for contiguous DamageCounterAny blocks.
  - `ptcg_rl/seq/data.py` / `ptcg_rl/seq/model.py`: batch now carries
    `target_context`; training has `--damage-counter-weight`; metrics now
    report `dcaN/dcaR/dcaTop1/dcaF1`.
  - Added `tools/v14_audit_damage_counter_any.py` for fast block-level DCA
    audit instead of using the slow generic audit for this purpose.
- Verification:
  - Local and remote `py_compile` passed for changed modules.
  - Remote CPU smoke train on old 0813 Dragapult corpus succeeded with
    `--damage-counter-weight 4.0`; progress printed `dcaN=19 dcaR=0.297
    dcaTop1=0.053 dcaF1=0.053`, proving the new DCA metrics are wired.
- Started the main new runner on `ks`:
  - script: `/tmp/run_v14_dca_0808_0813_20260814.sh`
  - log: `logs/v14_dca_0808_0813_20260814.runner.nohup.log`
  - corpus: `data/seq_corpus_v14_dca_0808_0813`
  - checkpoints: `checkpoints/v14_dca_0808_0813/`
  - extraction uses 0808-0813 episodes, 6 zip workers, future horizon 16, and
    leaderboard CSV
    `logs/info_pull_20260813/leaderboard_full/pokemon-tcg-ai-battle-publicleaderboard-2026-08-13T09:36:23.csv`.
  - after extraction it builds `train_manifest_top3_allbands.csv`, audits
    Dragapult DCA blocks, then trains three models in parallel on explicitly
    free GPUs:
    `v14dca_dragapult_cc2e_w512_dca6` on GPU1,
    `v14dca_dragapult_140f_w512_dca6` on GPU2, and
    `v14dca_alakazam_7f9a_w384_ctx` on GPU3.
  - The runner automatically runs random g500 after training. Deck paths used:
    `logs/ladder_pool_0812_all_v13_20260813/decks/cc2e995b5ad0_dragapult_kh0a.csv`,
    `logs/ladder_pool_0812_all_v13_20260813/decks/140f7d8b2f09_dragapult_dipam_chakraborty.csv`,
    and `logs/ladder_pool_0802_all/decks/7f9a538936e3_alakazam_yushin_ito.csv`.
- Status when started: old `v14_sequence_0808_0812` population still had
  Marnie b8f, Alakazam 7f, and Mega Lopunny f144 running, but GPU usage showed
  only GPU0 busy; the new runner binds new jobs to GPU1/2/3 after extraction.
  Monitor with:
  `ssh ks 'cd /data/jie/ptcg_rl_git_v7_baseline_20260804 && tail -f logs/v14_dca_0808_0813_20260814.runner.nohup.log'`

Update 2026-08-14 16:55 CST:

- Old v14 population `logs/v14_sequence_0808_0812/pop_parallel3.nohup.log`
  completed: `done=23 failed=0`. It uses the pre-DCA feature set
  (`opt_feat_dim=64`, `ledger_feat_dim=64`) on corpus
  `data/seq_corpus_v14_0808_0812`.
- Fixed v14 live/eval compatibility after DCA feature expansion:
  `TorchSequencePolicy` now builds inference batches with the dimensions from
  checkpoint `model_config`, so old 64/64 checkpoints do not silently fail
  after current code expanded to 72/80. `tools/v14_eval_random.py` now reports
  policy fallback errors. The random audit showed `policy_errors=0`.
- Backed up core old-v14 artifacts from `ks` to local:
  `remote_backups/ks_20260814_v14_core/`
  including `checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3`
  (~2.1 GB), old v14 logs, DCA logs, random summaries, and RR manifest.
- Old v14 random g300 summary:
  - Strong vs random: Mega Lucario `43d6` 100.0, N's Zoroark `aa80` 100.0,
    Alakazam `7f9a` 99.7, Cynthia `52f4` 99.7, Marnie `a3b0` 99.7,
    Mega Lopunny `f144` 99.7, Marnie `b8f` 99.0, Ogerpon `8bc` 98.7,
    Ogerpon `d175` 97.3.
  - Dragapult remains weak for the stated goal: `cc2e` only 96.7, and
    `3535` only 71.3. Training logs explain this: `cc2e` has train top1 about
    0.825 but validation top1 only about 0.671 by epoch 8; `3535` overfits
    badly (train top1 0.974, val top1 about 0.648). Old v14 did not yet model
    the six-step DamageCounterAny allocation as one tactical sub-plan.
  - Clearly poor random models: Team Rocket Mewtwo `958e` 33.0, Mega Starmie
    `1756` 22.3, Festival `41ff` 41.0, Cynthia `bfeb` 57.3, Mega Lucario
    `8388` 63.7, Dragapult `3535` 71.3.
- Started filtered RR for old v14 only after random filtering:
  - manifest:
    `logs/v14_sequence_0808_0812/rr_v14_pop_filtered_random96_20260814/manifest.csv`
  - log:
    `logs/v14_sequence_0808_0812/rr_v14_pop_filtered_random96_20260814/rr_g40.log`
  - output:
    `logs/v14_sequence_0808_0812/rr_v14_pop_filtered_random96_20260814/rr_g40.csv`
  - command uses `--manifest`, `--games 40`, `--workers 8`, 13 entries,
    78 pairs. First completed pair showed Alakazam `5a5` lost heavily to
    Alakazam `7f9` (7-33), matching the random/data-size signal.
- Original DCA runner stalled because `tools/v14_audit_damage_counter_any.py`
  re-read object arrays (`of_arr`/`oc`) while scanning all rows. That made the
  audit take more than two hours before training. The old runner also had
  obsolete random-eval CLI (`--policy`, `--out-csv`) that would have failed.
  Killed the stuck audit/runner and replaced it with a continuation runner:
  `/tmp/run_v14_dca_continue_20260814.sh`.
- DCA audit was rewritten to use extracted numeric fields (`dca_group_index`,
  `dca_pos`, `dca_len`, `dca_selected_slot`, `dca_group_unique_slots`,
  `dca_group_focus_frac`) instead of object-array reconstruction. A 200k-row
  smoke finished in 81s and confirmed Dragapult DCA groups are length 6:
  `length_counts={6: 5495}`. Most groups focus one target:
  `unique_slot_counts={1: 4013, 2: 1301, 3: 151, 4: 27, 5: 3}`.
- DCA continuation status:
  - log: `logs/v14_dca_0808_0813_20260814.continue.nohup.log`
  - fast audit cap: 500k rows, then trains three models in parallel:
    `v14dca_dragapult_cc2e_w512_dca6`, `v14dca_dragapult_140f_w512_dca6`,
    `v14dca_alakazam_7f9a_w384_ctx`.
  - random eval command was corrected to the positional checkpoint CLI.

Update 2026-08-15 00:10 CST:

- User explicitly requested: do not continue fixing remaining signal/architecture
  issues overnight. First run stable long training and automated tests; resume
  signal fixes tomorrow.
- Current branch: `sequence-decision-pipeline-v14`. Remote workspace:
  `ks:/data/jie/ptcg_rl_git_v7_baseline_20260804` (not a git repo).
- Local code changes after commit `de514c8`:
  - `ptcg_rl/seq/model.py`: added training-only future action-type head and
    DCA block-plan heads; loss now includes `next_type_weight` and
    `dca_plan_weight`. DCA spread BCE now uses dynamic positive weighting
    capped at 8 because spread labels are minority.
  - `tools/v14_train_sequence_policy.py`: training logs now show next-type,
    DCA-plan, no-action/reversed-history future deltas, DCA positive weight,
    and corrected `grad_norm` moving averages. The latest `grad_norm` fix was
    synced to ks before starting the long runner.
  - `ptcg_rl/seq/torch_policy.py` and `tools/v14_probe_sequence_policy.py` load
    checkpoints with `strict=False` so new training-only heads do not break old
    v14 inference.
- Smoke result:
  - `batch-size=512` in `logs/v14_seqplan_smoke3_20260814/` OOMed near 80GB.
    Do not use this for overnight runs.
  - `batch-size=256` in `logs/v14_seqplan_smoke4_20260814/` completed for
    Dragapult `cc2e`, Alakazam `7f9`, Mega Lopunny `f144`, and Marnie `b8f`.
    GPU memory was about 30-37GB. This is the safe nightly setting.
  - smoke4 showed future-type history sensitivity is visible
    (`revhist_nextD` around -0.07 to -0.15), but current-action logits still
    barely use history order (`revhist_delta` near 0 and high agreement). This
    is the main signal fix to revisit tomorrow.
  - Dragapult DCA plan no longer fully collapses after positive weighting:
    smoke4 `cc2e` val `dplan_f1=0.334`, `spread_rate=0.16`,
    `pred_spread_rate=0.32`; still weak and needs structural work later.
- Nightly long runner started on ks:
  - script: `/tmp/run_v14_seqplan_nightly_20260814.sh`
  - runner pid printed by ssh: `2505620`
  - log directory: `logs/v14_seqplan_nightly_20260814/`
  - checkpoint directory: `checkpoints/v14_seqplan_nightly_20260814/`
  - corpus: `data/seq_corpus_v14_0801_0813_hq`
  - training is four parallel specialist-from-scratch jobs, all `seq_len=48`,
    `width=384`, `layers=4`, `heads=6`, `batch=256`, no AMP, all score bands
    600+ through 1200+:
    - GPU0 Dragapult `cc2e995b5ad0`, 8 epochs, `damage_counter_weight=6`,
      deck `logs/ladder_pool_0812_all_v13_20260813/decks/cc2e995b5ad0_dragapult_kh0a.csv`
    - GPU1 Dragapult `7b80ccf70f9f`, 16 epochs, `damage_counter_weight=6`,
      deck `logs/ladder_pool_0812_all_v13_20260813/decks/7b80ccf70f9f_dragapult_sixth_sense.csv`
    - GPU2 Alakazam `7f9a538936e3`, 6 epochs,
      deck `logs/ladder_pool_0812_all_v13_20260813/decks/7f9a538936e3_alakazam_yushin_ito.csv`
    - GPU3 Alakazam `589269aab27e`, 14 epochs,
      deck `logs/ladder_pool_0812_all_v13_20260813/decks/589269aab27e_alakazam_bart_edwyn.csv`
  - After each training job, the runner automatically runs:
    - `tools/eval_bc.py ... --games 500 --workers 16`
    - candidate-only RR vs
      `logs/v14_sequence_0808_0812/rr_v14_pop_filtered_random96_20260814/manifest.csv`
      with `--games 80 --workers 16`
  - Final summary will be written to
    `logs/v14_seqplan_nightly_20260814/summary.txt`, and completion marker to
    `logs/v14_seqplan_nightly_20260814/runner.done`.
- Morning TODO:
  - First inspect `logs/v14_seqplan_nightly_20260814/summary.txt`, train logs,
    and RR CSVs. Pull key logs/checkpoints back to local before risky changes.
  - Do not judge only by random/RR. Inspect training-time signal health:
    `revhist_delta`, `revhist_nextD`, `noact_nextD`, `dplan_f1`,
    `dca_plan_pred_spread_rate`, `attach/evolve/attack` top1, ambiguous option
    top1, and current-vs-prefix loss.
  - Then continue structural signal fixes: current action must actually depend
    on useful history/order, not only future auxiliary heads.

Update 2026-08-15 07:45 CST:

- Nightly runner is still active because Alakazam `7f9` is training. Remote
  time check showed:
  - active process: `v14seqlong_alakazam_7f9_0801_0813_seqplan`
  - GPU2 about 36GB used, other GPUs free.
  - the runner will automatically evaluate random/RR and write `summary.txt`
    after `7f9` finishes.
- Completed and tested so far:
  - `v14seqlong_dragapult_7b80_0801_0813_seqplan`: random g500 only 48.4%
    (242/500), filtered old-v14 RR mean 0.052. This model is broken/overfit;
    do not submit.
  - `v14seqlong_dragapult_cc2e_0801_0813_seqplan`: random g500 82.4%
    (412/500), filtered old-v14 RR mean 0.212, worst matchups include old
    Alakazam `7f9`, Crustle, Lopunny, Ogerpon, Marnie. Still not submit-grade.
  - `v14seqlong_alakazam_5892_0801_0813_seqplan`: random g500 92.8%
    (464/500), filtered old-v14 RR mean 0.238. It is better than the bad
    Dragapult runs but still weak against the filtered pool; not submit-grade
    unless `7f9` also fails and a low-risk ablation is needed.
- Training-time interpretation from completed jobs:
  - Full-history/future objectives are now measurable and nonzero, but small
    sig datasets overfit hard. `7b80` train top1 climbed to 0.85 while val
    stayed near 0.63 and DCA/multi gaps grew; extra epochs made it worse.
  - `cc2e` generalizes better than `7b80` but DCA block planning remains
    insufficient: final val `dcaTop1` about 0.862 and `dplan_f1` about 0.330;
    random/RR show this is not enough for Dragapult.
- Tomorrow, do not continue this exact small-sig high-epoch recipe. Resume
  structural fixes and consider stronger data splitting/regularization or
  a different objective before more full overnight runs.

## Update: Known-Info / Reveal Ledger 2026-08-15

- Added public-information memory on top of the old BC history features.
  `SequenceLedger.observe_public_logs()` now consumes engine logs and tracks
  opponent cards that were publicly revealed or searched into hand.
- `SequenceBatch` now carries `known_opp_cards`, `known_opp_counts`, and
  `known_opp_mask`. These are extracted into corpus files, batched during
  training, and passed through live inference.
- `SequencePolicyNet` now consumes known-info in two ways:
  - it can be folded into the ledger summary;
  - it also has an explicit `known_query` path that conditions action scoring.
- New training-time diagnostics are available:
  - `known_opp_rate`, `known_opp_slots_mean`, `known_opp_count_mean`
  - `known_query_ratio`
  - `val_noknown=...`
  - warning `known_info_not_affecting_current`
- Ablation meanings:
  - `noact` removes action-ledger history but keeps reveal/known memory.
  - `noknown` removes reveal/known memory but keeps ordinary action history.
  - `stateless` removes both explicit history and known memory.
- If `known_opp_rate` is nontrivial but `noknown_delta≈0`, the model is still
  not using public reveal information in a measurable way.

## Update: 2026-08-15 Afternoon Signal Diagnostics

- Local/remote branch remains `sequence-decision-pipeline-v14`; remote working
  tree is still `ks:/data/jie/ptcg_rl_git_v7_baseline_20260804`.
- Critical rule from user: do not keep doing BC fine-tune / tiny filtered-data
  adjustments. BC is effectively locked as a baseline; only structural changes
  or retraining with new objectives are worth trying.
- `v14_opportunity_diag_20260815` and `v14_margin_diag_20260815` were negative.
  Opportunity type/margin and exact rank-margin harden current-row CE but do not
  fix the root issue.
- `v14_rank_diag_20260815` completed and was pulled locally to:
  - `artifacts/ks_pull_20260815_v14_rank_diag/logs`
  - `artifacts/ks_pull_20260815_v14_rank_diag/checkpoints`
- Rank diagnostic final result:
  - Alakazam `7f9` rank-margin 0.0: val `1.9605`, val top1 `0.768`, current
    `0.762/0.877`, random smoke `43/48`.
  - Alakazam `7f9` rank-margin 0.35: val `2.2632`, val top1 `0.767`, current
    `0.761/0.873`, random smoke `46/48`.
  - Dragapult `cc2e` rank-margin 0.0: val `2.2010`, val top1 `0.670`, current
    `0.680/0.862`, random smoke `40/48`.
  - Dragapult `cc2e` rank-margin 0.35: val `2.5047`, val top1 `0.666`, current
    `0.676/0.860`, random smoke `33/48`.
  - Interpretation: exact rank-margin is not the right lever. It worsens loss
    and does not address `history_order_not_affecting_current`.
- New code added after rank diagnostics:
  - `SequenceCorpus` now precomputes same-turn next-action labels:
    `turn_next_exists`, `turn_next_type`, `turn_next_card`,
    `turn_next_card2`, `turn_next_attack`, `turn_next_context`.
  - `SequencePolicyNet` now has same-turn concrete next-action heads and an
    optional `turn_next_condition_scale` path into the action scorer.
  - Trainer exposes `--turn-next-plan-weight`,
    `--turn-next-condition-scale`, and subweights for type/card/attack/context.
  - Training logs now include `turnNext`, `cur_turn_next`,
    `val_turnNextL`, `cond=.../tn...`, and warnings for weak/collapsed
    turn-next learning.
- Remote short diagnostic currently running:
  - PID `3524727`
  - script `/tmp/run_v14_turn_next_diag_20260815.sh`
  - logs `logs/v14_turn_next_diag_20260815/`
  - checkpoints `checkpoints/v14_turn_next_diag_20260815/`
  - four parallel jobs: Alakazam/Dragapult loss-only and
    loss+`turn_next_condition_scale=0.50`.
  - Early signal at batch 40: turn-next labels are real and learnable
    (`turn_next_rate` about 0.86-0.91; positive type acc already about
    0.40-0.52). Card prediction is still weak. Conditioning is active
    (`tn≈0.30`) but has not yet visibly changed current-action metrics.
- Next decision after this diagnostic:
  - If `turnNext` learns but `revhist_delta` and current metrics remain nearly
    unchanged, do not tune weights. Jump to a stronger plan-token architecture:
    make predicted/teacher same-turn next actions explicit plan tokens that
    cross-attend to legal options, and add ablation showing action logits change
    when plan tokens are removed.
  - If `turnNext` itself fails, debug labels from corpus rows before training
    any longer run.
