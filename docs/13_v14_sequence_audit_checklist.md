# v14 Sequence Pipeline Audit Checklist

Last updated: 2026-08-14.

This checklist exists because v14 is a large rewrite. It should be read before
launching long training or interpreting any result.

Status meanings:

- `OK`: checked in code and smoke/probe where possible.
- `FIXED`: bug found and patched.
- `NEEDS_PROBE`: code exists, but real training must prove it works.
- `RISK`: known design risk that can affect results.

## A. Extraction And Label Alignment

1. Observation/action temporal alignment
   - Code: `tools/v14_extract_sequences.py`
   - Requirement: label must be the action answering the previous ACTIVE
     observation for that same player.
   - Status: `OK`
   - Evidence: local smoke over first 20 games of each available zip extracted
     `52099` rows with `bad=0 err=0`; pending observation is cleared after the
     later action is consumed.

2. Deck-selection actions excluded
   - Code: `valid_action()` in `tools/v14_extract_sequences.py`
   - Requirement: 60-card deck list must not become a policy label.
   - Status: `OK`

3. Game grouping key is unique enough
   - Code: `game_key=f"{episode_id}:{player_index}"`
   - Requirement: windows must not mix players or episodes.
   - Status: `OK`

4. Decision order inside game
   - Code: `decision_index`, `SequenceCorpus(... rows sorted by decision_index)`
   - Requirement: causal order must match actual sequence of player decisions.
   - Status: `OK`

5. Ledger is live-compatible
   - Code: `SequenceLedger`, `torch_policy.py`
   - Requirement: offline training ledger must use only previous own actions and
     current public/current observation, not future labels.
   - Status: `OK`
   - Note: opponent public information is in current state/options, not in a
     hidden offline opponent-action stream.

6. Future-plan target uses future expert actions
   - Code: `future_plan_targets()`
   - Requirement: allowed only as training auxiliary target, never as inference
     input.
   - Status: `OK`

7. Leaderboard score bands
   - Code: `--lb-csv` in extractor
   - Requirement: without CSV, all scores become `600-699`.
   - Status: `RISK`
   - Action: always pass leaderboard CSV for real runs. Smoke run intentionally
     omitted it.

8. Variable option arrays
   - Code: object arrays in extractor, padded in `SequenceCorpus.collate`
   - Requirement: per-step legal option identities must remain aligned with
     action indices.
   - Status: `NEEDS_PROBE`
   - Probe: audit random rows by checking selected indices fall inside `ot`.

## B. Model Objectives And Loss

1. Current first-action classification
   - Code: `action_logits`, `target_first`
   - Status: `OK`

2. Multi-select set learning
   - Code: `target_multi`, BCE over option logits
   - Status: `OK`
   - Risk: BCE on same logits as first-action softmax may underweight secondary
     picks unless count head is also good.

3. Selection count prediction
   - Code: `count_head`, `count_loss`, `TorchSequencePolicy.select`
   - Status: `FIXED`
   - Bug found: initial v14 model had multi-select BCE but no count head, so
     inference defaulted to one selected option. This directly harmed multi-pick
     decisions.

4. Ordered multi-select prediction
   - Code: `target_order`, `order_logits`, `TorchSequencePolicy.select`
   - Status: `FIXED`
   - Bug found during checklist audit: count and set BCE still did not teach
     selection order. Added per-position option logits and live ordered top-k
     inference with duplicate removal.

5. Action type head
   - Code: `type_head`, `target_type`
   - Status: `OK`
   - Purpose: gives a coarse decision-mode signal independent of exact option id.

6. Future-plan head
   - Code: `plan_head`, `future_plan`
   - Status: `NEEDS_PROBE`
   - Required evidence: plan loss improves with longer prefix and worsens under
     prefix shuffle/zero.

7. Outcome head
   - Code: `outcome_head`
   - Status: `RISK`
   - Risk: outcome labels are noisy and can be dominated by deck strength or
     matchup. Keep low weight until real probes show value.

8. Causal sequence mask with left padding
   - Code: `SequencePolicyNet.forward`
   - Status: `FIXED`
   - Bug found: using `src_key_padding_mask` together with causal mask produced
     NaNs for all-masked padding queries. Fixed by zeroing padded inputs/outputs
     and using `step_mask` in loss.

9. Loss masking against padded steps
   - Code: `sequence_policy_loss`
   - Status: `OK`
   - Note: avoid multiplying NaN by zero; model forward must avoid NaNs first.

10. AMP-safe logit masks
    - Code: `NEG_INF` in `ptcg_rl/seq/model.py`
    - Status: `FIXED`
    - Bug found during CUDA population launch: `masked_fill(..., -1e9)` cannot
      write into fp16 logits under AMP. The mask value is now `-1e4`, which is
      large enough for softmax/BCE masking and representable in fp16.

11. Ordered-selection scorer memory
    - Code: `order_scores` in `SequencePolicyNet.forward`
    - Status: `FIXED`
    - Bug found during 8-way population launch: the first implementation
      expanded `[B,T,K,N,W]` and concatenated three copies, causing OOM with
      two jobs per A800. It now uses a position-conditioned dot-product scorer
      that materializes only `[B,T,K,N]`.

## C. Inference Semantics

1. v14 live policy keeps per-game prefix
   - Code: `TorchSequencePolicy.buffer`, `reset_history`, `remember_encoded`
   - Status: `OK`

2. Reset at deck selection
   - Code: `main.py` calls `policy.reset_history()` when `select is None`
   - Status: `OK`

3. Rule overlay updates history with final action
   - Code: `main.py`, `policy.remember_decision(obs_dict, picks)`
   - Status: `OK`

4. Count-aware top-k inference
   - Code: `TorchSequencePolicy.select`
   - Status: `FIXED`
   - Requirement: use predicted count clipped by min/max count.

5. Legacy policy API compatibility
   - Code: `TorchSequencePolicy.select`
   - Status: `FIXED`
   - Bug found during checklist audit: `main.py` and old eval call
     `select(..., temperature=..., update_history=False)`. v14 initially did
     not accept `temperature`, which would have forced exception fallback and
     random actions in packaged/eval use.

6. Empty legal action / maxCount zero
   - Code: `TorchSequencePolicy.select`, `main.py`
   - Status: `OK`

7. Torch submission runtime
   - Code: `main.py`, `package_submission.py`
   - Status: `RISK`
   - Note: `.pt` can now be packaged, but Kaggle runtime/size has not been
     validated. Local validation first.

8. Legacy MCTS switch compatibility
   - Code: `TorchSequencePolicy.select_mcts`
   - Status: `FIXED`
   - Bug found during checklist audit: old eval/submission paths can call
     `select_mcts` when a switch is enabled. v14 has no trained value head, but
     raising here would silently fall through to random/legal fallback. The shim
     now returns greedy sequence-policy actions.

## D. Evaluation And Probing

1. Legacy eval accepts `.pt`
   - Code: `ptcg_rl/policy_loader.py`, `tools/eval_bc.py`,
     `tools/eval_round_robin.py`
   - Status: `OK`

2. Sequence-use probe exists
   - Code: `tools/v14_probe_sequence_policy.py`
   - Status: `OK`
   - Must compare `base`, `last_only`, `last_zero_prefix_zero_ledger`,
     `last_shuffle_prefix`, and prefix lengths.

3. Probe acceptance condition
   - Status: `NEEDS_PROBE`
   - Real model should show base/full-prefix better than ablations. If not,
     the model is still behaving like a single-step classifier.

4. Corpus audit exists
   - Code: `tools/v14_audit_sequence_corpus.py`
   - Status: `OK`
   - Must inspect game length, action types, opponent coverage, deck sig
     concentration, future-plan means before training.

5. Random/RR compatibility
   - Code: `tools/eval_bc.py`, `tools/eval_round_robin.py`
   - Status: `FIXED`
   - Bug found during checklist audit: parallel RR worker loading still called
     an undefined `NumpyPolicy.load`, and would not load `.pt` policies anyway.
     Workers now use `ptcg_rl.policy_loader.load_policy()`.
   - Need to run actual games after first real checkpoints finish.

6. Training progress telemetry
   - Code: `tools/v14_train_sequence_policy.py`,
     `tools/v14_train_population.py`
   - Status: `OK`
   - Requirement: long training must show progress and enough metrics to stop
     early if the pipeline is broken. Training lines now include loss parts,
     top1/type/count accuracy, count MAE, predicted/target selection count,
     set-F1, ordered-selection accuracy, outcome accuracy, ETA, samples/s, and
     CUDA memory. Population runner prints the latest metric line per running
     job on every poll.

## Immediate Open Items

1. Validate `TorchSequencePolicy` random games with the smoke checkpoint after
   count/order patch.
2. If remote extraction finishes before this patch is synced, stop population
   training and restart. As of this checklist, extraction was still running and
   the count-head patch was synced to ks.
