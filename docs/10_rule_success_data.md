# 10 - Rule And Success-Data Experiments

Last updated: 2026-08-06.

This note records the current direction for improving complex matchup play
without blindly widening BC or turning on old PPO/MCTS.

## Principle

BC remains the default policy. Rules should be narrow rerank or veto layers that
target trace-confirmed mistakes. A rule is useful only if it passes:

- random sanity;
- focused weak-pool delta;
- broad balanced-shadow delta;
- trace review showing fewer bad decision patterns.

If a rule or search variant starts producing wins in a hard matchup, preserve
those rollouts as generated successful trajectories and distill them back into
BC or use them as an RL anchor.

## External Sources

Public PTCG sites are useful as hypothesis sources, not direct labels:

- Limitless deck explorer: `https://play.limitlesstcg.com/decks`
- Limitless API docs: `https://docs.limitlesstcg.com/`
- TrainerHill meta analysis: `https://www.trainerhill.com/analysis/meta`
- PokemonMeta win rates: `https://www.pokemonmeta.com/winrates/`

Use these to ask questions such as "which attacker matters in this matchup" or
"which setup step is usually prioritized". Do not assume those heuristics are
valid in the Kaggle simulator; deck signatures, available cards, and simulator
behavior differ. Every external idea must be validated by local trace/RR.

Initial external-search hypotheses:

- Limitless has dedicated deck/matchup pages for Marnie's Grimmsnarl ex and
  Ogerpon Box style decks. These are useful priors for matchup direction and
  common support-card packages.
- Limitless Ogerpon Box lists include tech basics such as Mega Kangaskhan ex,
  Meowth ex, and Lillie's Clefairy ex. This matches the local finding that
  Kaggle `2a5072194fdf` is a different Ogerpon box, not a small variation of
  `5899/697`.
- Crustle results should be treated as anti-ex wall matchups. The goal is not
  just "attack earlier"; it may require choosing a non-ex or tech attacker,
  avoiding ability loops, and timing switches/retreats.

## Success-Data Classes

For each weak matchup, first audit how many winning demonstrations exist:

- `same_sig_success_bc_ok`: enough same-signature wins exist. Build win/loss
  subsets, compare action patterns, then try outcome-aware BC or a narrow rule.
- `same_sig_sparse_use_cross_sig_or_generate`: the target signature has a few
  wins, but not enough. Compare same-sig wins with losses, then use cross-sig
  teacher data only for shared cards or shared strategic contexts.
- `cross_sig_teacher_needed`: the target signature has no wins, but the
  archetype has winning signatures. Use teacher traces and deck intersection;
  direct imitation is unsafe.
- `generate_success_needed`: the corpus has no meaningful wins. Use local search,
  rule probes, or later RL to create successful trajectories before expecting
  BC to learn the counter-plan.

## Tools

Audit available success data:

```bash
python3 tools/audit_matchup_success_data.py \
  --corpus data/bc_corpus_banded_v11_0724_0804 \
  --weak-plan logs/eval_next_v11_specialists_20260805/weak_pair_corpus_plan.csv \
  --limit 24 \
  --top-sigs 4 \
  --out-csv logs/rule_success_20260805/matchup_success_counts_top24.csv \
  --out-plan-csv logs/rule_success_20260805/success_data_plan_top24.csv
```

Build a filtered win/loss subset:

```bash
python3 tools/build_bc_subset.py \
  --corpus data/bc_corpus_banded_v11_0724_0804 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --deck-sig b8f251a476e7 \
  --opponent-archetype "Teal Mask Ogerpon" \
  --outcome win \
  --out data/bc_success_subsets_v11_20260805 \
  --out-band marnie_b8f_ogerpon_success \
  --name marnie_b8f_vs_ogerpon_wins
```

Compare success vs failure action patterns:

```bash
python3 tools/compare_bc_subsets.py \
  --a data/bc_success_subsets_v11_20260805/Marnie_Grimmsnarl/marnie_b8f_ogerpon_success/marnie_b8f_vs_ogerpon_wins.npz \
  --a-label success \
  --b data/bc_success_subsets_v11_20260805/Marnie_Grimmsnarl/marnie_b8f_ogerpon_losses/marnie_b8f_vs_ogerpon_losses.npz \
  --b-label loss \
  --out-dir logs/rule_success_20260805/marnie_ogerpon_success_vs_loss
```

## Current Findings

`logs/rule_success_20260805/success_data_plan_top24.csv` audits the first 24
weak-pair rows from the current RR-derived weak plan over all score bands.

Top-sig recommendation counts:

- `same_sig_success_bc_ok`: 13 deck-sig rows.
- `same_sig_sparse_use_cross_sig_or_generate`: 37 rows.
- `same_sig_sparse_generate_more`: 26 rows.
- `cross_sig_teacher_needed`: 5 rows.
- `generate_success_needed`: 8 rows.

Important examples:

- Marnie `b8f251a476e7` vs Teal Mask Ogerpon has 738 wins and 60,042 winning
  decisions. This has enough same-sig data for success-vs-loss mining.
- Teal Mask Ogerpon `5899c772bace` and `697a82e582d5` vs Crustle Wall have only
  1 and 5 wins respectively in the 12-day corpus. BC cannot learn a robust
  counter-plan from those same-sig labels.
- Ogerpon `2a5072194fdf` vs Crustle has 81 wins and 5,547 winning decisions, but
  that deck is materially different from `5899/697` and includes Mega
  Kangaskhan ex, Meowth ex, and Lillie's Clefairy ex. Use it as a teacher only
  after filtering to shared cards or abstract rule hypotheses.
- Mega Starmie vs Mega Lucario has no usable top-sig data in this audit. It is a
  generated-success-data target, not a BC reweighting target.

Ogerpon-vs-Crustle success/loss comparison shows the direct labels are dominated
by the `2a507` tech plan: non-Ogerpon active/switch choices and lower repeated
Teal Mask Ogerpon ability share. The transferable hypothesis is an anti-loop
rule around high-option MAIN states and attack-window timing; direct imitation
of Kangaskhan/Meowth decisions is invalid for `5899/697`.

## Next Experiments

1. Build win/loss subset pairs for the `same_sig_success_bc_ok` rows.
2. Run `compare_bc_subsets.py` and `trace_outcome_gap_report.py` for each pair.
3. Turn only high-confidence, shared-card differences into local rule-overlay
   probes.
4. If a rule wins a hard matchup, save its rollout trace as generated success
   data and distill it with BC.
5. For rows marked `generate_success_needed`, do not train BC until search/RL
   has produced positive trajectories.
