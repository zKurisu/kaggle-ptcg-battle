# 11 - Human Matchup Strategy Ingestion

Last updated: 2026-08-10.

The goal is to turn human PTCG matchup knowledge into testable local policy
improvements. This is not a global "play better" rule layer. Each idea must
move through the same pipeline:

1. External source or episode/replay evidence.
2. A simulator-observable trigger.
3. A candidate action bias or veto.
4. Focused local validation.
5. Broad/random validation.
6. Distillation into BC or generated success data only if it actually wins.

## Source Tiers

- Matchup statistics: Limitless matchup pages, PokemonMeta, TrainerHill, and
  Kaggle episode/replay aggregates. These identify weak pairs and likely
  pressure points.
- Human deck guides and tournament articles: TCGplayer, PokeBeach, Limitless
  deck pages, notes/articles from strong players, and tournament writeups.
  These provide sequencing and game-plan hypotheses.
- Card text: official Pokemon card pages or the local `data/EN_Card_Data.csv`.
  These define what a rule can legally reason about.
- Local traces: `trace_matchup_decisions.py`, success/loss subset comparison,
  and focused baseline-delta. These decide whether the human idea applies to
  the Kaggle simulator and exact deck signature.

External strategy is a hypothesis source, not a label. If a guide says "use X
to buy turns", the model needs an observable condition such as "X is an
available PLAY/ATTACK/SWITCH option while the opponent is setting up Y".

Keep source format drift explicit. Paper Standard, PTCG Live, PTCG Pocket, and
Kaggle simulator lists can disagree on legal cards, exact card pool, and common
tech choices. A strategy is usable only after confirming that the named cards
exist in the target deck CSV and in `data/EN_Card_Data.csv`. For example, a
human answer involving another Ogerpon mask is not transferable to a pure Teal
Mask Ogerpon list unless that exact mask is present in the submitted deck.

## Kaggle Community Constraints

Relevant Kaggle discussion findings as of 2026-08-10:

- `Top players' methods, revealed by 30,000 games`
  (`https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/724362`)
  argues from action-time traces that much of the field is rule based, while
  the very top appears to combine a loaded model with bounded search or RL.
  A comment also warns that search helps only when the value estimate is good.
  Our failed online search-guided probe matches that warning: root search found
  locally higher-scoring actions but did not compose into wins.
- `Sharing my Reinforcement Learning journey`
  (`https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717697`)
  emphasizes representation, curriculum, replay failure analysis, and broad
  card/deck exposure. It also says random self-play is not enough.
- `Differences Between the Official Pokemon TCG Rules and the Simulator
  Behavior`
  (`https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586`)
  confirms simulator behavior is authoritative. Human TCG rules can seed ideas,
  but every rule must be validated against legal simulator options and replay
  outcomes.

Implication for this repo: do not use single-step online search directly for
submission. Use community strategy plus replay traces to build route-level
rules, use search only as an offline label source after strict filtering, and
train new scratch policies from generated/planner data rather than fine-tuning
the locked BC checkpoints.

## Translation Modes

- `rerank_guard`: nudge or force a legal option in a narrow state. Use this for
  high-confidence single-action mistakes such as wrong active choice.
- `matchup_bc`: filter real winning examples or overweight relevant contexts.
  Use only when enough same-signature or safely transferable wins exist.
- `teacher_rollout`: run a rules/search policy locally to create successful
  games, then distill decisions back into BC. Use when real wins are too sparse.
- `deck_sig_shadow`: train the exact opponent game plan better. Use when the
  weakness may be a poor local opponent approximation rather than our deck.
- `do_not_train`: record that data is too sparse or contradictory.

## Initial Human Strategy Seeds

The structured seed file is:

```text
data/matchup_strategy_seeds_v1.csv
```

The seed-to-card mapping file is:

```text
data/matchup_strategy_seed_cards_v1.csv
```

Use the mapping file to separate "the article says this card matters" from
"this exact Kaggle deck can use this card". Basic Energy names may come from
the simulator runtime rather than `data/EN_Card_Data.csv`, so they should be
validated from traces/deck IDs before becoming hard rules.

Current seeds:

- Marnie Grimmsnarl vs Ogerpon: Limitless matchup data confirms this is weak
  for Marnie. Marnie guides describe the core engine as Grimmsnarl plus
  Froslass/Munkidori, held together by Spikemuth Gym, Secret Box, and
  Technical Machine: Evolution. Candidate rules should prioritize getting the
  Marnie line online, preserving a second attacker, and converting spread
  damage with Munkidori/Boss/Devolution rather than delaying with low-impact
  utility actions.
- Marnie Grimmsnarl general board planning: some guides warn that Froslass is
  not always correct, especially into decks that can exploit the damage
  counters with their own Munkidori. This should become a trace feature, not a
  blanket rule: count opponent Munkidori-style threats before benching/evolving
  Froslass.
- Crustle vs ex-heavy decks: Crustle's Mysterious Rock Inn prevents attack
  damage from Pokemon ex. Human articles frame the deck as an anti-ex wall with
  healing/HP buffs, resource pressure, and opportunistic Superb Scissors KOs.
  For our Crustle policies this suggests fast Dwebble-to-Crustle setup, keeping
  Crustle active, attaching enough energy to attack, and avoiding unnecessary
  switches away from the wall.
- Ogerpon vs Crustle: if the Ogerpon list lacks a non-ex answer, a pure BC
  model cannot learn a robust counter-plan from mostly losing labels. The
  human-strategy translation should focus on early Dwebble punish windows,
  avoiding futile ex attacks into established Crustle, using tank/draw active
  choices to buy turns when no KO path exists, and constructing successful
  teacher rollouts where Dwebble is removed before the wall stabilizes.
- Cynthia vs Crustle: local success/loss mining shows `Cynthia's Spiritomb`
  appears in successful switch/active decisions. The human-card-text rationale
  is that Spiritomb's Raging Curse is a non-ex attacker whose damage can scale
  from benched Cynthia Pokemon damage and ignores Weakness. Test a narrow
  `primary_or_counter_active` rule before any training.
- Dragapult vs Marnie/Crustle: success mining points to correct Dragapult ex
  active choice, Drakloak attach-from usage, and early Dreepy setup. Human deck
  guide ingestion should focus on spread setup and Devolution-style prize maps
  before adding more generic BC weights.
- Mega Lucario route planning: official strategy material frames the deck as an
  engine route around Solrock, Lunatone, Fighting Gong, Poke Pad, and Premium
  Power Pro before cashing in with Mega Lucario ex. Treat this as a sequence
  planner seed: assemble engine, cycle/support, then commit payoff attacks.
- Ogerpon box route planning: Teal Mask Kangaskhan/Ogerpon articles emphasize
  precise support sequencing and secondary attackers such as Mega Kangaskhan ex
  and Meowth ex. Into Crustle, this should become a route that either removes
  Dwebble early or builds a secondary/search route before accepting blank ex
  attacks into an established wall.
- Mega Starmie route planning: the 0804/0805 ladder signatures use Staryu into
  Mega Starmie ex with Duskull/Dusclops/Dusknoir pressure, Hilda/Grand Tree
  evolution search, and Wally's Compassion sustain. This was added after
  rule-guided rollout selected Starmie weak pairs while `DeckPlan` had no
  Starmie route, which made `resource_plan` a no-op for that archetype.

## Public Source Catalog

The structured catalog is:

```text
data/public_strategy_sources_v1.csv
```

Use it before writing new strategy rules. The workflow is:

1. Start with a confirmed weak pair from Kaggle replay, filtered RR, or
   Limitless-style statistics.
2. Search high-priority article/video/forum sources for that exact archetype
   pair and for the key card interaction.
3. Validate that the named cards exist in the exact Kaggle deck signature.
4. Translate only simulator-observable opportunities into rules or teacher
   rollouts.
5. Record the source and whether the idea is direct, transferable, or blocked
   by decklist mismatch.

Important source notes:

- Limitless gives matchup rates and lists, but not turn-by-turn winning lines.
  Treat a 30% matchup win rate as evidence that some route exists somewhere in
  the public archetype, not proof that our exact Kaggle deck can execute it.
- TCGplayer, PokeBeach, Japanese `note.com`, and YouTube VODs are the strongest
  sources for sequencing and matchup plans.
- Reddit, Discord, and X/Twitter can reveal practical counterplay, but use only
  public/manual content and record uncertainty. Do not scrape private channels
  or bypass paid guides.
- Official card text remains mandatory for every hard rule. For example,
  Cornerstone Mask Ogerpon ex can attack through effects on the opponent Active,
  while pure Teal Mask Ogerpon ex cannot break Crustle by direct ex damage.

Current Ogerpon-vs-Crustle caveat:

- Kaggle Ogerpon `5899c772bace` and `697a82e582d5` contain Teal Mask Ogerpon ex,
  Boss/Judge-style disruption, but no Cornerstone Mask Ogerpon ex. Public
  Ogerpon Box advice that relies on Cornerstone/Demolish is therefore not
  directly transferable to those signatures.
- Ogerpon `2f538fcfa698` does contain Cornerstone Mask Ogerpon ex, and
  `2a5072194fdf` contains a broader box route with Wellspring, Mega Kangaskhan,
  Meowth, and Lillie's Clefairy. Those signatures need separate route rules.

## Validation Template

For each seed, run this sequence:

1. Confirm the human target exists in the exact deck CSV and card database.
2. Run trace on the weak matchup and count states where the desired action was
   available but not selected.
3. Implement the narrowest rule/rerank mode behind `--rules-entry`.
4. Run random sanity first.
5. Run focused baseline-delta against a weakness pool.
6. Run broad balanced-shadow delta.
7. If focused improves and broad does not collapse, generate teacher rollouts
   and distill them into a mixed BC corpus.

Do not submit a rule or specialist solely because focused delta improves. The
previous complex/card-weight/success-FT experiments often improved supervised
or narrow metrics while failing broad RR.

Do not fine-tune existing BC checkpoints for these ideas. BC is locked. The
allowed training path is:

1. Validate a rule/planner with random sanity and weakness-pool RR.
2. Generate rollout data from a rule/search/planner teacher.
3. Train a fresh scratch policy on replay corpus plus generated teacher corpus,
   with generated data intentionally overweighted.
4. Compare against the locked BC baseline in random, focused weakness pools,
   balanced shadow RR, and Kaggle replay-derived opponent pools.

## Top-Player Episode Mining

Use `tools/mine_top_player_strategy.py` when a strong Kaggle team has enough
episode data and we want to translate its play into traceable rule/teacher
hypotheses. The tool compares a target cohort, such as one team name or exact
deck signature, against a control cohort in the same archetype/matchup.

It writes:

- `games.csv`: one row per target/control trajectory with setup and tempo
  metrics.
- `target_game_keys.csv` and `control_game_keys.csv`: whole-game selectors for
  `tools/build_bc_subset.py`.
- `metric_gaps.csv`: trajectory-level differences such as attack timing,
  primary attacker board time, evolve count, and early-end count.
- `event_gaps.csv`: selected action, board-state, target-card, and 2/3-gram
  sequence differences.
- `opportunity_gaps.csv`: states where an action type or tracked card was
  legal, plus how often target vs control chose it.
- `rule_candidates.csv` and `summary.md`: ranked hypotheses for narrow rerank,
  trace, teacher rollout, or trajectory-BC follow-up.

Example: compare a top Marnie team winning games against same-signature control
losses:

```bash
python3 tools/mine_top_player_strategy.py \
  --corpus data/bc_corpus_banded_v11_0701_0804 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1200+ 1100-1199 1000-1099 \
  --deck-sig b8f251a476e7 \
  --target-team-name "LiamK" \
  --target-outcome win \
  --control-outcome loss \
  --opponent-archetype "Teal Mask Ogerpon" \
  --min-games 10 \
  --min-rate-gap 0.08 \
  --min-choose-gap 0.10 \
  --top 40 \
  --out-dir logs/top_player_strategy_20260808/marnie_liamk_vs_ogerpon
```

If the target team is only an example or has too few games, first run
`tools/build_team_deck_trajectories.py` to find high-support team/deck pairs.
Treat positive `opportunity_gap` rows as rule-probe candidates only after
checking trace examples. Treat `event_gap` n-grams and `metric_gap` rows as
teacher/trajectory hypotheses, not direct action rules.

## Seed-Driven Job Planner

Use `tools/plan_strategy_seed_jobs.py` to turn the seed tables into concrete
trace, rule-probe, and teacher-spec tasks:

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

The planner writes:

- `strategy_seed_tasks.csv`: one row per concrete candidate-vs-opponent seed
  task, including exact commands.
- `strategy_seed_skipped.csv`: seeds or pairs skipped because they cannot be
  expanded or fail hard card validation.
- `teacher_specs.jsonl`: structured future teacher-rollout specs.
- `run_strategy_seed_traces.sh`: runnable trace and available rule-probe shell
  script.

Run the generated shell script from the repo root on `ks`:

```bash
bash logs/strategy_seed_jobs_20260806/run_strategy_seed_traces.sh
```

Planner statuses:

- `rule_probe_available`: an existing `rule_overlay` mode can be tested now.
- `needs_rule_implementation`: human strategy is a rerank/veto idea, but the
  narrow rule mode does not exist yet.
- `needs_teacher_policy`: strategy likely needs successful generated rollouts.
- `trace_then_matchup_bc`: first verify the trace gap, then build a filtered or
  weighted matchup BC corpus.
- `trace_first`: gather evidence before choosing an intervention.

Summarize a completed planner directory:

```bash
python3 tools/summarize_strategy_seed_jobs.py \
  --job-dir logs/strategy_seed_jobs_20260806 \
  --weak-wr 0.35 \
  --strong-wr 0.85 \
  --top 20
```

The summary writes:

- `strategy_seed_summary_matchups.csv`
- `strategy_seed_summary_seeds.csv`
- `strategy_seed_summary.md`

Rule-only probe directories are supported too. If a rule has negative or tiny
`avg_rule_delta`, mark it as `do_not_scale_current_rule` and move back to trace,
teacher rollout, or matchup-conditioned BC.

## Rule-Guided Rollout Training

`generate_rollout_bc.py` now supports true stateful `resource_plan` actors. The
old parser accepted `+rules:resource_plan`, but it only called the stateless
overlay and therefore did not actually use route memory. As of 2026-08-10, both
`generate_rollout_bc.py` and `build_weakness_state_bank.py` instantiate
`ResourcePlanner` per game when `resource_plan` is selected.

Use `resource_plan` for aggressive route-level teacher data:

```bash
python3 tools/plan_rollout_teacher_jobs.py \
  --weakness-csv logs/eval_v11_0724_0804/rr_candidates_pop_top3_shadow_ge097_g100.csv \
  --candidate-manifest logs/eval_v11_0724_0804/candidate_manifest_pop_top3_shadow_ge097.csv \
  --opponent-manifest logs/eval_v11_0724_0804/shadow_pools_20260805/mixed_shadow_popfallback_environment_balanced.csv \
  --max-win-rate 0.45 \
  --max-jobs 24 \
  --max-per-archetype 3 \
  --max-per-candidate 2 \
  --rule-mode resource_plan \
  --games 1200 \
  --workers 24 \
  --parallel-jobs 4 \
  --keep-outcomes nonloss \
  --actor-scope game \
  --epsilon-random 0.02 \
  --flush-every-games 40 \
  --out-root data/generated_rollout_bc_resource_plan_20260810 \
  --out-band weak_route_nonloss \
  --log-dir logs/rule_guided_rollout_20260810 \
  --out-csv logs/rule_guided_rollout_20260810/rollout_teacher_plan.csv \
  --skipped-csv logs/rule_guided_rollout_20260810/rollout_teacher_skipped.csv \
  --out-sh logs/rule_guided_rollout_20260810/run_rollout_teacher_jobs.sh
```

Then run:

```bash
bash logs/rule_guided_rollout_20260810/run_rollout_teacher_jobs.sh
```

Train fresh scratch policies, not fine-tunes, by mixing the generated corpus as
an overweighted auxiliary corpus. Example for Marnie:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v12_0701_0807 \
  --aux-corpus data/generated_rollout_bc_resource_plan_20260810 \
  --aux-repeat 8 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands 1000-1099 1100-1199 1200+ \
  --deck-sig b8f251a476e7 \
  --arch cross_attn \
  --state-layers 2 \
  --width 4 \
  --epochs 8 \
  --batch 4096 \
  --lr 0.0008 \
  --winner-weight 1.5 \
  --loser-weight 0.4 \
  --draw-weight 0.8 \
  --split-by-game \
  --save checkpoints/rule_guided_20260810/bc2_marnie_b8f_resource_plan_scratch_w4.npz
```

For search-teacher output, first filter labels:

```bash
python3 tools/filter_search_teacher_labels.py \
  --best-csv 'logs/search_teacher_20260810/*_teacher_best.csv' \
  --min-delta-score 0.20 \
  --min-best-score 0.05 \
  --min-motif-count 2 \
  --out-csv logs/search_teacher_20260810/high_conf_teacher_labels.csv \
  --out-jsonl logs/search_teacher_20260810/high_conf_teacher_labels.jsonl
```
