# 11 - Human Matchup Strategy Ingestion

Last updated: 2026-08-06.

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
