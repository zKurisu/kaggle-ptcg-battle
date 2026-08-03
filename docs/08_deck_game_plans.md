# Deck Game Plans

This document records the current deck-plan assumptions used for BC diagnostics
and future plan-aware features. These are not complete hand-written agents; they
are guardrails for interpreting BC errors and selecting deck-specific features.

## External Signals

Kaggle discussion suggests the field is not solved by generic BC alone:

- Some competitors are exploring RL from rule-based teachers and positive
  examples.
- Heuristic-only agents have reportedly reached near top-150, and several
  players suspect some high-rank agents use strong rules or rule/model hybrids.
- Submission score can get stuck in the 800-1000 band depending on early
  matchups and current ladder composition.
- Meta analysis shows deck win rate is environment-dependent. A deck can be good
  when rare and lose edge after the field adapts.

Implication for this repo: random win rate is only a sanity check. For submission
candidates, train deck-signature specialists, evaluate against the current ladder
pool, and inspect plan-tagged failures.

## Current Plan Registry

The machine-readable version is in `ptcg_rl/deck_plans.py`.

| Archetype | Core plan | Current status |
| --- | --- | --- |
| Marnie Grimmsnarl | Build the Impidimp -> Morgrem -> Grimmsnarl ex line, then use Punk Up to attach dark energy and convert setup into attacks. | Mixed is acceptable because data is large, but keep deck-sig ablations. |
| Teal Mask Ogerpon | Use basic-ex tempo, Teal Dance energy/draw acceleration, and support basics for consistency/matchups. | Must stay deck-sig/top-k. Mixed v8 collapsed on Kaggle. |
| Mega Lopunny | Set up Mega Lopunny with consistency support, then attack with a coherent top2 signature plan. | Random is strong, core matchups weak. Keep as population until matchup training improves. |
| Mega Lucario | Sample-limited single-signature Mega attacker plan with Solrock/Lunatone/Hariyama support. | Needs deck-sig and correct score bands. v9 features improved random but data remains thin. |
| Alakazam | Stage-2 control/bench attack plan using Kadabra draw and correct deck pairing. | Policy/deck mismatch was the main failure mode; registry auto-deck is mandatory. |
| Dragapult | Stage-2 spread plan with Drakloak filtering and Dragapult ex damage placement. | Not ready; needs damage-counter and discard-choice work. |
| Festival Lead | Enable Festival Lead/Dipplin repeated attacks, with Thwackey search and Festival Grounds dependency. | Top1 specialist works vs random; mixed only for population. |
| Crustle Wall | Anti-ex wall plan using Crustle's damage prevention against Pokemon ex. | Matchup-specific counter, especially useful in ex-heavy fields. |
| Cynthia Garchomp | Linear Stage-2 plan with Gabite search into Garchomp ex. | Mixed or top1 acceptable; validate by current pool. |
| Team Rocket Mewtwo | Fill board with Team Rocket Pokemon before Mewtwo can attack; Spidops accelerates from discard. | Strong vs random but not enough on Kaggle/core. |

## Next Feature Direction

Plan-aware features should answer questions that generic v9 features cannot:

- Is the primary attacker in play, in hand, or still missing?
- Is the evolution chain blocked at basic, stage1, or stage2?
- Is the deck's engine online this turn?
- Is this option selecting a primary attacker, setup basic, engine card, energy
  acceleration piece, or unrelated card?
- Is the model missing attacks or ending before the deck's main plan is online?
- In multi-choice contexts, did the selected set preserve the plan even if the
  exact card order differs?

Use `tools/deck_plan_report.py` before training/eval to catch policy/deck
mismatch and `tools/bc2_failure_report.py` to inspect plan-tagged mistakes.

## Trajectory Specialists

For unstable or data-mixed archetypes, prefer a teacher trajectory over a broad
archetype pool:

```bash
python3 tools/build_team_deck_trajectories.py \
  --corpus data/bc_corpus_banded_v9 \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" "800-899" \
  --out logs/team_deck_trajectories_v9.csv
```

Then train against a selected `team_name + deck_sig` pair:

```bash
python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig 697a82e582d5 \
  --team-name "Majkel1337" \
  --winner-only \
  --epochs 12 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --save checkpoints/bc2_ogerpon_majkel1337_traj_v9_w2.npz
```

Use trajectory specialists when:

- the same team uses the same signature across multiple dates;
- the trajectory has enough decisions and games;
- mixed/top-k training looks strong vs random but weak in core matchups;
- a card's plan depends on long-term setup choices that broad BC averages away.

## Rule Overlay Experiments

`ptcg_rl/rule_overlay.py` provides an experimental guard layer for local tests.
It is disabled by default and should not be enabled for submissions until it
beats the BC baseline in round-robin.

Random test:

```bash
python3 tools/eval_bc.py checkpoints/bc2_ogerpon_top1_v9_gameplan_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --games 500 --workers 8 --rules conservative
```

Candidate-only matchup test:

```bash
python3 tools/eval_round_robin.py \
  --entry candidate=checkpoints/bc2_ogerpon_top1_v9_gameplan_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --entry crustle=checkpoints/bc2_crustle_wall_top5_v8_topdeck_w2.npz:logs/ladder_pool_0802_all/decks/47756cdfd20f_crustle_wall_flg.csv \
  --rules-entry candidate=conservative \
  --games 500 --workers 8 --max-turns 700
```

The first target for rules is not to improve random win rate. The target is to
reduce systematic errors such as early END, missed attack windows, discarding
plan-critical cards, or failing known matchup plans.
