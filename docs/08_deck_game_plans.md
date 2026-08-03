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
