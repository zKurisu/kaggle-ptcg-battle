# Expert Close-Read Playbooks

This file records full-game lessons from high-score Kaggle episode traces.  Do
not replace these notes with aggregate random/RR summaries.  When a matchup has
a structural local weakness, inspect at least one complete expert win and one
complete expert loss before changing rules or route policies.

## 2026-08-16: 1200+ Dragapult vs Crustle Wall

Source traces:

- Current 1200+ pool: `logs/ladder_pool_0814_0815_current_20260816/`
- Expert wins: `logs/teacher_trace_20260816/current_1200_dragapult_vs_crustle/wins/`
- Close reads: `logs/teacher_trace_20260816/current_1200_dragapult_vs_crustle/close_read/`
- Key exact hard-wall win: `2026-08-14_52c825f8-97dc-11f1-9286-0242ac130203_p0_cc2e995b5ad0_vs_7ee600c6f769_close_read.md`

Observed route from the exact `cc2e995b5ad0` 1200+ win over `7ee600c6f769`:

1. Setup starts with Dreepy, not a support-only opener when a line is available.
2. Early Poke Pad/Poffin are used to create multiple Dreepy/Drakloak lines.
3. Drakloak ability is used before stage-2 commitment and before later search
   actions.  The expert does not rush every possible Dragapult ex immediately
   if a Drakloak engine action is still available.
4. R/P energy is reserved for the Dragapult line.  Dark energy is reserved for
   Munkidori once the Dragapult route is viable.
5. Crushing Hammer and Jamming Tower are part of the plan, not optional noise.
   The win repeatedly removes Mist/Rock Fighting energy and replaces Crustle's
   stadium tempo.
6. Boss is used to pull Dwebble or avoid active wall targets when the bench
   gives a vulnerable target.
7. Dragapult ex pressure is used to keep damage moving, but the wall is not
   solved by repeatedly attacking active Crustle for zero damage.
8. Damage counters go to Dwebble/Crustle when they create a future KO route.
   Counters to Kangaskhan/Ogerpon happen when that is the active pressure line,
   but the closing route is usually wall-line damage.
9. Munkidori is a central conversion piece.  It transfers damage from Dragapult
   ex to Dwebble/Crustle, then sometimes attacks to finish active Crustle.  Rules
   that only prioritize bench DCA miss the late-game active-Crustle conversion.

Implementation implication:

- The current `dragapult_vs_crustle_plan` must be treated as a multi-turn route:
  setup line -> energy discipline -> disruption -> DCA/Munkidori conversion.
- Do not promote the Blaziken/Chi-Yu route as a generic `cc2e` solution.  That
  route belongs to `7ac0181b46a0`-style lists with Chi-Yu/Torchic/Blaziken ex.
- A useful rule hit should be visible in trace as one of:
  `munkidori_ready`, `munkidori_transfer_wall`, `dark_to_munkidori`,
  `boss_dwebble`, `dca_wall_line`, `jamming_or_hammer_before_wall_attack`.
- Any patch that only changes final random/RR win rate but does not show these
  trace hits is not a real solution.
