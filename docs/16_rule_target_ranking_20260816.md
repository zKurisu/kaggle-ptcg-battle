# Rule Target Ranking 2026-08-16

This note is the current target-selection record after the pivot away from
Dragapult-first work. The active rule/strategy target is Alakazam. Dragapult is
still important in the ladder, but current local bases are too weak and too
complex for rule overlays to rescue reliably.

## Inputs

- Current Kaggle ladder pool: `logs/ladder_pool_0814_0815_current_20260816`.
- Current local sanity/RR:
  - `logs/random_loss_probe_20260816/summary.txt`.
  - `logs/rr_probe_after_random100_20260816/*_vs_current_models_g120.csv`.
- Historical local notes from v7-v15 in `AGENT_HANDOFF.md` and `docs/07_*`,
  `docs/08_*`, `docs/09_*`, `docs/10_*`, `docs/11_*`.
- External strategy/meta priors:
  - Kaggle topic 735346: "The Meta on the Eve of the Final Deadline:
    Grimmsnarl ex Down, Dragapult ex Up".
  - Limitless deck explorer:
    `https://play.limitlesstcg.com/decks`. Current public Standard data shows
    Dragapult, Festival Lead, Alakazam Dudunsparce, Grimmsnarl Froslass, Mega
    Lucario, Cynthia's Garchomp, Rocket's Mewtwo, Crustle, and Starmie variants
    as real archetypes rather than Kaggle-only artifacts.
  - Official Pokemon Mega Lucario articles:
    `https://www.pokemon.com/us/strategy/pokemon-tcg-mega-evolution-battle-pass-deck-strategies`
    and
    `https://www.pokemon.com/uk/strategy/pokemon-tcg-deck-list-and-strategy-building-a-mega-lucario-ex-deck`.
    These make Lucario rule candidates concrete: Solrock/Lunatone engine,
    Aura Jab energy setup, Premium Power Pro, Hariyama/Boss gust timing, and
    single-prize prize-trade management.
  - TCGplayer Dragapult ex/Dusknoir guide:
    `https://www.tcgplayer.com/content/article/Dragapult-ex-Dusknoir-Deck-Guide-Pok%C3%A9mon-TCG-December-2025/5ae9f646-c024-4f38-8b3c-4aa65d2089f4/`.
    It reinforces why Dragapult is strategically strong but hard for our local
    base: Budew slow-down, Drakloak draw, Phantom Dive, Dusknoir/Cursed Blast,
    and multi-prize damage maps.
  - TCGplayer Marnie's Grimmsnarl ex guide:
    `https://www.tcgplayer.com/content/article/Marnie-s-Grimmsnarl-ex-Deck-Guide-Pok%C3%A9mon-TCG-July-2025/8c493445-7f49-4281-9c98-37f0938c239f/`.
    It reinforces Marnie rule candidates: Spikemuth Gym / TM Evolution setup,
    Froslass and Munkidori spread conversion, Devolution windows, and avoiding
    Froslass in matchups where the opponent can exploit damage counters.
  - Pokemon community Alakazam ex strategy thread:
    `https://community.pokemon.com/en-us/discussion/9389/best-pokemon-trading-card-game-strategy-039-s`.
    The useful transferable idea is not the exact old list, but the bench-attack
    / wall-active patience pattern: Alakazam attacks from the bench while the
    active slot buys time and protects the route.

External sources are hypotheses only; simulator traces and local RR decide
whether a rule is retained.

## 0815 Kaggle Ladder Snapshot

`logs/ladder_pool_0814_0815_current_20260816/archetype_stats.csv`:

| Archetype | Decks | Games | Weight | Comment |
| --- | ---: | ---: | ---: | --- |
| Dragapult | 25 | 6928 | 17070.9580 | By far the largest current ladder pressure. |
| Teal Mask Ogerpon | 33 | 4795 | 10333.1234 | Still very common, but sig-sensitive. |
| Alakazam | 28 | 3697 | 5648.0902 | High enough prevalence and has a usable base. |
| Mega Lopunny | 6 | 3334 | 5577.7012 | One top sig carries most support. |
| Other | 24 | 2123 | 5048.7083 | Needs separate classification before training. |
| Marnie Grimmsnarl | 11 | 3274 | 4862.3967 | Still has 1200+ b8f, despite community "down" note. |
| Crustle Wall | 15 | 1228 | 2278.9459 | Strong counter role, lower share. |
| Mega Lucario | 5 | 906 | 1966.8092 | Balanced candidate; smaller data. |
| Festival Lead | 11 | 617 | 1043.4965 | Relevant opponent, not first rule target. |
| Team Rocket Mewtwo | 5 | 134 | 234.6510 | Low current share and poor local base. |
| N's Zoroark | 2 | 77 | 152.2356 | Low current local coverage. |
| Cynthia Garchomp | 1 | 103 | 114.9867 | Low current share but simple enough to monitor. |
| Mega Starmie | 1 | 8 | 7.8598 | Insufficient Kaggle/local data. |
| Archaludon | 2 | 4 | 4.3650 | Insufficient Kaggle/local data. |

## Complexity Ranking

Complexity means how much correct play depends on multi-turn route planning,
hidden/resource memory, damage-counter/prize mapping, or matchup-specific
sequencing. Lower is easier for a rule/base hybrid.

| Rank | Archetype | Complexity | Reason |
| ---: | --- | --- | --- |
| 1 | Cynthia Garchomp | Low-medium | Mostly linear Stage-2 route and attack timing. |
| 2 | Mega Lucario | Medium | Engine route is explicit: Solrock/Lunatone, Fighting Gong, attacker choice. |
| 3 | Crustle Wall | Medium | Simple wall identity, but resource/stadium/prize timing matters. |
| 4 | Teal Mask Ogerpon | Medium | Direct tempo for some sigs, but box/tech sigs diverge strongly. |
| 5 | Team Rocket Mewtwo | Medium | Team Rocket board setup is explicit, but current base is poor. |
| 6 | Marnie Grimmsnarl | Medium-high | Spread, Munkidori/Froslass, Devolution, and Ogerpon weakness. |
| 7 | Mega Lopunny | Medium-high | Tank/heal/retreat loops and 3-prize risk management. |
| 8 | Alakazam | High | Bench attacker, wall/active choice, Rare Candy/Kadabra timing, disruption. |
| 9 | Festival Lead | High | Stadium/engine dependency and repeated-attack sequencing. |
| 10 | N's Zoroark | High | Transform/route choices and current low local coverage. |
| 11 | Mega Starmie | High | Starmie/Froslass/Munkidori two-axis plan, low local base. |
| 12 | Dragapult | Very high | Stage-2 setup, Drakloak timing, DCA allocation, Dusknoir/Munkidori prize maps. |
| 13 | Archaludon | Unknown | Too little current data. |

## Rule-Improvement Suitability

Suitability requires both rule leverage and a usable local base. A deck with
great human strategy but a weak random/RR base is not a near-term rule target.

| Rank | Archetype | Suitability | Evidence |
| ---: | --- | --- | --- |
| 1 | Alakazam | High | v14seq 7f9 random 500/500; current RR strong except Marnie; clear traceable mistakes. |
| 2 | Mega Lucario | High | Historically balanced, random 100 in v14, explicit official route. |
| 3 | Mega Lopunny | Medium-high | Current ladder weight high and v14 base strong; rules can target tank/heal/pivot timing. |
| 4 | Crustle Wall | Medium-high | Rules align well with wall/stadium/resource play; hard-counter risk remains. |
| 5 | Marnie Grimmsnarl | Medium | Usable bases exist, but Ogerpon structural weakness has resisted BC/rule tweaks. |
| 6 | Teal Mask Ogerpon | Medium-low | Some strong historical sigs, but recent sig drift and Crustle weakness are structural. |
| 7 | Cynthia Garchomp | Medium-low | Simpler route, but low current share and limited fresh base evidence. |
| 8 | Festival Lead | Low-medium | Relevant opponent; base/rule target not yet strong enough. |
| 9 | Team Rocket Mewtwo | Low | Current local base poor despite readable plan. |
| 10 | N's Zoroark | Low | Low current coverage. |
| 11 | Mega Starmie | Low | Human strategy exists, but local base and Kaggle data are too weak. |
| 12 | Dragapult | Low near-term, high long-term | Current ladder says important, but local RR remains poor even after random gate. |
| 13 | Archaludon | Unknown/low | Too little data. |

## No-Hard-Counter Ranking

This is not pure public TCG matchup strength. It combines historical Kaggle
episode edges, current local RR, and current 0815 prevalence.

| Rank | Archetype | Hard-counter risk | Notes |
| ---: | --- | --- | --- |
| 1 | Mega Lucario | Low | Historical notes found no strong 12-day high-score weakness; current base feasible. |
| 2 | Alakazam | Medium-low locally | Local v14seq 7f9 is broad; public/episode data warn about Marnie/TRM/Dragapult. |
| 3 | Mega Lopunny | Medium | Weak into Mega Lucario/Cynthia historically, but current base is usable. |
| 4 | Dragapult | Medium in real meta, high locally | Public meta strong, but our base loses many local RR pairs. |
| 5 | Crustle Wall | Medium-high | Excellent counter role but loses to Alakazam/Mega Lopunny/Lucario/Starmie-style answers. |
| 6 | Marnie Grimmsnarl | Medium-high | Ogerpon remains a dominant structural weakness. |
| 7 | Teal Mask Ogerpon | High | Crustle/Lopunny/Alakazam are recurring issues; sig tech matters. |
| 8 | Cynthia Garchomp | High/uncertain | Ogerpon/Alakazam weakness priors; low current support. |
| 9 | Team Rocket Mewtwo | High/uncertain | Can beat Alakazam in data, but local base and current share are poor. |
| 10 | Festival Lead | High/uncertain | Several bad matchups and lower local support. |
| 11 | N's Zoroark | Unknown | Not enough current Kaggle/local data. |
| 12 | Mega Starmie | Unknown/high | Very low local support; high complexity. |
| 13 | Archaludon | Unknown | Too little data. |

## Current Decision

1. Continue with Alakazam as the active rule/strategy target.
   - Base: `checkpoints/v14_sequence_0808_0812/pop_top2_allbands_parallel3/v14seq_alakazam_7f9a5389_1.pt`.
   - Deck: `logs/ladder_pool_0812_all_v13_20260813/decks/7f9a538936e3_alakazam_yushin_ito.csv`.
   - Primary weak trace target: Marnie `b8f251a476e7`, then TRM/Festival if
     fresh RR confirms.
2. If the new Alakazam rebase jobs produce a better random/RR base, switch the
   rule isolation base to that checkpoint.
3. If Alakazam fails the rule loop, next candidates are:
   - Mega Lucario `43d6d8b0fce9`: best combined balance/rule-readability.
   - Mega Lopunny `f1445356c3a7`: current ladder weight plus usable base.
   - Crustle Wall: strong counter deck, but use only with explicit hard-counter
     accounting.
4. Do not return to Dragapult-first work until a Dragapult base can pass both
   random 100% and a broad RR sanity check. Use Dragapult as an important
   opponent/shadow target meanwhile.

## Practical Rule-Design Implications For Alakazam

- Rule work should focus on route-level opportunities that traces can verify:
  - preserve and establish Abra/Kadabra/Alakazam line;
  - choose Rare Candy vs Kadabra route correctly;
  - use Kadabra draw before evolving when legal and beneficial;
  - keep the correct wall/utility active while Alakazam attacks from bench;
  - avoid late missed attack windows;
  - against Marnie, reduce avoidable damage-spread/devolution collapse by
    preserving backup line and not over-benching liabilities.
- Rules must be isolated by owner. Do not enable Dragapult-side or opponent-side
  experimental rules when measuring Alakazam-specific improvement.
