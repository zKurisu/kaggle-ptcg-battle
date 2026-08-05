# 09 - Matchup Relations

Last updated: 2026-08-05.

This note records matchup priors from Kaggle episode data plus the latest pulled
Kaggle replay probes. These are policy-and-deck relations from the public
ladder, not pure card matchup odds. Policy quality, exact deck signature, first
player bias, and timeout behavior are part of the observed signal.

## Generated Artifacts

Reusable tool:

```bash
python3 tools/analyze_episode_matchups.py --help
```

Remote and local generated outputs:

```text
logs/matchup_notes_20260805/0804_score900/
logs/matchup_notes_20260805/0724_0804_score900/
```

The output files are:

- `archetype_matchups.csv`: directional archetype win rates.
- `archetype_counter_edges.csv`: archetype edges above the configured threshold.
- `deck_sig_matchups.csv`: directional deck signature win rates.
- `deck_sig_counter_edges.csv`: deck signature edges above the configured threshold.
- `replay_*`: same idea, but only for the latest pulled live Kaggle replays.
- `matchup_summary.md`: generated human-readable summary.

Commands used:

```bash
python3 tools/analyze_episode_matchups.py \
  --episodes-dir /home/jie/Do/0_PTCG/workspace/episodes_raw \
  --date-from 2026-08-04 \
  --date-to 2026-08-04 \
  --deck-manifest logs/ladder_pool_0804_all/pool_manifest.csv \
  --deck-manifest logs/ladder_pool_0802_all/pool_manifest.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/jie_55252321_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/jie_55241767_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/by_55254351_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/by_55252351_rows.csv \
  --out-dir logs/matchup_notes_20260805/0804_score900 \
  --source-label episodes_0804_score900 \
  --score-floor 900 \
  --min-games 20 \
  --min-deck-games 5 \
  --edge-threshold 0.08 \
  --progress-every 1000
```

```bash
python3 tools/analyze_episode_matchups.py \
  --episodes-dir /home/jie/Do/0_PTCG/workspace/episodes_raw \
  --date-from 2026-07-24 \
  --date-to 2026-08-04 \
  --deck-manifest logs/ladder_pool_0804_all/pool_manifest.csv \
  --deck-manifest logs/ladder_pool_0802_all/pool_manifest.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/jie_55252321_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/jie_55241767_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/by_55254351_rows.csv \
  --replay-rows logs/kaggle_replay_latest2_20260805/by_55252351_rows.csv \
  --out-dir logs/matchup_notes_20260805/0724_0804_score900 \
  --source-label episodes_0724_0804_score900 \
  --score-floor 900 \
  --min-games 100 \
  --min-deck-games 20 \
  --edge-threshold 0.06 \
  --workers 12
```

## Data Windows

`0804_score900`:

- Episode files: 4,811.
- Games used after filters: 4,382.
- Score-filtered games: 270.
- Other-archetype filtered games: 159.
- Latest replay rows: 212.

`0724_0804_score900`:

- Episode files: 54,105.
- Games used after filters: 46,566.
- Score-filtered games: 7,103.
- Other-archetype filtered games: 436.
- Latest replay rows: 212.

The score filter uses the current known deck manifests from `0804` and `0802`.
Unknown signatures or currently sub-900 signatures are excluded when
`--score-floor 900` is used.

## Stable Archetype Edges

These edges appear in both `0804_score900` and `0724_0804_score900`.

| Favored archetype | Weak archetype | 0804 games | 0804 WR | 12d games | 12d WR |
| --- | --- | ---: | ---: | ---: | ---: |
| Mega Lucario | Mega Lopunny | 101 | 0.861 | 219 | 0.895 |
| Mega Lopunny | Crustle Wall | 83 | 0.843 | 397 | 0.826 |
| Cynthia Garchomp | Mega Lopunny | 60 | 0.817 | 182 | 0.819 |
| Mega Lopunny | Festival Lead | 46 | 0.870 | 174 | 0.816 |
| Crustle Wall | Teal Mask Ogerpon | 66 | 0.909 | 471 | 0.800 |
| Mega Lopunny | Teal Mask Ogerpon | 129 | 0.690 | 609 | 0.772 |
| Teal Mask Ogerpon | Cynthia Garchomp | 41 | 0.805 | 258 | 0.729 |
| Teal Mask Ogerpon | Marnie Grimmsnarl | 313 | 0.767 | 3,182 | 0.725 |
| Alakazam | Crustle Wall | 84 | 0.798 | 681 | 0.674 |
| Alakazam | Teal Mask Ogerpon | 128 | 0.703 | 597 | 0.662 |
| Alakazam | Cynthia Garchomp | 71 | 0.648 | 560 | 0.648 |
| Crustle Wall | Dragapult | 30 | 0.700 | 224 | 0.603 |
| Dragapult | Alakazam | 95 | 0.632 | 343 | 0.586 |
| Mega Lopunny | Marnie Grimmsnarl | 401 | 0.626 | 2,131 | 0.585 |
| Festival Lead | Marnie Grimmsnarl | 105 | 0.648 | 1,432 | 0.584 |
| Marnie Grimmsnarl | Alakazam | 454 | 0.610 | 6,048 | 0.567 |

## 12-Day Edges Not Clear In 0804 Alone

These have enough 12-day support but did not pass the stricter 0804 single-day
threshold. Treat them as useful priors, not as current-day certainties.

| Favored archetype | Weak archetype | 12d games | 12d WR |
| --- | --- | ---: | ---: |
| Team Rocket Mewtwo | Alakazam | 372 | 0.777 |
| Teal Mask Ogerpon | Team Rocket Mewtwo | 143 | 0.650 |
| Crustle Wall | Festival Lead | 196 | 0.622 |
| Mega Lucario | Alakazam | 129 | 0.597 |
| Alakazam | Mega Lopunny | 503 | 0.573 |
| Cynthia Garchomp | Team Rocket Mewtwo | 122 | 0.566 |
| Festival Lead | Teal Mask Ogerpon | 199 | 0.563 |
| Crustle Wall | Cynthia Garchomp | 272 | 0.562 |

## Per-Archetype Weakness Priors

Use this as the first pass for weakness pool construction. Prefer the largest
negative edges, then validate with local RR/trace before training overlays or
RL fine-tunes.

| Archetype to improve | Main weakness priors |
| --- | --- |
| Alakazam | Team Rocket Mewtwo is the strongest 12-day weakness; Mega Lucario, Dragapult, and Marnie are secondary. |
| Crustle Wall | Mega Lopunny is the dominant weakness; Alakazam is also strong into Crustle. |
| Cynthia Garchomp | Teal Mask Ogerpon and Alakazam are the main weaknesses; Crustle is a smaller 12-day edge. |
| Dragapult | Crustle Wall is the stable weakness; Mega Lucario is a lower-sample concern. |
| Festival Lead | Mega Lopunny and Crustle Wall are the main weaknesses; Dragapult is a strong lower-sample concern. |
| Marnie Grimmsnarl | Teal Mask Ogerpon is the dominant weakness; Mega Lopunny and Festival Lead are secondary. |
| Mega Lopunny | Mega Lucario and Cynthia Garchomp are dominant weaknesses; Alakazam is a smaller but sampled edge. |
| Teal Mask Ogerpon | Crustle Wall, Mega Lopunny, and Alakazam are the main weaknesses; Festival and Dragapult are mild/lower-confidence. |
| Team Rocket Mewtwo | Teal Mask Ogerpon and Cynthia Garchomp are the main sampled weaknesses. |
| Mega Lucario | No strong 12-day high-score weakness was established in this filtered data. |
| Mega Starmie | Too little high-score episode data in the filtered window; Marnie over Mega Starmie is low-confidence only. |
| N's Zoroark | Too little high-score episode data in the filtered window. |

## Strong Deck-Sig Edges

These are useful for selecting concrete shadow opponents once the archetype
weakness is chosen.

| Favored deck sig | Favored archetype | Weak deck sig | Weak archetype | Games | WR |
| --- | --- | --- | --- | ---: | ---: |
| `47756cdfd20f` | Crustle Wall | `697a82e582d5` | Teal Mask Ogerpon | 76 | 0.987 |
| `276707c0fdb4` | Mega Lopunny | `e82dcbe62260` | Festival Lead | 46 | 0.978 |
| `f1445356c3a7` | Mega Lopunny | `5899c772bace` | Teal Mask Ogerpon | 25 | 0.960 |
| `7ee600c6f769` | Crustle Wall | `e82dcbe62260` | Festival Lead | 20 | 0.950 |
| `b141ae295739` | Crustle Wall | `697a82e582d5` | Teal Mask Ogerpon | 20 | 0.950 |
| `43d6d8b0fce9` | Mega Lucario | `276707c0fdb4` | Mega Lopunny | 64 | 0.906 |
| `43d6d8b0fce9` | Mega Lucario | `f1445356c3a7` | Mega Lopunny | 137 | 0.891 |
| `697a82e582d5` | Teal Mask Ogerpon | `b8f251a476e7` | Marnie Grimmsnarl | 886 | 0.844 |
| `5899c772bace` | Teal Mask Ogerpon | `b8f251a476e7` | Marnie Grimmsnarl | 265 | 0.823 |
| `962a164ee798` | Team Rocket Mewtwo | `7f9a538936e3` | Alakazam | 120 | 0.792 |
| `f0bac971c56d` | Team Rocket Mewtwo | `7f9a538936e3` | Alakazam | 67 | 0.746 |
| `7f9a538936e3` | Alakazam | `697a82e582d5` | Teal Mask Ogerpon | 154 | 0.708 |

## Latest Replay Notes

The latest replay pull only covers the newest two submissions per account at
the time of the run. It should be used as live probing evidence, not as a full
metagame matrix.

Current active replay rows showed:

- `b8f251a476e7` Marnie shadow vs `7f9a538936e3` Alakazam: 6/7 wins in the
  pulled live replay sample.
- `7f9a538936e3` Alakazam active submissions vs `b8f251a476e7` Marnie:
  combined 12/26 wins. This is weaker than the full 12-day episode prior, which
  also favors Marnie over Alakazam.
- `e82dcbe62260` Festival shadow vs `b8f251a476e7` Marnie: 9/15 wins. This
  matches the episode prior that Festival has an edge into Marnie, but the live
  sample is still small.
- `e82dcbe62260` Festival shadow vs Dragapult: 0/3 in replay and 0804/12-day
  low-sample signals also suggest Dragapult pressure. Treat this as a trace
  priority.
- `7f9a538936e3` Alakazam vs Crustle was 2/9 in the pulled replay rows, which
  conflicts with the 0804/12-day episode prior that Alakazam beats Crustle.
  Check exact Crustle signatures before using this as a training target.

## V11 Submission Replay Check

On 2026-08-05, `jie` submitted:

- `55264182`: `bc: pop_v11all_marnie_grimmsnarl_b8f251a4_1`, score observed
  around 889.2 after the early batch.
- `55264151`: `bc: pop_v11all_teal_mask_ogerpon_5899c772_2`, score observed
  around 751.1 after the early batch.

Replay pull saved at:

```text
logs/kaggle_replay_v11_submit_20260805/
```

The replay sample was still small:

- Marnie `55264182`: 23 attributed games, 15/8, WR 0.652.
- Ogerpon `55264151`: 24 attributed games, 15/9, WR 0.625.

Interpretation:

- Marnie did not underperform the local weighted RR win-rate estimate; local
  weighted RR for this candidate was 0.652, matching the pulled replay sample.
  The lower Kaggle score is likely early rating variance plus opponent-rating
  effects, not immediate evidence that the model is bad.
- Ogerpon did underperform its local weighted RR estimate. The main reason is
  current live opponent composition: Ogerpon was 0/4 into Crustle Wall in the
  replay sample, and this exactly matches the known hard weakness from episode
  and RR data. It also lost 2/3 to a live Mega Lucario signature
  `ab089ccfad1a`, which was not represented in the candidate RR pool.
- Many live opponent signatures were not in `candidate_manifest_pop_top3_shadow_ge097.csv`.
  The local pool is useful for relative testing, but it is not a reliable
  Kaggle score predictor until live replay opponent signatures are added as
  deck-sig shadows or otherwise weighted into the environment pool.

Action:

- Add live replay opponent signatures, especially Ogerpon losses to Crustle
  (`8e4fb0aa3e67`, `21218b184038`, `14f6f8138286`, `1aeea67ee0a7`) and Mega
  Lucario (`ab089ccfad1a`), to the next shadow/deck-sig coverage pass.
- Treat RR ranking as a candidate filter, then validate against a live replay
  opponent pool before spending more submissions.

## Usage For Training

1. Build weakness pools from the stable archetype edges first.
2. Select concrete opponent policies by deck signature from
   `deck_sig_counter_edges.csv`, then intersect with the audited shadow pool.
3. Run local RR and trace on the selected weakness pool before deciding between
   matchup-conditioned BC, a rule overlay, deck-sig shadow training, or a small
   RL fine-tune.
4. Replay-only conflicts should trigger trace inspection, not immediate
   training changes.
