# 03 — Roadmap

## Phase 1: Rule-Based Opponent Pool (current)
- [x] Mega Lucario (760 score, existing)
- [x] Alakazam (615 score, existing)
- [x] Marnie Grimmsnarl (v3, 90% vs Random)
- [ ] Crustle Wall
- [ ] Archaludon
- [ ] Dragapult (verify kiyotah notebook)
- [ ] Cynthia Garchomp
- [ ] Mega Lopunny
- [ ] Remaining top-14

## Phase 2: BC Imitation Learning (current)
- [x] Download Kaggle episode datasets (4 days, ~17K episodes)
- [x] Extract state→action pairs with score band tags
- [ ] Train BC policies for top archetypes
- [ ] Deploy as opponent strategies

## Phase 3: Population Training
- [ ] Solo PPO for each opponent (50 iters)
- [ ] Population training: our agent vs trained opponents
- [ ] Curriculum: 600→800→1000→1200+

## Phase 4: Kaggle Submission Pipeline
- [ ] Pack trained model as Kaggle submission
- [ ] Submit and iterate
