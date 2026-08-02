# 01 — Pipeline Architecture

## Components

```
train.py          → PPOTrainer (self-play + PPO update)
main.py           → Kaggle submission agent
ptcg_rl/
  encoder.py      → FastEncoder: obs dict → numpy, 0.21ms/decision
  model.py        → PolicyValueNet: pointer-style, 501K params
  trainer.py      → PPOTrainer: self-play collection + PPO
  numpy_policy.py → NumpyPolicy: torch-free Kaggle inference + MCTS
tools/
  bc_extract_v2.py → Extract state→action pairs from Kaggle episodes
  deck_battle.py   → Local deck vs deck matchup testing
  ladder_decks.py  → Leaderboard deck classification
  convert_deck.py  → PTCGL text → Card ID converter
```

## Data Flow

```
Training:
  deck.csv → PPOTrainer → self-play games → GAE → PPO update → checkpoint

Inference (Kaggle):
  policy.npz → NumpyPolicy.select_mcts() → action indices
  (falls back: MCTS → greedy → random)

BC Corpus:
  Kaggle episode ZIPs → bc_extract_v2.py → data/bc_corpus_banded/
  <Archetype>/<ScoreBand>/<date>.npz
```
