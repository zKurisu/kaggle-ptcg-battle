# 04 — BC Extraction Pipeline

## Data Source
Kaggle daily episode datasets: `pokemon-tcg-ai-battle-episodes-2026-MM-DD.zip`
- ~4000-4500 episodes per day (~710MB compressed, 21GB uncompressed)

## Extraction
```
episodes_raw/*.zip
  → bc_extract_v2.py
    → data/bc_corpus_banded/<Archetype>/<ScoreBand>/<date>.npz
```

Each .npz contains:
- board[12], hand[25], feats[32] — state encoding
- ot, oc, oc2, oa, of — option encoding
- action — selected option indices
- min_c, max_c — selection constraints

## Score Bands
1200+, 1100-1199, 1000-1099, 900-999, 800-899, 700-799, 600-699

## Usage for BC Training
```python
import numpy as np
# Load only high-score decisions
data = np.load("data/bc_corpus_banded/Marnie_Grimmsnarl/1200+/2026-08-01.npz", allow_pickle=True)
# Train: state → action
```
