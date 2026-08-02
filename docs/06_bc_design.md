# 06 — BC Training Design

## Core Idea

Train PolicyValueNet to imitate human decisions from high-score Kaggle replays.
Same model architecture, different training objective: supervised cross-entropy
instead of PPO policy gradient.

## Data → Training Loop

```
data/bc_corpus_banded/
  Marnie_Grimmsnarl/
    1200+/2026-08-01.npz    ← only top players
    1100-1199/2026-07-30.npz

For each training step:
  1. Sample batch of decisions from .npz files
  2. Forward pass: state → option logprobs + value
  3. Loss = cross_entropy(logprobs, human_action) + value_loss
  4. Backward, update
```

## Multi-Select Handling

```
Decision: pick 3 options from 10 (minCount=0, maxCount=3)

Human picked: [2, 7, STOP]  (3 sequential picks)

Training loss = 
  cross_entropy(logprobs, 2)    ← first pick
  + cross_entropy(logprobs, 7)  ← second pick (option 2 masked)
  + cross_entropy(logprobs, STOP) ← third pick (STOP at index 10)

Same as model.act() logic, but with known targets.
```

## Batch Collation

Variable-length decisions → fixed tensor batch:

```
All decisions in batch share:
  - Same number of options → pad to max_N
  - State features: stack [batch, feat_dim]
  - Option features: pad to [batch, max_N, opt_feat_dim]
  - Action targets: pad to [batch, max_picks] with STOP sentinel
  - Mask: which options are legal [batch, max_N + 1]
```

## Training Config

```python
BC_TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 128,
    "epochs": 10,            # fixed corpus, can overfit
    "dropout": 0.1,          # regularization
    "value_coef": 0.5,       # train value head too
    "score_filter": "1100+",  # only top players
    "datasets": ["2026-07-28", "2026-07-29", "2026-07-30", "2026-08-01"],
}
```

## Implementation (bc_trainer.py)

```python
class BCTrainer:
    def __init__(self, model, corpus_dir, archetype="Marnie Grimmsnarl",
                 score_bands=["1200+", "1100-1199"]):
        # Load all .npz files matching archetype + score bands
        self.samples = self._load_corpus(corpus_dir, archetype, score_bands)

    def _load_corpus(self, dir, arch, bands):
        samples = []
        for band in bands:
            for npz in glob(f"{dir}/{arch}/{band}/*.npz"):
                data = np.load(npz, allow_pickle=True)
                for i in range(len(data['board'])):
                    samples.append({k: data[k][i] for k in data.files})
        return samples

    def train_step(self, batch):
        # Forward: reuse model.evaluate_actions()
        logprobs, entropies, values = self.model.evaluate_actions(batch)
        # Loss: cross-entropy (maximize probability of human action)
        policy_loss = -logprobs.mean()
        value_loss = (values - batch['outcome']).pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        self.optimizer.step()
```

## Key Differences from PPO Training

| | PPO | BC |
|---|---|---|
| Data source | Self-play games | Kaggle replays |
| Action selection | Model samples | Human chose |
| Policy loss | Clipped PPO objective | Cross-entropy |
| Value target | GAE returns | Game outcome (win/loss) |
| Exploration | entropy bonus | Fixed dataset |
| Overfitting risk | Low (fresh data) | High (fixed corpus) |

## Output

After BC training: `policy_bc.npz` — a model that plays like a top-1100+ player.
Then fine-tune with PPO self-play to surpass human performance.

## Implementation Status

- [x] `tools/bc_trainer.py` — working, stable pol_loss ~3-5
- [x] Data loading: 304K decisions, 0s load time
- [x] Score band filtering (1200+, 1100-1199, etc.)
- [x] GPU training with inline collation (no CPU round-trip)
- [x] progress bar (per-batch)

## Key Bug Fixed

**STOP token masked when `len(action) < minCount`**
- Some replay data has inconsistent action/minCount
- Fix: only set STOP target if `len(acts[i]) >= mn[i]`
- Without fix: pol_raw = 7,812,503 (NEG_INF/128)
- With fix: pol_raw = 3-5 (normal)

## Next Steps

1. Run full BC training on Marnie Grimmsnarl 1100+ (304K decisions, ~1.5h/epoch)
2. Export policy.npz → submit to Kaggle as BC baseline
3. BC+PPO finetune
4. Repeat for other archetypes (Alakazam, Crustle, etc.)
