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
  1. Sample batch indices from in-memory .npz arrays
  2. Collate variable-length options/actions into padded numpy arrays
  3. Move each padded field to the target device once
  4. Forward pass: state → sequential option logprobs
  5. Loss = mean cross_entropy(logprobs, human_action/STOP)
  6. Backward, update
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

The implementation keeps each `.npz` file as a loaded dict of numpy arrays. It
does not expand the corpus into one Python dict per decision, and it does not
write one sample at a time into GPU tensors. Batch collation happens in CPU
numpy arrays first, then each field is transferred to the device once.

## Training Config

```python
BC_TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 2048,
    "epochs": 30,            # fixed corpus, can overfit
    "width": 2.0,
    "score_filter": "1100+",  # only top players
    "include_empty": False,   # default filters empty actions
    "datasets": ["2026-07-28", "2026-07-29", "2026-07-30", "2026-08-01"],
}
```

## Implementation (bc_trainer.py)

```python
class BCTrainer:
    def __init__(self, corpus_dir, archetype="Marnie Grimmsnarl",
                 score_bands=["1200+", "1100-1199"], width=2.0):
        # Load all .npz files matching archetype + score bands
        self.npz_data, self.groups = self._load_corpus(corpus_dir, archetype, score_bands)

    def _load_corpus(self, dir, arch, bands):
        npz_data, groups = [], []
        for band in bands:
            for npz in glob(f"{dir}/{arch}/{band}/*.npz"):
                with np.load(npz, allow_pickle=True) as z:
                    data = {k: z[k] for k in z.files}
                di = len(npz_data)
                idxs = [i for i in range(len(data["board"])) if len(data["action"][i]) > 0]
                npz_data.append(data)
                groups.append([(di, i) for i in idxs])
        return npz_data, groups

    def train_step(self, indices):
        batch = self._collate(indices)
        loss = self._compute_loss(batch)
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
- [x] Data loading: each `.npz` is materialized once, then reused from memory
- [x] Score band filtering (1200+, 1100-1199, etc.)
- [x] Empty actions filtered by default; `--include-empty` trains legal STOP
- [x] GPU training with numpy-first batch collation and one transfer per field
- [x] progress bar (per-batch)

## Key Bug Fixed

**STOP token masked when `len(action) < minCount`**
- Some replay data has inconsistent action/minCount
- Fix: only set STOP target if `len(acts[i]) >= mn[i]`
- Without fix: pol_raw = 7,812,503 (NEG_INF/128)
- With fix: pol_raw = 3-5 (normal)

## Performance Fix

A800 utilization was near zero because the old trainer mixed lazy `.npz`
member access with per-sample GPU tensor writes. That made every batch wait on
Python/object-array access and many tiny host-to-device copies.

The current trainer:
- loads each `.npz` member into normal numpy arrays once;
- keeps only `(file_index, sample_index)` references in training indices;
- pads a whole batch in CPU numpy arrays;
- sends each batch field to the device once;
- updates selected-option state with tensor indexing instead of per-sample GPU
  writes.

## Next Steps

1. Run full BC training on Marnie Grimmsnarl 1100+ with `--batch-size 2048`
2. Export policy.npz → submit to Kaggle as BC baseline
3. BC+PPO finetune
4. Repeat for other archetypes (Alakazam, Crustle, etc.)
