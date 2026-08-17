# 06 - BC 训练实操

这一章从“我已经有 corpus”开始，讲如何训练第一个可用 BC、如何读训练日志，以及训练后必须做哪些验证。

## 1. 训练前先确定目标

不要一上来就跑全量训练。先回答四个问题：

1. 训练哪个 `archetype`？
2. 是否限定 `deck_sig`？
3. 用哪些 `score-bands`？
4. 这次产物是 submission 候选、shadow opponent，还是普通 population baseline？

这四个问题决定了数据过滤、模型大小和评测方式。

## 2. 先看 corpus 分布

用 `bc_corpus_stats.py` 看样本量、deck sig 和队伍分布：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --top 30 \
  --out logs/stats_marnie_0801_0815.csv
```

重点看：

- 哪些 `deck_sig` 样本最多。
- 样本是否来自少数稳定高分队伍。
- 胜局和败局数量是否极端不平衡。
- 某个 sig 是否只在一两天出现。

如果这里都看不清，后面的训练结果很难解释。

## 3. 单个 specialist 训练

下面是一个 deck-sig specialist 模板：

```bash
mkdir -p checkpoints/bc_tutorial logs/bc_tutorial
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b8f251a476e7 \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --arch pointer \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --checkpoint-every 1 \
  --save checkpoints/bc_tutorial/bc2_marnie_b8f_w3.npz \
  2>&1 | tee logs/bc_tutorial/bc2_marnie_b8f_w3.log
```

如果你只是做 smoke test，把 `--epochs` 调到 1，`--batch-size` 调小即可。

## 4. population 并行训练

当你要一次训练多个 archetype，用 `train_bc_population.py`：

```bash
python3 -u tools/train_bc_population.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Alakazam" \
  --archetype "Dragapult" \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --epochs 8 \
  --batch-size 4096 \
  --width 3.0 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --checkpoint-dir checkpoints/pop_tutorial \
  --log-dir logs/pop_tutorial \
  --tag score900_w3 \
  --poll-seconds 30 \
  2>&1 | tee logs/pop_tutorial/runner.log
```

`train_bc_population.py` 会为每个 archetype 启一个 `bc2_train.py` 子进程。训练长任务一定要保留 `runner.log`，后续才能知道失败点。

## 5. 训练日志怎么看

训练时至少要关注这些信号：

- `train` / `val` loss：是否持续下降，是否过拟合。
- best checkpoint 是第几个 epoch：如果很早最佳，后续可能在退化。
- `first_action`：第一步动作是否学会。
- `set` / `multi-select`：多选是否真的在改善。
- `trajectory` / `step_plan`：长期计划辅助项是否有效。
- `accuracy`：离线标签拟合度，但不能单独作为提交依据。
- GPU 显存和吞吐：显存空着且训练慢，可以加 batch size 或并行任务。

项目的历史教训是：最终 random/RR 才发现问题太晚。新的信号和诊断必须在训练日志里能看到。

## 6. 训练后第一关：random

用训练时对应的 deck 测随机基线：

```bash
python3 tools/eval_bc.py checkpoints/bc_tutorial/bc2_marnie_b8f_w3.npz \
  --deck logs/ladder_pool_latest/decks/b8f251a476e7_marnie_grimmsnarl.csv \
  --games 300 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50
```

经验阈值：

- 很多直接卡组应该接近 100%。
- 复杂卡组低于 97% 时，通常不应提交。
- random 高不代表强，但 random 低基本说明基础策略或 deck 对齐有问题。

## 7. 第二关：RR 和 baseline-delta

RR 看候选之间的直接胜负：

```bash
python3 tools/eval_round_robin.py \
  --manifest logs/candidate_manifest.csv \
  --games 80 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/rr_candidates_g80.csv
```

baseline-delta 看候选是否比基线更适合当前对手池：

```bash
python3 tools/eval_baseline_delta.py \
  --baseline base=checkpoints/base.npz:decks/base.csv \
  --candidate cand=checkpoints/bc_tutorial/bc2_marnie_b8f_w3.npz:decks/marnie.csv \
  --opponent-manifest logs/shadow_manifest.csv \
  --games 80 \
  --workers 16 \
  --max-turns 700 \
  --skip-bad-entries \
  --out-csv logs/baseline_delta_marnie.csv
```

## 8. 第三关：失败 trace

如果 RR 显示某个 matchup 明显差，直接看单局决策：

```bash
python3 tools/trace_matchup_decisions.py \
  --candidate cand=checkpoints/bc_tutorial/bc2_marnie_b8f_w3.npz:decks/marnie.csv \
  --opponent weak=checkpoints/weak.npz:decks/weak.csv \
  --games 20 \
  --max-turns 700 \
  --out-prefix logs/trace_marnie_vs_weak
```

不要只看胜负。要看每回合是否错过：

- attack window
- attach
- evolve
- search target
- active/bench 选择
- 关键资源保留

## 9. 打包 submission

只有通过基本验证后再打包：

```bash
mkdir -p submission
python3 tools/package_submission.py \
  --policy checkpoints/bc_tutorial/bc2_marnie_b8f_w3.npz \
  --deck decks/pool_329_marnie_s_grimmsnarl.csv \
  --cg-dir "$CG_DIR" \
  --out submission/submission_marnie_b8f_w3.tar.gz
```

## 10. 常见错误

- `deck.csv` 和 `deck_sig` 不对应：random 会异常低。
- 训练用了 history，但 submission 推理没有同等 history：线上会退化。
- mixed 数据太杂：loss 好看，game plan 被平均掉。
- winner-only 数据太少：过拟合胜场事故。
- RR 池太弱：候选看起来强，但 Kaggle ladder 被真实强模型打爆。

下一章：[07 - specialist、population、shadow](07_population_shadow.md)
