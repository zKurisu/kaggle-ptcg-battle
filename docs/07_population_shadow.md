# 07 - specialist、population、shadow

上一章训练了单个 BC。本章解释三类模型的区别：

- `specialist`：为某个 archetype / deck_sig / team 专门训练，通常用于 submission 候选。
- `population`：为多个 archetype 训练一批基本可用模型，用于本地环境覆盖。
- `shadow`：模拟 Kaggle ladder 中具体强队或具体 deck_sig，用于 RR、baseline-delta 和弱点分析。

三者不要混用。提交模型追求稳定和强度；shadow 追求环境覆盖；population 追求广度。

## 1. 什么时候用 specialist

优先用 specialist 的情况：

- 同一 archetype 下 `deck_sig` 差异很大。
- 线上历史显示 mixed 策略明显变差。
- 某个高分队伍长期使用同一 signature。
- 你准备把它作为 Kaggle submission 候选。

典型例子：

- Ogerpon：top2/top3 或高质量 sig 很关键，mixed 容易冲掉打法。
- Crustle：counter deck 的具体构筑影响很大。
- Alakazam：deck 对齐错误时 random 都会掉。
- Dragapult：动作细节多，不适合粗 mixed。

Marnie 这类数据量巨大的卡组可以尝试 mixed，但历史上 deck-sig specialist 仍然经常更稳。

## 2. 什么时候用 population

population 适合做：

- 本地 RR 候选池。
- shadow 初始化。
- 每个 archetype 的最低可用模型。
- 对照实验。

它不一定适合直接提交。population 的价值是让你看到整体生态，而不是追求单个卡组最强。

## 3. 什么时候用 shadow

shadow 是“环境模型”。它应该尽量模拟 Kaggle ladder 中真实会遇到的对手。

好的 shadow 至少要有：

- 对应 `archetype`
- 对应 `deck_sig`
- 对应 `team_name`
- score 或 score band
- deck CSV
- random 质量审计

如果 shadow 自己打 random 都很差，它在 RR 中会把别人的胜率虚高，导致环境判断失真。

## 4. 先选 signature，不要先训练

先查数据：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --top 40 \
  --out logs/stats_ogerpon_0801_0815.csv
```

选择 sig 时看四项：

- 决策样本数。
- 胜局决策数。
- 出现日期跨度。
- 该 sig 对应队伍是否在 Kaggle replay 里稳定。

不要只按样本数选。样本多但主要来自输局，可能会污染 specialist。

## 5. 手动训练 top-k specialist

top1：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig 697a82e582d5 \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --save checkpoints/ogerpon_top1_697_w3.npz \
  2>&1 | tee logs/ogerpon_top1_697_w3.log
```

top2/top3 只需要重复 `--deck-sig`：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig 697a82e582d5 \
  --deck-sig 2a5072194fdf \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --save checkpoints/ogerpon_top2_w3.npz \
  2>&1 | tee logs/ogerpon_top2_w3.log
```

top-k 的判断标准不是“sig 越多越好”。如果加入第三个 sig 后 RR 或 Kaggle replay 变差，就说明它带来了冲突路线。

## 6. 并行训练 population

```bash
python3 -u tools/train_bc_population.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Alakazam" \
  --archetype "Crustle Wall" \
  --archetype "Dragapult" \
  --archetype "Marnie Grimmsnarl" \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --epochs 8 \
  --batch-size 4096 \
  --width 3.0 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --checkpoint-dir checkpoints/pop_0801_0815 \
  --log-dir logs/pop_0801_0815 \
  --tag score900_w3 \
  --poll-seconds 30 \
  2>&1 | tee logs/pop_0801_0815/runner.log
```

如果某个卡组 failed，先看它自己的 log，而不是只看 runner 状态：

```bash
tail -200 logs/pop_0801_0815/bc2_dragapult_score900_w3.log
```

## 7. 构建 shadow pool

第一步生成 manifest：

```bash
python3 tools/build_shadow_pool.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --known-decks-dir logs/ladder_pool_latest/decks \
  --min-decisions 3000 \
  --min-episodes 3 \
  --top-per-archetype 12 \
  --checkpoint-dir checkpoints/shadow_0801_0815 \
  --epochs 6 \
  --batch-size 4096 \
  --width 3.0 \
  --cuda-memory-gb 24 \
  --out logs/shadow_pool_manifest_0801_0815.csv
```

第二步按 manifest 训练：

```bash
python3 -u tools/train_shadow_manifest.py \
  logs/shadow_pool_manifest_0801_0815.csv \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --batch-size 4096 \
  --log-dir logs/shadow_0801_0815 \
  --skip-existing \
  --poll-seconds 30 \
  2>&1 | tee logs/shadow_0801_0815/runner.log
```

## 8. shadow 必须做 random 审计

```bash
python3 tools/eval_manifest_random.py \
  --manifest logs/shadow_pool_manifest_0801_0815.csv \
  --games 300 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 50 \
  --skip-bad-entries \
  --out-csv logs/shadow_0801_0815/random_g300.csv
```

解释结果时要谨慎：

- shadow random 低，说明它不可靠。
- shadow random 高，也不代表它能模拟 Kaggle 高分策略。
- 同一 archetype 的 shadow 要覆盖多个强 sig，不能只留下最好打的。

## 9. 本章判断标准

训练完一批模型后，把它们分三类：

- submission candidate：random 高、RR 稳、符合当前 ladder 环境。
- environment shadow：random 及格，代表真实 deck/team，哪怕不适合提交也保留。
- bad / quarantine：random 差、deck 对不上、日志异常，不能进入 RR 主池。

下一章：[08 - game plan、history、trajectory](08_plan_history_trajectory.md)
