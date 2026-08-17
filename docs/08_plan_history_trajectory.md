# 08 - game plan、history、trajectory

前面的 BC 主要关注“当前 observation 下选什么”。这一章关注更长的东西：一套卡组在一整局里想达成什么路线。

## 1. 什么是 game plan

game plan 是一局牌的主线，例如：

- setup：铺 Basic、找 engine、保证不会断线。
- tempo attack：尽快开始有效攻击。
- wall / stall：让对方难以造成有效伤害。
- prize race：围绕奖赏卡交换节奏。
- disrupt：破坏对手关键资源或手牌。
- finish：寻找最后 KO 路线。

BC 如果只看单步动作，会在这些路线之间摇摆。game plan 的目标是让模型知道“当前动作服务于哪条路线”。

## 2. 本项目已有 plan 入口

先看已有卡组计划：

```bash
python3 tools/deck_plan_report.py --list
```

查看某个 deck：

```bash
python3 tools/deck_plan_report.py \
  --deck decks/pool_329_marnie_s_grimmsnarl.csv \
  --archetype "Marnie Grimmsnarl"
```

机器可读计划主要在：

```text
ptcg_rl/deck_plans.py
```

它不是强规则 agent，而是给训练、trace 和诊断提供“哪些卡是主线资源”的知识。

## 3. history 有哪些来源

Kaggle observation 能提供的历史主要来自三类：

- 自己过去动作：最近选过哪些 action type、card、target。
- 对手过去动作：对手公开打出、attach、evolve、attack 的内容。
- public logs：抽牌、移动、reveal、damage、KO 等公开事件。

抽取时可以保存：

```bash
python3 -u tools/bc_extract_v2.py "$EPISODES" \
  --out data/bc_corpus_banded_v11_history \
  --lb-csv "$LB_CSV" \
  --workers 12 \
  --progress-every 1000 \
  --action-history-k 32 \
  --log-history-k 128 \
  --board-history-k 12
```

训练时也必须打开对应参数：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_history \
  --archetype "Dragapult" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig cc2e995b0000 \
  --history-k 32 \
  --log-history-k 128 \
  --board-history-k 12 \
  --history-summary \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --save checkpoints/dragapult_history_w3.npz
```

如果抽取端没有这些字段，训练端打开参数也没有意义。

## 4. reveal 信息怎么用

有些卡牌效果会 reveal 卡面。这等价于对手公开了一部分手牌或牌库信息。

当前模型只有在 history/log 特征把 reveal 事件编码进去时，才能长期利用这些信息。否则模型只在 reveal 当下看见，下一步就忘了。

处理 reveal 的原则：

1. 先确认 `logs` 里事件是否公开。
2. 再确认抽取脚本是否保存。
3. 再确认 `history_features.py` 是否编码。
4. 最后确认 `NumpyPolicy` 推理端是否维护同样的历史。

训练端有，推理端没有，就是最常见的 history 假改进。

## 5. trajectory label 的作用

trajectory 不是“把整局塞进模型”。它更像给每个决策点打一个阶段标签，让模型知道这步属于哪类路线。

生成轨迹目标：

```bash
python3 tools/build_trajectory_targets.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Dragapult" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig cc2e995b0000 \
  --opponent-archetype "Alakazam" \
  --progress-every 1000 \
  --out-csv logs/trajectory_dragapult_vs_alakazam.csv
```

训练时再启用：

```bash
python3 tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Dragapult" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig cc2e995b0000 \
  --hierarchical-plan \
  --step-plan \
  --trajectory-csv logs/trajectory_dragapult_vs_alakazam.csv \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --save checkpoints/dragapult_traj_plan_w3.npz
```

如果脚本参数有变化，先以当前帮助为准：

```bash
python3 tools/bc2_train.py --help | rg "trajectory|plan|history"
```

## 6. 如何判断 history/trajectory 是否真的生效

不要等 RR 结束才判断。训练日志里应能看到：

- history 相关输入维度被识别。
- trajectory / step-plan loss 有非零样本。
- 多选指标不被单选样本完全淹没。
- history-sensitive context 的 loss 或 accuracy 有单独统计。
- 推理导出的 `.npz` 包含对应元数据。

如果这些信号没有出现，说明模型可能根本没用上你以为加入的 history。

## 7. 典型卡组计划例子

- Dragapult：先保证 Dreepy/Drakloak/Dragapult ex 路线，再处理 bench damage 分配和 Duskull/Dusclops/Dusknoir 配合。
- Alakazam：需要 Stage-2 进化线、正确使用 draw/search，并利用 bench attack 或控制节奏。
- Marnie Grimmsnarl：Marnie line、能量加速、第二攻击手和 disruption 都影响连续路线。
- Crustle Wall：尽快让 wall 成立，同时保证有能量和攻击窗口。
- Ogerpon Box：具体 sig 差异很大，是否存在非 ex/tech attacker 会决定对 Crustle 的可行路线。

## 8. 本章练习

选择一个你关心的 deck，做三步：

1. 用 `deck_plan_report.py` 看已有标签。
2. 对一个弱 matchup 跑 `build_trajectory_targets.py`。
3. 训练一版带 history/plan 的模型，并检查训练日志里这些信号是否出现。

下一章：[09 - matchup、random、RR、baseline-delta](09_evaluation_matchups.md)
