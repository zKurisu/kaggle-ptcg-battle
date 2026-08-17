# 05 - BC 到底在学什么

BC 是 Behavior Cloning，也就是行为克隆。它的目标不是“推导最优策略”，而是在给定局面和合法动作时，模仿训练数据中高分选手或高分 agent 的选择。

在这个项目里，BC 是最稳定的基座。后面的 specialist、shadow、规则层、teacher rollout 和 RL，基本都要先和一个稳定 BC 比较。

## 1. BC 与普通分类的区别

普通分类任务通常是：

```text
输入样本 -> 预测一个类别
```

PTCG 决策更像：

```text
observation + legal options + 选择上下限 + 历史信息 -> 返回一个合法 option index 列表
```

因此 BC 不能只学“下一步选哪个动作”。它至少要处理四类情况：

- 单选：例如是否攻击、选择某个 MAIN 动作。
- 多选：例如搜索牌库加入多张牌、丢弃多张牌。
- 顺序多选：例如某些效果会按选择顺序结算。
- 可选 STOP：`minCount` 到 `maxCount` 允许选 0 到多张时，要学会什么时候停。

## 2. 训练标签来自哪里

第 04 章抽出的 corpus 里，最关键的标签是：

- `action`：真实 episode 中选择的 option index 列表。
- `min_c` / `max_c`：这次选择的数量约束。
- `won` / `draw` / `reward`：这局最后结果。
- `deck_sig` / `team_name` / `score`：这条样本来自哪类 deck、哪个队伍、什么分段。
- `opponent_*`：对手的 archetype、signature、team、score band。
- `history_*`：可选的历史动作、公开日志、board snapshot。

这些字段决定了 BC 能不能学到稳定策略。字段越乱，模型越容易学成平均行为。

## 3. sequence loss 为什么重要

多选标签天然是序列：

```text
选择 option A -> 选择 option B -> 选择 STOP
```

如果只监督第一个动作，模型可能会学会“开头看起来对”，但后续选择错掉。当前训练逻辑会围绕 sequence 计算损失，并辅以：

- `first_action`：第一步是否选对。
- `set loss`：多选集合是否接近。
- `option loss`：每个 option 的局部质量。
- `trajectory` / `step_plan`：更长期的计划辅助标签。
- `value`：把结果信号接入模型，用作辅助判断。

这些项不是为了让日志更复杂，而是为了防止模型在复杂局面里退化成单步分类器。

## 4. 多选在 PTCG 里为什么常见

下面这些都是高价值多选或准多选局面：

- 牌库搜索：选哪张 Basic、哪张进化、哪张 supporter。
- 弃牌：为了代价丢什么，不该丢什么。
- attach/evolve target：同一张资源给谁。
- damage counter 分配：例如 Dragapult 的 bench damage。
- switch/active 选择：谁上前，谁留在 bench。
- prize 或 revealed card 相关选择。

对 Dragapult、Alakazam、Marnie 这类卡组，多选质量经常比单纯“是否攻击”更关键。

## 5. deck_sig 为什么经常比 archetype 更重要

`archetype` 只是大类，例如 `Teal Mask Ogerpon`。`deck_sig` 是具体 60 张牌的签名。

同一 archetype 下不同 `deck_sig` 可能有不同路线：

- 是否有非 ex secondary attacker。
- 是否带某张 tech card。
- engine 数量不同。
- supporter / stadium / tool 配比不同。
- 打某个 matchup 时是否有破局牌。

把差异很大的 sig 混在一起，模型会学到“平均路线”。平均路线在训练 loss 上可能不错，但在 Kaggle ladder 上常常变差。

因此提交候选通常先考虑：

1. 高分、样本足、胜率稳定的单 sig。
2. 高质量 top2/top3 sig。
3. 再考虑 mixed population。

## 6. BC 能学会什么，不能学会什么

BC 擅长学：

- 常规启动顺序。
- 常见搜索/贴能/进化选择。
- 同一 deck-sig 的稳定习惯。
- 高分样本里反复出现的 matchup 处理。

BC 不擅长学：

- 数据里几乎没有出现过的破局路线。
- 大量胜场来自对手事故或运气的“伪成功轨迹”。
- 需要探索才能发现的反直觉策略。
- 训练/推理特征不一致时的 history 依赖。

这也是为什么后面要引入 `success data`、`rule overlay`、`teacher rollout` 和 RL。

## 7. 当前代码里 BC 的调用位置

先看四个文件：

```bash
sed -n '1,220p' ptcg_rl/encoder.py
sed -n '1,220p' ptcg_rl/model.py
sed -n '1,220p' ptcg_rl/bc2/data.py
sed -n '1,220p' ptcg_rl/bc2/losses.py
```

最短链路是：

```text
corpus batch
  -> BCCorpus.collate()
  -> PolicyValueNet
  -> sequence_loss_parts()
  -> export_numpy()
  -> NumpyPolicy.select()
```

训练和提交都必须使用同一套特征语义。只改训练端不改 `numpy_policy.py`，通常就是假改进。

## 8. 本章练习

先打开一个 corpus 文件：

```bash
python3 - <<'PY'
import glob, numpy as np
paths = glob.glob("data/bc_corpus_*/*/*/*.npz")
print("files:", len(paths))
if paths:
    z = np.load(paths[0], allow_pickle=True)
    print(paths[0])
    print(z.files)
    print("action example:", z["action"][0])
    print("min/max:", z["min_c"][0], z["max_c"][0])
PY
```

再看训练入口：

```bash
python3 tools/bc2_train.py --help
```

下一章：[06 - BC 训练实操](06_bc_training.md)
