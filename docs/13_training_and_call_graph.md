# Training And Call Graph

这篇文档只做一件事：把“episode -> corpus -> model -> policy -> eval -> submission”的调用关系说清楚。

## 1. 总体数据流

```text
Kaggle daily episode zip
  -> tools/bc_extract_v2.py
  -> data/.../*.npz corpus
  -> tools/bc2_train.py / tools/train_bc_population.py
  -> ptcg_rl/model.py
  -> export_numpy() -> policy.npz
  -> ptcg_rl/numpy_policy.py
  -> main.py
  -> Kaggle submission tarball
```

离线评测和 RR 走的是同一条引擎链，只是对手和目标不同：

```text
policy.npz + deck.csv
  -> tools/eval_bc.py / tools/eval_round_robin.py
  -> ptcg_rl/numpy_policy.py
  -> cg.game
  -> cg.api / cg.sim
  -> ptcg_engine C++ library
```

RL/self-play 走的是：

```text
ptcg_rl/trainer.PPOTrainer
  -> ptcg_rl.model.PolicyValueNet
  -> cg.game
  -> self-play games
  -> PPO update on GPU
```

## 2. BC 训练是怎么串起来的

### 2.1 抽取

入口：

```bash
python3 tools/bc_extract_v2.py ...
```

它会读取 Kaggle episode zip 和 leaderboard CSV，把每一个 decision point 抽成 `.npz` 样本。样本里通常包含：

- `board`
- `hand`
- `feats`
- `option` related arrays
- `action`
- `reward`
- `won`
- `score`
- `team_name`
- `deck_sig`
- opponent metadata
- history / log / board history（如果启用）

抽取阶段的责任是“把引擎 observation 变成固定特征”，不是做模型判断。

### 2.2 训练

主要入口：

```bash
python3 tools/bc2_train.py ...
```

主链路：

1. `tools/bc2_train.py` 解析训练参数。
2. `ptcg_rl.bc2.data.BCCorpus` 打开 corpus `.npz`。
3. `ptcg_rl.bc2.losses.sequence_loss_parts()` 计算 sequence NLL / set loss / plan loss。
4. `ptcg_rl.model.PolicyValueNet` 做前向。
5. 训练结束后 `export_numpy()` 导出 `policy.npz`。
6. `ptcg_rl.numpy_policy.NumpyPolicy` 负责把这个 checkpoint 变成离线推理器。

### 2.3 评测

BC 训练后先做两类离线评测：

```bash
python3 tools/eval_bc.py ...
python3 tools/eval_round_robin.py ...
```

`eval_bc.py` 是“对 random 或指定 deck 打随机基线”；`eval_round_robin.py` 是“对 policy pool 互打”。

先 random 再 RR 的原因很简单：

- random 不稳，说明基础合法动作和局部策略都没学会。
- random 稳但 RR 差，说明模型可能只学会了基础执行，没有学会 matchup / resource / plan。
- RR 高但 Kaggle 低，通常是对手池不对、shadow 质量不够、或真实 ladder 环境偏移。

## 3. RL / PPO 是怎么串起来的

### 3.1 入口

直接训练入口通常是：

```bash
python3 tools/rl_train_league.py ...
```

或者 legacy wrapper：

```bash
python3 train.py ...
```

### 3.2 训练循环

`ptcg_rl.trainer.PPOTrainer` 负责：

1. 生成 self-play games。
2. 收集每一步 `EncodedDecision`。
3. 计算 GAE。
4. 用 `PolicyValueNet.evaluate_actions()` 做 PPO update。
5. 周期性保存 checkpoint。

这里的关键不是“打得更快”，而是：

- `cg.game` 必须一直保持和训练时一致。
- `FastEncoder` 和 `PolicyValueNet` 的特征维度必须和导出/推理一致。
- 任何 history、plan、rule overlay 的改动都必须同时影响训练和推理，否则会出现训练/提交不一致。

## 4. 代码调用关系

### 4.1 submission 入口

```text
Kaggle -> main.py -> ptcg_rl.numpy_policy.NumpyPolicy -> cg.game -> cg C++ engine
```

`main.py` 的责任只应该是：

- 读 `deck.csv` 和 `policy.npz`
- 恢复规则 overlay（如果启用）
- 在没有模型动作时给安全 fallback
- 返回 legal option index list

### 4.2 BC 推理

```text
obs_dict
  -> FastEncoder.encode()
  -> numpy arrays
  -> NumpyPolicy.select()
  -> action indices
```

### 4.3 训练期前向

```text
corpus batch
  -> PolicyValueNet.encode_state()
  -> PolicyValueNet.encode_options()
  -> PolicyValueNet.option_logits()
  -> sequence_loss_parts()
```

### 4.4 多选 / 顺序 / plan

如果一个 decision 需要多选，训练时必须保留：

- 选了几个
- 选了哪些 index
- 顺序是否敏感
- 该决策是否对应 plan / trajectory label

否则模型会把复杂局面压扁成单步分类，最后在 `Dragapult`、`Dusknoir`、`search`、`attach`、`evolve` 这种局面上失真。

## 5. 文件结构怎么理解

### 5.1 训练相关

- `tools/bc_extract_v2.py`：抽取。
- `tools/bc2_train.py`：BC 训练。
- `tools/train_bc_population.py`：population / specialist 训练。
- `tools/train_shadow_manifest.py`：shadow 训练。
- `tools/eval_manifest_random.py`：manifest 的随机门槛。

### 5.2 推理相关

- `ptcg_rl/encoder.py`：状态编码。
- `ptcg_rl/model.py`：PyTorch 网络。
- `ptcg_rl/numpy_policy.py`：NumPy 推理和 MCTS。
- `main.py`：Kaggle entrypoint。

### 5.3 评测相关

- `tools/eval_bc.py`
- `tools/eval_round_robin.py`
- `tools/eval_baseline_delta.py`
- `tools/analyze_kaggle_replays.py`
- `tools/trace_matchup_decisions.py`

### 5.4 RL / search 相关

- `ptcg_rl/trainer.py`
- `ptcg_rl/mcts.py`
- `tools/rl_train_league.py`
- `tools/rl_finetune_vs_pool.py`
- `train.py`：旧的 RL/PPO 包装器，逻辑上只是转发到 `rl_finetune_vs_pool.main`。
- `tools/search_action_teacher.py`

## 6. 推荐改代码的顺序

1. 先改 `bc2_extract_v2.py` / corpus schema。
2. 再改 `encoder.py` / `history_features.py`。
3. 再改 `model.py`。
4. 再改 `numpy_policy.py`。
5. 最后改 `main.py` 和 submission packaging。

原因：

- 抽取一旦变了，训练、推理、评测都要跟着变。
- 编码一旦变了，模型和导出都必须同时更新。
- 只改 `main.py` 而不改训练和 corpus，通常只能得到假改进。

## 7. 什么时候该删文件，什么时候不该删

适合删或合并的：

- 明显重复的历史 wrapper。
- 已被 README/教程索引完全替代的零散说明文。
- 只服务一次性实验且没有被 handoff 引用的临时脚本。

不适合轻易删的：

- `main.py`
- `ptcg_rl/numpy_policy.py`
- `ptcg_rl/encoder.py`
- `ptcg_rl/model.py`
- `ptcg_rl/trainer.py`
- 当前使用中的 `tools/*`

如果要做真正的目录重整，先改文档，再改入口，再慢慢移脚本，不要先大面积删文件。
