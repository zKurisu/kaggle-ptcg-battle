# Tutorial Index

这是一份按教程顺序阅读的入口页。先从比赛和引擎的最小闭环开始，再看数据、BC、RL、评测和提交。

## 推荐阅读顺序

1. [PTCG Gameplay And CG Engine Guide](docs/12_ptcg_gameplay_and_cg_engine.md)
   - 先知道一局牌怎么跑起来。
   - 认识 Kaggle files、`cg` wrapper、`ptcg_engine` C++ 源码。
2. [BC Extraction Guide](docs/04_bc_extraction.md)
   - 了解 episode 如何被抽成 `.npz` corpus。
   - 搞清楚 state、option、action、history、score、deck sig 如何存。
3. [BC Design Guide](docs/06_bc_design.md)
   - 了解模型输入、输出、训练目标、权重和评测方式。
4. [Specialist BC Recipes](docs/07_specialist_bc_recipes.md)
   - 了解为什么有些卡组适合 top1/top2/top3 或 deck-sig specialist。
5. [Deck Game Plans](docs/08_deck_game_plans.md)
   - 了解卡组 game plan、资源规划和连续决策想表达什么。
6. [Matchup Relations](docs/09_matchup_relations.md)
   - 了解 weak matchup、strong matchup、counter pool 的来源。
7. [Rule Success Data](docs/10_rule_success_data.md)
   - 了解规则/成功轨迹数据如何进入训练或评测。
8. [Human Matchup Strategy](docs/11_human_matchup_strategy.md)
   - 了解如何把真实 PTCG 讨论、打法文章和比赛经验转成规则。
9. [RL Training](docs/05_rl_training.md)
   - 了解 BC 之后如何接 RL、PPO、搜索或 self-play。

## 代码阅读顺序

1. `main.py`
   - Kaggle submission 入口，最终要返回 legal option index list。
2. `ptcg_rl/numpy_policy.py`
   - `.npz` checkpoint 的推理实现，也是 submission 侧最重要的 runtime。
3. `ptcg_rl/encoder.py`
   - 把 observation 转成 state/option 特征。
4. `ptcg_rl/model.py`
   - PyTorch policy/value 网络和 action scorer。
5. `ptcg_rl/bc2/data.py`
   - episode corpus 的加载、过滤和 batch collation。
6. `ptcg_rl/bc2/losses.py`
   - BC 的 sequence loss、set loss、plan loss。
7. `ptcg_rl/trainer.py`
   - PPO/self-play 的训练骨架。
8. `tools/bc_extract_v2.py`
   - episode -> corpus 的抽取入口。
9. `tools/bc2_train.py`
   - BC 训练主入口。
10. `tools/eval_bc.py` / `tools/eval_round_robin.py`
    - 离线评测和 RR。
11. `tools/package_submission.py`
    - 打包成 Kaggle submission tarball。
12. `train.py`
    - 旧的 RL/PPO 包装器，当前更推荐直接看 `tools/rl_train_league.py`。

## 最小闭环

如果只想先跑通最小闭环，按下面顺序：

```bash
python3 tools/bc_extract_v2.py ...
python3 tools/bc2_train.py ...
python3 tools/eval_bc.py ...
python3 tools/package_submission.py ...
```

如果只想验证引擎和 Python wrapper：

```bash
python3 - <<'PY'
from cg.game import battle_start, battle_select, battle_finish
PY
```

## 读文档时的判断标准

- 先看“数据如何产生”，再看“模型如何消费这些数据”。
- 先看“legal option 和 observation 的真实结构”，再看“我们想让模型学什么”。
- 先看离线 trace 和随机测试，再看 Kaggle 分数。
- 先看调用链，再看具体实现细节。
