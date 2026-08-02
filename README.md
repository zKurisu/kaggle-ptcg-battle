# PTCG RL Training Pipeline

Pokémon TCG AI Battle 强化学习训练管线。

## 环境搭建

```bash
# 依赖
pip install torch numpy

# 确保 cg/ 引擎在 workspace 根目录
ls /home/jie/Do/0_PTCG/workspace/cg/libcg.so  # 应该存在

# 克隆
cd /home/jie/Do/0_PTCG/workspace
git clone <repo> ptcg_rl_git
cd ptcg_rl_git
```

## 目录结构

```
ptcg_rl_git/
├── README.md              ← 本文件
├── docs/                  ← 文档
│   ├── 01_architecture.md
│   ├── 02_meta_analysis.md
│   ├── 03_roadmap.md
│   ├── 04_bc_extraction.md
│   ├── 05_rl_training.md
│   └── 06_bc_design.md
├── tools/                 ← 工具脚本
│   ├── bc_extract_v2.py   ← 提取 Kaggle episode → 训练数据
│   ├── bc_trainer.py      ← BC 模仿学习训练
│   ├── deck_battle.py     ← 本地卡组对战测试
│   ├── ladder_decks.py    ← 天梯卡组分类
│   └── convert_deck.py    ← PTCGL 文本 → deck.csv
├── agents/                ← 规则型 agent
│   └── marnie_grimmsnarl.py
├── ptcg_rl/               ← 核心库
│   ├── encoder.py         ← FastEncoder (0.21ms/decision)
│   ├── model.py           ← PolicyValueNet (501K params)
│   ├── trainer.py         ← PPO 训练器
│   └── numpy_policy.py    ← Kaggle 提交推理 (无 torch)
├── deck.csv               ← 主卡组 (Marnie Grimmsnarl)
├── decks/                 ← 对手卡组池 (30 套)
├── train.py               ← RL 训练入口
└── main.py                ← Kaggle 提交 agent
```

## 数据准备

### 1. 下载 Kaggle Episode 数据

```bash
# 在 workspace 根目录
mkdir episodes_raw && cd episodes_raw

# 下载最近几天（每天 ~710MB 压缩，解压后 ~21GB）
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-01
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-30
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-29
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-28
```

### 2. 提取决策数据（按卡组 + 分数段分类）

```bash
cd ptcg_rl_git

# 下载排行榜（用于分数段标记）
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p /tmp/lb
unzip -o /tmp/lb/pokemon-tcg-ai-battle.zip -d /tmp/lb/

# 提取（~10 分钟/天，500K-1.5M decisions）
LB_CSV=$(ls /tmp/lb/*.csv | head -1)
python3 -u tools/bc_extract_v2.py ../episodes_raw/ \
    --out data/bc_corpus_banded/ \
    --lb-csv "$LB_CSV"
```

输出结构：
```
data/bc_corpus_banded/
  Marnie_Grimmsnarl/
    1200+/            ← 顶尖高手决策
    1100-1199/
    1000-1099/
    ...
  Alakazam/
    1200+/
    ...
```

## 训练流程

### BC2 模仿学习（当前主线）

`bc2` 是新的干净训练入口：仍然输出可被 `main.py`/`numpy_policy.py`
直接加载的 `.npz`，但数据加载、过滤、mask、loss 和诊断已经拆成独立模块。
默认会提高第一步动作和大候选集合的权重，因为当前弱点主要集中在
`TO_HAND`、`ATTACK`、`ABILITY` 和 6+ 候选的排序。

2026-08-02 之后的 encoder 修复了 `Play`/`Skill` option 没有卡牌身份的问题，
并且模型新增了 `context/area/index` embedding。建议重新抽取到 v4：

```bash
LB_CSV=$(ls /tmp/lb/*.csv | head -1)
python3 -u tools/bc_extract_v2.py ../episodes_raw \
    --out data/bc_corpus_banded_v4 \
    --lb-csv "$LB_CSV" \
    --workers 4 \
    > logs/bc_extract_v4.log 2>&1
```

训练前先审计 raw option 字段，确认低分场景的 option 身份是否已被暴露：

```bash
python3 tools/audit_episode_options.py ../episodes_raw \
    --max-episodes 2000 \
    --option-types PLAY ABILITY ATTACK SKILL CARD
```

```bash
mkdir -p logs checkpoints
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
    --corpus data/bc_corpus_banded_v4 \
    --archetype "Marnie Grimmsnarl" \
    --score-bands "1200+" "1100-1199" "1000-1099" \
    --epochs 12 --batch-size 4096 --width 2.0 --device cuda:0 \
    --first-action-weight 1.5 --option-weight 0.15 \
    --checkpoint-every 1 \
    --save checkpoints/bc2_marnie_1000_w2.npz \
    > logs/bc2_marnie_1000_w2.log 2>&1
```

离线 first-action/分场景诊断：

```bash
python3 tools/bc2_accuracy.py checkpoints/bc2_marnie_1000_w2.npz \
    --corpus data/bc_corpus_banded_v4 \
    --archetype "Marnie Grimmsnarl" \
    --score-bands "1200+" "1100-1199" "1000-1099" \
    --max-samples 50000 --batch-size 4096 --progress-every 5000
```

对 random 实战：

```bash
python3 tools/eval_bc.py checkpoints/bc2_marnie_1000_w2.npz \
    --deck deck.csv --games 200
```

### 批量训练 BC Population

先用同一份 v4 corpus 训练多个有足够 replay 数据的 archetype，形成后续
round-robin 和 RL opponent pool。脚本启动前会统计每个卡组在所选
score bands 下的 corpus 文件数和 decisions 数；低于 `--min-decisions`
会直接跳过，避免子进程启动后才失败。

默认列表会避开当前 v4 corpus 中只有几百到几千 decisions 的卡组，包括：
Marnie、Alakazam、Crustle、Team Rocket Mewtwo、Teal Mask Ogerpon、
Mega Lopunny、Dragapult、Festival Lead、Cynthia Garchomp。

先 dry-run 检查命令和输出路径：

```bash
python3 tools/train_bc_population.py \
    --corpus data/bc_corpus_banded_v4 \
    --gpus 0,1,2,3 \
    --epochs 8 \
    --batch-size 4096 \
    --width 2.0 \
    --tag v4_1000_w2 \
    --min-decisions 20000 \
    --dry-run
```

正式启动 4 卡并行训练；每个 job 完成后会自动跑一次 `bc2_accuracy`：

```bash
python3 -u tools/train_bc_population.py \
    --corpus data/bc_corpus_banded_v4 \
    --gpus 0,1,2,3 \
    --epochs 8 \
    --batch-size 4096 \
    --width 2.0 \
    --tag v4_1000_w2 \
    --min-decisions 20000 \
    --accuracy-samples 50000 \
    --poll-seconds 30 \
    > logs/train_bc_population_v4_1000_w2.log 2>&1
```

只训练指定卡组时重复 `--archetype`：

```bash
python3 -u tools/train_bc_population.py \
    --archetype "Crustle Wall" \
    --archetype "Team Rocket Mewtwo" \
    --archetype "Teal Mask Ogerpon" \
    --corpus data/bc_corpus_banded_v4 \
    --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
    --gpus 0,1,2 \
    --tag v4_900_w2 \
    --min-decisions 20000
```

输出命名示例：

```text
checkpoints/bc2_marnie_grimmsnarl_v4_1000_w2.npz
logs/bc2_marnie_grimmsnarl_v4_1000_w2.log
logs/bc2_marnie_grimmsnarl_v4_1000_w2_accuracy.log
```

### 旧 BC 模仿学习

从高分 replay 学习人类决策：

```bash
cd ptcg_rl_git
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc_trainer.py \
    --archetype "Marnie Grimmsnarl" \
    --score-bands "1200+" "1100-1199" \
    --epochs 30 --batch-size 2048 --width 2.0 --device cuda:0 \
    --checkpoint-every 5 \
    --save checkpoints/bc_marnie_1100.npz
```

输出 `checkpoints/bc_marnie_1100.npz` 可直接用于 Kaggle 提交。

`bc_trainer.py` 默认过滤空 action，避免无梯度样本浪费 batch。需要学习
“不选择任何 option 直接 STOP”的场景时，显式加 `--include-empty`。

当前 BC 训练是单进程单卡；`CUDA_VISIBLE_DEVICES=1 --device cuda:0` 表示使用
可见设备列表中的第 0 张卡，也就是物理 GPU 1。4 卡 A800 不会自动并行，
需要另做 DDP/多进程训练。

### RL 训练（PPO + MCTS）

```bash
python3 -u train.py \
    --iterations 500 --games 32 --device cuda:0 \
    --mcts --mcts-sims 32
```

### 本地评估

```bash
# 小规模对战测试
python3 tools/deck_battle.py battle deck.csv decks/ --games 10

# 单 checkpoint 对 legal random
python3 tools/eval_bc.py checkpoints/bc2_marnie_1000_w2_v4_ep003.npz \
    --deck deck.csv \
    --games 500 \
    --progress-every 25

# population round-robin：行胜列的胜率矩阵
python3 tools/eval_round_robin.py \
    --include-random \
    --policy checkpoints/bc2_marnie_1000_w2_v4_ep003.npz \
    --policy checkpoints/bc2_marnie_1000_w2_v4.npz \
    --deck deck.csv \
    --games 100 \
    --progress-every 10 \
    --out-csv logs/round_robin_marnie.csv

# 不同卡组/不同 checkpoint 时，用 NAME=POLICY:DECK
python3 tools/eval_round_robin.py \
    --entry marnie=checkpoints/bc2_marnie_1000_w2_v4.npz:deck.csv \
    --entry lucario=checkpoints/bc2_lucario_1000_w2.npz:decks/lucario.csv \
    --entry random=random:deck.csv \
    --games 100 \
    --progress-every 10
```

`eval_round_robin.py` 会交替先后手，draw/error 计入总局数但不算任一方胜。
这个矩阵比只看 random 更接近 Kaggle ladder：random 100% 只说明 agent
基本可用，能否过 800-1000 分段要看它对其他 BC/规则型 population 的胜率。

### Slot-Aware BC v5

v5 模型默认启用 slot-aware board encoder：不再把 active 和 bench 混在一个
pool 里，而是分别编码我方 active、我方 bench、对方 active、对方 bench、手牌。
这不需要重建 corpus，直接复用 `data/bc_corpus_banded_v4`。

适合重点改善 `ATTACH`、`RETREAT`、`DISCARD` 和多候选选择。旧 checkpoint
仍可加载；需要回退旧结构做 ablation 时加 `--legacy-state-pool`。

```bash
python3 -u tools/train_bc_population.py \
    --corpus data/bc_corpus_banded_v4 \
    --gpus 0,1,2,3 \
    --epochs 8 \
    --batch-size 4096 \
    --width 2.0 \
    --tag v5_slot_1000_w2 \
    --min-decisions 20000 \
    --accuracy-samples 50000 \
    --poll-seconds 30 \
    > logs/train_bc_population_v5_slot_1000_w2.log 2>&1
```

### Expanded Features v6

v6 在重新抽取 corpus 时把标量特征从 `state=32/options=16` 扩到
`state=48/options=32`。新增特征只使用 observation 中稳定存在的信息：
active/target 的 HP、retreat cost、stage、ex/mega、tool 数、异常状态、
discard 数量，以及 option 指向 active/bench/self/opponent 的标志。

旧 v4/v5 checkpoint 仍可加载；推理和 accuracy 会按 checkpoint 维度自动截断
或补零。新特征需要重新抽取 corpus：

```bash
python3 -u tools/bc_extract_v2.py ../episodes_raw \
    --out data/bc_corpus_banded_v6wide \
    --workers 9 \
    --progress-every 500 \
    > logs/bc_extract_v6wide.log 2>&1
```

## Kaggle 提交

```bash
# 打包 tar.gz
python3 tools/package_submission.py \
    --policy checkpoints/bc2_marnie_1000_w2_v4_ep003.npz \
    --deck deck.csv \
    --out submission.tar.gz

# 提交
kaggle competitions submit pokemon-tcg-ai-battle \
    -f submission.tar.gz \
    -m "BC2 Marnie v4 epoch3"
```

### 记录 Kaggle 分数变化

simulation 分数会随匹配继续波动，早期冲高、回落速度、最终稳定区间都应记录。
提交后的前几小时建议 60 秒采样，稳定后再改成 5-15 分钟：

```bash
python3 -u tools/track_kaggle_scores.py \
    --watch \
    --interval 60 \
    --out logs/kaggle_submission_scores.csv \
    > logs/kaggle_score_watch.log 2>&1
```

只查看不写 CSV：

```bash
python3 tools/track_kaggle_scores.py --no-append
```

### 分析 Kaggle Replay

提交分数只给总分，不告诉输给谁。用 replay 分析脚本可以拉取某个
submission 的公开 episodes，自动缓存回放，并按对手 deck/team 聚合胜率：

```bash
python3 tools/analyze_kaggle_replays.py 55188444 \
    --deck decks/pool_341_crustle_mysterious_rock_inn.csv \
    --known-decks-dir decks \
    --group-by opponent_deck_name \
    --out logs/kaggle_replay_analysis_55188444.csv \
    --write-opponent-decks \
    --progress-every 5
```

如果没有本地 deck 可用于自动识别我方 agent，就显式传：

```bash
python3 tools/analyze_kaggle_replays.py 55188444 --team-name "Jie Orkarin"
```

## 常用命令

```bash
# 查看 GPU
nvidia-smi

# 查看训练进度
tail -f train.log

# 查看提取进度  
tail -f bc_extract_v2.log

# 查看卡组分布
python3 tools/ladder_decks.py stats data/bc_corpus_banded/
```
