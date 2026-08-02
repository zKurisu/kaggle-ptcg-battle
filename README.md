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
```

## Kaggle 提交

```bash
# 打包
mkdir -p /tmp/submit/ptcg_rl
cp main.py deck.csv checkpoints/bc_marnie_1100.npz /tmp/submit/
cp policy.npz /tmp/submit/ 2>/dev/null  # 或 bc 产出的权重
cp ptcg_rl/*.py /tmp/submit/ptcg_rl/
cp -r /home/jie/Do/0_PTCG/workspace/cg /tmp/submit/
cd /tmp/submit
tar czf submission.tar.gz *

# 提交
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "BC Marnie 1100+"
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
