# 01 - 项目与环境

本章先回答三个问题：

1. 这个仓库里什么是主线，什么是实验线。
2. 哪些目录会不断长大，哪些目录只是输入。
3. 你每次开工应该先看什么。

## 1. 仓库职责

这个仓库不是“一个模型文件夹”，而是一个完整的训练和验证系统：

- `main.py`：Kaggle submission 入口。
- `ptcg_rl/`：模型、编码器、BC/RL 训练逻辑、NumPy 推理。
- `tools/`：抽取、训练、评测、打包、分析、trace 工具。
- `docs/`：教程和实验说明。
- `decks/`：项目内可用 deck CSV。
- `data/`：原始资料、引擎文件、抽取 corpus。
- `logs/`：训练、评测、分析日志。
- `checkpoints/`：模型和中间 checkpoint。

## 2. 每类目录的角色

### 输入

- Kaggle episodes：`episodes_raw/`
- leaderboard CSV：用于 score band 标注
- Kaggle files：规则、卡牌数据、引擎源码、sample submission

### 处理中间件

- `data/bc_corpus_*`
- `logs/ladder_pool_*`
- `logs/shadow_pool_*`
- `artifacts/`

### 最终产物

- `checkpoints/*.npz`
- `submission/*.tar.gz`

## 3. 最小环境变量

在仓库根目录执行：

```bash
export REPO=${REPO:-$(pwd)}
export EPISODES=${EPISODES:-$REPO/../episodes_raw}
export CG_DIR=${CG_DIR:-$REPO/../cg}
export LB_DIR=${LB_DIR:-logs/leaderboard_$(date +%Y%m%d)}
mkdir -p data logs checkpoints
```

检查引擎和 Python 环境：

```bash
test -f "$CG_DIR/libcg.so"
python3 -m py_compile main.py tools/bc2_train.py tools/package_submission.py train.py
nvidia-smi
```

## 4. 为什么要先理解目录

因为后面每章都会用到这些路径：

- 抽取章节会写到 `data/bc_corpus_*`
- BC 章节会读 `data/bc_corpus_*`
- 评测章节会读 `checkpoints/*.npz` 和 `decks/*.csv`
- submission 章节会把 `main.py`、`policy.npz`、`deck.csv`、`cg/` 打包

如果你在命令里写错路径，实验通常不是“效果不好”，而是“根本没在跑正确的文件”。

## 5. 一次实验前先做的三件事

1. `git status --short`
2. `nvidia-smi`
3. `ls -lh data logs checkpoints`

然后再看下一章。

下一章：[02 - PTCG 基础玩法与 Kaggle 文件](02_game_and_kaggle_files.md)
