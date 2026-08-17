# 04 - 数据抽取与 corpus 构建

这章的目标很明确：把 Kaggle episode zip 变成可以训练的 `.npz` corpus。

## 1. 抽取解决什么问题

原始 episode 不能直接喂给 BC/RL，因为它们只是完整对局记录，太大、太杂、而且一局里有很多非决策片段。抽取脚本负责把它压成“每个决策点一条样本”。

换句话说：

- 输入：整局 episode + leaderboard 分数 + 牌组信息
- 输出：一组结构化 decision 样本

## 2. 抽取脚本

主入口是：

```bash
python3 tools/bc_extract_v2.py --help
```

常用参数：

- `episodes_dir`：episode zip 所在目录
- `--out`：输出 corpus 根目录
- `--lb-csv`：leaderboard CSV
- `--workers`：并行处理 zip 的 worker 数
- `--progress-every`：进度输出
- `--action-history-k`：保存最近动作历史
- `--log-history-k`：保存最近公开日志
- `--board-history-k`：保存最近 board 快照
- `--max-episodes`：调试时限制每个 zip 读取多少局

## 3. 典型命令

```bash
export LB_CSV=$(find "$LB_DIR" -name '*.csv' | head -1)
python3 -u tools/bc_extract_v2.py "$EPISODES" \
  --out data/bc_corpus_banded_v11_0801_0815 \
  --lb-csv "$LB_CSV" \
  --workers 12 \
  --progress-every 1000 \
  --action-history-k 32 \
  --log-history-k 128 \
  --board-history-k 12
```

如果你只想先跑烟雾测试：

```bash
python3 -u tools/bc_extract_v2.py "$EPISODES" \
  --out data/bc_corpus_smoke \
  --lb-csv "$LB_CSV" \
  --workers 2 \
  --progress-every 50 \
  --max-episodes 2
```

## 4. corpus 里通常有什么

一个样本通常会保存：

- `board`
- `hand`
- `feats`
- `ot`, `oc`, `oc2`, `oa`, `of_arr`
- `action`
- `min_c`, `max_c`
- `deck_sig`
- `team_name`
- `score`
- `episode_id`
- `player_index`
- `reward`
- `won`
- `draw`
- `final_status`
- `game_steps`
- `opponent_*` 元数据
- `history_*` 或 `board_*` 派生字段

这些字段的关键是：

1. 让训练时的输入和推理时的输入一致。
2. 让我们后面能按 `deck_sig`、`team_name`、score band、对手类型做切分。

## 5. score band 为什么重要

同一个 archetype 在不同分段的打法差异很大。

常用分段：

- `1200+`
- `1100-1199`
- `1000-1099`
- `900-999`
- `800-899`
- `700-799`
- `600-699`

用分段的原因不是“分数越高越好”这么简单，而是：

- 高分段更接近稳定策略。
- 中分段能补充从 600+ 往上爬时常见的启动和失误。
- 低分段容易混进很多噪声，不适合直接当专家标签。

## 6. 什么时候要重新抽取

只要你改了下面任一项，就应该重新抽：

- `encoder.py` 的状态特征维度
- `history_features.py` 的历史字段
- `bc_extract_v2.py` 输出字段
- `option` 编码方式
- opponent metadata 的存法

如果你只改了：

- loss 权重
- epoch
- batch size
- LR

那通常不需要重新抽。

## 7. 看一个抽出的文件

```bash
python3 - <<'PY'
import glob, numpy as np
paths = glob.glob("data/bc_corpus_banded_v11_0801_0815/*/*/*.npz")
print("files:", len(paths))
z = np.load(paths[0], allow_pickle=True)
print("keys:", z.files)
print("state feat:", np.asarray(z["feats"][0]).shape)
print("option feat:", np.asarray(z["of_arr"][0]).shape)
PY
```

如果 shapes 和你预期不一致，先不要训练，先回头看抽取逻辑。

## 8. 本章练习

先做三步：

1. 下载一天 episode。
2. 抽一个很小的 corpus。
3. 打开一个 `.npz` 看 key 和 shape。

下一章：[05 - BC 到底在学什么](05_bc_concepts.md)
