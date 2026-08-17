# 02 - PTCG 基础玩法与 Kaggle 文件

这一章先把“游戏是什么”和“我们能从 Kaggle 下载到什么”讲清楚。

## 1. 一局 PTCG 的基本对象

你只需要先记住这些词：

- `deck`：60 张牌。
- `active`：当前出战位。
- `bench`：备战区。
- `hand`：手牌。
- `discard`：弃牌区。
- `prize`：奖赏卡。
- `energy`：能量。
- `trainer`：Item / Supporter / Stadium / Tool。
- `evolve`：进化。
- `attack`：攻击并结束回合。

对局的核心不是“每一步都选对单卡”，而是：

1. 先把 engine 立起来。
2. 再把主攻手准备好。
3. 再考虑 prize race、资源保留和对手节奏。

## 2. Kaggle 比赛能提供什么

当前比赛静态文件主要分三类：

- 卡牌数据：`EN Card Data.csv`、`JP Card Data.csv`
- 规则/玩法说明：`Card_ID List_EN_.pdf`、`Card_ID List_JP_.pdf`
- 引擎和 sample submission：`ptcg_engine/ptcgProgram 22/*`、`sample_submission/sample_submission/*`

可以先列文件：

```bash
mkdir -p data/kaggle_files
kaggle competitions files pokemon-tcg-ai-battle --page-size 200 -v \
  | tee data/kaggle_files/files.csv
```

当前没有单独的 rulebook PDF 文件。官方规则入口通常看 Kaggle competition rules，以及官方 Play! Pokémon resources。

## 3. 先下载再看

下载静态文件：

```bash
kaggle competitions download pokemon-tcg-ai-battle -p data/kaggle_files
unzip -o data/kaggle_files/pokemon-tcg-ai-battle.zip -d data/kaggle_files
```

下载 episode：

```bash
mkdir -p "$EPISODES"
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-16 -p "$EPISODES"
```

建议把 leaderboard 也下载一份，后面抽 corpus 时要用：

```bash
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p "$LB_DIR"
unzip -o "$LB_DIR/pokemon-tcg-ai-battle.zip" -d "$LB_DIR"
```

## 4. 为什么要看这些文件

因为 Kaggle 的训练不是凭空写模型，而是把真实 observation、合法动作和卡牌文字变成数据。

- `Card Data.csv` 告诉你卡名和卡牌效果。
- `ptcg_engine` 告诉你引擎如何合法执行。
- episode 数据告诉你别的 agent 真正怎么打。
- leaderboard 告诉你天梯环境在什么分数段更常见什么 deck。

## 5. 最小练习

先做两个检查：

```bash
test -f data/kaggle_files/pokemon-tcg-ai-battle.zip
test -f "$EPISODES"/pokemon-tcg-ai-battle-episodes-2026-08-16.zip
```

再看看你自己的 deck：

```bash
wc -l deck.csv
head deck.csv
```

`deck.csv` 必须正好 60 行。

下一章：[03 - cg 引擎、observation 与合法动作](03_cg_engine_observation.md)
