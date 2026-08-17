# PTCG RL / BC Training Handbook
这个仓库用于 kaggle Pokemon TCG AI Battle 的比赛记录, 包含数据抽取、BC 训练、离线评测、Kaggle replay 分析、submission 打包等, [AGENT_HANDOFF.md](AGENT_HANDOFF.md) 中包含长期实验细节、长任务和临时判断的记录.

README 默认你已经在一台有 GPU、Kaggle CLI 和 Python 环境的训练机器中工作。所有命令都从仓库根目录执行，不依赖特定用户名、机器名或外部目录布局。

如果你是第一次看这个仓库，先在仓库根目录完成一次最小初始化，再按教程顺序阅读。后续文档中的 `cg` 引擎验证、submission 打包和本地对局都依赖这个 submodule。

```bash
export REPO=${REPO:-$(pwd)}
export EPISODES=${EPISODES:-$REPO/raw_episode}
export CG_DIR=${CG_DIR:-$REPO/external/kaggle-environments/kaggle_environments/envs/cabt/cg}
mkdir -p raw_episode data logs checkpoints
git submodule update --init --recursive external/kaggle-environments
test -f "$CG_DIR/libcg.so"
```

如果 GitHub 下载很慢，可以临时给 `git submodule` 加代理，例如 `http://127.0.0.1:20171`。

初始化后建议先读这些入口：

1. [docs/00_tutorial_index.md](docs/00_tutorial_index.md) - 教程总目录，按顺序带你读完整个项目。
2. [docs/01_project_setup.md](docs/01_project_setup.md) - 项目目录、环境变量和最小健康检查。
3. [docs/12_cg_engine_source.md](docs/12_cg_engine_source.md) - PTCG 基础玩法、Kaggle files、`cg` 引擎和 Python wrapper。
4. [docs/13_call_graph_submission.md](docs/13_call_graph_submission.md) - 数据、BC、RL、评测和 submission 的调用链。
5. [docs/decks/00_deck_index.md](docs/decks/00_deck_index.md) - 本地 deck 模板、Kaggle ladder 强签名、卡组打法、视频和卡面素材来源。

## Kaggle 比赛信息

官方入口:

- Kaggle Simulation competition: <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle>
- PTCG AI Battle Challenge 官网: <https://ptcg-abc.pokemon.co.jp/>
- simulator / cabt API 文档: <https://matsuoinstitute.github.io/cabt/>

如果需要先了解 PTCG 基本玩法、Kaggle competition files、`cg` Python wrapper 和 C++ simulator 源码阅读路线，见 [docs/12_cg_engine_source.md](docs/12_cg_engine_source.md)。

这个项目对应的是 PTCG AI Battle Challenge 的 Simulation track。参赛者提交一个能玩 Pokemon Trading Card Game 的 AI agent，Kaggle 会在自动天梯中持续安排 agent 对战，并用 skill rating 更新 leaderboard。另有 Strategy / Hackathon track，需要提交策略报告；Simulation track 的提交成绩会影响该方向的评估，但本仓库主要记录 Simulation ladder 的训练和提交。

关键规则和约束:

- Simulation track 的 Kaggle submission 是 `.tar.gz`，顶层需要有 `main.py` 和 `deck.csv`；本项目的 `tools/package_submission.py` 会把 policy、deck 和 `cg/` 引擎打成符合要求的包。
- `main.py` 暴露 agent 函数。每个决策点收到 observation，其中包含 public game logs、当前 board state 和 legal options；agent 返回要选择的 legal option index 列表。引擎只会给合法动作，但 agent 仍必须有 fallback，不能崩溃。
- 每个新 submission 会先跑 self-validation，对自己打验证局。验证失败会标记为 Error，可以下载 logs 排查；验证通过后从初始 rating 加入天梯。
- 每天最多提交 5 次；Kaggle 只跟踪最近 2 个 active submissions 继续参与最终评估。leaderboard 显示最好分数，但 Submissions 页面可以看每个提交的分数变化。
- rating 是带不确定性的 skill estimate。赢会提高 rating、输会降低 rating、平局会让双方 rating 靠近；单局赢多少分不直接影响 rating 更新。
- 提交资源限制很紧：2 vCPU、约 12.2 GiB RAM、约 11.8 GiB disk、submission 包大小上限约 197.7 MiB。因此提交侧推理必须轻量，不能依赖训练时的大规模 PyTorch pipeline。
- Kaggle Simulation track 官方时间线是 2026-06-16 开始，2026-08-16 UTC final submission deadline；PTCGABC 官网以日本时间显示为 2026-08-17 08:59 JST。之后约两周继续跑最终评估直到 leaderboard 收敛。

这些规则直接影响本项目的方法选择：训练阶段可以使用更大的模型和大量离线 RR，但 submission 侧只能保留轻量推理；random/RR 只是代理指标，最终仍要通过 Kaggle replay 和 active ladder 反馈验证。

## PTCG 背景资料和外部参考

Kaggle episode 只能告诉我们某个 agent 在当前天梯中做了什么，不一定能解释为什么这样打、如何打弱势 matchup。需要把真实 PTCG 资料当作 strategy seed，再通过离线 trace/RR 验证。

### 资料源

| 来源 | 适合拿什么 | 使用方式 | 注意事项 |
| --- | --- | --- | --- |
| [Limitless TCG](https://limitlesstcg.com/) | 大赛结果、top decklists、meta share、卡组骨架 | 找真实高分构筑、确认 archetype 主流 engine、导出/改写 decklist | 真实 Standard 环境不等于 Kaggle ladder，先转成 `deck.csv` 后用引擎测合法性 |
| [Limitless Labs](https://labs.limitlesstcg.com/) | 更细的 tournament standings、metagame analysis、matchup 信息 | 用来判断“现实中某弱势局是否仍有 30% 左右胜率”、找可参考胜局 | 官方页面也提示其统计是站外计算，可能有误差，不能直接当训练标签 |
| [Play Limitless](https://play.limitlesstcg.com/) | 线上赛 decklists、metagame、pairings、win rate | 找更接近线上玩家的构筑和 side tech | 线上赛样本质量参差，需要筛 top cut / 高胜率玩家 |
| [Play! Pokémon Resources](https://play.pokemon.com/en-us/resources/documents/?filter=all) / [宝可梦中文玩法规则](https://www.pokemon.cn/tcg-rules-howtoplay) | 官方规则书、玩法规则、赛场规则、errata | 补 PTCG 基础规则和真实赛制背景 | Kaggle simulator 可能与真实规则有差异，最终以 `cg` legal options 为准 |
| [Limitless Docs](https://docs.limitlesstcg.com/player/decklists.html) | decklist 文本格式、PTCGL/Limitless 导出说明 | 用来规范 PTCG Live/Limitless decklist 到本项目 `deck.csv` 的转换 | PTCGL ALT 卡号可能需要用普通版本替换或删掉编号再匹配 |
| [Trainer Hill](https://trainerhill.com/) / [Trainer Hill GitHub](https://github.com/Trainer-Hill) | meta overview、matchup data、decklist analysis、skeleton list 思路 | 查某 archetype 的核心卡、tech cards、对局热图 | 部分功能可能需要账号/付费；只作为外部先验 |
| [PokeDeck Architect](https://pokedeckarchitect.com/meta) | 基于 Limitless 的 meta、win rate、top-8 转化、tech trend | 快速确认近期 meta 和可疑 hard counter | 第三方聚合，需回到原 decklist 或离线测试验证 |
| [Pokemon.com TCG Live](https://www.pokemon.com/us/pokemon-video-games/pokemon-trading-card-game-live) | 官方线上客户端、Standard/Casual/Ranked/Test Deck | 人工测试卡组启动顺序、关键资源管理、常见误操作 | PTCG Live 与 Kaggle simulator 的接口和可见信息不同，不能直接导出训练数据 |
| 官方/社区策略文章、YouTube、Reddit `r/pkmntcg` / `r/PTCGL`、Limitless Discord | matchup guide、pilot notes、tech 选择、对局思路 | 把自然语言打法拆成可验证规则: setup priority、attach priority、evolve timing、target priority、resource preserve | 讨论质量差异很大，不能只凭单帖改模型；必须用 trace 和 RR 验证 |

### 如何把资料变成本项目可用信息

1. 先从 Limitless/Trainer Hill/PokeDeck Architect 找 archetype、主流 decklist、tech cards 和 matchup 方向。
2. 把 decklist 转成本项目可用 `deck.csv`，放入 `logs/.../decks/` 或 `artifacts/.../decks/`，再用 `tools/eval_bc.py` 或 `tools/eval_round_robin.py` 验证引擎能加载。
3. 对外部资料中的打法写成结构化 notes，例如：
   - 开局优先找哪些 Basic / engine card；
   - 哪些牌不能过早 discard；
   - 先 ability 后 evolve，还是先 evolve；
   - 贴能优先级；
   - 对某 archetype 的攻击目标和 bench damage 分配；
   - 什么时候应该 stall、什么时候必须 race prize。
4. 把 notes 映射到代码或数据:
   - `ptcg_rl/deck_plans.py`: deck-specific card tags、主 engine、资源优先级；
   - `ptcg_rl/rule_overlay.py`: 可以明确判断的强规则或 action rerank；
   - `tools/mine_strategy_trajectories.py` / `tools/build_trajectory_targets.py`: 生成 strategy labels；
   - `tools/build_shadow_pool.py`: 为新 decklist 训练 shadow opponent；
   - `tools/trace_matchup_decisions.py`: 对比规则前后同一 fixed seed matchup 对局。
5. 只有当规则/资料能在离线 trace 中改变坏决策，并在 random/RR 或 Kaggle replay loss pool 中改善，才进入 submission 候选。

### PTCG Live 如何结合

PTCG Live 更适合做人类策略验证，不适合直接当训练数据来源。

推荐流程:

1. 从 Limitless 或 Kaggle 高分 deck 导入/手动复刻到 PTCG Live。
2. 用 Test Deck / Casual / Ranked 观察真实玩家或 AI 环境下的启动顺序，特别是 Dragapult、Alakazam、Marnie 这类依赖多回合资源规划的卡组。
3. 记录关键局面，不要只记录胜负:
   - 第几回合开始攻击；
   - 哪张 engine card 没启动会导致崩盘；
   - 哪些 card reveal 暴露了对手手牌/计划；
   - 哪些攻击目标或 bench damage 分配是关键；
   - 哪些回合应该保留资源而不是立即打出。
4. 把这些记录写入 handoff 或专门的 strategy markdown，再转成 rule/plan/teacher 任务。
5. 在 Kaggle simulator 中复现同类局面。PTCG Live 看到的打法只有通过离线 engine trace 验证后，才算进入本项目 pipeline。

关键 caveat:

- PTCG Live 当前 Ranked 主要是 Standard format；Kaggle simulator 的可用卡池、bug、行动接口和计时约束可能不同。
- PTCG Live 对局不能自动导出逐步训练标签。它的价值是帮助人理解“正确连续策略”，然后人工转成规则、plan label 或测试用 trace。
- 外部资料中的真实 TCG matchup win rate 不能直接替代 Kaggle RR，因为 Kaggle agent 的错误分布和真实玩家不同。

## 0. 提交历史和版本结论

下面是 Kaggle submission 的部分提交历史，分数来自项目保存日志和 2026-08-17 通过 `kaggle competitions submissions pokemon-tcg-ai-battle -v` 刷新的 public/private score。Kaggle 分数会随天梯环境变化，表里的分数是对应时间点的有效记录，不代表模型绝对强度。

### 0.1 关键提交历史

| 日期 | Submission | 模型/版本 | 关键改动 | 记录分数 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 2026-07-31 | `55125297` | `first try` | 初始提交 | 299.0 | 只能验证打包链路 |
| 2026-08-01 | `55151898` | Mega Lucario rollout | 搜索 rollout + 改进 eval | 740.1 | 早期规则/搜索有一定基础，但不稳定 |
| 2026-08-02 | `55187025` | Marnie v4 | 早期 BC2 Marnie | 597.3 | BC 管线尚不成熟 |
| 2026-08-02 | `55189149` | Crustle v5 | Crustle 早期 specialist | 690.2 | 可以合法执行，但强度不足 |
| 2026-08-03 | `55206814` | Ogerpon top2 v7sig | 固定高质量 top2 sig + win-weighted | 965.9 | v7 是第一个可靠 Ogerpon 高点 |
| 2026-08-03 | `55207693` | Ogerpon v8 mixed | mixed corpus | 742.8 | mixed 污染 game plan，离线好不等于线上好 |
| 2026-08-03 | `55212562` | Ogerpon v9 gameplan | 加 gameplan/trajectory | 545.0 | 新标签没有正确转化，明显退化 |
| 2026-08-04 | `55227640` | Ogerpon v9 repro w2 | v9 repro | 640.0 | 没复现 v7 |
| 2026-08-04 | `55227671` | Ogerpon v7sig resubmit | 重新提交 v7 | 959.2 | 证明 v7 思路仍优于 v8/v9 |
| 2026-08-04 | `55241740` | Marnie v10pop | v10 population | 969.2 | Marnie BC 一度非常强 |
| 2026-08-04 | `55241767` | Alakazam v10pop | v10 population | 907.5 | Alakazam 可用，但后续环境不稳 |
| 2026-08-04 | `55252321` | Festival v10shadow | v10 shadow specialist | 918.6 | shadow/specialist 可产生可提交模型 |
| 2026-08-05 | `55264182` | Marnie v11all pop | v11 all-date deck specialist | 931.9 | v11 数据量提升有效 |
| 2026-08-05 | `55264151` | Ogerpon v11all 5899 | v11 all-date Ogerpon | 830.7 | Ogerpon 对环境敏感，不如历史 top2 |
| 2026-08-05 | `55276246` | Crustle v11all 477 | v11 all-date Crustle | 904.3 | Crustle 可到 900+，但环境波动大 |
| 2026-08-06 | `55302986` | Marnie w4 sig1 b8f | deck-sig specialist w4 | 986.2 | 历史最佳 Marnie 之一 |
| 2026-08-06 | `55303028` | Crustle w4 sig1 3cd | deck-sig specialist w4 | 903.1 | 可用但后续再训复现弱 |
| 2026-08-07 | `55328252` | Marnie history-k init | history-k init | 916.7 | history 并未明显超过旧 BC |
| 2026-08-08 | `55349364` | Festival v12 seqplan | seqplan scratch | 813.9 | sequence plan 未形成稳定优势 |
| 2026-08-09 | `55370364` | Marnie v12hist bigbatch | v12 history + big batch | 975.4，用户观测峰值约 1050 | 有线上高点，但 history 整体不稳定 |
| 2026-08-10 | `55395773` | Marnie good test | Marnie 重提/变体 | 940.2 | Marnie 当时仍适合环境 |
| 2026-08-11 | `55419189` | Marnie v12hist resubmit | v12hist bigbatch 重提 | 959.1 | 能复现部分强度 |
| 2026-08-12 | `55452390` | Alakazam lossopt 7f9 | lossopt/histplan 7f9 | 929.4 | Alakazam 可到 900+，但不稳 |
| 2026-08-12 | `55455876` | high900win old Marnie b8f | old recipe + 900+ winner | 953.4 | 不加复杂 history 反而更稳 |
| 2026-08-12 | `55461178` | Marnie livehard W2 | live-hard 数据增强 | 901.8 | 有改善但不如 old Marnie 高点 |
| 2026-08-13 | `55478159` | Alakazam lossopt resubmit | lossopt 7f9 重提 | 922.6 | 能短期上 900，但环境适配有限 |
| 2026-08-13 | `55484585` | v13 shadow Marnie | v13 shadow | 789.6 | v13 shadow 泛化差 |
| 2026-08-14 | `55505115` | v14 Lopunny f144 | old v14 Lopunny | 890.4 | Lopunny 可用，但重提不稳 |
| 2026-08-15 | `55527660` | Alakazam lossopt rerere | 旧 Alakazam 重提 | 794.4 | 环境改变后掉分 |
| 2026-08-16 | `55550625` | retrain Alakazam 0801-0815 | 最新 8 月复训 | 834.7 | 新数据复训未恢复历史高点 |
| 2026-08-16 | `55551578` | retrain Crustle 3cd | 最新 8 月复训 | 641.7 | Crustle 当前复训失败 |
| 2026-08-16 | `55552876` | retrain Marnie b8f w3 | 最新 8 月复训 | 855.0 | 有潜力但低于历史 Marnie |
| 2026-08-16 | `55565531` | Ogerpon current top6 ab7e | v10-style top6/current retrain | 737.6 | Ogerpon 当前环境吃力 |

完整提交列表可随时刷新：

```bash
kaggle competitions submissions pokemon-tcg-ai-battle -v
```

### 0.2 版本演化和效果差异

| 版本/阶段 | 主要变化 | 代表结果 | 经验结论 |
| --- | --- | --- | --- |
| v4-v5 | 早期 BC2、少量 specialist | Marnie v4 597.3，Crustle v5 690.2 | 管线能跑，但数据和评测都不足 |
| v7 | 固定高质量 deck sig，win-weighted，少混合 | Ogerpon top2 v7sig 965.9 | 纯净 sig + 合理 win/loss 权重非常重要 |
| v8 | mixed corpus | Ogerpon v8 mixed 742.8 | mixed 容易破坏 deck-specific game plan |
| v9 | gameplan/trajectory 特征 | Ogerpon v9 gameplan 545.0 | 新标签没对齐时会严重退化 |
| v10 | 修复 v8/v9 问题，population 和 shadow 初步成型 | Marnie v10pop 969.2，Festival v10shadow 918.6 | 稳定 BC 仍是主力，shadow 有价值但质量不均 |
| v11 | 80/64 matchup-aware 特征，更多日期，deck-sig specialist | Marnie 931.9，Crustle 904.3 | 数据量和 sig-specific 明显重要 |
| w4 specialist | 更大模型/更强 old objective | Marnie w4 986.2，Crustle w4 903.1 | 某些卡组更适合 top1/top2 specialist，而不是 mixed |
| v12 history | action/log/board history、big batch、seqplan | Marnie v12hist 975.4，history-k Lucario 643.8 | history 只有在抽取、训练、推理完全一致时才可能有用；多数实验未稳定增益 |
| v13 shadow | shadow/team specialist 恢复 | Marnie shadow 789.6，Lucario shadow 511.4 | shadow 训练质量不足时会误导 RR |
| v14 | sequence pipeline、训练期诊断、DCA/turn-plan | Lopunny 890.4，但 Lucario 569.1 | 新 pipeline 需要先看训练信号，不应直接提交 |
| v15 | block/plan 重写、fixed-seed random trace | 研究中 | 目标是连续决策，不再做小幅微调；先要求 random 100% 和可解释 trace |

## 1. 当前项目原则

1. `bc2_train.py` 是当前最稳定的 BC 主线，产物是 `.npz`。
2. `v14_*` 和 `v15_*` 是连续决策/计划学习实验线，产物通常是 `.pt`，不能只看最终 random/RR，要先看训练期诊断信号。
3. Kaggle 提交次数有限。提交候选至少要通过：训练指标正常、random gate、RR 或 baseline-delta、失败 trace 审查。
4. `random 100%` 只说明基础执行稳定，不等于 Kaggle 强；但如果 random 都不稳，通常不应提交。
5. 历史提交只作为离线比较基线，不再用于提交。

## 2. 目录和环境

在仓库根目录设置少量相对路径变量。原始 episode zip 放在项目内 `raw_episode/`。`cg` 引擎源码和 runtime wrapper 来自 `external/kaggle-environments/` submodule。

```bash
export REPO=${REPO:-$(pwd)}
export EPISODES=${EPISODES:-$REPO/raw_episode}
export CG_DIR=${CG_DIR:-$REPO/external/kaggle-environments/kaggle_environments/envs/cabt/cg}
mkdir -p raw_episode data logs checkpoints
git submodule update --init --recursive external/kaggle-environments
```

确认引擎和 Kaggle CLI 可用：

```bash
test -f "$CG_DIR/libcg.so"
kaggle competitions submissions pokemon-tcg-ai-battle -v | head
python3 - <<'PY'
import numpy, torch
print("numpy", numpy.__version__)
print("torch", torch.__version__)
PY
```

训练和大量评测前限制 CPU 线程，避免 worker 乘上 BLAS 线程后打满机器：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
```

查看 GPU：

```bash
nvidia-smi
```

## 3. 获取 Kaggle 数据

### 3.1 下载 leaderboard

leaderboard CSV 用于给 episode 标记 score 和 score band。每次重建 corpus 前建议下载一份新的。

```bash
export LB_DIR=logs/leaderboard_$(date +%Y%m%d)
mkdir -p "$LB_DIR"
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p "$LB_DIR"
unzip -o "$LB_DIR/pokemon-tcg-ai-battle.zip" -d "$LB_DIR"
export LB_CSV=$(find "$LB_DIR" -name '*.csv' | head -1)
test -f "$LB_CSV"
echo "$LB_CSV"
```

### 3.2 下载 daily episode zip

不要解压 episode zip。抽取脚本会直接读 zip。

下载 2026-08-01 到 2026-08-15：

```bash
mkdir -p "$EPISODES"
for d in $(seq -w 1 15); do
  kaggle datasets download "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-$d" -p "$EPISODES"
done
ls -lh "$EPISODES"/pokemon-tcg-ai-battle-episodes-2026-08-*.zip
```

只补某一天，例如 2026-08-16：

```bash
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-16 -p "$EPISODES"
```

### 3.3 下载 Kaggle 静态比赛文件

Kaggle competition files 里包含卡牌 ID PDF、卡牌数据 CSV、`ptcg_engine` C++ 源码、sample submission 和 `cg` Python wrapper。`cg` runtime 也可以从 `external/kaggle-environments/kaggle_environments/envs/cabt/cg` 读取。详细说明见 [12 - PTCG 玩法与 cg/C++ 引擎深读](docs/12_cg_engine_source.md)。

```bash
mkdir -p data/kaggle_files
kaggle competitions files pokemon-tcg-ai-battle --page-size 200 -v \
  | tee data/kaggle_files/files.csv
kaggle competitions download pokemon-tcg-ai-battle -p data/kaggle_files
unzip -o data/kaggle_files/pokemon-tcg-ai-battle.zip -d data/kaggle_files
```

## 4. 构建稳定 BC corpus

当前稳定 BC corpus 使用 `tools/bc_extract_v2.py`。它保存基础状态、legal option、动作标签、胜负、分数、team、deck signature、opponent metadata，并可保存 history/log/board history。

先做 smoke test：

```bash
python3 -u tools/bc_extract_v2.py "$EPISODES" \
  --out data/bc_corpus_smoke \
  --lb-csv "$LB_CSV" \
  --workers 2 \
  --max-episodes 20 \
  --action-history-k 12 \
  --log-history-k 16 \
  --board-history-k 4 \
  --board-history-feat-dim 80 \
  --progress-every 10 \
  2>&1 | tee logs/extract_smoke.log
```

正式抽取 2026-08-01 到 2026-08-15：

```bash
export CORPUS=data/bc_corpus_v12hist_0801_0815
python3 -u tools/bc_extract_v2.py "$EPISODES" \
  --out "$CORPUS" \
  --lb-csv "$LB_CSV" \
  --workers 12 \
  --action-history-k 12 \
  --log-history-k 16 \
  --board-history-k 4 \
  --board-history-feat-dim 80 \
  --progress-every 1000 \
  2>&1 | tee logs/extract_v12hist_0801_0815.log
```

检查输出字段：

```bash
python3 - <<'PY'
import glob, numpy as np
paths = glob.glob("data/bc_corpus_v12hist_0801_0815/*/*/*.npz")
print("files:", len(paths))
assert paths, "no corpus npz found"
z = np.load(paths[0], allow_pickle=True)
print("file:", paths[0])
print("state feat:", np.asarray(z["feats"][0]).shape)
print("option feat:", np.asarray(z["of_arr"][0]).shape)
print("opponent keys:", [k for k in z.files if k.startswith("opponent_")])
print("history keys:", [k for k in z.files if "history" in k][:20])
print("keys:", z.files)
PY
```

如果改了 encoder、option 特征、opponent metadata、history 字段，就必须重新抽取。只改 loss、权重、epoch、batch size 时不需要重新抽取。

## 5. 分析当前天梯环境

### 5.1 构建 ladder deck pool

`build_ladder_pool.py` 输出的是 deck pool，不是 policy pool。它适合分析环境和找 deck CSV，但不能直接代表强对手策略。

```bash
export POOL=logs/ladder_pool_0801_0815_900p
python3 tools/build_ladder_pool.py \
  --episodes-dir "$EPISODES" \
  --out "$POOL" \
  --lb-csv "$LB_CSV" \
  --min-score 900 \
  --top 240 \
  --min-games 1 \
  --workers 12 \
  --progress-every 1000
```

查看分布：

```bash
column -s, -t "$POOL/archetype_stats.csv" | head -40
python3 tools/summarize_ladder_manifest.py \
  "$POOL/pool_manifest.csv" \
  --top 20 \
  --out "$POOL/band_archetype_summary.csv"
column -s, -t "$POOL/band_archetype_summary.csv" | head -80
```

### 5.2 查看某个 archetype 的 deck signature

例如 Marnie：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus "$CORPUS" \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --top 30 \
  --out-csv logs/stats_marnie_0801_0815_900p.csv
column -s, -t logs/stats_marnie_0801_0815_900p.csv | head -40
```

例如 Alakazam：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus "$CORPUS" \
  --archetype "Alakazam" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --top 30 \
  --out-csv logs/stats_alakazam_0801_0815_900p.csv
column -s, -t logs/stats_alakazam_0801_0815_900p.csv | head -40
```

判断方法：

- `deck_sig` 多且构筑差异大：优先 deck-specific 或 team-specific。
- 一两个强 sig 占主导：可以 top1/top2。
- 样本少但天梯常见：放宽到 900+，必要时跟踪高分 team 从 600+ 上分轨迹。
- winner-only 不一定更强；样本量不足或胜局靠运气时会学坏。

## 6. 稳定 BC 训练

### 6.1 单 deck signature 训练

示例：Marnie `b8f251a476e7`，900+，w3。

```bash
mkdir -p checkpoints/bc_aug_0801_0815 logs/bc_aug_0801_0815
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b8f251a476e7 \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --best-metric policy_raw \
  --save checkpoints/bc_aug_0801_0815/bc2_marnie_b8f_w3_900p_0801_0815.npz \
  2>&1 | tee logs/bc_aug_0801_0815/train_marnie_b8f_w3_900p_0801_0815.log
```

### 6.2 Top-k deck signature 训练

示例：Ogerpon top2。历史经验是不要把所有 Ogerpon 混在一起，all-mixed 容易破坏 game plan。

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig 697a82e582d5 \
  --deck-sig 2a5072194fdf \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --best-metric policy_raw \
  --save checkpoints/bc_aug_0801_0815/bc2_ogerpon_top2_w2_900p_0801_0815.npz \
  2>&1 | tee logs/bc_aug_0801_0815/train_ogerpon_top2_w2_900p_0801_0815.log
```

### 6.3 Team-specific 训练

先查 team trajectory：

```bash
python3 tools/build_team_deck_trajectories.py \
  --corpus "$CORPUS" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --min-decisions 1000 \
  --min-episodes 5 \
  --top 80 \
  --out logs/team_deck_trajectories_0801_0815.csv
column -s, -t logs/team_deck_trajectories_0801_0815.csv | head -80
```

然后使用精确 team name 训练。下面命令只是格式示例，`--team-name` 必须替换为 corpus 中存在的名称。

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype "Alakazam" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig 7f9a538936e3 \
  --team-name "LiamK" \
  --epochs 10 \
  --batch-size 4096 \
  --width 3.0 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --best-metric policy_raw \
  --save checkpoints/bc_aug_0801_0815/bc2_alakazam_7f9_liamk_w3_900p_0801_0815.npz \
  2>&1 | tee logs/bc_aug_0801_0815/train_alakazam_7f9_liamk_w3_900p_0801_0815.log
```

### 6.4 批量 population 训练

先 dry-run：

```bash
python3 tools/train_bc_population.py \
  --corpus "$CORPUS" \
  --archetype "Alakazam" \
  --archetype "Marnie Grimmsnarl" \
  --archetype "Teal Mask Ogerpon" \
  --archetype "Crustle Wall" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --tag aug900p_w2 \
  --checkpoint-dir checkpoints/pop_aug_0801_0815 \
  --log-dir logs/pop_aug_0801_0815 \
  --dry-run
```

正式启动：

```bash
python3 -u tools/train_bc_population.py \
  --corpus "$CORPUS" \
  --archetype "Alakazam" \
  --archetype "Marnie Grimmsnarl" \
  --archetype "Teal Mask Ogerpon" \
  --archetype "Crustle Wall" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --tag aug900p_w2 \
  --checkpoint-dir checkpoints/pop_aug_0801_0815 \
  --log-dir logs/pop_aug_0801_0815 \
  --accuracy-samples 50000 \
  --poll-seconds 30 \
  2>&1 | tee logs/pop_aug_0801_0815/runner.log
```

## 7. Shadow / RR policy pool

离线 RR 要用 policy pool，而不是只有 deck 的 ladder pool。低质量 shadow 会抬高候选胜率，所以必须先做 random 审计。

生成 shadow manifest：

```bash
export SHADOW_MANIFEST=logs/shadow_manifest_aug900p_w2.csv
python3 tools/build_shadow_pool.py \
  --corpus "$CORPUS" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --known-decks-dir "$POOL/decks" \
  --top-per-archetype 3 \
  --min-decisions 5000 \
  --min-episodes 3 \
  --checkpoint-dir checkpoints/shadow_aug_0801_0815 \
  --epochs 6 \
  --batch-size 4096 \
  --width 2.0 \
  --cuda-memory-gb 24 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --best-metric policy_raw \
  --label-score-in-name \
  --out "$SHADOW_MANIFEST"
```

训练 shadow manifest：

```bash
python3 -u tools/train_shadow_manifest.py "$SHADOW_MANIFEST" \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --cuda-memory-gb 24 \
  --batch-size 4096 \
  --log-dir logs/shadow_aug_0801_0815 \
  --poll-seconds 30 \
  2>&1 | tee logs/shadow_aug_0801_0815/runner.log
```

审计 shadow random：

```bash
python3 tools/eval_manifest_random.py \
  --manifest "$SHADOW_MANIFEST" \
  --games 300 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 50 \
  --skip-bad-entries \
  --out-csv logs/shadow_aug_0801_0815/random_g300.csv
```

## 8. 单模型测试流程

假设要测试刚训练出的 Marnie：

```bash
export POLICY=checkpoints/bc_aug_0801_0815/bc2_marnie_b8f_w3_900p_0801_0815.npz
export DECK=$(find "$POOL/decks" -name 'b8f251a476e7_*.csv' | head -1)
test -f "$POLICY"
test -f "$DECK"
echo "$POLICY"
echo "$DECK"
```

Accuracy：

```bash
python3 tools/bc2_accuracy.py "$POLICY" \
  --corpus "$CORPUS" \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b8f251a476e7 \
  --max-samples 50000 \
  --batch-size 4096 \
  --progress-every 5000 \
  --out-csv logs/bc_aug_0801_0815/acc_marnie_b8f.csv
```

Random gate：

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 300 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 50 \
  2>&1 | tee logs/bc_aug_0801_0815/random_marnie_b8f_g300.log
```

定位 random 输局可以先跑更长 random 测试，再用 matchup trace 固定 random opponent 观察决策：

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 500 \
  --workers 16 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 50 \
  2>&1 | tee logs/bc_aug_0801_0815/random_marnie_b8f_g500.log
```

固定 seed 输出对 random 的 trace：

```bash
python3 tools/trace_matchup_decisions.py \
  --candidate "candidate=$POLICY:$DECK" \
  --opponent "random=random:$DECK" \
  --games 50 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 10 \
  --out-prefix logs/bc_aug_0801_0815/trace_random_marnie_b8f
```

候选打 shadow pool：

```bash
python3 tools/eval_round_robin.py \
  --entry "candidate=$POLICY:$DECK" \
  --manifest "$SHADOW_MANIFEST" \
  --candidate-only \
  --manifest-limit 120 \
  --skip-bad-entries \
  --games 80 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/bc_aug_0801_0815/rr_marnie_b8f_vs_shadow_g80.csv
```

汇总 RR：

```bash
python3 tools/summarize_round_robin.py \
  logs/bc_aug_0801_0815/rr_marnie_b8f_vs_shadow_g80.csv \
  --manifest "$SHADOW_MANIFEST" \
  --top 40 \
  --out logs/bc_aug_0801_0815/rr_marnie_b8f_vs_shadow_summary.csv
column -s, -t logs/bc_aug_0801_0815/rr_marnie_b8f_vs_shadow_summary.csv | head -60
```

按 archetype 看矩阵：

```bash
python3 tools/rr_archetype_matrix.py \
  --rr logs/bc_aug_0801_0815/rr_marnie_b8f_vs_shadow_g80.csv \
  --manifest "$SHADOW_MANIFEST" \
  --out logs/bc_aug_0801_0815/rr_marnie_b8f_archetype_matrix.csv
column -s, -t logs/bc_aug_0801_0815/rr_marnie_b8f_archetype_matrix.csv | head -80
```

## 9. 历史连续决策实验线

仓库里曾经有过 v14/v15 连续决策实验。那些脚本已经不是当前分支的活动入口，旧日志中的 `v14_*` / `v15_*` 名称只作为历史记录保留，不要直接照抄为当前命令。

当前可复现入口已经收敛到这些工具和教程：

- 连续决策、history、trajectory：见 [docs/08 - game plan、history、trajectory](docs/08_plan_history_trajectory.md)。
- 随机门槛和 RR：`tools/eval_bc.py`、`tools/eval_round_robin.py`、`tools/eval_baseline_delta.py`。
- 失败局 trace：`tools/trace_matchup_decisions.py`。
- 规则/teacher/RL：见 [docs/10 - 规则层、成功数据与人类策略](docs/10_rules_success_data.md) 和 [docs/11 - RL、search 与 teacher rollout](docs/11_rl_search_teacher.md)。

## 10. Kaggle replay 分析

查看提交列表：

```bash
kaggle competitions submissions pokemon-tcg-ai-battle -v | head -40
```

分析某个 submission，例如 `55303028`：

```bash
export SUB_ID=55303028
export SUB_DECK="$DECK"
test -f "$SUB_DECK"
mkdir -p logs/kaggle_replay_$SUB_ID
python3 tools/analyze_kaggle_replays.py "$SUB_ID" \
  --deck "$SUB_DECK" \
  --assume-agent-index 0 \
  --known-decks-dir "$POOL/decks" \
  --cache-dir logs/kaggle_replay_$SUB_ID/cache \
  --out logs/kaggle_replay_$SUB_ID/episodes.csv \
  --summary-out logs/kaggle_replay_$SUB_ID/summary_by_arch.csv \
  --group-by opponent_deck_name \
  --download-logs \
  --write-opponent-decks \
  --opponent-decks-dir logs/kaggle_replay_$SUB_ID/opponent_decks \
  --progress-every 20
column -s, -t logs/kaggle_replay_$SUB_ID/summary_by_arch.csv | head -80
```

如果某个新 opponent deck 反复击败我们，把它加入项目内 deck pool：

```bash
python3 tools/build_ladder_pool.py \
  --episodes-dir "$EPISODES" \
  --out logs/ladder_pool_with_personal_losses_$SUB_ID \
  --lb-csv "$LB_CSV" \
  --personal-loss-dir logs/kaggle_replay_$SUB_ID/opponent_decks \
  --personal-loss-weight 25 \
  --min-score 600 \
  --top 240 \
  --workers 12 \
  --progress-every 1000
```

## 11. 打包和提交

先确认 policy 与 deck 对应：

```bash
export POLICY=checkpoints/bc_aug_0801_0815/bc2_marnie_b8f_w3_900p_0801_0815.npz
export DECK=$(find "$POOL/decks" -name 'b8f251a476e7_*.csv' | head -1)
test -f "$POLICY"
test -f "$DECK"
```

打包：

```bash
export SUB_DIR=submission/$(date +%Y%m%d)
mkdir -p "$SUB_DIR"
python3 tools/package_submission.py \
  --policy "$POLICY" \
  --deck "$DECK" \
  --out "$SUB_DIR/submission_marnie_b8f_w3_900p_0801_0815.tar.gz"
ls -lh "$SUB_DIR/submission_marnie_b8f_w3_900p_0801_0815.tar.gz"
```

提交到 Kaggle：

```bash
kaggle competitions submit pokemon-tcg-ai-battle \
  -f "$SUB_DIR/submission_marnie_b8f_w3_900p_0801_0815.tar.gz" \
  -m "bc: marnie_b8f_w3_900p_0801_0815"
```

提交后立即记录：

```bash
kaggle competitions submissions pokemon-tcg-ai-battle -v | head -20
```

## 12. 长任务模板

长训练建议写成脚本再用 `nohup` 或 `tmux` 运行，避免 SSH/终端断开导致任务退出。下面模板假设脚本在当前服务器执行。

```bash
cat >run_ptcg_job.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export REPO=${REPO:-$PWD}
export EPISODES=${EPISODES:-$REPO/raw_episode}
export CG_DIR=${CG_DIR:-$REPO/external/kaggle-environments/kaggle_environments/envs/cabt/cg}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
cd "$REPO"
nvidia-smi
python3 --version
SH
chmod +x run_ptcg_job.sh
```

后台执行并查看日志：

```bash
nohup bash run_ptcg_job.sh >logs/run_ptcg_job.nohup.log 2>&1 &
echo $!
tail -f logs/run_ptcg_job.nohup.log
```

关键产物建议复制到稳定备份目录：

```bash
mkdir -p artifacts/backup_$(date +%Y%m%d)
cp -a checkpoints/bc_aug_0801_0815 artifacts/backup_$(date +%Y%m%d)/ 2>/dev/null || true
cp -a logs/bc_aug_0801_0815 artifacts/backup_$(date +%Y%m%d)/ 2>/dev/null || true
```

## 13. 结果记录和提交代码

每次完成重要实验后更新 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)，至少写：

- 日期、机器、分支、commit。
- 数据窗口，例如 `2026-08-01..2026-08-15`。
- corpus 路径、checkpoint 路径、日志路径。
- 训练配置：archetype、deck_sig、score band、winner-only/weight、width、epoch、batch、seed。
- random、RR、Kaggle replay 结果。
- 明确结论：保留、提交、废弃、需要 trace、需要重训。

提交代码：

```bash
git status --short
git diff --check
python3 -m py_compile $(git ls-files '*.py')
git add README.md AGENT_HANDOFF.md
git commit -m "Update reproducible training handbook"
```

如果 `py_compile` 因历史脚本依赖环境失败，改为只检查本次改动的 Python 文件，不要因为旧实验脚本阻塞 README 更新。

## 14. 常见问题

### 15.1 为什么离线 RR 高，Kaggle 低？

常见原因：

- RR pool 混入弱 shadow，抬高候选胜率。
- ladder 前期 600-900 分也可能遇到刚提交的强模型。
- deck signature 对不上 checkpoint。
- top-k/mixed 数据污染了 deck-specific game plan。
- 只看均值，没有看 hard counter 和分段分布。
- random/RR 没有覆盖最新 Kaggle 环境。

### 15.2 为什么很多新架构没有提升？

之前的经验是：如果新信号只作为很小辅助 loss，通用单步 BC 会淹没它。v14/v15 必须在训练日志中证明 history、plan、multi-select、known info、same-turn sequence 的样本覆盖和 loss 都真实存在，否则最后 RR 才发现无效已经太晚。

### 15.3 什么时候用 winner-only？

只有在胜局样本足够、且胜局不是明显靠运气时才用。对结构性弱势 matchup，winner-only 常常样本太少，容易学到偶然胜局。更稳妥的流程是：先找到高质量 team/deck trajectory，再检查 trace，最后决定是否 winner-only。

### 15.4 什么时候回到历史代码？

如果目标是最后提交 sprint，优先用历史高光代码和当前 August 数据快速复现，例如 v10/v11/v12 的 deck-sig specialist。具体 commit、旧配置、线上高点记录看 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)。如果目标是研究长期能力，再用当前 v14/v15。

### 15.5 当前需要重点覆盖哪些 archetype？

至少保留这些常见 archetype 的数据、shadow 和 RR 覆盖：

```text
Alakazam
Crustle Wall
Dragapult
Festival Lead
Marnie Grimmsnarl
Mega Lopunny
Mega Lucario
Teal Mask Ogerpon
Team Rocket Mewtwo
Cynthia Garchomp
Iono Bellibolt
N's Zoroark
Raging Bolt
```

环境每天会变。提交前一定先用最新 episode 和 Kaggle replay 更新 ladder 分布。
