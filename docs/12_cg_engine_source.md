# 12 - PTCG 玩法与 cg/C++ 引擎深读

上一章：[11 - RL、search 与 teacher rollout](11_rl_search_teacher.md)

这份文档用于补齐两类基础信息：

1. 不熟悉 PTCG 时，先知道一局牌的基本结构，以及这些结构在 Kaggle observation 里如何出现。
2. 需要查 `cg` simulator 行为时，知道 Kaggle files 能下载什么、Python wrapper 怎么用、C++ 源码应该从哪里开始读。

本文默认命令都在仓库根目录执行。

## 1. PTCG 基本玩法

### 1.1 一局游戏的核心对象

- Deck：每个玩家 60 张牌。比赛里的 `deck.csv` 是 60 行 Card ID。
- Active Spot：当前出战宝可梦。没有 Active Pokemon 通常会输。
- Bench：备战区。大多数卡组需要在 bench 上提前铺 Basic、进化线和后续攻击手。
- Prize Cards：奖赏卡。击倒对方 Pokemon 后拿 prize；拿完全部 prize 是最常见胜利条件。
- Hand / Deck / Discard：手牌、牌库、弃牌区。自己的手牌可见，对方手牌通常不可见。
- Energy：能量。通常每回合只能手贴一次，但卡牌效果可以额外贴能。
- Trainer：Item、Supporter、Stadium、Tool 等训练家牌。Supporter 通常每回合一次，Stadium 场上通常只有一张。
- Evolution：进化。大多数 Pokemon 不能在刚进场当回合进化；具体限制由引擎生成 legal options。
- Attack：攻击通常会结束自己的回合。攻击能量需求、伤害、附加效果由 card implementation 决定。
- Special Conditions：中毒、灼伤、睡眠、麻痹、混乱等特殊状态。

引擎是最终规则来源。不要在模型或规则 overlay 里硬写“理论上可以做什么”，要以 `select.option` 中实际出现的合法选项为准。

### 1.2 一回合的常见流程

引擎会处理洗牌、抽牌、mulligan、先后手、setup active/bench、回合开始、自动触发效果和胜负判定。agent 只在引擎需要玩家选择时收到 observation。

常见选择点：

- setup：选择是否 mulligan、选择 Active、选择 Bench。
- main：打出手牌、贴能、进化、使用 ability、撤退、攻击、结束回合。
- effect sub-selection：某张牌或攻击要求选目标、选数量、选要加入手牌/弃牌/回牌库的牌。
- coin / yes-no：硬币或是否发动效果。

比赛 agent 的输出永远是 legal option 的下标列表，例如 `[3]` 或 `[0, 2, 5]`。不能返回 card id，也不能返回自然语言动作。

### 1.3 胜负条件

从引擎的 `LogType.RESULT` 和 C++ `FinishReason` 看，主要结束原因包括：

- 某玩家拿完 prize。
- 某玩家在回合开始时牌库为空，需要抽牌但无法抽。
- 某玩家没有 Active Pokemon。
- 某些卡牌效果直接导致胜负或平局。

离线评测里如果一局超过 `--max-turns`，通常被工具当作 loss/error；这不是 PTCG 真实胜负，而是我们评测脚本的保护机制。

## 2. Kaggle Competition Files

### 2.1 查看文件列表

先列出 Kaggle 静态文件。`--page-size 200` 足够覆盖当前比赛文件列表；后续如果 Kaggle 更新，重新跑一次。

```bash
mkdir -p data/kaggle_files
kaggle competitions files pokemon-tcg-ai-battle --page-size 200 -v \
  | tee data/kaggle_files/files.csv
```

截至 2026-08-17，文件列表中关键内容包括：

- `Card_ID List_EN_.pdf`
- `Card_ID List_JP_.pdf`
- `EN Card Data.csv`
- `JP Card Data.csv`
- `ptcg_engine/ptcgProgram 22/*.h`
- `ptcg_engine/ptcgProgram 22/Export.cpp`
- `ptcg_engine/ptcgProgram 22/README.md`
- `ptcg_engine/ptcgProgram 22/LICENSES/...`
- `sample_submission/sample_submission/cg/api.py`
- `sample_submission/sample_submission/cg/game.py`
- `sample_submission/sample_submission/cg/sim.py`
- `sample_submission/sample_submission/cg/utils.py`
- `sample_submission/sample_submission/cg/libcg.so`
- `sample_submission/sample_submission/cg/libcg-arm64.so`
- `sample_submission/sample_submission/cg/libcg.dylib`
- `sample_submission/sample_submission/cg/cg.dll`
- `sample_submission/sample_submission/main.py`
- `sample_submission/sample_submission/deck.csv`

当前 Kaggle file list 没有单独的 PTCG 规则书 PDF。Kaggle overview 的 “How to Play Pokemon TCG” 区域会指向官方玩法/规则资源；可以从 Kaggle overview、官方 Play! Pokémon resources 或中文玩法页面补 PTCG 基础规则：

```text
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
https://play.pokemon.com/en-us/resources/documents/?filter=all
https://www.pokemon.cn/tcg-rules-howtoplay
```

官方比赛规则入口是 Kaggle rules page：

```text
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules
```

卡牌资料、卡牌 ID、卡牌文本和 simulator 行为主要来自 `Card_ID List_*.pdf`、`* Card Data.csv`、`ptcg_engine/` 和 sample submission 的 `cg/` wrapper。如果 Kaggle 后续增加 rulebook 文件，用同样的 `kaggle competitions files` 先确认文件名，再用 `-f` 下载。

### 2.2 下载全部静态文件

```bash
export STATIC_DIR=data/kaggle_files
mkdir -p "$STATIC_DIR"
kaggle competitions download pokemon-tcg-ai-battle -p "$STATIC_DIR"
unzip -o "$STATIC_DIR/pokemon-tcg-ai-battle.zip" -d "$STATIC_DIR"
```

下载后建议设置两个路径变量：

```bash
export ENGINE_SRC=$(find "$STATIC_DIR" -type d -path '*/ptcg_engine/ptcgProgram 22' | head -1)
export CG_SAMPLE=$(find "$STATIC_DIR" -type d -path '*/sample_submission/sample_submission/cg' | head -1)
test -d "$ENGINE_SRC"
test -d "$CG_SAMPLE"
echo "$ENGINE_SRC"
echo "$CG_SAMPLE"
```

如果只想下载某个文件：

```bash
kaggle competitions download pokemon-tcg-ai-battle \
  -f "EN Card Data.csv" \
  -p "$STATIC_DIR"

kaggle competitions download pokemon-tcg-ai-battle \
  -f "ptcg_engine/ptcgProgram 22/Api.h" \
  -p "$STATIC_DIR"
```

路径里有空格时必须加引号。

### 2.3 不要提交 Kaggle 静态文件

`ptcg_engine`、card data、card text 和二进制库属于比赛提供材料。它们用于参赛、训练和测试，不应复制进公开仓库或二次分发。仓库里只记录下载命令、源码阅读方法和我们自己的训练代码。

## 3. Python `cg` 接口

Kaggle sample submission 里的 `cg/` 是 Python wrapper，加上平台对应的动态库：

- Linux x86_64：`libcg.so`
- Linux arm64：`libcg-arm64.so`
- macOS：`libcg.dylib`
- Windows：`cg.dll`

本项目打包时通常通过 `tools/package_submission.py --cg-dir "$CG_DIR"` 把 `cg/` 放进 submission。

### 3.1 最小 smoke test

如果 `CG_DIR` 指向 `cg/` 目录本身，Python import 需要把它的父目录加入 `sys.path`。

```bash
export CG_DIR=${CG_DIR:-$REPO/../cg}
python3 - <<'PY'
import os, sys
cg_dir = os.environ["CG_DIR"]
sys.path.insert(0, os.path.dirname(cg_dir))

from cg.api import all_card_data, all_attack
cards = all_card_data()
attacks = all_attack()
print("cards:", len(cards), "attacks:", len(attacks))
print("first card:", cards[0].cardId, cards[0].name)
PY
```

### 3.2 对战接口

`cg.game` 是最常用的对战接口：

- `battle_start(deck0, deck1)`：启动一局，两个 deck 都是 60 个 Card ID。
- `battle_select(select_list)`：提交 legal option 下标列表并推进到下一个选择点。
- `battle_finish()`：释放引擎内存。
- `visualize_data()`：拿到可视化用 JSON。

最小自战循环：

```bash
python3 - <<'PY'
import os, sys, random
cg_dir = os.environ.get("CG_DIR", "../cg")
sys.path.insert(0, os.path.dirname(os.path.abspath(cg_dir)))

from cg.game import battle_start, battle_select, battle_finish

with open("deck.csv") as f:
    deck = [int(x.strip()) for x in f if x.strip()]

obs, sd = battle_start(deck, deck)
assert obs is not None, (sd.errorPlayer, sd.errorType)

try:
    for step in range(200):
        current = obs.get("current") or {}
        if current.get("result", -1) != -1:
            print("result:", current["result"], "steps:", step)
            break
        select = obs.get("select")
        if select is None:
            print("initial deck selection")
            break
        opts = select.get("option", [])
        mn = int(select.get("minCount", 0))
        mc = int(select.get("maxCount", 0))
        k = min(max(mn, 0), mc, len(opts))
        action = random.sample(range(len(opts)), k) if k > 0 else []
        obs = battle_select(action)
finally:
    battle_finish()
PY
```

### 3.3 Observation 结构

`cg.api.py` 定义了 dataclass 和 enum。训练、trace、规则 overlay 主要读这些字段：

- `Observation.select`
  - `type`：这次选择属于 main/card/energy/yes-no 等哪类。
  - `context`：选择发生在什么语境，例如 setup active、to hand、discard、damage counter、attack、evolve。
  - `minCount` / `maxCount`：必须选择多少个 option。
  - `option`：legal options 数组。agent 返回的是这个数组的下标。
  - `deck`：只有在某些搜索/查看牌库效果中才非空。
  - `contextCard` / `effect`：当前触发选择的卡。
- `Observation.current`
  - `turn`、`turnActionCount`、`yourIndex`、`firstPlayer`
  - `supporterPlayed`、`stadiumPlayed`、`energyAttached`、`retreated`
  - `players[yourIndex]` 和 `players[1 - yourIndex]`
  - active、bench、discard、prize、deckCount、hand/handCount、特殊状态
- `Observation.logs`
  - 只包含上一次选择之后发生的公开事件。
  - `MOVE_CARD_REVERSE`、`DRAW_REVERSE` 等代表隐藏信息或只公开数量。
  - 如果想利用对手 reveal 过的牌，agent 需要自己在跨 turn 状态里记住 logs。
- `Observation.search_begin_input`
  - 搜索 API 使用的序列化状态输入。普通 BC 推理通常不用。

### 3.4 Legal action 规则

agent 必须满足：

- 返回 `list[int]`。
- 每个 index 在 `[0, len(select.option))`。
- 不重复。
- 数量满足 `minCount <= len(action) <= maxCount`。

C++ `State::checkPlayerSelect()` 会检查这些条件。不满足时 Python wrapper 会抛错，Kaggle agent 很可能失败。

不要假设 `maxCount > 1` 时选项顺序无关。某些 context 下顺序可能进入 `selectedList` 或 `SkillOrder`，应先看引擎和 trace 再决定是否排序。

## 4. C++ `ptcg_engine` 源码阅读路线

源码目录通常是：

```bash
data/kaggle_files/ptcg_engine/ptcgProgram 22
```

建议设置：

```bash
export ENGINE_SRC=${ENGINE_SRC:-"data/kaggle_files/ptcg_engine/ptcgProgram 22"}
test -d "$ENGINE_SRC"
```

### 4.1 从 Python 调到 C++ 的路径

1. `cg/sim.py`
   - 选择并加载 `libcg.so` / `libcg-arm64.so` / `libcg.dylib` / `cg.dll`。
   - 通过 `ctypes` 声明 `BattleStart`、`Select`、`GetBattleData`、`SearchBegin` 等 C ABI。
2. `cg/game.py`
   - `battle_start()` 调 C++ `BattleStart`。
   - `battle_select()` 调 C++ `Select`。
   - `_get_battle_data()` 调 C++ `GetBattleData` 并返回 JSON dict。
3. `cg/api.py`
   - 定义 Python enum/dataclass。
   - 提供 `all_card_data()`、`all_attack()`、`to_observation_class()`、`search_begin()`、`search_step()`。
4. C++ `Export.cpp`
   - 暴露 `extern "C"` 函数给 Python 动态库调用。
5. C++ `Api.h`
   - 实现 `ApiBattleStart`、`ApiGetBattleData`、`ApiSelect`、`ApiSearchBegin`、`ApiSearchStep`。

### 4.2 C++ 文件速查表

| 文件 | 读它是为了什么 |
| --- | --- |
| `README.md` | 官方对 engine 包用途、授权、构建方式的说明 |
| `Export.cpp` | C ABI 导出入口；Python wrapper 最终调用这里 |
| `Api.h` | battle/search API 的核心实现；deck 合法性、隐藏信息序列化、select 推进 |
| `ApiType.h` | `SelectType`、`SelectContext`、`SelectOptionType`、`LogType` 的 C++ enum |
| `State.h` | 完整游戏状态、选择状态、legal action 验证、序列化 |
| `ToJson.h` | C++ state/log/option 如何转成 Kaggle observation JSON |
| `GameProc.h` | 主游戏流程；main action 如何执行；attack/turn end 等 |
| `SetupProc.h` | 开局、mulligan、active/bench setup |
| `SelectProc.h` | effect sub-selection 如何处理 |
| `AddOption.h` | legal options 如何被添加 |
| `TargetList.h` | 卡牌效果候选目标如何生成和过滤 |
| `EffectProc.h` | 效果执行管线 |
| `EffectInstant.h` | 即时效果实现，例如抽牌、贴能、伤害、搜索 |
| `EffectContinual.h` | 持续效果、状态类效果 |
| `SatisfyCondition.h` | 条件判断，例如能否攻击、能否选择某目标 |
| `CardImpl.h` | 卡牌具体实现，按 card name/card id 搜索最常用 |
| `CreateCard.h` / `InitializeCard.h` | 卡牌表构造和初始化 |
| `CardMove.h` | 牌在区域之间移动 |
| `AddLog.h` | 公开 log 如何记录 |
| `Search.h` | search API 的状态复制、step、release |
| `Types.h` | 能量、区域、卡牌类型、效果类型、目标类型等底层 enum |

### 4.3 常用源码定位命令

查某张牌的实现：

```bash
rg -n "Dragapult|Dusknoir|Alakazam|Marnie's Grimmsnarl|Crustle|Ogerpon" \
  "$ENGINE_SRC/CardImpl.h"
```

查 legal option 是哪里生成的：

```bash
rg -n "addOption|setSelect|SelectContext::Damage|SelectOptionType::Attack" \
  "$ENGINE_SRC"
```

查 observation JSON 字段：

```bash
rg -n "appendCommaKeyValue|ToJsonApi|ToJsonSelect|ToJsonLog" \
  "$ENGINE_SRC/ToJson.h"
```

查 select 验证和多选约束：

```bash
rg -n "checkPlayerSelect|selectMin|selectMax|selectedList|SkillOrder" \
  "$ENGINE_SRC/State.h" "$ENGINE_SRC/SelectProc.h" "$ENGINE_SRC/EffectInstant.h"
```

查胜负判定：

```bash
rg -n "FinishReason|GameResult|Result|isFinish|prize|deck" \
  "$ENGINE_SRC/State.h" "$ENGINE_SRC/GameProc.h" "$ENGINE_SRC/AddLog.h"
```

## 5. 面向训练/规则开发的排查指南

### 5.1 不知道为什么某个 action 不合法

1. 先在 trace 里打印当前 `select.type`、`select.context`、`minCount`、`maxCount`、`option`。
2. 去 `ApiType.h` 找 enum 名称。
3. 去 `AddOption.h` 和 `TargetList.h` 查 legal option 是如何生成的。
4. 去 `State.h::checkPlayerSelect` 查是否是 index、重复、数量约束失败。
5. 如果是卡牌效果内的选择，再去 `SelectProc.h` 和 `EffectInstant.h` 看 selected option 如何被消费。

### 5.2 需要理解某张牌为什么这样结算

1. 在 `CardImpl.h` 用英文卡名或 card id 搜索。
2. 找到 `.textEn(...)` 附近的 effect builder。
3. 顺着 `effect...` helper 名称去 `EffectInstant.h`、`EffectProc.h`、`EffectContinual.h`。
4. 如果涉及目标，查 `TargetList.h` 和 `SatisfyCondition.h`。
5. 如果涉及移动区域，查 `CardMove.h`。
6. 如果涉及公开/隐藏信息，查 `AddLog.h` 和 `ToJson.h`。

### 5.3 需要处理 reveal 和对手手牌信息

当前 observation 的 `current.players[opponent].hand` 通常是 `None`，但 logs 可能公开某些 reveal、search、move 事件。正确做法：

1. 在 agent 内维护一个跨 turn memory。
2. 每次收到 observation 时扫描 `logs`。
3. 对 `MOVE_CARD`、`PLAY`、`ATTACH`、`EVOLVE`、`DISCARD`、`ATTACK`、`HP_CHANGE`、`COIN` 等公开事件更新 memory。
4. 对 `MOVE_CARD_REVERSE`、`DRAW_REVERSE` 只更新数量或不确定性，不要当作知道具体牌。
5. 训练抽取时也必须生成同样的 memory feature，否则训练/推理不一致。

这也是 history/sequence pipeline 的关键：不能只把最近动作编码进去，还要把“曾经 reveal 过什么、对方 engine 是否已启动、关键资源是否进 discard、已贴能/已进化/已攻击节奏”聚合成模型能直接使用的状态。

### 5.4 多选和顺序

`minCount` / `maxCount` 允许一次选择多张牌或多个目标。典型场景：

- setup bench 选择多个 Basic。
- 搜索牌库加入手牌。
- 弃多张牌。
- 选择多个能量支付 retreat/attack cost。
- 分配多个 damage counters。
- 选择技能触发顺序。

训练标签和 policy 输出必须保留：

- 选择数量。
- 每个 option index。
- 在顺序敏感 context 下的顺序。

如果把多选压成单个 action，模型会在 Dragapult bench damage、Dusknoir damage counter、搜索/弃牌/贴能等场景中持续失真。

### 5.5 Search API 何时使用

`cg.api.search_begin()` / `search_step()` 可以复制当前状态并做 lookahead，适合 MCTS 或 teacher rollout。但它需要为隐藏信息提供预测：

- 自己 deck / prize。
- 对方 deck / prize / hand。
- 对方 face-down active。

因此 search 结果强依赖 hidden-state assumption。BC 主线不要默认认为 search 是真值；用它构造 teacher 时必须记录 hidden-state 采样方式和失败率。

## 6. 与本项目代码的连接点

- `main.py`：Kaggle submission 入口。收到 `obs_dict`，返回 legal option index list。
- `ptcg_rl/numpy_policy.py`：`.npz` policy 的 NumPy 推理实现。
- `ptcg_rl/encoder.py`、`ptcg_rl/history_features.py`：把 observation 编成模型特征。
- `ptcg_rl/rule_overlay.py`、`ptcg_rl/resource_planner.py`：规则/资源规划 overlay。
- `tools/bc_extract_v2.py`：从 episode 里抽取 observation、legal options、动作标签、history/log metadata。
- `tools/eval_bc.py`：用 `cg.game` 跑 policy vs random。
- `tools/eval_round_robin.py`：用 `cg.game` 跑 policy pool 对战。
- `tools/trace_matchup_decisions.py`：逐步 trace matchup 决策，适合定位坏决策。

当模型表现异常时，优先用 `tools/trace_matchup_decisions.py` 找到一局完整失败过程，再按上面的源码路线查：当前 observation 是否缺信号、legal option 是否被编码错、标签是否丢了多选/顺序、规则 overlay 是否改错 action。

下一章：[13 - 调用链与 submission 打包](13_call_graph_submission.md)
