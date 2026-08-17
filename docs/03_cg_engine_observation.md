# 03 - cg 引擎、observation 与合法动作

这一章回答：模型在对局时到底看到了什么，为什么输出必须是合法动作下标。

开始本章代码前，先确认 submodule 已经拉取：

```bash
export CG_DIR=${CG_DIR:-$(pwd)/external/kaggle-environments/kaggle_environments/envs/cabt/cg}
git submodule update --init --recursive external/kaggle-environments
test -f "$CG_DIR/libcg.so"
```

## 1. 运行时链路

比赛提交的最短链路是：

```text
Kaggle -> main.py -> ptcg_rl.numpy_policy.NumpyPolicy -> cg.game -> ptcg_engine
```

其中：

- `main.py` 负责加载 `policy.npz`、`deck.csv`、可选规则层。
- `NumpyPolicy` 负责把 observation 编成 NumPy 特征并给出动作。
- `cg.game` 负责启动、推进、结束对局。
- `ptcg_engine` 才是真正的游戏规则实现。

## 2. observation 里有什么

你会经常看到三个部分：

- `select`：当前要做什么选择。
- `current`：当前完整游戏状态。
- `logs`：从上次选择到现在发生的公开事件。

重点字段：

- `select.type` / `select.context`
- `select.minCount` / `select.maxCount`
- `select.option`
- `current.turn`
- `current.players[...]`
- `current.result`

## 3. 为什么输出是 option index

因为引擎已经把合法动作列出来了。模型不是“自由发明动作”，而是：

1. 在 `select.option` 里找出有希望的条目。
2. 返回这些条目的下标。
3. 下标必须合法、去重、数量满足上下限。

如果你返回的是 card id 或者越界值，`State::checkPlayerSelect()` 会直接报错。

## 4. 最小对局循环

先看一个最小例子。这里用当前目录的 `deck.csv` 自战，并在每个选择点提交一个满足 `minCount/maxCount` 的最小合法下标列表；真实模型会把这个占位动作换成策略动作。

```bash
python3 - <<'PY'
import os, sys
cg_dir = os.environ.get("CG_DIR", "external/kaggle-environments/kaggle_environments/envs/cabt/cg")
sys.path.insert(0, os.path.dirname(os.path.abspath(cg_dir)))

from cg.game import battle_start, battle_select, battle_finish

with open("deck.csv") as f:
    deck = [int(x.strip()) for x in f if x.strip()]

obs, sd = battle_start(deck, deck)
assert obs is not None, (sd.errorPlayer, sd.errorType)

try:
    for step in range(50):
        sel = obs.get("select")
        cur = obs.get("current") or {}
        if cur.get("result", -1) != -1:
            print("finished:", cur["result"], "steps:", step)
            break
        if sel is None:
            print("waiting for initial deck selection or no selectable action")
            break
        opts = sel.get("option", [])
        mn = int(sel.get("minCount", 0))
        mc = int(sel.get("maxCount", 0))
        k = min(max(mn, 0), mc, len(opts))
        obs = battle_select(list(range(k)))
    else:
        print("stopped after smoke-test step cap")
finally:
    battle_finish()
PY
```

## 5. 常见选择类型

- `MAIN`：主阶段动作。
- `CARD` / `ATTACHED_CARD` / `ENERGY`：卡牌选择。
- `ATTACK`：选攻击。
- `EVOLVE`：选进化。
- `YES_NO`：是否。
- `COUNT`：数量。

这就是为什么多选和顺序很重要。一次选择多个对象时，模型不是在做“单个分类”，而是在做一个顺序敏感的组合决策。

## 6. logs 为什么重要

`logs` 能告诉你：

- 抽到了什么
- 公开打出了什么
- 贴了什么能
- 进化了什么
- 攻击造成了什么后果

但注意：

- 对手手牌通常不可见。
- reveal 的牌要靠历史日志自己记。
- `logs` 不等于完整真值，只是公共信息流。

## 7. 本章练习

先用命令看接口：

```bash
python3 - <<'PY'
import os, sys
cg_dir = os.environ.get("CG_DIR", "external/kaggle-environments/kaggle_environments/envs/cabt/cg")
sys.path.insert(0, os.path.dirname(os.path.abspath(cg_dir)))

from cg.game import battle_start, battle_select, battle_finish, visualize_data
print("cg.game:", battle_start.__name__, battle_select.__name__, battle_finish.__name__, visualize_data.__name__)
PY
```

再看一次提交入口：

```bash
sed -n '1,220p' main.py
```

下一章：[04 - 数据抽取与 corpus 构建](04_bc_extraction.md)
