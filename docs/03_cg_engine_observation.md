# 03 - cg 引擎、observation 与合法动作

这一章回答：模型在对局时到底看到了什么，为什么输出必须是合法动作下标。

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

先看一个最小例子：

```python
from cg.game import battle_start, battle_select, battle_finish

obs, sd = battle_start(deck, deck)
assert obs is not None
while True:
    sel = obs.get("select")
    cur = obs.get("current") or {}
    if cur.get("result", -1) != -1:
        break
    if sel is None:
        break
    obs = battle_select([])
battle_finish()
```

实际模型做的事情只是把上面的空动作换成策略动作。

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
from cg.api import all_card_data, all_attack
print(len(all_card_data()), len(all_attack()))
PY
```

再看一次提交入口：

```bash
sed -n '1,220p' main.py
```

下一章：[04 - 数据抽取与 corpus 构建](04_bc_extraction.md)
