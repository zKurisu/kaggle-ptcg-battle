# Kaggle PTCG AI Battle — 提交前检查清单

> 长期提升方案见 [ROADMAP.md](ROADMAP.md)

## 每次提交前必须检查

### 1. 牌组数量（最常见错误！）

```bash
# 检查硬编码 DECK 是否 60 张
python3 -c "from main import DECK; assert len(DECK)==60, f'DECK={len(DECK)}'; print('DECK: OK')"

# 检查 deck.csv 是否 60 行
python3 -c "cards=[l.strip() for l in open('deck.csv') if l.strip()]; assert len(cards)==60, f'deck.csv={len(cards)}'; print('deck.csv: OK')"
```

### 2. 本地自对弈（模拟 Kaggle Validation Episode）

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from cg.game import battle_start, battle_select, battle_finish
from main import agent
DECK = [int(l.strip()) for l in open('deck.csv') if l.strip()]
obs, sd = battle_start(DECK, DECK)
assert obs is not None, f'start failed: errorType={sd.errorType}'
turn = 0
while True:
    sel = obs.get('select')
    res = obs.get('current',{}).get('result',-1)
    if res != -1: print(f'OK: {turn} turns, result={res}'); break
    if sel is None: print('ERROR: select=None unexpectedly'); break
    obs = battle_select(agent(obs)); turn += 1
    if turn > 500: print('ERROR: 500 turn timeout'); break
battle_finish()
"
```

### 3. 模块导入无 crash

```bash
# import 不能有异常，不能挂死
timeout 5 python3 -c "import sys; sys.path.insert(0,'.'); from main import agent; print('Import OK')"
```

### 4. agent 返回格式正确

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from main import agent

# 初始牌组提交：select=None，返回 60 张卡
deck = agent({'select': None})
assert len(deck) == 60, f'Deck selection returned {len(deck)} cards'
assert all(isinstance(c, int) for c in deck), 'Not all ints'
print('Initial deck: OK')

# 正常选择：返回 list[int]，长度在 minCount~maxCount 之间
obs = {'select': {'option': [{'type': 1}, {'type': 2}], 'minCount': 1, 'maxCount': 1, 
        'context': 41, 'type': 9, 'remainDamageCounter': 0, 'remainEnergyCost': 0,
        'deck': None, 'contextCard': None, 'effect': None},
       'current': None, 'logs': []}
result = agent(obs)
assert isinstance(result, list), f'Not a list: {type(result)}'
assert len(result) == 1, f'Expected 1, got {len(result)}'
assert 0 <= result[0] < 2, f'Index {result[0]} out of range [0,2)'
print('Selection: OK')
"
```

### 5. 打包内容正确

```bash
# tar.gz 内容必须是 main.py、deck.csv、cg/ 在顶层
tar tzf submission.tar.gz | sort
# 预期输出:
#   cg/
#   cg/__init__.py
#   cg/api.py
#   cg/game.py
#   cg/libcg.so (或对应平台的 .so/.dylib/.dll)
#   cg/sim.py
#   cg/utils.py
#   deck.csv
#   main.py
```

### 6. 无模块级副作用

```bash
# main.py 不应该在 import 时崩溃、超时或产生大量输出
# 已确认的做法：将 all_card_data()/all_attack() 等 DLL 调用延迟到 agent() 首次调用时
grep -n "all_card_data\|all_attack" main.py
# 确保不在模块顶层直接调用（应该在函数内）
```

### 7. 绝对不能修改 cg/ 目录内容

```bash
# cg/ 必须和 sample_submission 中的完全一致
diff -r cg/ data/sample_submission/sample_submission/cg/ --exclude=__pycache__
```

---

## 常见错误总结

| 症状 | 根因 | 检查方法 |
|------|------|---------|
| `SubmissionStatus.ERROR`，日志 ~0.09s，无 stderr | 牌组不是 60 张 | `len(DECK) == 60` |
| `SubmissionStatus.ERROR`，日志有 Python traceback | 代码逻辑错误 | 本地自对弈 |
| `SubmissionStatus.ERROR`，日志显示 timeout | import 太慢 / 死循环 | 检查模块级代码 |
| 分数低（<400） | agent 返回无效选择 | 检查 minCount/maxCount/索引范围 |
| 分数低，自对弈正常 | 策略太弱 | 参考高分方案优化 |

---

## Kaggle 提交命令

```bash
# 打包
python package_submission.py

# 提交
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "描述"

# 查看结果
kaggle competitions submissions pokemon-tcg-ai-battle

# 查看错误日志（如有 ERROR）
kaggle competitions episodes <submission_id>
kaggle competitions logs <episode_id> 0
kaggle competitions logs <episode_id> 1
```
