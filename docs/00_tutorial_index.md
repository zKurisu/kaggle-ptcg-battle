# PTCG 教程总目录

这是一条按顺序阅读和实操的路线。目标不是“看完概念”，而是能一步一步搭起一条完整实验链：

`Kaggle episode -> corpus -> BC -> evaluation -> RR -> rule/search/RL -> submission`

先在仓库根目录做一次最小初始化。后续教程里的 `cg` import、episode 下载目录和 submission 打包都默认这些变量已经存在。

```bash
export REPO=${REPO:-$(pwd)}
export EPISODES=${EPISODES:-$REPO/raw_episode}
export CG_DIR=${CG_DIR:-$REPO/external/kaggle-environments/kaggle_environments/envs/cabt/cg}
mkdir -p raw_episode data logs checkpoints
git submodule update --init --recursive external/kaggle-environments
test -f "$CG_DIR/libcg.so"
```

建议顺序：

1. [01 - 项目与环境](01_project_setup.md)
2. [02 - PTCG 基础玩法与 Kaggle 文件](02_game_and_kaggle_files.md)
3. [03 - cg 引擎、observation 与合法动作](03_cg_engine_observation.md)
4. [04 - 数据抽取与 corpus 构建](04_bc_extraction.md)
5. [05 - BC 到底在学什么](05_bc_concepts.md)
6. [06 - BC 训练实操](06_bc_training.md)
7. [07 - specialist、population、shadow](07_population_shadow.md)
8. [08 - game plan、history、trajectory](08_plan_history_trajectory.md)
9. [09 - matchup、random、RR、baseline-delta](09_evaluation_matchups.md)
10. [10 - 规则层、成功数据与人类策略](10_rules_success_data.md)
11. [11 - RL、search 与 teacher rollout](11_rl_search_teacher.md)
12. [12 - PTCG 玩法与 cg/C++ 引擎深读](12_cg_engine_source.md)
13. [13 - 调用链与 submission 打包](13_call_graph_submission.md)
14. [decks - 卡组资料索引](decks/00_deck_index.md)

每一章都按这个格式组织：

- 先解释概念。
- 再给最小命令。
- 再指向源码。
- 最后写常见误区。

如果你只想先做第一轮实操，按这个顺序跑帮助命令，确认当前入口参数：

```bash
python3 tools/bc_extract_v2.py --help
python3 tools/bc2_train.py --help
python3 tools/eval_bc.py --help
python3 tools/eval_round_robin.py --help
python3 tools/package_submission.py --help
```

如果你想先验证引擎，使用下面这个自包含 smoke test。不要直接 `from cg...`，因为 `CG_DIR` 指向的是 `cg/` 目录本身，Python import 需要把它的父目录加入 `sys.path`。

```bash
python3 - <<'PY'
import os, sys
cg_dir = os.environ.get("CG_DIR", "external/kaggle-environments/kaggle_environments/envs/cabt/cg")
sys.path.insert(0, os.path.dirname(os.path.abspath(cg_dir)))

from cg.game import battle_start, battle_select, battle_finish
print("cg.game import ok:", battle_start.__name__, battle_select.__name__, battle_finish.__name__)
PY
```

读教程时不要跳章太多。前一章没有看懂的字段，通常会在后一章的命令、源码和日志里再次出现。
