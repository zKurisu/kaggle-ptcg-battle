# 10 - 规则层、成功数据与人类策略

BC 只能模仿数据中已有行为。结构性弱点通常需要三类额外信息：

- 人类 PTCG 策略资料。
- 成功对局或成功局面。
- 明确可验证的规则/teacher。

这一章讲如何把这些信息变成可测试的改进，而不是凭感觉加规则。

## 1. 规则不是另一个完整 agent

本项目里的规则层应该优先做窄范围 rerank 或 veto：

- rerank：把某个合法 option 的分数抬高。
- veto：排除明显错误的 option。
- guard：在极少数确定局面强制保护关键资源。

不要一开始就写完整规则型 bot。那会很快变成不可维护的分支。

## 2. 规则必须满足的验证链

每条规则至少过四关：

1. random sanity：不能破坏基础胜率。
2. focused weakness pool：必须改善目标弱 matchup。
3. broad RR：不能明显损坏其他高频 matchup。
4. trace：确实改变了之前看见的坏决策。

如果只在 focused pool 变好，但 broad RR 崩了，不能提交。

## 3. 外部资料怎么用

可用来源包括：

- Limitless TCG / Play Limitless：decklist、meta、matchup 方向。
- Trainer Hill / PokeDeck Architect / PokemonMeta：meta share、matchup 统计、tech trend。
- PTCG Live：人工测试启动路线和资源规划。
- 官方卡牌文本：确认规则是否合法。
- Kaggle discussion：确认当前比赛里别人常用的思路和 simulator caveat。

外部资料只能当 hypothesis。必须再确认：

- 这张卡是否在 Kaggle deck CSV 中。
- 这张卡是否在 `EN Card Data.csv` 中。
- simulator 的 legal option 是否真的允许这样操作。
- 这个思路是否能在本地 trace 中改变坏决策。

## 4. 先找成功数据是否存在

弱势局有两种情况：

- 真实 corpus 里有足够同 sig 胜局：可以挖成功/失败差异。
- 几乎没有同 sig 胜局：不能指望 BC 从稀疏胜局里学会，应该找 teacher 或生成成功对局。

先审计：

```bash
python3 tools/audit_matchup_success_data.py --help
python3 tools/find_matchup_teachers.py --help
```

找某个弱 pair 的 teacher：

```bash
python3 tools/find_matchup_teachers.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --weak-pair "Teal Mask Ogerpon=>Crustle Wall" \
  --min-pair-games 20 \
  --min-pair-wins 5 \
  --min-teacher-win-decisions 500 \
  --top-per-pair 10 \
  --progress-every 1000 \
  --out-csv logs/teachers_ogerpon_crustle.csv \
  --out-pair-csv logs/teacher_pairs_ogerpon_crustle.csv
```

输出里如果 teacher 和目标 deck_sig 差异很大，只能转成抽象规则，不能直接模仿整局动作。

## 5. 挖高分选手的打法差异

对同一 archetype，比较强队胜局与普通失败局：

```bash
python3 tools/mine_top_player_strategy.py \
  --corpus data/bc_corpus_banded_v11_0801_0815 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig b8f251a476e7 \
  --opponent-archetype "Teal Mask Ogerpon" \
  --target-team-name "LiamK" \
  --target-outcome win \
  --control-outcome loss \
  --min-games 10 \
  --min-rate-gap 0.08 \
  --min-choose-gap 0.10 \
  --top 40 \
  --out-dir logs/mine_marnie_ogerpon_liamk
```

重点看：

- setup timing 差异。
- attack timing 差异。
- 关键卡被选择的频率差异。
- 2-gram / 3-gram 行动序列差异。
- 某些 option 有机会但没选的机会差异。

这些比“最后赢了”更接近可迁移策略。

## 6. 从 strategy seed 生成任务

结构化 seed 文件通常放在：

```text
data/matchup_strategy_seeds_v1.csv
data/matchup_strategy_seed_cards_v1.csv
```

生成 trace/rule_probe 任务：

```bash
python3 tools/plan_strategy_seed_jobs.py \
  --candidate-manifest logs/candidate_manifest.csv \
  --opponent-manifest logs/shadow_manifest.csv \
  --out-dir logs/strategy_seed_jobs \
  --expand-all \
  --limit-candidates 20 \
  --limit-opponents 80 \
  --command-scope all \
  --games 80 \
  --rule-games 120 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 20
```

汇总结果：

```bash
python3 tools/summarize_strategy_seed_jobs.py \
  --job-dir logs/strategy_seed_jobs \
  --out-prefix logs/strategy_seed_jobs/summary \
  --weak-wr 0.35 \
  --strong-wr 0.55 \
  --min-rule-delta 0.05 \
  --top 40
```

## 7. 成功数据如何进入训练

可选路径：

- `matchup_bc`：同 sig 有足够胜局时，抽成功/失败子集，重新训练一版 scratch BC。
- `teacher_rollout`：规则/search 能赢时，生成 rollouts，再蒸馏成 BC 数据。
- `rule_overlay`：如果规则非常窄且稳定，可以 submission 侧启用。
- `do_not_train`：胜局像运气或数据矛盾时，先记录，不训练。

生成 teacher rollout 任务：

```bash
python3 tools/plan_rollout_teacher_jobs.py \
  --weakness-csv logs/rr_candidates_g100.csv \
  --candidate-manifest logs/candidate_manifest.csv \
  --opponent-manifest logs/shadow_manifest.csv \
  --max-win-rate 0.35 \
  --max-jobs 24 \
  --max-per-archetype 4 \
  --games 200 \
  --workers 16 \
  --parallel-jobs 4 \
  --max-turns 700 \
  --keep-outcomes win \
  --rule-mode conservative \
  --out-root data/generated_teacher_rollouts \
  --out-csv logs/teacher_rollout_plan.csv \
  --out-sh logs/run_teacher_rollouts.sh
```

执行前必须打开 `logs/run_teacher_rollouts.sh` 检查路径和并发数。

## 8. 本章底线

不要把“别人说应该这样打”直接写成提交规则。正确流程是：

```text
资料/胜局 -> 可观察触发条件 -> 规则或 teacher -> focused eval -> broad eval -> trace -> 再训练或提交
```

下一章：[11 - RL、search 与 teacher rollout](11_rl_search_teacher.md)
