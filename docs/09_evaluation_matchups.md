# 09 - matchup、random、RR、baseline-delta

这一章讲评测。核心原则是：不要用一个平均胜率判断模型。PTCG ladder 是分段环境，早期爬分、当前 meta、具体 deck_sig 和弱势 matchup 都会改变提交结果。

## 1. 三层评测

### random

random 是基础行为门槛：

```bash
python3 tools/eval_bc.py checkpoints/candidate.npz \
  --deck decks/candidate.csv \
  --games 300 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50
```

random 主要回答：

- agent 是否能合法稳定执行。
- 启动顺序是否足够基本。
- 是否频繁错过明显 attack/evolve/attach。

它不能回答“能不能打赢高分模型”。

### round robin

RR 用候选池互打：

```bash
python3 tools/eval_round_robin.py \
  --manifest logs/candidate_manifest.csv \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --skip-bad-entries \
  --out-csv logs/rr_candidates_g100.csv
```

RR 主要回答：

- 谁在本地池里强。
- 哪些 matchup 是结构性弱点。
- 新模型是否只是吃了弱 shadow 的分。

### baseline-delta

baseline-delta 用同一批 opponent 同时打 baseline 和 candidate：

```bash
python3 tools/eval_baseline_delta.py \
  --baseline base=checkpoints/base.npz:decks/base.csv \
  --candidate cand=checkpoints/candidate.npz:decks/candidate.csv \
  --opponent-manifest logs/shadow_manifest.csv \
  --games 80 \
  --workers 16 \
  --max-turns 700 \
  --skip-bad-entries \
  --out-csv logs/baseline_delta_candidate.csv
```

它更适合回答“candidate 相比已有最好模型是否真的变强”。

## 2. RR 池必须过滤

如果 RR 池里有很多弱 shadow，强模型会被虚高。

构建主池时建议保留：

- random 质量达标的 shadow。
- 当前 ladder 里真实出现的 archetype。
- 每个 archetype 至少一个强 sig。
- 历史提交中稳定强的基线模型。
- 不同 score band 的代表模型。

不要只留下“最好打”的 shadow。那会让本地 RR 与 Kaggle ladder 偏离。

## 3. 生成 archetype 矩阵

RR CSV 不直观，可以转成 deck x opponent archetype 矩阵：

```bash
python3 tools/rr_archetype_matrix.py \
  --rr logs/rr_candidates_g100.csv \
  --manifest logs/candidate_manifest.csv \
  --random logs/random_candidates_g300.csv \
  --out logs/rr_candidates_archetype_matrix.csv \
  --counts-out logs/rr_candidates_archetype_counts.csv
```

baseline-delta 也能转矩阵：

```bash
python3 tools/baseline_delta_archetype_matrix.py \
  --delta logs/baseline_delta_candidate.csv \
  --opponent-manifest logs/shadow_manifest.csv \
  --out logs/baseline_delta_candidate_matrix.csv \
  --include-baseline
```

看矩阵时重点不是平均值，而是：

- 是否有 10% 级别的极弱 matchup。
- 是否在当前高频 archetype 上明显亏。
- 是否只是赢了低频弱卡组。
- 是否对 early-climb 分段常见卡组稳定。

## 4. 分析 episode matchup

Kaggle daily episode 能提供环境先验：

```bash
python3 tools/analyze_episode_matchups.py \
  --episodes-dir "$EPISODES" \
  --date-from 2026-08-01 \
  --date-to 2026-08-15 \
  --deck-manifest logs/ladder_pool_latest/pool_manifest.csv \
  --out-dir logs/matchups_0801_0815_score900 \
  --source-label episodes_0801_0815_score900 \
  --score-floor 900 \
  --min-games 50 \
  --min-deck-games 10 \
  --edge-threshold 0.06 \
  --workers 12 \
  --progress-every 1000
```

输出通常包括：

- `archetype_matchups.csv`
- `archetype_counter_edges.csv`
- `deck_sig_matchups.csv`
- `deck_sig_counter_edges.csv`
- `matchup_summary.md`

这些是环境先验，不是模型强度。真实提交还会受当前 active submission、评分不确定性和新 deck 影响。

## 5. Kaggle replay 为什么重要

Kaggle ladder 每个提交从低分开始爬，早期几局的胜负对分数影响很大。replay 能回答：

- 低分段输给了谁。
- 输的是没见过的 deck_sig，还是已知弱点。
- 是连续坏 matchup，还是模型自己启动失败。
- 本地 RR 是否缺少真实对手。

常用入口：

```bash
kaggle competitions submissions pokemon-tcg-ai-battle -v
kaggle competitions episodes pokemon-tcg-ai-battle --help
python3 tools/analyze_kaggle_replays.py --help
```

注意：Kaggle 通常只持续匹配最新的 active submissions，所以 replay 数据是局部窗口，不是完整 ladder。

## 6. 从评测走向训练任务

评测输出应该转成后续任务：

1. `random` 低：先修基础行为、deck 对齐、encoder。
2. `RR` 某 archetype 极弱：建 weakness pool。
3. `baseline-delta` 负：新模型不要提交，找具体退化 matchup。
4. `episode matchup` 与本地 RR 冲突：先补 shadow，而不是盲目训练。
5. replay 出现新 sig：加入 ladder/shadow manifest 后重测。

下一章：[10 - 规则层、成功数据与人类策略](10_rules_success_data.md)
