# PTCG RL Training Pipeline

Pokemon TCG AI Battle 的 BC/RL 训练、评测、Kaggle 提交流程。

这份 README 是操作手册，按真实工作顺序组织：

1. 下载 Kaggle replay / episode 数据
2. 抽取 BC corpus
3. 统计环境和卡组分布
4. 选择 deck signature 和训练方案
5. 训练单个 checkpoint 或批量 population
6. 建立 checkpoint -> deck registry
7. 做 accuracy / random / round-robin / Kaggle replay 分析
8. 打包和提交
9. 根据结果迭代

## 0. 环境和目录

推荐在远端训练机运行，路径约定如下：

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git
mkdir -p logs checkpoints data
```

引擎目录需要能被打包脚本找到，通常在 repo 邻近目录：

```bash
ls ../cg/libcg.so
```

最小依赖：

```bash
pip install torch numpy
```

常用目录：

```text
../episodes_raw/                 Kaggle daily episode zip
data/bc_corpus_banded_v*/        抽取后的 BC corpus
logs/ladder_pool_*/              ladder 对手池、deck manifest、统计
logs/opp_decks_*/                从 Kaggle replay 提取的对手 deck
checkpoints/*.npz                BC checkpoint
submission.tar.gz                Kaggle 提交包
```

### 0.1 当前主线状态（2026-08-04）

本分支当前是 `v7-baseline-20260804`，目标不是单次冲榜，而是继续完善 BC pipeline、本地 ladder pool、shadow pool 和 matchup 诊断。

已确认的 v10 结论：

- Ogerpon `v10_fixed_top2`（`697a82e582d5` + `2a5072194fdf`，非 winner-only，win/loss/draw=`1.5/0.4/0.8`）已经基本回到 v7 水平；Kaggle final 951.8，用户观察峰值约 1040。
- Ogerpon `v10_fixed_top3` 加入 `5899c772bace` 后明显变差，final 720.5；暂时不要用 top3 替代 top2。
- Random 胜率只能检查 agent 是否可执行，不能代表 Kaggle 强度。后续每个候选都至少要走 random + category round-robin + failure trace。
- 0803 v10 population 的本地探针结果：Alakazam random 99.4% 但 RR avg 0.613；Crustle random 93.4%、RR avg 0.697；Marnie random 100%、RR avg 0.689；Team Rocket Mewtwo random 100%、RR avg 0.505。重点深挖低胜率 matchup，而不是只筛掉弱 shadow。

当前失败画像：

- Marnie Grimmsnarl 的全局 imitation 不差，但 `ATTACH_TO` 和开局 setup 对 Ogerpon 很敏感。
- Team Rocket Mewtwo 需要更强的 Team Rocket in-play / Mewtwo Power Saver / setup 识别。
- Alakazam 和 Crustle 的 random 表现不能解释真实强度，MAIN 阶段 PLAY/ATTACH/ABILITY/ATTACK 混淆仍然明显。
- 旧 shadow pool 能模拟部分合法策略，但还不是 Kaggle 高分策略；所有卡组都应保留并改进训练，不要过早过滤成少数“强 shadow”。

特征维度注意：

- 0803 已抽取的 v10 corpus 是 `state=64`、`option=48`。
- 当前 encoder 已扩展到 `state=80`、`option=64`，用于下一轮 matchup-aware/v11 抽取。
- 旧 v10 checkpoint 可继续评测和打包；`NumpyPolicy` 会按 checkpoint 输入维度截断新特征。
- 用旧 v10 corpus 训练且没有 `--init` 时，显式传 `--state-feat-dim 64 --opt-feat-dim 48`。
- `bc2_train.py` 在非 `--init-partial` 的精确 init 模式下会从 init checkpoint 自动推断旧维度；如果想用 v10 checkpoint 初始化 v11 80/64 模型，必须加 `--init-partial --state-feat-dim 80 --opt-feat-dim 64`。

## 1. 下载 Kaggle Episode 数据

在 repo 外层放原始 zip，避免误提交大文件：

```bash
cd /home/jie/Do/0_PTCG/workspace
mkdir -p episodes_raw
cd episodes_raw

kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-02
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-01
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-31
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-30
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-29
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-28
```

不要手动解压。`bc_extract_v2.py`、`build_ladder_pool.py` 会直接读取 zip。

如果在本机下载更快，再传到训练机：

```bash
scp /home/jie/Do/0_PTCG/raw_episode/*.zip \
  ks:/home/jie/Do/0_PTCG/workspace/episodes_raw/
```

## 2. 下载 Leaderboard CSV

Leaderboard 用来给 episode 标记 score band。

```bash
mkdir -p /tmp/lb
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p /tmp/lb
unzip -o /tmp/lb/pokemon-tcg-ai-battle.zip -d /tmp/lb
LB_CSV=$(ls /tmp/lb/*.csv | head -1)
echo "$LB_CSV"
```

如果 `bc_extract_v2.py --lb-csv` 省略，脚本会尝试自动下载；但为了可复现，推荐显式传 `LB_CSV`。

## 3. 抽取 BC Corpus

当前主线使用 `bc_extract_v2.py`。它会写入：

- `state` / `options` / `actions`
- `archetype`
- `score_band`
- `deck_sig`
- `team_name`
- `score`
- `episode_id`
- `player_index`
- `reward` / `won` / `draw`
- `final_status` / `game_steps`
- `opponent_deck_sig` / `opponent_archetype` / `opponent_team_name`
- `opponent_score` / `opponent_score_band`
- `feature_version` / `state_feat_dim` / `opt_feat_dim`

这些字段是 deck-specific、winner-aware、trajectory-aware 训练的基础。

```bash
cd /home/jie/Do/0_PTCG/workspace/ptcg_rl_git
mkdir -p logs data

python3 -u tools/bc_extract_v2.py ../episodes_raw \
  --out data/bc_corpus_banded_v9 \
  --lb-csv "$LB_CSV" \
  --workers 9 \
  --progress-every 500 \
  2>&1 | tee logs/bc_extract_v9.log
```

抽取前后可以审计原始 option 身份是否完整：

```bash
python3 tools/audit_episode_options.py ../episodes_raw \
  --max-episodes 2000 \
  --option-types PLAY ABILITY ATTACK SKILL CARD \
  --progress-every 200
```

如果改了 encoder、option 特征、deck metadata、game-plan 特征，就需要重新抽取 corpus；只改训练 loss 或权重不需要重新抽取。

### 3.1 v11 Matchup-aware 抽取

当前 encoder 已经新增 matchup/shadow 需要的 80/64 维特征。不要把 v11 输出覆盖到 v10 corpus；新抽取放新目录。

```bash
python3 -u tools/bc_extract_v2.py ../episodes_raw \
  --out data/bc_corpus_banded_v11_matchup_0803 \
  --lb-csv "$LB_CSV" \
  --workers 9 \
  --progress-every 500 \
  2>&1 | tee logs/bc_extract_v11_matchup_0803.log
```

抽完先检查维度和 opponent metadata：

```bash
python3 - <<'PY'
import glob, numpy as np
p = glob.glob("data/bc_corpus_banded_v11_matchup_0803/*/*/*.npz")[0]
z = np.load(p, allow_pickle=True)
print("file:", p)
print("state feat:", np.asarray(z["feats"][0]).shape)
print("option feat:", np.asarray(z["of_arr"][0]).shape)
print("feature meta:", z.get("feature_version"), z.get("state_feat_dim"), z.get("opt_feat_dim"))
print("opponent keys:", [k for k in z.files if k.startswith("opponent_")])
PY
```

matchup-conditioned 统计示例：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v11_matchup_0803 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --opponent-archetype "Teal Mask Ogerpon" \
  --top 20 \
  --out-csv logs/bc_corpus_stats_marnie_vs_ogerpon_v11.csv
```

matchup-conditioned failure report 示例：

```bash
python3 tools/bc2_failure_report.py checkpoints/pop/bc2_marnie_grimmsnarl_v10pop_all0803_set_w2.npz \
  --corpus data/bc_corpus_banded_v11_matchup_0803 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --opponent-archetype "Teal Mask Ogerpon" \
  --device cpu \
  --max-samples 50000 \
  --out-prefix logs/bc_failure_marnie_vs_ogerpon_v11
```

如果用 v10 checkpoint 作为 v11 初始化，必须使用 partial init：

```bash
python3 tools/build_shadow_pool.py \
  --corpus data/bc_corpus_banded_v11_matchup_0803 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --opponent-archetype "Teal Mask Ogerpon" \
  --init-template "checkpoints/pop/bc2_{archetype_slug}_v10pop_all0803_set_w2.npz" \
  --init-partial \
  --state-feat-dim 80 \
  --opt-feat-dim 64 \
  --out logs/shadow_pool_manifest_marnie_vs_ogerpon_v11.csv
```

## 4. 看当天环境和卡组分布

### 4.1 构建 Ladder Pool

`build_ladder_pool.py` 从 daily episodes 和个人 Kaggle replay loss deck 中抽取对手池。

```bash
python3 tools/build_ladder_pool.py \
  --episodes-dir ../episodes_raw \
  --out logs/ladder_pool_0802_all \
  --lb-csv "$LB_CSV" \
  --top 120 \
  --min-games 1 \
  --workers 9 \
  --progress-every 1000
```

如果已有自己提交输过的对手 deck，加入 personal loss 权重：

```bash
python3 tools/build_ladder_pool.py \
  --episodes-dir ../episodes_raw \
  --out logs/ladder_pool_today \
  --lb-csv "$LB_CSV" \
  --personal-loss-dir logs/opp_decks_55207693/55207693 \
  --personal-loss-dir logs/opp_decks_55212562/55212562 \
  --personal-loss-weight 25 \
  --top 120 \
  --workers 9
```

主要输出：

```text
logs/ladder_pool_today/archetype_stats.csv
logs/ladder_pool_today/pool_manifest.csv
logs/ladder_pool_today/decks/*.csv
```

### 4.2 查看各分段卡组分布

```bash
column -s, -t logs/ladder_pool_today/archetype_stats.csv | head -40
```

按 score band 汇总 manifest：

```bash
python3 tools/summarize_ladder_manifest.py \
  logs/ladder_pool_today/pool_manifest.csv \
  --top 12 \
  --out logs/ladder_pool_today/band_archetype_summary.csv
```

读法：

- `games`：该 archetype 在 replay 中出现次数。
- `decks`：不同 deck signature 数。
- `weight`：构建对手池时的权重，通常比 raw games 更适合决定测试优先级。
- 如果一个 archetype deck 很多，优先做 deck-specific，不要直接 mixed。

## 5. 从环境统计中选择训练目标

先看某个 archetype 里 deck signature 是否混杂：

```bash
python3 tools/bc_corpus_stats.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --top 20 \
  --out-csv logs/bc_corpus_stats_ogerpon_v9.csv
```

常见判断：

- top deck sig 占比很高，且 random/round-robin 弱：训练 top1 或该 team trajectory。
- top2/top3 都是高质量变体：训练 top-k high-quality sig。
- 多个 sig 胜率、队伍、构筑差异很大：不要 `mixed all`，先 deck-specific。
- 样本很少但 ladder 出现频繁：考虑放宽 score bands 到 `900-999` 或 `800-899`，并优先 winner-only / team-name。
- random 强但 Kaggle 弱：说明本地对手池缺真实策略，先分析 Kaggle replay。

### 5.1 追踪高分用户/卡组路径

构建 team + deck signature trajectory 表：

```bash
python3 tools/build_team_deck_trajectories.py \
  --corpus data/bc_corpus_banded_v9 \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --min-decisions 1000 \
  --min-episodes 10 \
  --top 80 \
  --progress-every-files 10 \
  --out logs/team_deck_trajectories_v9.csv
```

用它找：

- 同一 team + deck_sig 的长期样本。
- max score 高、episodes 多、decision win rate 高的 trajectory。
- 适合 `--team-name` + `--deck-sig` 的 specialist。

## 6. 当前推荐训练方法

当前主线是 `bc2_train.py`。默认模型使用 slot-aware board encoder；旧 `bc_trainer.py` 只保留作历史参考。

### 6.1 单 deck signature 训练

适合 Alakazam、Lucario、Lopunny 等构筑差异大、mixed 会污染的卡组。

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1100-1199" "900-999" "800-899" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --epochs 30 \
  --batch-size 1024 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --context-weight MAIN=1.6 \
  --context-weight TO_HAND=2.0 \
  --context-weight ATTACH_FROM=3.0 \
  --context-weight ATTACH_TO=2.5 \
  --type-weight ATTACK=1.8 \
  --type-weight ATTACH=2.5 \
  --type-weight PLAY=1.4 \
  --type-weight EVOLVE=1.6 \
  --save checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz \
  2>&1 | tee logs/train_mega_lucario_top1_v9_gameplan_w2.log
```

### 6.2 Top-k high-quality deck sig 训练

适合 Ogerpon 这类有多个高质量变体、但 all mixed 会学坏的 archetype。

复现当前 Ogerpon 高分思路时，不要全 mixed，也不要只训 `697a82e582d5`。优先复现 top2：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig 697a82e582d5 \
  --deck-sig 2a5072194fdf \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --save checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz \
  2>&1 | tee logs/train_ogerpon_top2_repro_v9_w2.log
```

参考已知结果：

- `bc2_ogerpon_top2_v7sig_w2.npz`：曾到 Kaggle 965.9。
- 它训练的是两个高质量 sig 的 winweighted 混合，不是全 Ogerpon mixed。
- `bc2_teal_mask_ogerpon_v8_mixed_1000_w2.npz` 离线 accuracy 高，但 Kaggle 低，说明 all mixed 会丢 game plan。

### 6.3 Team trajectory specialist

适合追踪某个高分用户的固定构筑。

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig 2a5072194fdf \
  --team-name "James Cox & Henry Chao" \
  --winner-only \
  --epochs 12 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --save checkpoints/bc2_ogerpon_box_james_traj_v9_w2.npz \
  2>&1 | tee logs/train_ogerpon_box_james_traj_v9_w2.log
```

注意：trajectory specialist 不一定比 top-k 更强。它更纯，但样本更少；必须经过 random 和 failure-pool 测试。

### 6.4 Win-weighted mixed

只适合构筑高度一致、deck sig 差异不大的 archetype。

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --save checkpoints/bc2_marnie_v9_winweighted_w2.npz \
  2>&1 | tee logs/train_marnie_v9_winweighted_w2.log
```

### 6.5 Value head 实验

`--value-weight` 当前是实验项，不是默认最佳。只在和 no-value 同 epoch 做 A/B 时使用：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --value-weight 0.005 \
  --save checkpoints/bc2_marnie_v9_value0005_w2.npz \
  2>&1 | tee logs/train_marnie_v9_value0005_w2.log
```

## 7. 批量训练 Population

### 7.1 Archetype-level 批量训练

适合先铺一组 baseline population。

```bash
python3 tools/train_bc_population.py \
  --corpus data/bc_corpus_banded_v9 \
  --gpus 0,1,2,3 \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --tag v9_1000_w2 \
  --min-decisions 20000 \
  --dry-run
```

正式启动：

```bash
python3 -u tools/train_bc_population.py \
  --corpus data/bc_corpus_banded_v9 \
  --gpus 0,1,2,3 \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --tag v9_1000_w2 \
  --min-decisions 20000 \
  --accuracy-samples 50000 \
  --poll-seconds 30 \
  2>&1 | tee logs/train_bc_population_v9_1000_w2.log
```

只训练指定 archetype：

```bash
python3 -u tools/train_bc_population.py \
  --archetype "Marnie Grimmsnarl" \
  --archetype "Crustle Wall" \
  --archetype "Team Rocket Mewtwo" \
  --corpus data/bc_corpus_banded_v9 \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --gpus 0,1,2,3 \
  --tag v9_core_w2 \
  --min-decisions 20000 \
  2>&1 | tee logs/train_bc_population_v9_core_w2.log
```

### 7.2 Deck-specific 自动计划

先为多个 archetype 生成 stats：

```bash
for arch in \
  "Marnie Grimmsnarl" "Teal Mask Ogerpon" "Mega Lopunny" \
  "Dragapult" "Festival Lead" "Mega Lucario"; do
  slug=$(echo "$arch" | tr "[:upper:] " "[:lower:]_" | tr -cd "a-z0-9_")
  python3 tools/bc_corpus_stats.py \
    --corpus data/bc_corpus_banded_v9 \
    --archetype "$arch" \
    --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
    --top 20 \
    --out-csv "logs/bc_corpus_stats_${slug}_v9.csv"
done
```

生成训练和评测脚本：

```bash
python3 tools/plan_deck_specific_bc.py \
  --stats-glob "logs/bc_corpus_stats_*_v9.csv" \
  --corpus data/bc_corpus_banded_v9 \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --tag v9_topdeck_w2 \
  --manifest logs/ladder_pool_0802_all/pool_manifest.csv \
  --registry logs/policy_deck_registry_v9_topdeck.csv \
  --out logs/deck_specific_bc_plan_v9.csv \
  --script logs/train_deck_specific_v9.sh \
  --eval-script logs/eval_deck_specific_v9.sh \
  --gpus 0,1,2,3 \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --random-games 500 \
  --ladder-games 100 \
  --workers 8
```

运行：

```bash
bash logs/train_deck_specific_v9.sh
bash logs/eval_deck_specific_v9.sh
```

如果脚本后期只剩单个 job，是正常的长尾调度；如需更高吞吐，增加 `--jobs-per-gpu` 或拆分重跑剩余 checkpoint。

## 8. 建立 Checkpoint -> Deck Registry

不要手动猜 checkpoint 对应 deck。先建 registry：

```bash
python3 tools/build_policy_registry.py \
  --checkpoint-glob "checkpoints/*v9*.npz" \
  --manifest logs/ladder_pool_0802_all/pool_manifest.csv \
  --out logs/policy_deck_registry_v9.csv
```

检查某个 checkpoint：

```bash
grep "ogerpon" logs/policy_deck_registry_v9.csv
```

如果 registry 没有匹配，测试和打包时显式传 `--deck`。特别注意：

- 文件名里没有 deck sig 时，registry 只能按 archetype 选 manifest top deck。
- `top-k` checkpoint 可能训练了多个 sig，但只能提交一套 deck；必须人工确认提交 deck 是否和训练目标兼容。
- Ogerpon 历史上出现过 `checkpoint 学 top2，但 registry 配到 top1 deck` 的情况；这类要在训练日志和 registry 之间人工核对。

## 9. 测试顺序

推荐顺序固定为：

1. `bc2_accuracy.py`：确认没有明显训练/抽取 bug。
2. `eval_bc.py` vs random：确认 agent 能正常执行、不会超时、基础胜率过线。
3. `eval_round_robin.py` vs core population：确认不是只会打 random。
4. `eval_round_robin.py` vs ladder/failure pool：确认能穿过当前低分环境。
5. Kaggle submit：最终验证。
6. `analyze_kaggle_replays.py` + `analyze_replay_decisions.py`：复盘输给谁、怎么输。

### 9.1 Accuracy

单 checkpoint：

```bash
python3 tools/bc2_accuracy.py checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig 697a82e582d5 \
  --deck-sig 2a5072194fdf \
  --max-samples 50000 \
  --batch-size 4096 \
  --progress-every 5000 \
  --out-csv logs/bc2_accuracy_ogerpon_top2_repro_v9_w2.csv
```

指标读法：

- `First action`：第一步选择准确率，优先看。
- `Top-3 first`：如果高而 top-1 低，说明候选排序接近，可以考虑采样/MCTS/rerank。
- `Length match`：多选长度是否学对。
- `By context`：重点看 `MAIN`、`TO_HAND`、`ATTACH_FROM`、`ATTACH_TO`、`DISCARD`。
- `By option count`：`11+` 和 `6-10` 低，说明大候选集合排序弱。

注意：accuracy 高不保证 Kaggle 高。`v8 Ogerpon mixed` 就是 accuracy 高但实战弱。

### 9.2 Random Test

显式 deck：

```bash
python3 tools/eval_bc.py checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --games 500 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50 \
  2>&1 | tee logs/eval_random_ogerpon_top2_repro_v9_w2.log
```

用 registry 自动找 deck：

```bash
python3 tools/eval_bc.py checkpoints/bc2_marnie_v9_winweighted_w2.npz \
  --registry logs/policy_deck_registry_v9.csv \
  --auto-deck \
  --games 500 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50
```

经验线：

- `<70%`：大概率 broken 或 deck mismatch。
- `70%-90%`：可用但弱。
- `90%-97%`：基本可测 round-robin。
- `98%-100%`：只说明能打 random，不代表能上分。

### 9.3 Core Round-robin

手写 entry 最可靠：

```bash
python3 tools/eval_round_robin.py \
  --entry candidate=checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --entry marnie=checkpoints/bc2_marnie_v8_mixed_w2.npz:logs/ladder_pool_0802_all/decks/b8f251a476e7_marnie_grimmsnarl_szlachetny_snieg.csv \
  --entry crustle=checkpoints/bc2_crustle_v8_mixed_w2.npz:logs/ladder_pool_0802_all/decks/47756cdfd20f_crustle_wall_flg.csv \
  --entry cynthia=checkpoints/bc2_cynthia_v8_mixed_w2.npz:logs/ladder_pool_0802_all/decks/52f467394857_cynthia_garchomp_junlee789.csv \
  --entry mewtwo=checkpoints/bc2_team_rocket_mewtwo_v8_mixed_w2.npz:logs/ladder_pool_0802_all/decks/f0bac971c56d_team_rocket_mewtwo_flg.csv \
  --include-random \
  --games 200 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/round_robin_ogerpon_top2_repro_vs_core.csv \
  2>&1 | tee logs/round_robin_ogerpon_top2_repro_vs_core.log
```

汇总：

```bash
python3 tools/summarize_round_robin.py \
  logs/round_robin_ogerpon_top2_repro_vs_core.csv \
  --top 20 \
  --out logs/round_robin_ogerpon_top2_repro_vs_core_summary.csv
```

指标读法：

- `avg_no_random`：总体强度。
- `min_no_random`：最差 matchup；决定是否容易被低分区拦截。
- `random_wr`：基础 sanity。
- `losses`：输给多少个对手。
- `worst`：下一个优先修复的 matchup。

### 9.4 Ladder Pool / Failure Pool

从 manifest 生成 opponent entries：

```bash
OPPS=$(python3 tools/emit_ladder_pool_entries.py \
  logs/ladder_pool_0802_all/pool_manifest.csv \
  --top 30 \
  --one-per-archetype)
```

候选只打对手，不让对手之间互打：

```bash
python3 tools/eval_round_robin.py \
  --entry candidate=checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  $OPPS \
  --candidate-only \
  --skip-bad-entries \
  --games 200 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/round_robin_ogerpon_top2_repro_vs_ladder_pool.csv \
  2>&1 | tee logs/round_robin_ogerpon_top2_repro_vs_ladder_pool.log
```

从 Kaggle 失败 replay deck 构建 failure pool：

```bash
python3 tools/make_kaggle_opp_round_robin_cmd.py \
  --policy-name candidate \
  --policy checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --opp-dir logs/opp_decks_55207693/55207693 \
  --opp-dir logs/opp_decks_55212562/55212562 \
  --games 200 \
  --progress-every 20 \
  --out-csv logs/round_robin_ogerpon_top2_repro_vs_kaggle_failures.csv
```

复制输出的命令执行即可。

注意：`random:deck` 对手只是 deck 合法随机策略，不能代表 Kaggle 真实提交策略。低分区突围必须结合 Kaggle replay loss pool。

### 9.5 Failure Report

用于定位具体 context/type 的弱点：

```bash
python3 tools/bc2_failure_report.py checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1100-1199" "900-999" "800-899" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --max-samples 50000 \
  --batch-size 4096 \
  --progress-every 5000 \
  --out-prefix logs/bc2_failure_mega_lucario_top1_v9_gameplan_w2
```

主要看：

- Worst contexts by exact accuracy
- Worst set-style contexts by F1
- `early_end`
- `miss_attack`
- examples CSV

### 9.6 Matchup Trace

`trace_matchup_decisions.py` 用于在本地直接复盘某个低胜率 matchup。它会输出 game summary、逐决策明细、choice type 聚合，适合回答“输局里具体坏在 setup、挂能量、进化、攻击还是 early END”。

```bash
python3 tools/trace_matchup_decisions.py \
  --candidate marnie=checkpoints/pop/bc2_marnie_grimmsnarl_v10pop_all0803_set_w2.npz:logs/ladder_pool_0802_all/decks/b8f251a476e7_marnie_grimmsnarl_raihan_ramadistra.csv \
  --opponent ogerpon=checkpoints/v10/bc2_ogerpon_v10_fixed_top2_w2.npz:logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --games 100 \
  --workers 8 \
  --max-turns 700 \
  --out-prefix logs/eval_v10/marnie_vs_ogerpon_trace_g100
```

优先看：

- `.summary.csv`：first attack turn、total attacks、setup active、win/loss 差异。
- `.choice_types.csv`：PLAY/ATTACH/ABILITY/ATTACK/END 的 win/loss 差异。
- `.decisions.csv`：loss 中高置信但明显错误的单步决策。

## 10. Kaggle 提交和分数追踪

### 10.1 打包

显式 deck：

```bash
python3 tools/package_submission.py \
  --policy checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz \
  --deck logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --out submission.tar.gz
```

用 registry：

```bash
python3 tools/package_submission.py \
  --policy checkpoints/bc2_marnie_v9_winweighted_w2.npz \
  --registry logs/policy_deck_registry_v9.csv \
  --auto-deck \
  --out submission.tar.gz
```

打包输出里必须核对：

```text
policy: ...
deck:   ...
cg:     ...
```

### 10.2 提交

```bash
kaggle competitions submit pokemon-tcg-ai-battle \
  -f submission.tar.gz \
  -m "bc: ogerpon_top2_repro_v9"
```

Kaggle 每日提交次数有限。提交前至少满足：

- random test 不低于 90%，强候选最好 97%+。
- core round-robin 不存在明显全输。
- ladder/failure pool 没有被低分区主流 deck 完全拦住。
- package deck 与训练 deck sig 已核对。

### 10.3 追踪分数

分数会随匹配继续波动，必须记录时间序列：

```bash
python3 -u tools/track_kaggle_scores.py \
  --watch \
  --interval 60 \
  --out logs/kaggle_submission_scores.csv \
  2>&1 | tee logs/kaggle_score_watch.log
```

只看一次：

```bash
python3 tools/track_kaggle_scores.py --no-append
```

分析时看：

- 初始能否冲出 600-800。
- 峰值分数。
- 回落速度。
- 稳定区间。
- 同一 checkpoint 在不同时间提交的差异。

## 11. Kaggle Replay 分析

提交分数只能说明结果，replay 才能告诉我们输给谁。

### 11.1 拉取并汇总 replay

```bash
python3 tools/analyze_kaggle_replays.py 55206814 \
  --deck logs/ladder_pool_0802_all/decks/697a82e582d5_teal_mask_ogerpon_majkel1337.csv \
  --known-decks-dir logs/ladder_pool_0802_all/decks \
  --known-decks-dir logs/ladder_pool_v2/decks \
  --cache-dir logs/kaggle_replays/55206814 \
  --out logs/kaggle_55206814_ogerpon_v7_rows.csv \
  --summary-out logs/kaggle_55206814_ogerpon_v7_by_deck.csv \
  --group-by opponent_deck_name \
  --write-opponent-decks \
  --opponent-decks-dir logs/opp_decks_55206814 \
  --progress-every 5 \
  --timeout 60
```

如果 deck 识别 ambiguous，可显式指定：

```bash
python3 tools/analyze_kaggle_replays.py 55206814 \
  --agent-index 0 \
  --cache-dir logs/kaggle_replays/55206814
```

### 11.2 按时间看上分路径

```bash
python3 - <<'PY'
import csv
path = "logs/kaggle_55206814_ogerpon_v7_rows.csv"
rows = list(csv.DictReader(open(path)))
rows.sort(key=lambda r: r["create_time"])
for r in rows:
    result = "W" if r["won"] == "1" else "L"
    print(r["create_time"], result, r["opponent_deck_sig"], r["opponent_deck_name"], "steps", r["steps"])
PY
```

这个视角可以判断：

- 是不是被低分区 counter 拦截。
- 是不是靠遇到某个主流高分 deck 快速上分。
- 哪些 opponent signature 需要加入本地 failure pool。

### 11.3 决策行为诊断

```bash
python3 tools/analyze_replay_decisions.py \
  --rows logs/kaggle_55206814_ogerpon_v7_rows.csv \
  --group-by opponent_deck_sig \
  --archetype "Teal Mask Ogerpon" \
  --out logs/kaggle_55206814_ogerpon_v7_decisions_by_sig.csv
```

指标解释：

- `miss_attack_rate`：有 ATTACK option 但没选 attack 的比例。它是行为信号，不是绝对错误。
- `miss_attach_rate`：有 ATTACH option 但没挂能量的比例。
- `early_end_rate`：有强动作时选择 END 的比例，这个高通常是 bug。
- examples：优先人工看 loss 中 “attack available but not chosen” 的回合。

## 12. 当前已知最佳实践

### 12.1 卡组类型和训练方式

| 类型 | 适合训练方式 | 说明 |
| --- | --- | --- |
| 构筑稳定、数据多 | winweighted mixed | Marnie、部分 Crustle |
| 多个强 deck sig | top-k high-quality sig | Ogerpon 当前应走 top2，不走 all mixed |
| 构筑差异大 | deck-specific | Alakazam、Lopunny、Lucario |
| 数据少但上限高 | 放宽 score band + winner-only + reweight | Mega Lucario |
| 规则/对局计划强依赖 | BC + rule overlay 或后续 RL | Crustle counter、anti-ex、复杂 Box |

### 12.2 当前重点 checkpoint 复现

Ogerpon 历史高分方向：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Teal Mask Ogerpon" \
  --score-bands "1200+" "1100-1199" "1000-1099" \
  --deck-sig 697a82e582d5 \
  --deck-sig 2a5072194fdf \
  --epochs 8 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --save checkpoints/bc2_ogerpon_top2_repro_v9_w2.npz
```

Mega Lucario 当前方向：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lucario" \
  --score-bands "1100-1199" "900-999" "800-899" \
  --deck-sig 43d6d8b0fce9 \
  --winner-only \
  --epochs 30 \
  --batch-size 1024 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --context-weight MAIN=1.6 \
  --context-weight TO_HAND=2.0 \
  --context-weight ATTACH_FROM=3.0 \
  --context-weight ATTACH_TO=2.5 \
  --type-weight ATTACK=1.8 \
  --type-weight ATTACH=2.5 \
  --type-weight PLAY=1.4 \
  --type-weight EVOLVE=1.6 \
  --save checkpoints/bc2_mega_lucario_top1_v9_gameplan_w2.npz
```

Mega Lopunny 当前方向：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u tools/bc2_train.py \
  --corpus data/bc_corpus_banded_v9 \
  --archetype "Mega Lopunny" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --deck-sig b0cb21e29406 \
  --deck-sig 276707c0fdb4 \
  --winner-only \
  --epochs 12 \
  --batch-size 4096 \
  --width 2.0 \
  --device cuda:0 \
  --first-action-weight 2.0 \
  --option-weight 0.35 \
  --multi-select-weight 1.5 \
  --save checkpoints/bc2_mega_lopunny_top2_v9_gameplan_w2.npz
```

### 12.3 什么时候考虑 RL

先不要过早上 RL。满足这些条件后再用当前 BC population 做 self-play/RL 更合理：

- 至少有 4-6 个 checkpoint 能稳定 95%+ 打 random。
- core round-robin 有清晰强弱关系，不是大量 broken agent。
- Kaggle replay 能稳定识别低分区拦截者。
- 目标卡组已有可用 BC，不会在 RL 初期只学随机探索。

RL 的首要目标不是从零学会玩，而是用强 BC 初始化后，针对已知 failure pool 改 matchup。

## 13. 常见问题

### Loss 很低但 random 弱

优先检查：

- checkpoint 和 deck 是否错配。
- 是否 all mixed 混入了不同 game plan。
- 是否 winner-only 样本太少。
- 是否训练 deck sig 和提交 deck sig 不一致。

### Accuracy 高但 Kaggle 分数低

说明 offline imitation 不是瓶颈。下一步：

1. 拉 Kaggle replay。
2. 看按时间的上分路径。
3. 提取 loss opponent decks。
4. 用 failure pool 评测。
5. 决定换 deck sig、top-k、规则 overlay 或 RL。

### Random 100% 但分数低

random 只验证基本合法动作和简单压制。Kaggle 低分区有真实策略提交和 counter deck，必须看 ladder/failure pool。

### 多选场景怎么处理

训练时提高：

```bash
--first-action-weight 2.0
--option-weight 0.35
--multi-select-weight 1.5
--context-weight TO_HAND=2.0
--context-weight ATTACH_TO=2.5
```

诊断时看：

- `Length match`
- set-style context F1
- `TO_HAND`、`DISCARD`、`ATTACH_TO`

### 4 卡 A800 如何利用

单个 `bc2_train.py` 是单进程单卡。用这些方式并行：

- `train_bc_population.py --gpus 0,1,2,3`
- `plan_deck_specific_bc.py` 生成多 GPU 脚本
- 手动开 4 个 shell，每个 `CUDA_VISIBLE_DEVICES=N --device cuda:0`

`CUDA_VISIBLE_DEVICES=3 --device cuda:0` 表示使用物理 GPU 3。

### 有大型程序时测试变慢

`eval_bc.py --workers 8` 是多进程 CPU 模拟，容易受 CPU 频率、NUMA、IO 和进程调度影响。大规模评测建议：

- `--workers 8` 到 `--workers 16` 之间试。
- 避免和大量 Python 多进程训练同时跑。
- 对慢 matchup 加 `--max-turns 700` 防止拖死。

## 14. 旧入口

`tools/bc_trainer.py` 和 `tools/bc_accuracy.py` 是旧 BC 流程，当前不作为主线。

旧训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/bc_trainer.py \
  --corpus data/bc_corpus_banded \
  --archetype "Marnie Grimmsnarl" \
  --score-bands "1200+" "1100-1199" \
  --epochs 30 \
  --batch-size 2048 \
  --width 2.0 \
  --device cuda:0 \
  --save checkpoints/bc_marnie_legacy.npz
```

除非做回归对比，否则优先使用 `bc2_train.py`。

## 15. 推荐日常循环

每天先做：

```bash
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p /tmp/lb
unzip -o /tmp/lb/pokemon-tcg-ai-battle.zip -d /tmp/lb
LB_CSV=$(ls /tmp/lb/*.csv | head -1)

python3 -u tools/bc_extract_v2.py ../episodes_raw \
  --out data/bc_corpus_banded_today \
  --lb-csv "$LB_CSV" \
  --workers 9 \
  --progress-every 500 \
  2>&1 | tee logs/bc_extract_today.log

python3 tools/build_ladder_pool.py \
  --episodes-dir ../episodes_raw \
  --out logs/ladder_pool_today \
  --lb-csv "$LB_CSV" \
  --top 120 \
  --workers 9
```

然后：

1. 看 `logs/ladder_pool_today/archetype_stats.csv`。
2. 对目标 archetype 跑 `bc_corpus_stats.py`。
3. 选择 top1/top-k/team trajectory。
4. 训练 checkpoint。
5. 跑 accuracy -> random -> core round-robin -> failure pool。
6. 打包提交。
7. 追踪分数。
8. 拉 replay，更新 failure pool 和下一轮训练目标。
