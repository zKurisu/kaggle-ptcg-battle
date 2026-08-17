# PTCG RL / BC Training Handbook

这个仓库用于 Pokemon TCG AI Battle 的数据抽取、BC 训练、本地评测、Kaggle replay 分析、submission 打包，以及 v14/v15 连续决策实验。

README 的目标是让第一次接手的人能从零开始复现一条完整流水线。长期实验结论、线上提交记录和临时远端任务放在 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)；开始任何新实验前先读 handoff，再读本文件。

## 0. 先记住当前项目原则

1. `bc2_train.py` 是当前最稳定的 BC 主线，产物是 `.npz`。
2. `v14_*` 和 `v15_*` 是连续决策/计划学习实验线，产物通常是 `.pt`，不能只看最终 random/RR，要先看训练期诊断信号是否真实生效。
3. Kaggle 提交次数有限。提交候选至少要通过：训练指标正常、random gate、RR 或 baseline-delta、失败 trace 审查。
4. `random 100%` 只说明基础执行稳定，不等于 Kaggle 强；但如果 random 都不稳，通常不应提交。
5. `DVH` 等历史提交只作为本地比较基线，不再用于提交。
6. 调用 Kaggle API 时只使用 `jie` 账号。
7. 长训练尽量在 `ks` 上跑，关键 checkpoint、日志、submission 要及时拉回本机备份。

## 1. 目录约定

在 `ks` 上推荐使用：

```bash
export PTCG_ROOT=/data/jie
export REPO=$PTCG_ROOT/ptcg_rl_git_v7_baseline_20260804
export EPISODES=$PTCG_ROOT/episodes_raw
export CG_DIR=$PTCG_ROOT/cg
cd "$REPO"
mkdir -p data logs checkpoints
```

在本机推荐使用：

```bash
export PTCG_ROOT=/home/jie/Do/0_PTCG
export REPO=$PTCG_ROOT/bak/ptcg_rl_git
export EPISODES=$PTCG_ROOT/raw_episode
export CG_DIR=$PTCG_ROOT/cg
cd "$REPO"
mkdir -p data logs checkpoints artifacts
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

训练或大量评测前固定 CPU 线程，避免 worker 乘上 BLAS 线程后把机器打满：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
```

查看 GPU 资源：

```bash
nvidia-smi
```

## 2. 获取 Kaggle 数据

### 2.1 下载 leaderboard

leaderboard CSV 用于给 episode 标记 score 和 score band。每次重建 corpus 前都建议重新下载一份。

```bash
export LB_DIR=$PTCG_ROOT/leaderboard_$(date +%Y%m%d)
mkdir -p "$LB_DIR"
kaggle competitions leaderboard pokemon-tcg-ai-battle --download -p "$LB_DIR"
unzip -o "$LB_DIR/pokemon-tcg-ai-battle.zip" -d "$LB_DIR"
export LB_CSV=$(find "$LB_DIR" -name '*.csv' | head -1)
test -f "$LB_CSV"
echo "$LB_CSV"
```

### 2.2 下载 daily episode zip

不要解压 episode zip。抽取脚本会直接读 zip。

下载 2026-08-01 到 2026-08-15：

```bash
mkdir -p "$EPISODES"
for d in $(seq -w 1 15); do
  kaggle datasets download "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-$d" -p "$EPISODES"
done
ls -lh "$EPISODES"/pokemon-tcg-ai-battle-episodes-2026-08-*.zip
```

如果只补某一天，例如 2026-08-16：

```bash
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-08-16 -p "$EPISODES"
```

本机下载后同步到 `ks`：

```bash
scp "$EPISODES"/pokemon-tcg-ai-battle-episodes-2026-08-*.zip ks:/data/jie/episodes_raw/
```

## 3. 构建稳定 BC corpus

当前稳定 BC corpus 使用 `tools/bc_extract_v2.py`。它会保存基础状态、legal option、动作标签、胜负、分数、team、deck signature、opponent metadata，并可保存 history/log/board history。

先做一个小样本 smoke test：

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

检查抽取文件是否包含期望字段：

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

## 4. 分析当前天梯环境

### 4.1 构建 ladder pool

`build_ladder_pool.py` 从 daily episode 中抽取 deck CSV 和卡组分布。它输出的是 deck pool，不是 policy pool。

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

### 4.2 查看某个 archetype 的 deck signature

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

读法：

- `deck_sig` 多且构筑差异大：优先 deck-specific 或 team-specific。
- 只有一两个强 sig：可以 top1/top2。
- 样本少但天梯常见：放宽到 900+，必要时跟踪高分 team 从 600+ 上分轨迹。
- winner-only 不一定更强；样本量不足或胜局靠运气时会学坏。

## 5. 稳定 BC 训练

### 5.1 单 deck signature 训练

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

### 5.2 Top-k deck signature 训练

示例：Ogerpon top2。不要把所有 Ogerpon 混在一起，历史上 all-mixed 经常线上变差。

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

### 5.3 Team-specific 训练

先用 stats 或 team trajectory 找到稳定 team，再训练。下面命令只是格式示例，`--team-name` 必须精确匹配 corpus 中的 team。

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

如果 team 名不确定，先不要跑训练，先查：

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

### 5.4 批量 population 训练

先 dry-run，确认会启动哪些 job：

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

## 6. Shadow / RR policy pool

本地 RR 要用 policy pool，而不是只有 deck 的 ladder pool。

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

只把质量合格的 shadow 放进高强度 RR。低质量 shadow 会抬高候选胜率，误导提交选择。

## 7. 单模型测试流程

假设要测试刚训练出的 Marnie：

```bash
export POLICY=checkpoints/bc_aug_0801_0815/bc2_marnie_b8f_w3_900p_0801_0815.npz
export DECK=$(find "$POOL/decks" -name 'b8f251a476e7_*.csv' | head -1)
test -f "$POLICY"
test -f "$DECK"
echo "$POLICY"
echo "$DECK"
```

### 7.1 Accuracy

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

重点看：

- `first action` 或 raw/policy first-action。
- 多 option 场景表现。
- `MAIN`、`ATTACH_FROM`、`ATTACH_TO`、`TO_HAND`、`DISCARD`。
- attach miss、evolve miss、attack miss、premature END 等诊断。

### 7.2 Random gate

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 300 \
  --workers 16 \
  --max-turns 700 \
  --progress-every 50 \
  2>&1 | tee logs/bc_aug_0801_0815/random_marnie_b8f_g300.log
```

如果 random 不稳定，先定位输局：

```bash
python3 tools/v15_find_random_losses.py "$POLICY" \
  --deck "$DECK" \
  --games 500 \
  --workers 16 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 50 \
  --recheck-losses \
  --out-csv logs/bc_aug_0801_0815/random_losses_marnie_b8f.csv
```

固定 seed 输出单局 trace：

```bash
python3 tools/v15_trace_game.py "$POLICY" \
  --deck "$DECK" \
  --games 200 \
  --seed 20260817 \
  --target-outcome loss \
  --max-turns 700 \
  --progress-every 20 \
  --out-md logs/bc_aug_0801_0815/trace_random_loss_marnie_b8f.md
```

trace 要能看清每回合做了什么：setup、贴能、进化、ability、攻击、资源浪费、提前 END、错误目标、对手 reveal 信息等。

### 7.3 RR against policy pool

对候选打 shadow pool：

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

## 8. v14 连续决策实验线

v14 的目标是让训练期能看到 sequence、history、same-turn plan、known opponent info、multi-select、DCA 等信号。不要把它当成稳定提交线。

抽取 sequence corpus：

```bash
export SEQ_CORPUS=data/seq_corpus_v14_0801_0815
python3 -u tools/v14_extract_sequences.py "$EPISODES" \
  --out "$SEQ_CORPUS" \
  --lb-csv "$LB_CSV" \
  --workers 12 \
  --future-horizon 8 \
  --progress-every 1000 \
  2>&1 | tee logs/v14_extract_sequences_0801_0815.log
```

审计 sequence corpus：

```bash
python3 tools/v14_audit_sequence_corpus.py \
  --corpus "$SEQ_CORPUS" \
  --archetype "Dragapult" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --progress-every 50000 \
  --out-csv logs/v14_sequence_corpus_audit_dragapult_0801_0815.csv
column -s, -t logs/v14_sequence_corpus_audit_dragapult_0801_0815.csv | head -60
```

训练一个 Dragapult v14 诊断模型，训练过程中每个 epoch 做小 random smoke：

```bash
export DRAG_DECK=$(find "$POOL/decks" -iname '*dragapult*.csv' | head -1)
test -f "$DRAG_DECK"
mkdir -p checkpoints/v14_diag logs/v14_diag
CUDA_VISIBLE_DEVICES=0 python3 -u tools/v14_train_sequence_policy.py \
  --corpus "$SEQ_CORPUS" \
  --archetype "Dragapult" \
  --score-bands "1200+" "1100-1199" "1000-1099" "900-999" \
  --seq-len 8 \
  --stride 1 \
  --width 256 \
  --layers 4 \
  --heads 4 \
  --batch-size 1024 \
  --epochs 6 \
  --device cuda:0 \
  --progress-every 200 \
  --diagnostic-ablation \
  --current-action-weight 1.0 \
  --prefix-action-weight 0.15 \
  --plan-weight 0.4 \
  --next-type-weight 0.3 \
  --turn-plan-weight 0.4 \
  --turn-next-plan-weight 0.4 \
  --turn-seq-plan-weight 0.4 \
  --known-action-weight 0.2 \
  --opportunity-type-weight 0.2 \
  --current-rank-margin-weight 0.2 \
  --current-complexity-weight 0.5 \
  --multi-target-weight 2.0 \
  --damage-counter-weight 2.0 \
  --random-smoke-deck "$DRAG_DECK" \
  --random-smoke-games 60 \
  --random-smoke-workers 6 \
  --random-smoke-every 1 \
  --random-smoke-max-turns 700 \
  --out checkpoints/v14_diag/dragapult_v14_seq_diag.pt \
  2>&1 | tee logs/v14_diag/train_dragapult_v14_seq_diag.log
```

v14 训练日志必须能回答这些问题：

- `target_k>1` 是否有足够样本，multi-select loss 是否非零。
- forced rows 是否占比过高，复杂决策是否被单步 forced label 淹没。
- current-only ablation 和 full sequence 是否有差异。
- plan、turn-next、turn-seq、known info 的 loss/accuracy 是否有样本覆盖。
- Dragapult 的 DamageCounterAny/DCA 相关统计是否出现。
- per-epoch random smoke 是否暴露基础执行退化。

如果这些信号缺失，不要等 RR 结束，先修抽取或 loss。

## 9. v15 重写实验线

v15 目标是把游戏拆成 block/plan 级训练，减少单步 BC 对连续策略的统治。仍处于研究线，必须先通过 random 100% 和 trace 审查。

抽取 block corpus：

```bash
export V15_CORPUS=data/v15_blocks_0801_0815
python3 -u tools/v15_extract_blocks.py "$EPISODES" \
  --out "$V15_CORPUS" \
  --lb-csv "$LB_CSV" \
  --workers 12 \
  --history-k 12 \
  --plan-steps 4 \
  --date-from 2026-08-01 \
  --date-to 2026-08-15 \
  --progress-every 1000 \
  2>&1 | tee logs/v15_extract_blocks_0801_0815.log
```

训练 Dragapult plan policy：

```bash
mkdir -p checkpoints/v15_plan logs/v15_plan
CUDA_VISIBLE_DEVICES=1 python3 -u tools/v15_train_plan_policy.py \
  --corpus "$V15_CORPUS" \
  --archetype "Dragapult" \
  --min-score 900 \
  --history-k 12 \
  --plan-steps 4 \
  --width 384 \
  --layers 4 \
  --heads 6 \
  --batch-size 1024 \
  --epochs 8 \
  --device cuda:0 \
  --progress-every 200 \
  --action-weight 1.0 \
  --within-type-weight 0.6 \
  --route-weight 0.5 \
  --count-weight 0.3 \
  --multi-weight 1.5 \
  --type-weight 0.5 \
  --history-type-weight 0.3 \
  --known-type-weight 0.2 \
  --context-weight 0.3 \
  --plan-type-weight 0.4 \
  --plan-card-weight 0.3 \
  --plan-attack-weight 0.3 \
  --plan-context-weight 0.3 \
  --continue-weight 0.4 \
  --mode-weight 0.3 \
  --outcome-weight 0.1 \
  --out checkpoints/v15_plan/dragapult_v15_plan_0801_0815.pt \
  2>&1 | tee logs/v15_plan/train_dragapult_v15_plan_0801_0815.log
```

v15 random gate：

```bash
python3 tools/v15_random_gate.py checkpoints/v15_plan/dragapult_v15_plan_0801_0815.pt \
  --deck "$DRAG_DECK" \
  --games 300 \
  --workers 16 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 50 \
  --out-dir logs/v15_plan/random_dragapult_v15_plan_g300
```

如果没有达到 random 100%，必须进入 fixed-seed trace，不要直接调参：

```bash
python3 tools/v15_find_random_losses.py checkpoints/v15_plan/dragapult_v15_plan_0801_0815.pt \
  --deck "$DRAG_DECK" \
  --games 500 \
  --workers 16 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 50 \
  --recheck-losses \
  --out-csv logs/v15_plan/random_losses_dragapult_v15_plan.csv

python3 tools/v15_trace_game.py checkpoints/v15_plan/dragapult_v15_plan_0801_0815.pt \
  --deck "$DRAG_DECK" \
  --games 200 \
  --seed 20260817 \
  --target-outcome loss \
  --max-turns 700 \
  --progress-every 20 \
  --out-md logs/v15_plan/trace_dragapult_v15_plan_random_loss.md
```

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

如果某个新 opponent deck 反复击败我们，把它加入本地 RR：

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
export SUB_DIR=$PTCG_ROOT/submission/$(date +%Y%m%d)
mkdir -p "$SUB_DIR"
python3 tools/package_submission.py \
  --policy "$POLICY" \
  --deck "$DECK" \
  --cg-dir "$CG_DIR" \
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

## 12. 远端长任务模板

长脚本先在 `/tmp` 写好，再上传到 `ks` 执行。下面是一个最小模板。

本机生成脚本：

```bash
cat >/tmp/run_ptcg_job.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export PTCG_ROOT=/data/jie
export REPO=$PTCG_ROOT/ptcg_rl_git_v7_baseline_20260804
export EPISODES=$PTCG_ROOT/episodes_raw
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
chmod +x /tmp/run_ptcg_job.sh
```

上传并后台执行：

```bash
scp /tmp/run_ptcg_job.sh ks:/tmp/run_ptcg_job.sh
ssh ks 'nohup bash /tmp/run_ptcg_job.sh >/tmp/run_ptcg_job.nohup.log 2>&1 & echo $!'
ssh ks 'tail -f /tmp/run_ptcg_job.nohup.log'
```

拉回关键产物：

```bash
mkdir -p artifacts/ks_sync_$(date +%Y%m%d)
scp -r ks:/data/jie/ptcg_rl_git_v7_baseline_20260804/checkpoints/bc_aug_0801_0815 artifacts/ks_sync_$(date +%Y%m%d)/
scp -r ks:/data/jie/ptcg_rl_git_v7_baseline_20260804/logs/bc_aug_0801_0815 artifacts/ks_sync_$(date +%Y%m%d)/
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

### 14.1 为什么本地 RR 高，Kaggle 低？

常见原因：

- RR pool 混入弱 shadow，抬高候选胜率。
- ladder 前期 600-900 分也可能遇到刚提交的强模型。
- deck signature 对不上 checkpoint。
- top-k/mixed 数据污染了 deck-specific game plan。
- 只看均值，没有看 hard counter 和分段分布。
- random/RR 没有覆盖最新 Kaggle 环境。

### 14.2 为什么很多新架构没有提升？

之前的经验是：如果新信号只作为很小辅助 loss，通用单步 BC 会淹没它。v14/v15 必须在训练日志中证明 history、plan、multi-select、known info、same-turn sequence 的样本覆盖和 loss 都真实存在，否则最后 RR 才发现无效已经太晚。

### 14.3 什么时候用 winner-only？

只有在胜局样本足够、且胜局不是明显靠运气时才用。对结构性弱势 matchup，winner-only 常常样本太少，容易学到偶然胜局。更稳妥的流程是：先找到高质量 team/deck trajectory，再检查 trace，最后决定是否 winner-only。

### 14.4 什么时候回到历史代码？

如果目标是最后提交 sprint，优先用历史高光代码和当前 August 数据快速复现，例如 v10/v11/v12 的 deck-sig specialist。具体 commit、旧配置、线上高点记录看 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)。如果目标是研究长期能力，再用当前 v14/v15。

### 14.5 当前需要重点覆盖哪些 archetype？

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
