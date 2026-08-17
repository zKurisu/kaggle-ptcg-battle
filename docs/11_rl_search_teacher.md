# 11 - RL、search 与 teacher rollout

这一章解释 BC 之后的三条路线：search、teacher rollout、RL。它们都比 BC 更昂贵，也更容易误导，因此必须建立在前面几章的评测和 trace 之上。

## 1. RL 是什么

RL 是 reinforcement learning，强化学习。它不再只模仿已有标签，而是让策略通过对局结果更新自己。

基本对象：

- policy：当前策略，给每个合法动作打分。
- value：估计当前局面的长期胜率或回报。
- reward：胜负和过程奖励。
- rollout：用当前策略实际打一批对局。
- advantage：某个动作比当前 value 预期好多少。
- PPO：一种限制策略更新幅度的 RL 算法，避免一步改太大。

BC 学“别人做了什么”。RL 学“我这么做之后结果是否更好”。

## 2. 为什么不能把 RL 当微调按钮

这个项目的历史经验是：小规模 PPO 或弱局微调通常不能修复结构性弱点，反而容易损坏通用能力。

原因：

- rollout 数太少，回报噪声很大。
- 对手池不准，学到本地 shadow 的漏洞。
- value head 没训练好，search/RL 会放大错误估值。
- 弱势局的偶然胜场不等于可复现策略。
- 微调会把原本稳定 BC 拉离高质量行为分布。

因此当前建议：

- BC checkpoint 可作为 baseline、opponent、模型形状模板。
- 若要 RL，优先 `--init-mode random` 或大改 scratch 训练。
- 不再做小幅 BC/RL fine-tune 来修结构性弱点。

## 3. search 适合做什么

search 可以在局部局面试动作，但它依赖后续 rollout 和 value。

适合：

- 生成 teacher label。
- 比较几个候选 root action。
- 验证某条规则是否真的有机会赢。
- 排查“当前模型是不是错过明显动作”。

不适合：

- 直接把慢 search 塞进 Kaggle submission。
- 在 value 很差时相信 search 结果。
- 只看单步局部收益。

动作老师入口：

```bash
python3 tools/search_action_teacher.py --help
```

典型流程是先建立 weakness state bank，再对这些状态跑 search：

```bash
python3 tools/build_weakness_state_bank.py --help
python3 tools/search_action_teacher.py \
  --bank-jsonl logs/weakness_bank.jsonl \
  --candidate cand=checkpoints/candidate.npz:decks/candidate.csv \
  --opponent weak=checkpoints/weak.npz:decks/weak.csv \
  --limit-states 200 \
  --rollouts-per-action 16 \
  --rollout-horizon 12 \
  --root-top-options 8 \
  --workers 16 \
  --progress-every 20 \
  --out-actions-csv logs/search_teacher_actions.csv \
  --out-best-csv logs/search_teacher_best.csv \
  --out-teacher-jsonl logs/search_teacher.jsonl
```

## 4. teacher rollout

teacher rollout 是中间路线：先用规则/search/planner 生成成功对局，再把这些对局蒸馏回训练数据。

这比“直接用稀疏胜局 BC”更可靠，因为 teacher 至少能保证目标行为在当前 simulator 中可执行。

生成任务见上一章的 `plan_rollout_teacher_jobs.py`。生成后要检查：

- win rate 是否真的提升到目标区间。
- 成功是否来自可复现路线，而不是对手事故。
- 生成数据是否覆盖多个 opener 和不同 opponent sig。
- 规则是否只在目标 matchup 生效，还是无意污染全局。

## 5. PPO 训练入口

当前主要入口是：

```bash
python3 tools/rl_train_league.py --help
```

一个 scratch PPO 模板：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u tools/rl_train_league.py \
  --policy-init checkpoints/base_template.npz \
  --init-mode random \
  --deck decks/candidate.csv \
  --save checkpoints/rl/candidate_scratch_best.npz \
  --save-policy best \
  --save-final checkpoints/rl/candidate_scratch_final.npz \
  --checkpoint-dir checkpoints/rl/candidate_scratch_ckpts \
  --metrics-csv logs/rl/candidate_scratch_metrics.csv \
  --opponent-manifest logs/weakness_or_league_manifest.csv \
  --skip-bad-entries \
  --opponent-weight-mode adaptive_lossrate \
  --iterations 200 \
  --games-per-iter 256 \
  --rollout-workers 32 \
  --rollout-temperature 1.0 \
  --rollout-temperature-final 0.3 \
  --rollout-top-k 12 \
  --rollout-top-k-final 6 \
  --ppo-epochs 3 \
  --minibatch 4096 \
  --lr 0.0001 \
  --target-kl 0.03 \
  --advantage-normalization opponent \
  --advantage-clip 5.0 \
  --entropy-coef 0.02 \
  --entropy-final-coef 0.005 \
  --schedule-iters 200 \
  --win-reward 1.0 \
  --loss-reward -1.0 \
  --draw-reward 0.0 \
  --shaping-weight 0.0 \
  --max-turns 700 \
  --save-every 10 \
  --progress-every 1 \
  --device cuda:0 \
  --cuda-memory-gb 24 \
  2>&1 | tee logs/rl/candidate_scratch.log
```

这只是模板，不是保证有效的配方。真正训练时要根据显存、CPU worker、对手池质量和中途指标调整。

## 6. PPO 中途必须看什么

不要等 200 iter 全跑完才看结果。中途看：

- 每个 iteration 的 rollout win rate。
- 对不同 opponent 的 win rate。
- adaptive weight 是否集中到真实弱点。
- policy entropy 是否过早塌缩。
- approximate KL 是否频繁超过 `target-kl`。
- value loss 是否失控。
- 长局/timeout 是否增加。

如果这些指标异常，继续训练只是在放大错误。

## 7. RL 后如何验证

用和 BC 相同的验证链：

```bash
python3 tools/eval_bc.py checkpoints/rl/candidate_scratch_best.npz \
  --deck decks/candidate.csv \
  --games 300 \
  --workers 8 \
  --max-turns 700 \
  --progress-every 50

python3 tools/eval_baseline_delta.py \
  --baseline bc=checkpoints/base_template.npz:decks/candidate.csv \
  --candidate rl=checkpoints/rl/candidate_scratch_best.npz:decks/candidate.csv \
  --opponent-manifest logs/weakness_or_league_manifest.csv \
  --games 80 \
  --workers 16 \
  --max-turns 700 \
  --skip-bad-entries \
  --out-csv logs/rl/candidate_vs_bc_delta.csv
```

然后对最差 matchup 做 trace。RL 如果不能解释“为什么赢得更多”，就不能信任。

## 8. Kaggle submission 约束

训练阶段可以很大，但提交端不能：

- 不能依赖 PyTorch。
- 不能跑大量 rollout。
- 不能超时。
- 不能假设有 GPU。

所以最终仍要导出轻量 `.npz`，由 `ptcg_rl/numpy_policy.py` 在 `main.py` 中推理。

## 9. 本章练习

先不要直接跑大 PPO。按顺序做：

1. 用第 09 章找一个明确弱点。
2. 用第 10 章确认是否有成功数据或规则 seed。
3. 用 search_action_teacher 在少量 state 上检查动作老师是否有价值。
4. 用 `rl_train_league.py --dry-run` 检查模型、deck 和 opponent manifest 能加载。
5. 只跑 5 到 10 个 iteration，看 metrics 是否合理。

下一章：[12 - PTCG 玩法与 cg/C++ 引擎深读](12_cg_engine_source.md)
