# Alakazam Powerful Hand / 胡地

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_350_alakazam_powerful_hand.csv`  
Kaggle 统计 archetype: `Alakazam`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|ecb67fcd9c0b|1163.2|196|0.551|Jonathan Coletti:66;LiamK:54;Ars Noveau:40;Bianco Chiu:24;mealck:5|
|75f1d900d851|1163.2|57|0.579|LiamK:57|
|7f9a538936e3|1127.1|673|0.435|fishcat:54;masspeaks:46;vvs:44;miya:44;yuto083:44|
|c044e6a70529|1111.8|13|0.692|Sixth Sense:13|
|8c9d940eb0f3|1099.9|63|0.540|THIRD PTCG Club:61;mikelou1:2|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|148|0.568|1163.2|75f1d900d851:57;ecb67fcd9c0b:54;7f9a538936e3:24;c044e6a70529:13|
|1000-1099|703|0.513|1099.9|7f9a538936e3:295;ecb67fcd9c0b:108;589269aab27e:105;1781d29b9962:98;8c9d940eb0f3:|
|900-999|329|0.429|997.6|7f9a538936e3:245;aca54383eee7:48;ca082d1c1eab:12;1781d29b9962:11;e6f7fc04c530:6|
|800-899|79|0.468|890.8|7f9a538936e3:49;ecb67fcd9c0b:29;350e3ce5bbf5:1|
|700-799|66|0.455|786.7|7f9a538936e3:53;1781d29b9962:9;9e3eb89c1d43:4|
|600-699|2|0.500|390.2|7f9a538936e3:2|
|unknown|5|0.400||7f9a538936e3:5|

## 打法摘要

Stage-2 bench/control。胡地类卡组不是单纯把 active 打满伤害，而是通过进化链、后排攻击/控制和资源循环拖慢对手，要求模型理解 bench、active、手牌检索和 prize race 的长期关系。

## 关键牌

- Abra
- Kadabra
- Alakazam
- Buddy-Buddy Poffin
- Poke Pad
- Boss's Orders
- stadium/search package

## 关键 combo / 决策点

- 早期要优先铺 Abra/Kadabra，避免把 search/support 用在低价值手牌整理上。
- Kadabra 类 engine 的抽/找牌动作要与 evolve timing 对齐；很多失败 trace 表现为有进化/攻击窗口但选择 END 或无关 trainer。
- 胡地对 Dragapult、Marnie 等主流局容易被 tempo 压制，训练时要审查 attack miss、evolve miss、target miss 是否真实存在。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Crustle Wall|681|0.674|
|高胜率|Teal Mask Ogerpon|597|0.662|
|高胜率|Cynthia Garchomp|560|0.648|
|高胜率|Mega Lopunny|503|0.573|
|高胜率|Festival Lead|225|0.511|
|低胜率/接近五五|Team Rocket Mewtwo|372|0.223|
|低胜率/接近五五|Mega Lucario|129|0.403|
|低胜率/接近五五|Dragapult|343|0.414|
|低胜率/接近五五|Marnie Grimmsnarl|6048|0.433|

## 训练和评测注意事项

7f9 曾在线上达到 900+，但环境变化后复训不稳。文档中强签名要跟随 0812/0813 高分段，而不是固定旧 7f9。

## 单独训练

目标是先训练一个 deck-sig specialist，而不是把多个不同 game plan 混在一起。当前自动填入的 `DECK_SIG=ecb67fcd9c0b` 来自 2026-08-13 ladder 强签名表。

```bash
export CORPUS=${CORPUS:-data/bc_corpus_banded_latest}
export ARCHETYPE='Alakazam'
export DECK_SIG=ecb67fcd9c0b
export DECK='decks/pool_350_alakazam_powerful_hand.csv'
export OUT_DIR=checkpoints/decks
export LOG_DIR=logs/deck_train
mkdir -p "$OUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Alakazam' \
  --deck-sig "$DECK_SIG" \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --date-from 2026-08-01 \
  --date-to 2026-08-15 \
  --epochs 8 \
  --batch-size 1024 \
  --width 512 \
  --arch pointer \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --split-by-game \
  --load-progress-every 200000 \
  --checkpoint-every 1 \
  --save "$OUT_DIR/bc2_alakazam_${DECK_SIG}_single.npz" \
  2>&1 | tee "$LOG_DIR/bc2_alakazam_${DECK_SIG}_single.log"
```

如果该 deck 是 Stage-2、control 或需要明显全局视角的卡组，可以追加一组对照，不覆盖上面的 baseline:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Alakazam' \
  --deck-sig "$DECK_SIG" \
  --score-bands 900-999 1000-1099 1100-1199 1200+ \
  --date-from 2026-08-01 \
  --date-to 2026-08-15 \
  --epochs 8 \
  --batch-size 768 \
  --width 768 \
  --arch cross_attn \
  --state-layers 2 \
  --step-plan \
  --step-plan-loss-weight 0.2 \
  --win-weight 1.5 \
  --loss-weight 0.4 \
  --draw-weight 0.8 \
  --split-by-game \
  --load-progress-every 200000 \
  --checkpoint-every 1 \
  --save "$OUT_DIR/bc2_alakazam_${DECK_SIG}_cross_stepplan.npz" \
  2>&1 | tee "$LOG_DIR/bc2_alakazam_${DECK_SIG}_cross_stepplan.log"
```

训练日志里要重点看: train/val 是否同步下降、best epoch 是否不是过早停止、`first_action`/`policy_raw` 是否恶化、样本数是否足够、是否过滤到目标 `deck_sig`。

## Random 测试

先用 300 局快速 gate，候选提交前再跑 500 或 1000 局。random 不能代表 Kaggle 强度，但如果这里明显不稳，通常说明基础启动、进化或攻击流程没学好。

```bash
export POLICY="$OUT_DIR/bc2_alakazam_${DECK_SIG}_single.npz"
mkdir -p logs/deck_eval/alakazam

python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 300 \
  --workers 16 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/alakazam/random_g300.log
```

候选提交前:

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 500 \
  --workers 32 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/alakazam/random_g500.log
```

## 推荐 RR 测试

优先测两类池:

- `balanced` 池: 每个主流 archetype 至少 1-2 个高质量 shadow，避免低质量卡组拉高平均胜率。
- `latest ladder` 池: 按最新 Kaggle 分段权重保留环境主流签名，模拟从 600 分往上爬时可能遇到的对手。

该 deck 当前优先关注的低胜率/接近五五对手: Team Rocket Mewtwo、Mega Lucario、Dragapult、Marnie Grimmsnarl。

先构建一个每类 top2 的轻量 RR 池:

```bash
export RR_MANIFEST=${RR_MANIFEST:-logs/rr_pool_latest/filtered_balanced.csv}
export FOCUS_MANIFEST=logs/deck_eval/alakazam/rr_pool_top2_per_arch.csv
mkdir -p logs/deck_eval/alakazam

python3 tools/select_manifest_top_per_archetype.py \
  --manifest "$RR_MANIFEST" \
  --max-per-arch 2 \
  --out "$FOCUS_MANIFEST"
```

candidate-only RR:

```bash
python3 tools/eval_round_robin.py \
  --entry alakazam="$POLICY:$DECK" \
  --manifest "$FOCUS_MANIFEST" \
  --candidate-only \
  --skip-bad-entries \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/deck_eval/alakazam/rr_top2_per_arch_g100.csv \
  2>&1 | tee logs/deck_eval/alakazam/rr_top2_per_arch_g100.log
```

如果 RR 暴露出具体坏 matchup，再用 fixed-seed trace 复现。`OPP_ENTRY` 可以直接从 manifest 的 `eval_entry` 列复制:

```bash
export OPP_ENTRY='opponent_name=checkpoints/opponent.npz:logs/opponent_deck.csv'

python3 tools/trace_matchup_decisions.py \
  --candidate alakazam="$POLICY:$DECK" \
  --opponent "$OPP_ENTRY" \
  --games 20 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 1 \
  --out-prefix logs/deck_eval/alakazam/trace_vs_bad_opp
```

## Kaggle episode 动态回放

可以引入，但不要把它当成可离线复现的唯一记录。Kaggle 网页动态 replay 依赖登录态、网页脚本和当前 UI，静态 Markdown 里通常只能放 episode/submission 链接；真正可复现的材料应下载 replay JSON 并写入本地日志。

人工查看流程:

1. 打开 Kaggle 比赛页面的 Submissions 或 Episodes。
2. 找到目标 `submission_id` 的 episode。
3. 打开 episode detail / replay 页面，逐回合看 setup、attach、evolve、ability、attack、target 和最终 status。
4. 把关键 episode id 写回本文件或 `AGENT_HANDOFF.md`，并下载 replay JSON 做本地 trace。

本地 replay 分析:

```bash
export SUB_ID=REPLACE_WITH_KAGGLE_SUBMISSION_ID
export REPLAY_DIR=logs/kaggle_replay_${SUB_ID}_alakazam
mkdir -p "$REPLAY_DIR"

python3 tools/analyze_kaggle_replays.py "$SUB_ID" \
  --deck "$DECK" \
  --known-decks-dir logs/ladder_pool_0805_all/decks \
  --cache-dir "$REPLAY_DIR/cache" \
  --out "$REPLAY_DIR/episodes.csv" \
  --summary-out "$REPLAY_DIR/summary_by_opponent_deck_sig.csv" \
  --group-by opponent_deck_sig \
  --max-episodes 200 \
  --write-opponent-decks \
  --opponent-decks-dir "$REPLAY_DIR/opponent_decks" \
  --progress-every 10
```

把 live replay 中遇到的新对手转成本地 RR:

```bash
python3 tools/make_kaggle_opp_round_robin_cmd.py \
  --policy-name alakazam \
  --policy "$POLICY" \
  --deck "$DECK" \
  --opp-dir "$REPLAY_DIR/opponent_decks" \
  --games 100 \
  --progress-every 20 \
  --out-csv "$REPLAY_DIR/local_vs_live_opponents_g100.csv" \
  > "$REPLAY_DIR/run_live_opponent_rr.sh"

bash "$REPLAY_DIR/run_live_opponent_rr.sh"
```

这样动态 replay 的结论会落到 `episodes.csv`、`summary_by_opponent_deck_sig.csv`、opponent deck CSV 和本地 RR 结果里，后续才能比较“线上输在哪里”和“本地是否能复现”。


评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Alakazam w/Toucannon - Pitch Black](https://www.youtube.com/watch?v=I4IsyYb7WtI)
- [YouTube search: Alakazam Powerful Hand gameplay](https://www.youtube.com/results?search_query=Alakazam+Powerful+Hand+Pokemon+TCG+gameplay+deck+profile)

如果链接是搜索入口，需要优先选择 2026 轮换后、与当前 Kaggle 可用卡池接近的视频；不要直接把旧环境打法写成强规则。

## 卡面素材

推荐只把卡图作为本地研究缓存或文档阅读辅助，不要把下载的卡图提交进仓库或 submission 包。

|来源|链接|用途|
|---|---|---|
|Pokémon TCG API|[Pokémon TCG API](https://docs.pokemontcg.io/api-reference/cards/card-object/)|`images.small` / `images.large` 字段可拿到卡图 URL，适合本地研究缓存。|
|pokemon-tcg-data|[pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data)|Pokémon TCG API 的原始 JSON 数据，可 clone 或下载 release；优先当 card metadata/image URL 索引用。|
|PokemonCard.io|[PokemonCard.io](https://pokemoncard.io/)|Deck 页面有 `Download all Images`，但页面版权说明显示卡图/卡文仍归 Pokémon/Nintendo/Game Freak 等权利方。|
|Limitless TCG|[Limitless TCG](https://limitlesstcg.com/)|适合查真实比赛 decklist、卡组占比和卡图预览；不要把卡图直接提交进仓库。|

## 后续可转成规则/trace 的问题

- 当前强签名是否真的覆盖了 600 -> 1100 的爬分阶段，而不只是高分段幸存局？
- 失败 trace 中是否存在明确 miss: setup/evolve/attach/ability/attack/target/reveal memory？
- 该 archetype 的强胜局是稳定策略，还是对手事故/抽牌运气？
- 如果要加 rule overlay，触发条件必须能从 observation/legal options 中稳定判断，且 fixed-seed replay 应证明行为改变。
