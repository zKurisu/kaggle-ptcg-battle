# Crustle Wall / 岩殿居蟹墙

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_341_crustle_mysterious_rock_inn.csv`  
Kaggle 统计 archetype: `Crustle Wall`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|3cd5039c59d2|1211.4|138|0.435|Dominic Peel & Rory Neville:53;tokk:43;MarineYY:39;Kailee Hamre:3|
|5489778f9e35|1163.2|61|0.607|LiamK:61|
|7ee600c6f769|1154.7|17|0.529|S4nkurero:6;palsystem:5;共逐荣光:5;JB Bryant:1|
|47756cdfd20f|1134.9|96|0.453|Zhu Liang:36;カントー地方マスター(KantoRegionMaster):36;ykhnkf:20;あなまか:3;tera358:1|
|b141ae295739|1134.9|79|0.620|カントー地方マスター(KantoRegionMaster):71;Katasonov Evgeny:8|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|53|0.453|1211.4|3cd5039c59d2:53|
|1100-1199|173|0.605|1163.2|b141ae295739:71;5489778f9e35:61;47756cdfd20f:36;7ee600c6f769:5|
|1000-1099|50|0.460|1027.7|3cd5039c59d2:43;7ee600c6f769:7|
|900-999|203|0.433|977.7|96d572411df0:65;3cd5039c59d2:42;3f267c892d7d:36;873baa869831:32;df868f5208e1:17|
|800-899|60|0.450|889.1|47756cdfd20f:36;4c9bdc537a23:16;b141ae295739:8|
|700-799|21|0.429|773.7|47756cdfd20f:20;a9d1e5ff787e:1|

## 打法摘要

Anti-ex wall。核心是让 Crustle 成型并保持 active，阻止对手 ex 攻击造成伤害，同时用 Superb Scissors 和资源压力拿奖。

## 关键牌

- Dwebble
- Crustle
- Mega Kangaskhan ex
- healing/HP buff package
- Boss's Orders

## 关键 combo / 决策点

- 对 ex-heavy decks，要优先保证 Dwebble -> Crustle，active 位置比普通 beatdown 更重要。
- 一旦 Crustle 建立，要避免无意义 retreat/switch，除非能直接创造奖赏。
- 对 Mega Lopunny/Alakazam 等能绕过或压制 wall 的局要提前准备 secondary route。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Teal Mask Ogerpon|471|0.800|
|高胜率|Festival Lead|196|0.622|
|高胜率|Dragapult|224|0.603|
|高胜率|Cynthia Garchomp|272|0.562|
|高胜率|Team Rocket Mewtwo|171|0.491|
|低胜率/接近五五|Mega Lopunny|397|0.166|
|低胜率/接近五五|Alakazam|681|0.326|
|低胜率/接近五五|Marnie Grimmsnarl|3704|0.476|

## 训练和评测注意事项

Crustle 对环境极敏感。近期复训被打爆时，不能只看 random；要看对 Lopunny/Alakazam/Ogerpon 的分段覆盖和高分 team 数据。

## 单独训练

目标是先训练一个 deck-sig specialist，而不是把多个不同 game plan 混在一起。当前自动填入的 `DECK_SIG=3cd5039c59d2` 来自 2026-08-13 ladder 强签名表。

```bash
export CORPUS=${CORPUS:-data/bc_corpus_banded_latest}
export ARCHETYPE='Crustle Wall'
export DECK_SIG=3cd5039c59d2
export DECK='decks/pool_341_crustle_mysterious_rock_inn.csv'
export OUT_DIR=checkpoints/decks
export LOG_DIR=logs/deck_train
mkdir -p "$OUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Crustle Wall' \
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
  --save "$OUT_DIR/bc2_crustle_wall_${DECK_SIG}_single.npz" \
  2>&1 | tee "$LOG_DIR/bc2_crustle_wall_${DECK_SIG}_single.log"
```

如果该 deck 是 Stage-2、control 或需要明显全局视角的卡组，可以追加一组对照，不覆盖上面的 baseline:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Crustle Wall' \
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
  --save "$OUT_DIR/bc2_crustle_wall_${DECK_SIG}_cross_stepplan.npz" \
  2>&1 | tee "$LOG_DIR/bc2_crustle_wall_${DECK_SIG}_cross_stepplan.log"
```

训练日志里要重点看: train/val 是否同步下降、best epoch 是否不是过早停止、`first_action`/`policy_raw` 是否恶化、样本数是否足够、是否过滤到目标 `deck_sig`。

## Random 测试

先用 300 局快速 gate，候选提交前再跑 500 或 1000 局。random 不能代表 Kaggle 强度，但如果这里明显不稳，通常说明基础启动、进化或攻击流程没学好。

```bash
export POLICY="$OUT_DIR/bc2_crustle_wall_${DECK_SIG}_single.npz"
mkdir -p logs/deck_eval/crustle_wall

python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 300 \
  --workers 16 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/crustle_wall/random_g300.log
```

候选提交前:

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 500 \
  --workers 32 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/crustle_wall/random_g500.log
```

## 推荐 RR 测试

优先测两类池:

- `balanced` 池: 每个主流 archetype 至少 1-2 个高质量 shadow，避免低质量卡组拉高平均胜率。
- `latest ladder` 池: 按最新 Kaggle 分段权重保留环境主流签名，模拟从 600 分往上爬时可能遇到的对手。

该 deck 当前优先关注的低胜率/接近五五对手: Mega Lopunny、Alakazam、Marnie Grimmsnarl、Team Rocket Mewtwo。

先构建一个每类 top2 的轻量 RR 池:

```bash
export RR_MANIFEST=${RR_MANIFEST:-logs/rr_pool_latest/filtered_balanced.csv}
export FOCUS_MANIFEST=logs/deck_eval/crustle_wall/rr_pool_top2_per_arch.csv
mkdir -p logs/deck_eval/crustle_wall

python3 tools/select_manifest_top_per_archetype.py \
  --manifest "$RR_MANIFEST" \
  --max-per-arch 2 \
  --out "$FOCUS_MANIFEST"
```

candidate-only RR:

```bash
python3 tools/eval_round_robin.py \
  --entry crustle_wall="$POLICY:$DECK" \
  --manifest "$FOCUS_MANIFEST" \
  --candidate-only \
  --skip-bad-entries \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/deck_eval/crustle_wall/rr_top2_per_arch_g100.csv \
  2>&1 | tee logs/deck_eval/crustle_wall/rr_top2_per_arch_g100.log
```

如果 RR 暴露出具体坏 matchup，再用 fixed-seed trace 复现。`OPP_ENTRY` 可以直接从 manifest 的 `eval_entry` 列复制:

```bash
export OPP_ENTRY='opponent_name=checkpoints/opponent.npz:logs/opponent_deck.csv'

python3 tools/trace_matchup_decisions.py \
  --candidate crustle_wall="$POLICY:$DECK" \
  --opponent "$OPP_ENTRY" \
  --games 20 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 1 \
  --out-prefix logs/deck_eval/crustle_wall/trace_vs_bad_opp
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
export REPLAY_DIR=logs/kaggle_replay_${SUB_ID}_crustle_wall
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
  --policy-name crustle_wall \
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

- [TCGMatch/YouTube VOD: NAIC 2026 Crustle article](https://tcgmatch.cl/blog/alloutblitzle-logra-ganar-el-naic-2026)
- [YouTube search: Crustle Mysterious Rock Inn gameplay](https://www.youtube.com/results?search_query=Crustle+Mysterious+Rock+Inn+Pokemon+TCG+gameplay)

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
