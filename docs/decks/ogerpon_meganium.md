# Ogerpon Meganium / 厄诡椪大竺葵

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_351_ogerpon_meganium.csv`  
Kaggle 统计 archetype: `Teal Mask Ogerpon`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|ab7e4b818773|1204.1|357|0.625|Dipam Chakraborty:89;Rmy:82;ANDPAD kaggler team:65;Majkel1337:40;EliKal:28|
|90abbfb0eee0|1154.7|90|0.489|palsystem:56;しんぴのしずく💧:22;THIRD PTCG Club:12|
|2bd9da52c43a|1115.6|108|0.635|James Cox & Henry Chao:58;e-toppo + kurupical:50|
|2a5072194fdf|1112.9|31|0.581|Oshbocker:31|
|081213bf731c|1111.8|53|0.585|Sixth Sense:53|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|65|0.492|1204.1|ab7e4b818773:65|
|1100-1199|290|0.579|1159.2|2bd9da52c43a:58;90abbfb0eee0:56;ab7e4b818773:56;081213bf731c:53;2a5072194fdf:31|
|1000-1099|930|0.535|1099.9|ab7e4b818773:227;5899c772bace:206;6205fc379380:111;2bd9da52c43a:50;d17573abc0e3:|
|900-999|284|0.479|993.5|5899c772bace:66;697a82e582d5:46;d17573abc0e3:44;7232316f9ca3:44;d6d4ab740380:39|
|800-899|8|0.250|853.9|0d77612de6ac:8|
|600-699|11|0.818|624.9|f4fe18b4203d:9;5899c772bace:1;28a52df999ce:1|

## 打法摘要

Grass engine 变体。和 Ogerpon Box 共享 Ogerpon 加速核心，但更强调进化/草系 engine 的稳定铺场。

## 关键牌

- Teal Mask Ogerpon ex
- Meganium line
- Energy Switch
- Grass-energy acceleration
- Boss's Orders

## 关键 combo / 决策点

- 先确保 Ogerpon 能稳定抽牌贴能，再决定是否转 Meganium route。
- 对快攻 matchup，不能为了完整 combo 牺牲第一轮攻击窗口。
- 对 wall/anti-ex matchup，需要尽早确认是否有非 ex 或抓 basic 的路。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Cynthia Garchomp|258|0.729|
|高胜率|Marnie Grimmsnarl|3182|0.725|
|高胜率|Team Rocket Mewtwo|143|0.650|
|高胜率|Dragapult|234|0.440|
|高胜率|Festival Lead|199|0.437|
|低胜率/接近五五|Crustle Wall|471|0.200|
|低胜率/接近五五|Mega Lopunny|609|0.223|
|低胜率/接近五五|Alakazam|597|0.338|

## 训练和评测注意事项

本地统计会和 Teal Mask Ogerpon 合并；如果要训练 Meganium specialist，应按 deck-sig 单独抽 corpus，不要和 Box/Ogerpon-Raging-Bolt 混在一起。

## 单独训练

目标是先训练一个 deck-sig specialist，而不是把多个不同 game plan 混在一起。当前自动填入的 `DECK_SIG=ab7e4b818773` 来自 2026-08-13 ladder 强签名表。

```bash
export CORPUS=${CORPUS:-data/bc_corpus_banded_latest}
export ARCHETYPE='Teal Mask Ogerpon'
export DECK_SIG=ab7e4b818773
export DECK='decks/pool_351_ogerpon_meganium.csv'
export OUT_DIR=checkpoints/decks
export LOG_DIR=logs/deck_train
mkdir -p "$OUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Teal Mask Ogerpon' \
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
  --save "$OUT_DIR/bc2_ogerpon_meganium_${DECK_SIG}_single.npz" \
  2>&1 | tee "$LOG_DIR/bc2_ogerpon_meganium_${DECK_SIG}_single.log"
```

如果该 deck 是 Stage-2、control 或需要明显全局视角的卡组，可以追加一组对照，不覆盖上面的 baseline:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 -u tools/bc2_train.py \
  --corpus "$CORPUS" \
  --archetype 'Teal Mask Ogerpon' \
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
  --save "$OUT_DIR/bc2_ogerpon_meganium_${DECK_SIG}_cross_stepplan.npz" \
  2>&1 | tee "$LOG_DIR/bc2_ogerpon_meganium_${DECK_SIG}_cross_stepplan.log"
```

训练日志里要重点看: train/val 是否同步下降、best epoch 是否不是过早停止、`first_action`/`policy_raw` 是否恶化、样本数是否足够、是否过滤到目标 `deck_sig`。

## Random 测试

先用 300 局快速 gate，候选提交前再跑 500 或 1000 局。random 不能代表 Kaggle 强度，但如果这里明显不稳，通常说明基础启动、进化或攻击流程没学好。

```bash
export POLICY="$OUT_DIR/bc2_ogerpon_meganium_${DECK_SIG}_single.npz"
mkdir -p logs/deck_eval/ogerpon_meganium

python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 300 \
  --workers 16 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/ogerpon_meganium/random_g300.log
```

候选提交前:

```bash
python3 tools/eval_bc.py "$POLICY" \
  --deck "$DECK" \
  --games 500 \
  --workers 32 \
  --progress-every 50 \
  --max-turns 700 \
  2>&1 | tee logs/deck_eval/ogerpon_meganium/random_g500.log
```

## 推荐 RR 测试

优先测两类池:

- `balanced` 池: 每个主流 archetype 至少 1-2 个高质量 shadow，避免低质量卡组拉高平均胜率。
- `latest ladder` 池: 按最新 Kaggle 分段权重保留环境主流签名，模拟从 600 分往上爬时可能遇到的对手。

该 deck 当前优先关注的低胜率/接近五五对手: Crustle Wall、Mega Lopunny、Alakazam、Festival Lead。

先构建一个每类 top2 的轻量 RR 池:

```bash
export RR_MANIFEST=${RR_MANIFEST:-logs/rr_pool_latest/filtered_balanced.csv}
export FOCUS_MANIFEST=logs/deck_eval/ogerpon_meganium/rr_pool_top2_per_arch.csv
mkdir -p logs/deck_eval/ogerpon_meganium

python3 tools/select_manifest_top_per_archetype.py \
  --manifest "$RR_MANIFEST" \
  --max-per-arch 2 \
  --out "$FOCUS_MANIFEST"
```

candidate-only RR:

```bash
python3 tools/eval_round_robin.py \
  --entry ogerpon_meganium="$POLICY:$DECK" \
  --manifest "$FOCUS_MANIFEST" \
  --candidate-only \
  --skip-bad-entries \
  --games 100 \
  --workers 32 \
  --max-turns 700 \
  --progress-every 20 \
  --out-csv logs/deck_eval/ogerpon_meganium/rr_top2_per_arch_g100.csv \
  2>&1 | tee logs/deck_eval/ogerpon_meganium/rr_top2_per_arch_g100.log
```

如果 RR 暴露出具体坏 matchup，再用 fixed-seed trace 复现。`OPP_ENTRY` 可以直接从 manifest 的 `eval_entry` 列复制:

```bash
export OPP_ENTRY='opponent_name=checkpoints/opponent.npz:logs/opponent_deck.csv'

python3 tools/trace_matchup_decisions.py \
  --candidate ogerpon_meganium="$POLICY:$DECK" \
  --opponent "$OPP_ENTRY" \
  --games 20 \
  --seed 20260817 \
  --max-turns 700 \
  --progress-every 1 \
  --out-prefix logs/deck_eval/ogerpon_meganium/trace_vs_bad_opp
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
export REPLAY_DIR=logs/kaggle_replay_${SUB_ID}_ogerpon_meganium
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
  --policy-name ogerpon_meganium \
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

- [YouTube search: Ogerpon Meganium deck](https://www.youtube.com/results?search_query=Ogerpon+Meganium+Pokemon+TCG+deck+gameplay)
- [Bilibili search: 厄诡椪 大竺葵](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E5%8E%84%E8%AF%A1%E6%A4%AA+%E5%A4%A7%E7%AB%BA%E8%91%B5+%E5%8D%A1%E7%BB%84)

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
