# Festival Lead / 祭典主唱

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_336_festival_lead.csv`  
Kaggle 统计 archetype: `Festival Lead`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|41ffa7894f40|1211.4|15|0.533|Dominic Peel & Rory Neville:15|
|e7a5089fca06|1071.6|9|0.444|EliKal:8;Gijs Smit:1|
|3bc4046822d2|1060.8|41|0.512|pokemon master:41|
|2cc31740f01d|1020.8|41|0.488|motono0223:41|
|e82dcbe62260|1014.6|53|0.340|Louis & Emile:27;sobameshi:12;Agent 33:10;hinox13:4|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|15|0.533|1211.4|41ffa7894f40:15|
|1000-1099|172|0.454|1071.6|3bc4046822d2:41;2cc31740f01d:41;f77b5c591d20:29;e82dcbe62260:27;2ca927a93744:26|
|900-999|40|0.450|974.3|e82dcbe62260:14;30fcc9b9b15a:11;663a9b4bd182:10;925421b795c4:2;5a7038d41d36:2|
|700-799|12|0.250|795.9|e82dcbe62260:12|

## 打法摘要

Festival engine 连击/节奏。重点是启用 Stadium/engine 后形成重复攻击，不能被普通 beatdown 的单步标签误导。

## 关键牌

- Festival Lead attacker
- Festival Grounds
- Dipplin/Applin line
- search/draw support

## 关键 combo / 决策点

- 优先找 Stadium/engine piece，而不是盲目把 attacker 推到 active。
- 一旦 engine 启动，要确保连续攻击和 prize race，不要因为低价值 draw 丢失攻击节奏。
- 对 Marnie/Ogerpon 有历史可打空间，但对 Lopunny/Crustle 会明显受压。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Marnie Grimmsnarl|1432|0.584|
|高胜率|Teal Mask Ogerpon|199|0.563|
|高胜率|Alakazam|225|0.489|
|高胜率|Crustle Wall|196|0.372|
|高胜率|Mega Lopunny|174|0.184|
|低胜率/接近五五|无足够非重复样本|-|-|

## 训练和评测注意事项

Festival 在 v10/v11 有过 900+ shadow/specialist 高点；样本少时更适合 top1 specialist。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Festival Lead - Pitch Black](https://www.youtube.com/watch?v=h_2c-O9HaBY)
- [YouTube search: Festival Lead gameplay](https://www.youtube.com/results?search_query=Festival+Lead+Pokemon+TCG+gameplay+deck+profile)

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
