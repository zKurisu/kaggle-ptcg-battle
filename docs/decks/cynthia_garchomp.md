# Cynthia's Garchomp ex / 竹兰烈咬陆鲨 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_332_cynthia_s_garchomp_ex.csv`  
Kaggle 统计 archetype: `Cynthia Garchomp`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|52f467394857|1125.7|76|0.539|Octavi Grau:67;Charmander & Meowth:5;Pokemon Garı Gacıyo:2;Yudai Ueno:2|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|67|0.567|1125.7|52f467394857:67|
|900-999|7|0.429|941.6|52f467394857:7|
|600-699|2|0.000|620.1|52f467394857:2|

## 打法摘要

Stage-2 linear tempo。Gabite/Garchomp 线要求早期稳定进化，部分弱局需要 Spiritomb 等非 ex/counter attacker 参与。

## 关键牌

- Cynthia's Gible
- Cynthia's Gabite
- Cynthia's Garchomp ex
- Cynthia's Spiritomb
- search engine

## 关键 combo / 决策点

- Gabite 的搜索/engine 需要在进化链中优先完成。
- 对 Crustle 时，Spiritomb active 的成功 trace 曾出现，说明不能只让 ex attacker 打进 wall。
- 对 Lopunny 有历史优势，但对 Ogerpon/Alakazam/Crustle 容易被结构克制。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Mega Lopunny|182|0.819|
|高胜率|Team Rocket Mewtwo|122|0.566|
|高胜率|Marnie Grimmsnarl|2278|0.538|
|高胜率|Dragapult|153|0.490|
|高胜率|Crustle Wall|272|0.438|
|低胜率/接近五五|Teal Mask Ogerpon|258|0.271|
|低胜率/接近五五|Alakazam|560|0.352|

## 训练和评测注意事项

样本量比主流 archetype 少，训练时要避免 winner-only 后数据过窄；RR 池中必须包含它作为 Lopunny counter。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Cynthia's Garchomp - Pitch Black](https://www.youtube.com/watch?v=GqeCAKzXMlg)
- [YouTube search: Cynthia Garchomp gameplay](https://www.youtube.com/results?search_query=Cynthia%27s+Garchomp+ex+Pokemon+TCG+gameplay+deck+profile)

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
