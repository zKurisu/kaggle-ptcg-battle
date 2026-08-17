# N's Zoroark ex / N 的索罗亚克 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_320_n_s_zoroark_ex.csv`  
Kaggle 统计 archetype: `N's Zoroark`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|aa80b030c069|1112.9|37|0.622|Oshbocker:36;da da:1|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|36|0.639|1112.9|aa80b030c069:36|
|900-999|1|0.000|976.7|aa80b030c069:1|

## 打法摘要

Dark evolution tempo。现实 Limitless 中占比很高，Kaggle 0813 样本不多但高分段有 aa80 签名。

## 关键牌

- N's Zorua
- N's Zoroark ex
- N's engine cards
- draw/search package

## 关键 combo / 决策点

- 早期优先铺 evolution basic，不要让 hand filter 消耗关键 evolution card。
- 对 Dragapult/Ogerpon 等主流局要记录 prize race 速度，而不是只看 random。
- 若要纳入 RR，需要为 aa80 或最新高分签名训练 shadow。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

本地 matchup 先验不足。需要从最新 episode 重算 `archetype_matchups.csv`。

## 训练和评测注意事项

以前 RR 覆盖不足，后续应作为 0813 之后环境池必备卡组。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: N's Zoroark ex gameplay](https://www.youtube.com/results?search_query=N%27s+Zoroark+ex+Pokemon+TCG+gameplay+deck+profile)
- [Bilibili search: N 的索罗亚克](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+N%E7%9A%84%E7%B4%A2%E7%BD%97%E4%BA%9A%E5%85%8B+ex+%E5%8D%A1%E7%BB%84)

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
