# Mega Starmie ex / 超级宝石海星 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_362_mega_starmie_ex.csv`  
Kaggle 统计 archetype: `Mega Starmie`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|a31f17401ea1|976.3|2|0.000|takusemba:2|
|f2c3693911b6|921.7|2|0.000|TakahiroHon:2|
|f8850204c5f3|865.0|2|0.500|Magikarp:2|
|b695906e69e0|738.1|1|0.000|Hiroyasu123:1|
|2b691196226a|393.4|3|0.000|shini47:3|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|900-999|4|0.000|976.3|a31f17401ea1:2;f2c3693911b6:2|
|800-899|2|0.500|865.0|f8850204c5f3:2|
|700-799|1|0.000|738.1|b695906e69e0:1|
|600-699|3|0.000|393.4|2b691196226a:3|

## 打法摘要

Mega attacker + Dusknoir damage-counter pressure。Kaggle 样本极少，更多是未来扩展池的候选。

## 关键牌

- Staryu
- Mega Starmie ex
- Duskull
- Dusclops
- Dusknoir
- Hilda
- Grand Tree
- Wally's Compassion

## 关键 combo / 决策点

- Staryu/Starmie 主线和 Duskull 伤害线需要同步建立。
- Hilda/Grand Tree 类进化搜索是关键，不应被当成普通 draw。
- Dusknoir damage counter 要和下一次攻击的 KO map 一起计算。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

本地 matchup 先验不足。需要从最新 episode 重算 `archetype_matchups.csv`。

## 训练和评测注意事项

样本不足时不要用它拉高/拉低 RR 总均值；先建立合法 shadow 和 random 质量。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: Mega Starmie ex gameplay](https://www.youtube.com/results?search_query=Mega+Starmie+ex+Pokemon+TCG+gameplay+deck+profile)
- [Bilibili search: 超级宝石海星](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E8%B6%85%E7%BA%A7%E5%AE%9D%E7%9F%B3%E6%B5%B7%E6%98%9F+ex+%E5%8D%A1%E7%BB%84)

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
