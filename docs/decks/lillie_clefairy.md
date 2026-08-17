# Lillie's Clefairy ex / 莉莉艾皮皮 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_326_lillie_s_clefairy_ex.csv`  
Kaggle 统计 archetype: `暂未单独识别`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

当前本地 0812/0813 ladder 分段统计没有稳定签名。先不要把该 archetype 作为高权重 RR 结论来源。

## 分段分布

无稳定分段统计。

## 打法摘要

support attacker / toolbox。常作为 Ogerpon/Box secondary route，也可能单独成 deck。

## 关键牌

- Lillie's Clefairy ex
- Lillie support package
- psychic/fairy-style support
- switching

## 关键 combo / 决策点

- 要判断它是主轴还是 secondary attacker。
- 若只是 Box 中的 coverage card，训练时不能让它污染 Ogerpon 主计划。
- 单独成 deck 时需要独立抽取 deck-sig corpus。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

本地 matchup 先验不足。需要从最新 episode 重算 `archetype_matchups.csv`。

## 训练和评测注意事项

Kaggle 当前没有稳定单独 archetype 统计；先从 deck sig 和 card counts 识别。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: Lillie's Clefairy ex gameplay](https://www.youtube.com/results?search_query=Lillie%27s+Clefairy+ex+Pokemon+TCG+gameplay+deck+profile)
- [Bilibili search: 莉莉艾 皮皮 ex](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E8%8E%89%E8%8E%89%E8%89%BE+%E7%9A%AE%E7%9A%AE+ex+%E5%8D%A1%E7%BB%84)

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
