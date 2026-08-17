# Slowking Seek Inspiration / 呆壳兽灵感检索

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_322_slowking_seek_inspiration.csv`  
Kaggle 统计 archetype: `暂未单独识别`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

当前本地 0812/0813 ladder 分段统计没有稳定签名。先不要把该 archetype 作为高权重 RR 结论来源。

## 分段分布

无稳定分段统计。

## 打法摘要

慢速 control/search。真实环境中 Slowking 有一定占比，但 Kaggle 本地分类暂未稳定覆盖。

## 关键牌

- Slowpoke
- Slowking
- Seek Inspiration
- control/search package

## 关键 combo / 决策点

- 核心是通过 Slowking 的检索/控制拉长游戏，而不是急于 race。
- 模型需要理解 hand/deck/resource 的长期价值，否则很容易把 search 用成随机抽牌。
- 先作为 RR 池补全和 rule seed，不优先提交。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

本地 matchup 先验不足。需要从最新 episode 重算 `archetype_matchups.csv`。

## 训练和评测注意事项

需要新增 deck_plans entry 和 archetype classifier 后再做训练。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: Slowking Seek Inspiration gameplay](https://www.youtube.com/results?search_query=Slowking+Seek+Inspiration+Pokemon+TCG+gameplay+deck+profile)
- [Bilibili search: 呆壳兽 卡组](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E5%91%86%E5%A3%B3%E5%85%BD+%E5%8D%A1%E7%BB%84+%E5%AF%B9%E6%88%98)

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
