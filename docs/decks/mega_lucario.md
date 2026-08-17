# Mega Lucario ex / 超级路卡利欧 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_345_mega_lucario_ex.csv`  
Kaggle 统计 archetype: `Mega Lucario`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|43d6d8b0fce9|1230.3|294|0.515|M Sato:35;Luca:33;wwwwwwwwwwwwwwwwwwwwwwwwwwwwww:28;Funky:28;Dipam Chakraborty:27|
|c62912d3e423|1163.2|29|0.517|LiamK:29|
|c570dd7eb87d|1111.8|11|0.727|Sixth Sense:11|
|8388b4e2ea06|1085.2|17|0.529|e-toppo + kurupical:17|
|509c3190ead7|963.1|1|0.000|Gyoukou:1|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|33|0.455|1230.3|43d6d8b0fce9:33|
|1100-1199|84|0.583|1163.2|43d6d8b0fce9:44;c62912d3e423:29;c570dd7eb87d:11|
|1000-1099|147|0.569|1085.2|43d6d8b0fce9:130;8388b4e2ea06:17|
|900-999|83|0.422|990.5|43d6d8b0fce9:78;d148ee0635b2:4;509c3190ead7:1|
|800-899|9|0.556|884.8|d148ee0635b2:8;43d6d8b0fce9:1|
|600-699|8|0.250|544.7|43d6d8b0fce9:8|

## 打法摘要

Fighting engine tempo。Lucario 在 Kaggle ladder 中表现相对均衡，关键是用 Solrock/Lunatone/Fighting Gong 一类 engine 保证 Mega Lucario 的攻击节奏。

## 关键牌

- Riolu
- Mega Lucario ex
- Lunatone
- Solrock
- Fighting Gong
- Premium Power Pro
- Poke Pad

## 关键 combo / 决策点

- 不要只把 Mega Lucario 当孤立 attacker；先建立 engine，再进入 prize race。
- Riolu 不同 card id 的签名差异很重要，训练和打包必须 deck-sig 对齐。
- 局面允许时，优先完成能保证下回合攻击的搜索/贴能，而不是当回合低价值攻击。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Mega Lopunny|219|0.895|
|高胜率|Alakazam|129|0.597|
|高胜率|Marnie Grimmsnarl|258|0.500|
|低胜率/接近五五|无足够非重复样本|-|-|

## 训练和评测注意事项

43d6 在高分段长期出现，是当前本地 RR 池和 shadow 池必须包含的 archetype。它也适合作为 climber 环境适配基线。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Mega Lucario w/Togekiss - Pitch Black](https://www.youtube.com/watch?v=wFwxEBBgDo0)
- [YouTube search: Mega Lucario ex gameplay](https://www.youtube.com/results?search_query=Mega+Lucario+ex+Pokemon+TCG+gameplay+deck+profile)

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
