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
