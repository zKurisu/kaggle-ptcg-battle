# Dragapult ex / 多龙巴鲁托 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_284_dragapult_ex.csv`  
Kaggle 统计 archetype: `Dragapult`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|cc2e995b5ad0|1218.2|1115|0.536|Petit Canard:132;Kh0a:117;Аzat Akhtyamov:71;JB Bryant:66;wwwwwwwwwwwwwwwwwwwwwwwwwwwwww:65|
|140f7d8b2f09|1163.2|125|0.584|LiamK:53;LumenLiquidity:52;Klein Houmani:20|
|3367e772eebc|1154.7|85|0.624|palsystem:85|
|3a1f338ace6e|1154.7|57|0.702|palsystem:57|
|7ac0181b46a0|1133.5|37|0.649|LumenLiquidity:37|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|64|0.578|1218.2|cc2e995b5ad0:64|
|1100-1199|611|0.625|1163.2|cc2e995b5ad0:191;7b80ccf70f9f:127;140f7d8b2f09:105;3367e772eebc:85;3a1f338ace6e:|
|1000-1099|783|0.512|1085.2|cc2e995b5ad0:556;c685c3674825:65;7b80ccf70f9f:44;d622a9f2bca0:34;b964eb134117:33|
|900-999|283|0.502|996.1|cc2e995b5ad0:230;140f7d8b2f09:20;d112d6fbe57d:13;d20580f80287:9;6763881ee2d5:8|
|800-899|53|0.509|898.6|cc2e995b5ad0:51;060811b4b6bf:1;1ca4cba76fe9:1|
|unknown|23|0.652||cc2e995b5ad0:23|

## 打法摘要

Stage-2 spread/race。核心是尽快建立 Dreepy -> Drakloak -> Dragapult ex，同时用 Drakloak 的抽牌/检索能力保证连续进化，再用 Phantom Dive 的主伤害和 bench damage 制造跨回合奖赏路线。

## 关键牌

- Dreepy
- Drakloak
- Dragapult ex
- Duskull/Dusclops/Dusknoir
- Crispin
- Counter Catcher

## 关键 combo / 决策点

- Drakloak 在线时，先尽量使用可用的抽/看牌能力，再进化成 Dragapult ex，避免提前进化导致少一次资源选择。
- bench damage 不应只看当回合击倒；要结合 Dusknoir/Dusclops 的伤害指示物、自身下一次 200 点主伤害和 Boss/Counter Catcher 的目标路线。
- 面对低 HP bench，优先考虑是否能用 60 点 bench damage 直接拿奖；面对 220 HP 目标，可以先铺 20/60 点，为后续 200 点攻击或自爆伤害创造 KO。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Alakazam|343|0.586|
|高胜率|Teal Mask Ogerpon|234|0.560|
|高胜率|Marnie Grimmsnarl|1569|0.526|
|高胜率|Cynthia Garchomp|153|0.510|
|高胜率|Mega Lopunny|232|0.509|
|低胜率/接近五五|Crustle Wall|224|0.397|

## 训练和评测注意事项

本项目历史上 Dragapult random 经常能接近 99%，但 RR 和线上仍弱，说明不能只做单步 attack/attach 标签；需要 sequence trace、伤害分配诊断和 fixed-seed loss replay。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Dragapult w/Dark Bell - Pitch Black](https://www.youtube.com/watch?v=7ZDRAA9FW_w)
- [PokeDronks Dragapult deck/video primer](https://pokemoncard.io/deck/dragapult-pokedrunks-121450)

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
