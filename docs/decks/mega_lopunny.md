# Mega Lopunny ex / 超级长耳兔 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_353_mega_lopunny_ex.csv`  
Kaggle 统计 archetype: `Mega Lopunny`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|f1445356c3a7|1107.3|938|0.483|DavidFreelan:105;dasfds:79;Yopta:77;M Sato:72;motono0223:71|
|276707c0fdb4|1041.0|104|0.500|kanno:89;Leolazz:15|
|25d1bd401fe7|992.4|11|0.364|djschmit:11|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|79|0.436|1107.3|f1445356c3a7:79|
|1000-1099|661|0.508|1098.7|f1445356c3a7:572;276707c0fdb4:89|
|900-999|227|0.420|999.4|f1445356c3a7:201;276707c0fdb4:15;25d1bd401fe7:11|
|800-899|37|0.568|855.2|f1445356c3a7:37|
|700-799|49|0.449|798.1|f1445356c3a7:49|

## 打法摘要

Mega attacker + consistency。Kaggle 中 f144 是主流高覆盖签名，核心是稳定铺场并持续用 Mega attacker 压 prize race。

## 关键牌

- Mega Lopunny ex
- Dunsparce
- Dudunsparce/Dudunsparce ex
- consistency engine
- Boss's Orders

## 关键 combo / 决策点

- Dunsparce/Dudunsparce engine 是稳定性来源，不能被单纯 attacker preference 淹没。
- 对 Crustle/Ogerpon/Festival 有历史优势，但对 Mega Lucario/Cynthia 等可能吃力。
- 训练时应保留 top1/top2 specialist，对 mixed 的收益要谨慎。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Crustle Wall|397|0.826|
|高胜率|Festival Lead|174|0.816|
|高胜率|Teal Mask Ogerpon|609|0.772|
|高胜率|Marnie Grimmsnarl|2131|0.585|
|高胜率|Dragapult|232|0.491|
|低胜率/接近五五|Mega Lucario|219|0.105|
|低胜率/接近五五|Cynthia Garchomp|182|0.181|
|低胜率/接近五五|Alakazam|503|0.427|

## 训练和评测注意事项

Lopunny 旧 v14 曾有可提交表现，但 0813 分段显示高分 ceiling 不如 Dragapult/Ogerpon/Lucario/Crustle。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: Mega Lopunny ex gameplay](https://www.youtube.com/results?search_query=Mega+Lopunny+ex+Pokemon+TCG+gameplay+deck+profile)
- [Bilibili search: 超级长耳兔 ex](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E8%B6%85%E7%BA%A7%E9%95%BF%E8%80%B3%E5%85%94+ex+%E5%8D%A1%E7%BB%84)

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
