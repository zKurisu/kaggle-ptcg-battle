# Alakazam Powerful Hand / 胡地

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_350_alakazam_powerful_hand.csv`  
Kaggle 统计 archetype: `Alakazam`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|ecb67fcd9c0b|1163.2|196|0.551|Jonathan Coletti:66;LiamK:54;Ars Noveau:40;Bianco Chiu:24;mealck:5|
|75f1d900d851|1163.2|57|0.579|LiamK:57|
|7f9a538936e3|1127.1|673|0.435|fishcat:54;masspeaks:46;vvs:44;miya:44;yuto083:44|
|c044e6a70529|1111.8|13|0.692|Sixth Sense:13|
|8c9d940eb0f3|1099.9|63|0.540|THIRD PTCG Club:61;mikelou1:2|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|148|0.568|1163.2|75f1d900d851:57;ecb67fcd9c0b:54;7f9a538936e3:24;c044e6a70529:13|
|1000-1099|703|0.513|1099.9|7f9a538936e3:295;ecb67fcd9c0b:108;589269aab27e:105;1781d29b9962:98;8c9d940eb0f3:|
|900-999|329|0.429|997.6|7f9a538936e3:245;aca54383eee7:48;ca082d1c1eab:12;1781d29b9962:11;e6f7fc04c530:6|
|800-899|79|0.468|890.8|7f9a538936e3:49;ecb67fcd9c0b:29;350e3ce5bbf5:1|
|700-799|66|0.455|786.7|7f9a538936e3:53;1781d29b9962:9;9e3eb89c1d43:4|
|600-699|2|0.500|390.2|7f9a538936e3:2|
|unknown|5|0.400||7f9a538936e3:5|

## 打法摘要

Stage-2 bench/control。胡地类卡组不是单纯把 active 打满伤害，而是通过进化链、后排攻击/控制和资源循环拖慢对手，要求模型理解 bench、active、手牌检索和 prize race 的长期关系。

## 关键牌

- Abra
- Kadabra
- Alakazam
- Buddy-Buddy Poffin
- Poke Pad
- Boss's Orders
- stadium/search package

## 关键 combo / 决策点

- 早期要优先铺 Abra/Kadabra，避免把 search/support 用在低价值手牌整理上。
- Kadabra 类 engine 的抽/找牌动作要与 evolve timing 对齐；很多失败 trace 表现为有进化/攻击窗口但选择 END 或无关 trainer。
- 胡地对 Dragapult、Marnie 等主流局容易被 tempo 压制，训练时要审查 attack miss、evolve miss、target miss 是否真实存在。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Crustle Wall|681|0.674|
|高胜率|Teal Mask Ogerpon|597|0.662|
|高胜率|Cynthia Garchomp|560|0.648|
|高胜率|Mega Lopunny|503|0.573|
|高胜率|Festival Lead|225|0.511|
|低胜率/接近五五|Team Rocket Mewtwo|372|0.223|
|低胜率/接近五五|Mega Lucario|129|0.403|
|低胜率/接近五五|Dragapult|343|0.414|
|低胜率/接近五五|Marnie Grimmsnarl|6048|0.433|

## 训练和评测注意事项

7f9 曾在线上达到 900+，但环境变化后复训不稳。文档中强签名要跟随 0812/0813 高分段，而不是固定旧 7f9。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Alakazam w/Toucannon - Pitch Black](https://www.youtube.com/watch?v=I4IsyYb7WtI)
- [YouTube search: Alakazam Powerful Hand gameplay](https://www.youtube.com/results?search_query=Alakazam+Powerful+Hand+Pokemon+TCG+gameplay+deck+profile)

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
