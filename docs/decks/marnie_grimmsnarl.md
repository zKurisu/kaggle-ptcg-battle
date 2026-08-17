# Marnie's Grimmsnarl ex / 玛俐长毛巨魔 ex

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_329_marnie_s_grimmsnarl_ex.csv`  
Kaggle 统计 archetype: `Marnie Grimmsnarl`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|b8f251a476e7|1127.1|1393|0.408|yumizu:88;hatry:65;Phil_Hellmuth:64;yoshitaka agent:54;Ken Zhou:50|
|2c22fa761816|1104.9|216|0.491|KawattaTaido:69;Mahog:62;JZ:31;213tubo:19;Mega Regigigas Ex Vmax:15|
|2986915b66b1|1003.8|25|0.440|tedo:25|
|a3b0c314bf8a|997.8|15|0.467|YumeNeko:15|
|b9ee5727d1c2|985.7|8|0.250|Lennox:8|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1100-1199|136|0.537|1127.1|b8f251a476e7:74;2c22fa761816:62|
|1000-1099|770|0.445|1092.1|b8f251a476e7:611;2c22fa761816:134;2986915b66b1:25|
|900-999|677|0.376|998.9|b8f251a476e7:558;024be26edfd2:49;2c22fa761816:16;a3b0c314bf8a:15;f3acd18e7e6d:9|
|800-899|119|0.412|898.6|b8f251a476e7:82;d44408da1c18:28;2c22fa761816:4;c9012702ae02:3;5bfd9c6bbd7a:1|
|700-799|56|0.321|798.3|b8f251a476e7:48;d3b885ec48ae:8|
|600-699|7|0.429|651.8|b8f251a476e7:7|
|unknown|13|0.615||b8f251a476e7:13|

## 打法摘要

Stage-2 damage-control。先建立 Marnie 进化线，再用 Grimmsnarl 的 Punk Up/Shadow Bullet、Froslass spread 和 Munkidori 伤害移动组成持续压力。

## 关键牌

- Marnie's Impidimp
- Marnie's Morgrem
- Marnie's Grimmsnarl ex
- Munkidori
- Snorunt
- Froslass
- Spikemuth Gym

## 关键 combo / 决策点

- 看到 Impidimp/Morgrem/Grimmsnarl 线时，优先保证进化链完整，而不是沉迷低价值 draw/search。
- Froslass 不是无脑上；面对 Munkidori/Dragapult 之类会利用伤害指示物的对手，需要确认对自己收益大于风险。
- 对 Ogerpon 明显劣势，不能指望 winner-only 微调解决；需要明确开局 setup、target 和资源保留策略。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Alakazam|6048|0.567|
|高胜率|Team Rocket Mewtwo|1602|0.524|
|高胜率|Crustle Wall|3704|0.522|
|高胜率|Mega Lucario|258|0.500|
|高胜率|Dragapult|1569|0.473|
|低胜率/接近五五|Teal Mask Ogerpon|3182|0.274|
|低胜率/接近五五|Mega Lopunny|2131|0.415|
|低胜率/接近五五|Festival Lead|1432|0.415|
|低胜率/接近五五|Cynthia Garchomp|2278|0.461|

## 训练和评测注意事项

历史最稳的一类模型来自 old BC/w3/w4 + 900+ winner 或 top sig 数据。history/trajectory 小修多次没有稳定超过 old recipe。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [AzulGG: Grimmsnarl - Pitch Black](https://www.youtube.com/watch?v=B0vkWIWIg9U)
- [Japanese VOD: Marnie Grimmsnarl vs Dragapult](https://www.youtube.com/watch?v=c4BHBU0T9A8)

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
