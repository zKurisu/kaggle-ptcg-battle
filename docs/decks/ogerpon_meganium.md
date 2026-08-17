# Ogerpon Meganium / 厄诡椪大竺葵

资料更新时间: 2026-08-17  
本地 deck 模板: `decks/pool_351_ogerpon_meganium.csv`  
Kaggle 统计 archetype: `Teal Mask Ogerpon`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

|deck_sig|max_score|rows|WR(no draw)|top teams|
|---|---|---|---|---|
|ab7e4b818773|1204.1|357|0.625|Dipam Chakraborty:89;Rmy:82;ANDPAD kaggler team:65;Majkel1337:40;EliKal:28|
|90abbfb0eee0|1154.7|90|0.489|palsystem:56;しんぴのしずく💧:22;THIRD PTCG Club:12|
|2bd9da52c43a|1115.6|108|0.635|James Cox & Henry Chao:58;e-toppo + kurupical:50|
|2a5072194fdf|1112.9|31|0.581|Oshbocker:31|
|081213bf731c|1111.8|53|0.585|Sixth Sense:53|

## 分段分布

|band|rows|WR|max_score|top sigs|
|---|---|---|---|---|
|1200+|65|0.492|1204.1|ab7e4b818773:65|
|1100-1199|290|0.579|1159.2|2bd9da52c43a:58;90abbfb0eee0:56;ab7e4b818773:56;081213bf731c:53;2a5072194fdf:31|
|1000-1099|930|0.535|1099.9|ab7e4b818773:227;5899c772bace:206;6205fc379380:111;2bd9da52c43a:50;d17573abc0e3:|
|900-999|284|0.479|993.5|5899c772bace:66;697a82e582d5:46;d17573abc0e3:44;7232316f9ca3:44;d6d4ab740380:39|
|800-899|8|0.250|853.9|0d77612de6ac:8|
|600-699|11|0.818|624.9|f4fe18b4203d:9;5899c772bace:1;28a52df999ce:1|

## 打法摘要

Grass engine 变体。和 Ogerpon Box 共享 Ogerpon 加速核心，但更强调进化/草系 engine 的稳定铺场。

## 关键牌

- Teal Mask Ogerpon ex
- Meganium line
- Energy Switch
- Grass-energy acceleration
- Boss's Orders

## 关键 combo / 决策点

- 先确保 Ogerpon 能稳定抽牌贴能，再决定是否转 Meganium route。
- 对快攻 matchup，不能为了完整 combo 牺牲第一轮攻击窗口。
- 对 wall/anti-ex matchup，需要尽早确认是否有非 ex 或抓 basic 的路。

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

|类型|对手 archetype|games|WR|
|---|---|---|---|
|高胜率|Cynthia Garchomp|258|0.729|
|高胜率|Marnie Grimmsnarl|3182|0.725|
|高胜率|Team Rocket Mewtwo|143|0.650|
|高胜率|Dragapult|234|0.440|
|高胜率|Festival Lead|199|0.437|
|低胜率/接近五五|Crustle Wall|471|0.200|
|低胜率/接近五五|Mega Lopunny|609|0.223|
|低胜率/接近五五|Alakazam|597|0.338|

## 训练和评测注意事项

本地统计会和 Teal Mask Ogerpon 合并；如果要训练 Meganium specialist，应按 deck-sig 单独抽 corpus，不要和 Box/Ogerpon-Raging-Bolt 混在一起。

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

- [YouTube search: Ogerpon Meganium deck](https://www.youtube.com/results?search_query=Ogerpon+Meganium+Pokemon+TCG+deck+gameplay)
- [Bilibili search: 厄诡椪 大竺葵](https://search.bilibili.com/all?keyword=%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%8D%A1%E7%89%8C+%E5%8E%84%E8%AF%A1%E6%A4%AA+%E5%A4%A7%E7%AB%BA%E8%91%B5+%E5%8D%A1%E7%BB%84)

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
