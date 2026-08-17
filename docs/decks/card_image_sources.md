# 卡面图片与外部素材

宝可梦卡面、卡文和商标通常受版权保护。这里的目标是为研究、trace 审阅和人工理解提供素材入口，不是把卡图打包进仓库或 submission。

## 推荐来源

|来源|链接|适合用途|
|---|---|---|
|Pokémon TCG API|[Pokémon TCG API](https://docs.pokemontcg.io/api-reference/cards/card-object/)|`images.small` / `images.large` 字段可拿到卡图 URL，适合本地研究缓存。|
|pokemon-tcg-data|[pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data)|Pokémon TCG API 的原始 JSON 数据，可 clone 或下载 release；优先当 card metadata/image URL 索引用。|
|PokemonCard.io|[PokemonCard.io](https://pokemoncard.io/)|Deck 页面有 `Download all Images`，但页面版权说明显示卡图/卡文仍归 Pokémon/Nintendo/Game Freak 等权利方。|
|Limitless TCG|[Limitless TCG](https://limitlesstcg.com/)|适合查真实比赛 decklist、卡组占比和卡图预览；不要把卡图直接提交进仓库。|

## 使用原则

- 可以把 API 返回的 `images.small` / `images.large` URL 保存成索引，用于本地 trace viewer 动态加载。
- 如需缓存图片，放到 gitignore 的本地目录，例如 `artifacts/card_images/`，并记录来源 URL、抓取时间和卡牌 id。
- 不要把下载的卡面图片 commit 到仓库，也不要放进 Kaggle submission 包。
- 规则和训练只依赖 card id、卡名、卡文和 engine observation；图片只能帮助人检查，不应成为模型输入。
- 如果后续要公开仓库，务必再次检查各数据源条款和 Pokémon/Nintendo/Game Freak 的版权声明。

## 可实现的小工具方向

1. 根据 `data/EN_Card_Data.csv` 的卡名生成 Pokémon TCG API 查询。
2. 把 API 的 image URL 缓存在 `data/card_image_index.generated.json`。
3. 在 trace viewer 中按 card id 显示卡名和外链缩略图；缺失时退化为文本。
4. 保留 `--no-image-download` 默认模式，只有人工调试时才下载图片缓存。
