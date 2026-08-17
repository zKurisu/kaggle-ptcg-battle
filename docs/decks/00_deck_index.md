# 卡组资料索引

这个目录把本地 `decks/` 模板、Kaggle ladder 分段、真实 PTCG 资料和视频入口放在一起。阅读顺序建议:

1. 先看本页的“强签名总表”，决定 RR/shadow 池要覆盖哪些 archetype。
2. 再看具体卡组文件里的打法、关键 combo、视频和 matchup 先验。
3. 按每个卡组文件里的“单独训练 / Random 测试 / 推荐 RR 测试 / Kaggle episode 动态回放”跑完整验证链。
4. 最后回到 `tools/trace_matchup_decisions.py`、`ptcg_rl/deck_plans.py`、`ptcg_rl/rule_overlay.py` 把策略变成可验证规则或训练信号。

## 强签名总表

数据主依据是 2026-08-13 ladder 分段统计。对于没有稳定样本的本地模板，只记录为待补充，不直接当作强环境结论。

|卡组文档|本地 deck|Kaggle archetype|当前强签名|
|---|---|---|---|
|[Dragapult ex / 多龙巴鲁托 ex](dragapult.md)|`decks/pool_284_dragapult_ex.csv`|Dragapult|cc2e995b5ad0 / max 1218.2 / rows 1115 / WR 0.536|
|[Alakazam Powerful Hand / 胡地](alakazam.md)|`decks/pool_350_alakazam_powerful_hand.csv`|Alakazam|ecb67fcd9c0b / max 1163.2 / rows 196 / WR 0.551|
|[Marnie's Grimmsnarl ex / 玛俐长毛巨魔 ex](marnie_grimmsnarl.md)|`decks/pool_329_marnie_s_grimmsnarl_ex.csv`|Marnie Grimmsnarl|b8f251a476e7 / max 1127.1 / rows 1393 / WR 0.408|
|[Teal Mask Ogerpon / 碧草面具厄诡椪](teal_mask_ogerpon.md)|`decks/pool_339_ogerpon_box.csv`|Teal Mask Ogerpon|ab7e4b818773 / max 1204.1 / rows 357 / WR 0.625|
|[Ogerpon Meganium / 厄诡椪大竺葵](ogerpon_meganium.md)|`decks/pool_351_ogerpon_meganium.csv`|Teal Mask Ogerpon|ab7e4b818773 / max 1204.1 / rows 357 / WR 0.625|
|[Crustle Wall / 岩殿居蟹墙](crustle_wall.md)|`decks/pool_341_crustle_mysterious_rock_inn.csv`|Crustle Wall|3cd5039c59d2 / max 1211.4 / rows 138 / WR 0.435|
|[Mega Lucario ex / 超级路卡利欧 ex](mega_lucario.md)|`decks/pool_345_mega_lucario_ex.csv`|Mega Lucario|43d6d8b0fce9 / max 1230.3 / rows 294 / WR 0.515|
|[Mega Lopunny ex / 超级长耳兔 ex](mega_lopunny.md)|`decks/pool_353_mega_lopunny_ex.csv`|Mega Lopunny|f1445356c3a7 / max 1107.3 / rows 938 / WR 0.483|
|[Festival Lead / 祭典主唱](festival_lead.md)|`decks/pool_336_festival_lead.csv`|Festival Lead|41ffa7894f40 / max 1211.4 / rows 15 / WR 0.533|
|[Cynthia's Garchomp ex / 竹兰烈咬陆鲨 ex](cynthia_garchomp.md)|`decks/pool_332_cynthia_s_garchomp_ex.csv`|Cynthia Garchomp|52f467394857 / max 1125.7 / rows 76 / WR 0.539|
|[Team Rocket's Mewtwo ex / 火箭队的超梦 ex](team_rocket_mewtwo.md)|`decks/pool_337_rocket_s_mewtwo_ex.csv`|Team Rocket Mewtwo|4c6c10c0d8d5 / max 1159.2 / rows 6 / WR 0.667|
|[N's Zoroark ex / N 的索罗亚克 ex](n_zoroark.md)|`decks/pool_320_n_s_zoroark_ex.csv`|N's Zoroark|aa80b030c069 / max 1112.9 / rows 37 / WR 0.622|
|[Mega Starmie ex / 超级宝石海星 ex](mega_starmie.md)|`decks/pool_362_mega_starmie_ex.csv`|Mega Starmie|a31f17401ea1 / max 976.3 / rows 2 / WR 0.000|
|[Archaludon ex / 铝钢桥龙 ex](archaludon.md)|`decks/pool_315_archaludon_ex.csv`|Archaludon|f2a48b323dd0 / max 980.8 / rows 2 / WR 0.000|
|[Raging Bolt ex / 猛雷鼓 ex](raging_bolt.md)|`decks/pool_280_raging_bolt_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Slowking Seek Inspiration / 呆壳兽灵感检索](slowking_seek_inspiration.md)|`decks/pool_322_slowking_seek_inspiration.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Lillie's Clefairy ex / 莉莉艾皮皮 ex](lillie_clefairy.md)|`decks/pool_326_lillie_s_clefairy_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Ethan's Typhlosion / 小响火暴兽](ethan_typhlosion.md)|`decks/pool_333_ethan_s_typhlosion.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Hydrapple ex / 蜜集大蛇 ex](hydrapple.md)|`decks/pool_352_hydrapple_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Team Rocket's Honchkrow / 火箭队乌鸦头头](rocket_honchkrow.md)|`decks/pool_356_rocket_s_honchkrow.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Metagross Metal Maker / 巨金怪 Metal Maker](metagross_metal_maker.md)|`decks/pool_361_metagross_metal_maker.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Hop's Trevenant / 赫普朽木妖](hop_trevenant.md)|`decks/pool_363_hop_s_trevenant.csv`|Hop Trevenant|暂无稳定 Kaggle 分段签名|
|[Mega Greninja ex / 超级甲贺忍蛙 ex](mega_greninja.md)|`decks/pool_370_mega_greninja_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Beedrill ex / 大针蜂 ex](beedrill.md)|`decks/pool_371_beedrill_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Sylveon Safeguard / 仙子伊布 Safeguard](sylveon_safeguard.md)|`decks/pool_373_sylveon_safeguard.csv`|待识别|暂无稳定 Kaggle 分段签名|
|[Mega Abomasnow ex / 超级暴雪王 ex](mega_abomasnow.md)|`decks/pool_400_mega_abomasnow_ex.csv`|待识别|暂无稳定 Kaggle 分段签名|

## 全局资料源

- [Limitless TCG](https://limitlesstcg.com/)：真实比赛 top deck、meta share、decklist。
- [Play Limitless](https://play.limitlesstcg.com/decks)：线上赛 decklist 和 win rate。
- [Pokémon TCG API card object](https://docs.pokemontcg.io/api-reference/cards/card-object/)：card metadata 与 card image URL。
- [pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data)：Pokémon TCG API 对应原始 JSON 数据。
- [PokemonCard.io](https://pokemoncard.io/)：decklist、deck primer、卡图下载入口；卡图版权需谨慎。

## Kaggle 动态回放怎么接入

Kaggle 网页 replay 可以作为人工逐回合检查入口，但它依赖登录态和网页脚本，不能直接作为本地可复现数据源。推荐做法是:

1. 在 Kaggle Submissions / Episodes 页面打开动态 replay，人工定位关键输局。
2. 记录 `submission_id` 和 episode id。
3. 用每个卡组文档里的 `tools/analyze_kaggle_replays.py` 命令下载 replay JSON、汇总 opponent deck-sig，并导出 opponent deck CSV。
4. 用 `tools/make_kaggle_opp_round_robin_cmd.py` 把真实线上对手转成本地 RR。
5. 对关键坏局再用 `tools/trace_matchup_decisions.py` 做 fixed-seed trace。

## 维护方法

重新生成文档:

```bash
python3 tools/build_deck_docs.py
```

更新时先替换或新增 ladder 分段统计 CSV，再运行脚本。对于视频/打法链接，直接更新 `tools/build_deck_docs.py` 里的 `ARCHETYPES` 条目。
