#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "docs" / "decks"
LADDER_DIR = ROOT / "logs" / "ladder_distribution_0812_0813_20260814"
SIG_SUMMARY = LADDER_DIR / "deck_sig_summary.csv"
BAND_SUMMARY = LADDER_DIR / "band_archetype_summary.csv"
MATCHUPS = ROOT / "logs" / "matchup_notes_20260805" / "0724_0804_score900" / "archetype_matchups.csv"


IMAGE_SOURCES = [
    (
        "Pokémon TCG API",
        "https://docs.pokemontcg.io/api-reference/cards/card-object/",
        "`images.small` / `images.large` 字段可拿到卡图 URL，适合本地研究缓存。",
    ),
    (
        "pokemon-tcg-data",
        "https://github.com/PokemonTCG/pokemon-tcg-data",
        "Pokémon TCG API 的原始 JSON 数据，可 clone 或下载 release；优先当 card metadata/image URL 索引用。",
    ),
    (
        "PokemonCard.io",
        "https://pokemoncard.io/",
        "Deck 页面有 `Download all Images`，但页面版权说明显示卡图/卡文仍归 Pokémon/Nintendo/Game Freak 等权利方。",
    ),
    (
        "Limitless TCG",
        "https://limitlesstcg.com/",
        "适合查真实比赛 decklist、卡组占比和卡图预览；不要把卡图直接提交进仓库。",
    ),
]


def yt(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


def bili(query: str) -> str:
    return "https://search.bilibili.com/all?keyword=" + quote_plus(query)


ARCHETYPES = [
    {
        "slug": "dragapult",
        "title": "Dragapult ex / 多龙巴鲁托 ex",
        "local_file": "decks/pool_284_dragapult_ex.csv",
        "kaggle_archetype": "Dragapult",
        "cn": "多龙巴鲁托 ex",
        "key_cards": ["Dreepy", "Drakloak", "Dragapult ex", "Duskull/Dusclops/Dusknoir", "Crispin", "Counter Catcher"],
        "plan": "Stage-2 spread/race。核心是尽快建立 Dreepy -> Drakloak -> Dragapult ex，同时用 Drakloak 的抽牌/检索能力保证连续进化，再用 Phantom Dive 的主伤害和 bench damage 制造跨回合奖赏路线。",
        "combos": [
            "Drakloak 在线时，先尽量使用可用的抽/看牌能力，再进化成 Dragapult ex，避免提前进化导致少一次资源选择。",
            "bench damage 不应只看当回合击倒；要结合 Dusknoir/Dusclops 的伤害指示物、自身下一次 200 点主伤害和 Boss/Counter Catcher 的目标路线。",
            "面对低 HP bench，优先考虑是否能用 60 点 bench damage 直接拿奖；面对 220 HP 目标，可以先铺 20/60 点，为后续 200 点攻击或自爆伤害创造 KO。",
        ],
        "training_notes": "本项目历史上 Dragapult random 经常能接近 99%，但 RR 和线上仍弱，说明不能只做单步 attack/attach 标签；需要 sequence trace、伤害分配诊断和 fixed-seed loss replay。",
        "videos": [
            ("AzulGG: Dragapult w/Dark Bell - Pitch Black", "https://www.youtube.com/watch?v=7ZDRAA9FW_w"),
            ("PokeDronks Dragapult deck/video primer", "https://pokemoncard.io/deck/dragapult-pokedrunks-121450"),
        ],
        "extra_refs": [("Limitless Dragapult decks", "https://limitlesstcg.com/decks/326")],
    },
    {
        "slug": "alakazam",
        "title": "Alakazam Powerful Hand / 胡地",
        "local_file": "decks/pool_350_alakazam_powerful_hand.csv",
        "kaggle_archetype": "Alakazam",
        "cn": "胡地 Powerful Hand",
        "key_cards": ["Abra", "Kadabra", "Alakazam", "Buddy-Buddy Poffin", "Poke Pad", "Boss's Orders", "stadium/search package"],
        "plan": "Stage-2 bench/control。胡地类卡组不是单纯把 active 打满伤害，而是通过进化链、后排攻击/控制和资源循环拖慢对手，要求模型理解 bench、active、手牌检索和 prize race 的长期关系。",
        "combos": [
            "早期要优先铺 Abra/Kadabra，避免把 search/support 用在低价值手牌整理上。",
            "Kadabra 类 engine 的抽/找牌动作要与 evolve timing 对齐；很多失败 trace 表现为有进化/攻击窗口但选择 END 或无关 trainer。",
            "胡地对 Dragapult、Marnie 等主流局容易被 tempo 压制，训练时要审查 attack miss、evolve miss、target miss 是否真实存在。",
        ],
        "training_notes": "7f9 曾在线上达到 900+，但环境变化后复训不稳。文档中强签名要跟随 0812/0813 高分段，而不是固定旧 7f9。",
        "videos": [
            ("AzulGG: Alakazam w/Toucannon - Pitch Black", "https://www.youtube.com/watch?v=I4IsyYb7WtI"),
            ("YouTube search: Alakazam Powerful Hand gameplay", yt("Alakazam Powerful Hand Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [("Limitless Alakazam decks", "https://limitlesstcg.com/decks/350")],
    },
    {
        "slug": "marnie_grimmsnarl",
        "title": "Marnie's Grimmsnarl ex / 玛俐长毛巨魔 ex",
        "local_file": "decks/pool_329_marnie_s_grimmsnarl_ex.csv",
        "kaggle_archetype": "Marnie Grimmsnarl",
        "cn": "玛俐长毛巨魔 ex",
        "key_cards": ["Marnie's Impidimp", "Marnie's Morgrem", "Marnie's Grimmsnarl ex", "Munkidori", "Snorunt", "Froslass", "Spikemuth Gym"],
        "plan": "Stage-2 damage-control。先建立 Marnie 进化线，再用 Grimmsnarl 的 Punk Up/Shadow Bullet、Froslass spread 和 Munkidori 伤害移动组成持续压力。",
        "combos": [
            "看到 Impidimp/Morgrem/Grimmsnarl 线时，优先保证进化链完整，而不是沉迷低价值 draw/search。",
            "Froslass 不是无脑上；面对 Munkidori/Dragapult 之类会利用伤害指示物的对手，需要确认对自己收益大于风险。",
            "对 Ogerpon 明显劣势，不能指望 winner-only 微调解决；需要明确开局 setup、target 和资源保留策略。",
        ],
        "training_notes": "历史最稳的一类模型来自 old BC/w3/w4 + 900+ winner 或 top sig 数据。history/trajectory 小修多次没有稳定超过 old recipe。",
        "videos": [
            ("AzulGG: Grimmsnarl - Pitch Black", "https://www.youtube.com/watch?v=B0vkWIWIg9U"),
            ("Japanese VOD: Marnie Grimmsnarl vs Dragapult", "https://www.youtube.com/watch?v=c4BHBU0T9A8"),
        ],
        "extra_refs": [
            ("TCGplayer Marnie's Grimmsnarl ex Deck Guide", "https://www.tcgplayer.com/content/article/Marnie-s-Grimmsnarl-ex-Deck-Guide-Pok%C3%A9mon-TCG-July-2025/8c493445-7f49-4281-9c98-37f0938c239f/"),
        ],
    },
    {
        "slug": "teal_mask_ogerpon",
        "title": "Teal Mask Ogerpon / 碧草面具厄诡椪",
        "local_file": "decks/pool_339_ogerpon_box.csv",
        "kaggle_archetype": "Teal Mask Ogerpon",
        "cn": "碧草面具厄诡椪 Box",
        "key_cards": ["Teal Mask Ogerpon ex", "Cornerstone/Wellspring Mask Ogerpon", "Passimian", "Raging Bolt ex", "Meowth ex", "Mega Kangaskhan ex", "Energy Switch"],
        "plan": "Basic ex tempo/toolbox。Ogerpon 用 Teal Dance 把草能转成抽牌和加速，再用不同 secondary attacker 处理 matchup。它在线上强度很依赖 deck sig 和环境。",
        "combos": [
            "不能把 Teal Dance 当成唯一目标；如果存在奖赏窗口，应该优先攻击/抓关键 basic。",
            "对 Crustle 时，ex 攻击进入已成型 Crustle 往往无效，实战计划是尽早惩罚 Dwebble 或建立非 ex/secondary route。",
            "Box 型 list 要识别 secondary attacker 的用途，不能混成只会 Ogerpon 自己循环的模型。",
        ],
        "training_notes": "v7 top2 sig 曾非常强，但 0812/0813 环境下 ab7e/2bd9/90ab 等新 sig 更值得关注。Ogerpon 的训练不宜盲目 top6/mix。",
        "videos": [
            ("AzulGG: Okidogi ex w/Teal Mask Ogerpon - Pitch Black", "https://www.youtube.com/watch?v=ThjcS0Thyw8"),
            ("YouTube search: Teal Mask Ogerpon Box gameplay", yt("Teal Mask Ogerpon Box Pokemon TCG gameplay 2026")),
        ],
        "extra_refs": [
            ("PokeBeach Teal Mask Kangaskhan Box article", "https://www.pokebeach.com/2026/04/stop-calling-it-slop-box-teal-mask-kangaskhan-decks-in-the-new-format"),
        ],
    },
    {
        "slug": "ogerpon_meganium",
        "title": "Ogerpon Meganium / 厄诡椪大竺葵",
        "local_file": "decks/pool_351_ogerpon_meganium.csv",
        "kaggle_archetype": "Teal Mask Ogerpon",
        "cn": "厄诡椪大竺葵",
        "key_cards": ["Teal Mask Ogerpon ex", "Meganium line", "Energy Switch", "Grass-energy acceleration", "Boss's Orders"],
        "plan": "Grass engine 变体。和 Ogerpon Box 共享 Ogerpon 加速核心，但更强调进化/草系 engine 的稳定铺场。",
        "combos": [
            "先确保 Ogerpon 能稳定抽牌贴能，再决定是否转 Meganium route。",
            "对快攻 matchup，不能为了完整 combo 牺牲第一轮攻击窗口。",
            "对 wall/anti-ex matchup，需要尽早确认是否有非 ex 或抓 basic 的路。",
        ],
        "training_notes": "本地统计会和 Teal Mask Ogerpon 合并；如果要训练 Meganium specialist，应按 deck-sig 单独抽 corpus，不要和 Box/Ogerpon-Raging-Bolt 混在一起。",
        "videos": [
            ("YouTube search: Ogerpon Meganium deck", yt("Ogerpon Meganium Pokemon TCG deck gameplay")),
            ("Bilibili search: 厄诡椪 大竺葵", bili("宝可梦卡牌 厄诡椪 大竺葵 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "crustle_wall",
        "title": "Crustle Wall / 岩殿居蟹墙",
        "local_file": "decks/pool_341_crustle_mysterious_rock_inn.csv",
        "kaggle_archetype": "Crustle Wall",
        "cn": "岩殿居蟹墙",
        "key_cards": ["Dwebble", "Crustle", "Mega Kangaskhan ex", "healing/HP buff package", "Boss's Orders"],
        "plan": "Anti-ex wall。核心是让 Crustle 成型并保持 active，阻止对手 ex 攻击造成伤害，同时用 Superb Scissors 和资源压力拿奖。",
        "combos": [
            "对 ex-heavy decks，要优先保证 Dwebble -> Crustle，active 位置比普通 beatdown 更重要。",
            "一旦 Crustle 建立，要避免无意义 retreat/switch，除非能直接创造奖赏。",
            "对 Mega Lopunny/Alakazam 等能绕过或压制 wall 的局要提前准备 secondary route。",
        ],
        "training_notes": "Crustle 对环境极敏感。近期复训被打爆时，不能只看 random；要看对 Lopunny/Alakazam/Ogerpon 的分段覆盖和高分 team 数据。",
        "videos": [
            ("TCGMatch/YouTube VOD: NAIC 2026 Crustle article", "https://tcgmatch.cl/blog/alloutblitzle-logra-ganar-el-naic-2026"),
            ("YouTube search: Crustle Mysterious Rock Inn gameplay", yt("Crustle Mysterious Rock Inn Pokemon TCG gameplay")),
        ],
        "extra_refs": [("Limitless Crustle decks", "https://limitlesstcg.com/decks/341")],
    },
    {
        "slug": "mega_lucario",
        "title": "Mega Lucario ex / 超级路卡利欧 ex",
        "local_file": "decks/pool_345_mega_lucario_ex.csv",
        "kaggle_archetype": "Mega Lucario",
        "cn": "超级路卡利欧 ex",
        "key_cards": ["Riolu", "Mega Lucario ex", "Lunatone", "Solrock", "Fighting Gong", "Premium Power Pro", "Poke Pad"],
        "plan": "Fighting engine tempo。Lucario 在 Kaggle ladder 中表现相对均衡，关键是用 Solrock/Lunatone/Fighting Gong 一类 engine 保证 Mega Lucario 的攻击节奏。",
        "combos": [
            "不要只把 Mega Lucario 当孤立 attacker；先建立 engine，再进入 prize race。",
            "Riolu 不同 card id 的签名差异很重要，训练和打包必须 deck-sig 对齐。",
            "局面允许时，优先完成能保证下回合攻击的搜索/贴能，而不是当回合低价值攻击。",
        ],
        "training_notes": "43d6 在高分段长期出现，是当前本地 RR 池和 shadow 池必须包含的 archetype。它也适合作为 climber 环境适配基线。",
        "videos": [
            ("AzulGG: Mega Lucario w/Togekiss - Pitch Black", "https://www.youtube.com/watch?v=wFwxEBBgDo0"),
            ("YouTube search: Mega Lucario ex gameplay", yt("Mega Lucario ex Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [
            ("Official Pokemon.com Mega Lucario ex strategy", "https://www.pokemon.com/us/strategy/pokemon-tcg-deck-list-and-strategy-building-a-mega-lucario-ex-deck"),
        ],
    },
    {
        "slug": "mega_lopunny",
        "title": "Mega Lopunny ex / 超级长耳兔 ex",
        "local_file": "decks/pool_353_mega_lopunny_ex.csv",
        "kaggle_archetype": "Mega Lopunny",
        "cn": "超级长耳兔 ex",
        "key_cards": ["Mega Lopunny ex", "Dunsparce", "Dudunsparce/Dudunsparce ex", "consistency engine", "Boss's Orders"],
        "plan": "Mega attacker + consistency。Kaggle 中 f144 是主流高覆盖签名，核心是稳定铺场并持续用 Mega attacker 压 prize race。",
        "combos": [
            "Dunsparce/Dudunsparce engine 是稳定性来源，不能被单纯 attacker preference 淹没。",
            "对 Crustle/Ogerpon/Festival 有历史优势，但对 Mega Lucario/Cynthia 等可能吃力。",
            "训练时应保留 top1/top2 specialist，对 mixed 的收益要谨慎。",
        ],
        "training_notes": "Lopunny 旧 v14 曾有可提交表现，但 0813 分段显示高分 ceiling 不如 Dragapult/Ogerpon/Lucario/Crustle。",
        "videos": [
            ("YouTube search: Mega Lopunny ex gameplay", yt("Mega Lopunny ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 超级长耳兔 ex", bili("宝可梦卡牌 超级长耳兔 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "festival_lead",
        "title": "Festival Lead / 祭典主唱",
        "local_file": "decks/pool_336_festival_lead.csv",
        "kaggle_archetype": "Festival Lead",
        "cn": "祭典主唱",
        "key_cards": ["Festival Lead attacker", "Festival Grounds", "Dipplin/Applin line", "search/draw support"],
        "plan": "Festival engine 连击/节奏。重点是启用 Stadium/engine 后形成重复攻击，不能被普通 beatdown 的单步标签误导。",
        "combos": [
            "优先找 Stadium/engine piece，而不是盲目把 attacker 推到 active。",
            "一旦 engine 启动，要确保连续攻击和 prize race，不要因为低价值 draw 丢失攻击节奏。",
            "对 Marnie/Ogerpon 有历史可打空间，但对 Lopunny/Crustle 会明显受压。",
        ],
        "training_notes": "Festival 在 v10/v11 有过 900+ shadow/specialist 高点；样本少时更适合 top1 specialist。",
        "videos": [
            ("AzulGG: Festival Lead - Pitch Black", "https://www.youtube.com/watch?v=h_2c-O9HaBY"),
            ("YouTube search: Festival Lead gameplay", yt("Festival Lead Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [("Limitless Festival Lead decks", "https://limitlesstcg.com/decks")],
    },
    {
        "slug": "cynthia_garchomp",
        "title": "Cynthia's Garchomp ex / 竹兰烈咬陆鲨 ex",
        "local_file": "decks/pool_332_cynthia_s_garchomp_ex.csv",
        "kaggle_archetype": "Cynthia Garchomp",
        "cn": "竹兰烈咬陆鲨 ex",
        "key_cards": ["Cynthia's Gible", "Cynthia's Gabite", "Cynthia's Garchomp ex", "Cynthia's Spiritomb", "search engine"],
        "plan": "Stage-2 linear tempo。Gabite/Garchomp 线要求早期稳定进化，部分弱局需要 Spiritomb 等非 ex/counter attacker 参与。",
        "combos": [
            "Gabite 的搜索/engine 需要在进化链中优先完成。",
            "对 Crustle 时，Spiritomb active 的成功 trace 曾出现，说明不能只让 ex attacker 打进 wall。",
            "对 Lopunny 有历史优势，但对 Ogerpon/Alakazam/Crustle 容易被结构克制。",
        ],
        "training_notes": "样本量比主流 archetype 少，训练时要避免 winner-only 后数据过窄；RR 池中必须包含它作为 Lopunny counter。",
        "videos": [
            ("AzulGG: Cynthia's Garchomp - Pitch Black", "https://www.youtube.com/watch?v=GqeCAKzXMlg"),
            ("YouTube search: Cynthia Garchomp gameplay", yt("Cynthia's Garchomp ex Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "team_rocket_mewtwo",
        "title": "Team Rocket's Mewtwo ex / 火箭队的超梦 ex",
        "local_file": "decks/pool_337_rocket_s_mewtwo_ex.csv",
        "kaggle_archetype": "Team Rocket Mewtwo",
        "cn": "火箭队的超梦 ex",
        "key_cards": ["Team Rocket's Mewtwo ex", "Team Rocket's Mimikyu", "Team Rocket's Tarountula", "Team Rocket's Spidops", "energy acceleration"],
        "plan": "Team Rocket board-count payoff。Mewtwo 往往需要足够火箭队宝可梦在场后才能进入主攻击节奏，因此 setup quality 比单回合打点更重要。",
        "combos": [
            "先铺 Team Rocket Pokémon 数量，再决定是否强行攻击。",
            "Spidops/辅助线承担能量和控制作用，不能被忽略。",
            "历史对 Alakazam 较强，对 Ogerpon/Cynthia/Marnie 不稳。",
        ],
        "training_notes": "random 可以很高但 Kaggle 转化差，说明模型可能学到基础启动但缺少 matchup plan。",
        "videos": [
            ("YouTube search: Team Rocket's Mewtwo ex gameplay", yt("Team Rocket's Mewtwo ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 火箭队的超梦", bili("宝可梦卡牌 火箭队的超梦 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "n_zoroark",
        "title": "N's Zoroark ex / N 的索罗亚克 ex",
        "local_file": "decks/pool_320_n_s_zoroark_ex.csv",
        "kaggle_archetype": "N's Zoroark",
        "cn": "N 的索罗亚克 ex",
        "key_cards": ["N's Zorua", "N's Zoroark ex", "N's engine cards", "draw/search package"],
        "plan": "Dark evolution tempo。现实 Limitless 中占比很高，Kaggle 0813 样本不多但高分段有 aa80 签名。",
        "combos": [
            "早期优先铺 evolution basic，不要让 hand filter 消耗关键 evolution card。",
            "对 Dragapult/Ogerpon 等主流局要记录 prize race 速度，而不是只看 random。",
            "若要纳入 RR，需要为 aa80 或最新高分签名训练 shadow。",
        ],
        "training_notes": "以前 RR 覆盖不足，后续应作为 0813 之后环境池必备卡组。",
        "videos": [
            ("YouTube search: N's Zoroark ex gameplay", yt("N's Zoroark ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: N 的索罗亚克", bili("宝可梦卡牌 N的索罗亚克 ex 卡组")),
        ],
        "extra_refs": [("Limitless N's Zoroark decks", "https://limitlesstcg.com/decks")],
    },
    {
        "slug": "mega_starmie",
        "title": "Mega Starmie ex / 超级宝石海星 ex",
        "local_file": "decks/pool_362_mega_starmie_ex.csv",
        "kaggle_archetype": "Mega Starmie",
        "cn": "超级宝石海星 ex",
        "key_cards": ["Staryu", "Mega Starmie ex", "Duskull", "Dusclops", "Dusknoir", "Hilda", "Grand Tree", "Wally's Compassion"],
        "plan": "Mega attacker + Dusknoir damage-counter pressure。Kaggle 样本极少，更多是未来扩展池的候选。",
        "combos": [
            "Staryu/Starmie 主线和 Duskull 伤害线需要同步建立。",
            "Hilda/Grand Tree 类进化搜索是关键，不应被当成普通 draw。",
            "Dusknoir damage counter 要和下一次攻击的 KO map 一起计算。",
        ],
        "training_notes": "样本不足时不要用它拉高/拉低 RR 总均值；先建立合法 shadow 和 random 质量。",
        "videos": [
            ("YouTube search: Mega Starmie ex gameplay", yt("Mega Starmie ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 超级宝石海星", bili("宝可梦卡牌 超级宝石海星 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "archaludon",
        "title": "Archaludon ex / 铝钢桥龙 ex",
        "local_file": "decks/pool_315_archaludon_ex.csv",
        "kaggle_archetype": "Archaludon",
        "cn": "铝钢桥龙 ex",
        "key_cards": ["Archaludon ex", "Metal Energy", "metal acceleration", "search/support package"],
        "plan": "Metal energy tempo。Kaggle 0812/0813 样本很少，但用户提交胡地时曾被该类卡组击败，因此 RR 池不能完全忽略。",
        "combos": [
            "重点监控能量 attach/acceleration 是否连续。",
            "对 Alakazam 等慢速 Stage-2，前期 tempo 压力可能比复杂 combo 更重要。",
            "先作为 shadow/opponent 补全，而不是立即作为提交候选。",
        ],
        "training_notes": "缺少高分样本，优先从 0801-0815 全部 900+ 或特定 high team 数据补充。",
        "videos": [
            ("YouTube search: Archaludon ex gameplay", yt("Archaludon ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 铝钢桥龙 ex", bili("宝可梦卡牌 铝钢桥龙 ex 卡组")),
        ],
        "extra_refs": [("Limitless deck ranking", "https://limitlesstcg.com/")],
    },
    {
        "slug": "raging_bolt",
        "title": "Raging Bolt ex / 猛雷鼓 ex",
        "local_file": "decks/pool_280_raging_bolt_ex.csv",
        "kaggle_archetype": "",
        "cn": "猛雷鼓 ex",
        "key_cards": ["Raging Bolt ex", "Teal Mask Ogerpon ex", "Sandy Shocks/energy engine", "Professor Sada's Vitality", "Boss's Orders"],
        "plan": "Basic ex burst/race。真实 PTCG 中常与 Ogerpon engine 绑定，通过大量能量和高打点快速拿奖。",
        "combos": [
            "关键是能量进入场和弃牌区的节奏，不是单纯每回合攻击。",
            "对 wall 或 prize denial 局，需要提前判断是否能绕过 active。",
            "Kaggle 分类可能被归入 Teal Mask Ogerpon 或 Other，需要按 deck signature 追踪。",
        ],
        "training_notes": "本地 ladder 分类没有单独稳定统计；若出现在 episode，应单独建 archetype classifier。",
        "videos": [
            ("YouTube search: Raging Bolt Ogerpon gameplay", yt("Raging Bolt Ogerpon Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 猛雷鼓 厄诡椪", bili("宝可梦卡牌 猛雷鼓 厄诡椪 卡组")),
        ],
        "extra_refs": [("Limitless Raging Bolt decks", "https://limitlesstcg.com/decks")],
    },
    {
        "slug": "slowking_seek_inspiration",
        "title": "Slowking Seek Inspiration / 呆壳兽灵感检索",
        "local_file": "decks/pool_322_slowking_seek_inspiration.csv",
        "kaggle_archetype": "",
        "cn": "呆壳兽 Seek Inspiration",
        "key_cards": ["Slowpoke", "Slowking", "Seek Inspiration", "control/search package"],
        "plan": "慢速 control/search。真实环境中 Slowking 有一定占比，但 Kaggle 本地分类暂未稳定覆盖。",
        "combos": [
            "核心是通过 Slowking 的检索/控制拉长游戏，而不是急于 race。",
            "模型需要理解 hand/deck/resource 的长期价值，否则很容易把 search 用成随机抽牌。",
            "先作为 RR 池补全和 rule seed，不优先提交。",
        ],
        "training_notes": "需要新增 deck_plans entry 和 archetype classifier 后再做训练。",
        "videos": [
            ("YouTube search: Slowking Seek Inspiration gameplay", yt("Slowking Seek Inspiration Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 呆壳兽 卡组", bili("宝可梦卡牌 呆壳兽 卡组 对战")),
        ],
        "extra_refs": [("Limitless Slowking decks", "https://limitlesstcg.com/decks/322")],
    },
    {
        "slug": "lillie_clefairy",
        "title": "Lillie's Clefairy ex / 莉莉艾皮皮 ex",
        "local_file": "decks/pool_326_lillie_s_clefairy_ex.csv",
        "kaggle_archetype": "",
        "cn": "莉莉艾皮皮 ex",
        "key_cards": ["Lillie's Clefairy ex", "Lillie support package", "psychic/fairy-style support", "switching"],
        "plan": "support attacker / toolbox。常作为 Ogerpon/Box secondary route，也可能单独成 deck。",
        "combos": [
            "要判断它是主轴还是 secondary attacker。",
            "若只是 Box 中的 coverage card，训练时不能让它污染 Ogerpon 主计划。",
            "单独成 deck 时需要独立抽取 deck-sig corpus。",
        ],
        "training_notes": "Kaggle 当前没有稳定单独 archetype 统计；先从 deck sig 和 card counts 识别。",
        "videos": [
            ("YouTube search: Lillie's Clefairy ex gameplay", yt("Lillie's Clefairy ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 莉莉艾 皮皮 ex", bili("宝可梦卡牌 莉莉艾 皮皮 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "ethan_typhlosion",
        "title": "Ethan's Typhlosion / 小响火暴兽",
        "local_file": "decks/pool_333_ethan_s_typhlosion.csv",
        "kaggle_archetype": "",
        "cn": "小响火暴兽",
        "key_cards": ["Ethan's Cyndaquil", "Ethan's Quilava", "Ethan's Typhlosion", "fire energy engine"],
        "plan": "Fire Stage-2 tempo。重点是火能资源和进化链的稳定衔接。",
        "combos": [
            "先保证 basic 和 Stage 1/2 链路，再考虑高打点回合。",
            "火能进入场/弃牌/手牌的位置决定后续是否能连续攻击。",
            "需要 trace 监控 attach miss 和 evolve miss。",
        ],
        "training_notes": "当前 Kaggle evidence 不足；属于待补 shadow 的环境覆盖卡组。",
        "videos": [
            ("YouTube search: Ethan's Typhlosion gameplay", yt("Ethan's Typhlosion Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 小响 火暴兽", bili("宝可梦卡牌 小响 火暴兽 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "hydrapple",
        "title": "Hydrapple ex / 蜜集大蛇 ex",
        "local_file": "decks/pool_352_hydrapple_ex.csv",
        "kaggle_archetype": "",
        "cn": "蜜集大蛇 ex",
        "key_cards": ["Applin", "Dipplin/Hydrapple ex", "grass energy engine", "healing/acceleration support"],
        "plan": "Grass evolution/resource deck。现实 Limitless 中有 meta 占比，Kaggle 分类暂未稳定单列。",
        "combos": [
            "草能和进化链要同步推进，不能只追 attacker。",
            "对快攻局要判断何时放弃完整 setup 进入 race。",
            "它和 Festival/Ogerpon 的草系 engine 有共性，但不能混训。",
        ],
        "training_notes": "后续可以用 Limitless strong list 先做 deck conversion，再训练 shadow。",
        "videos": [
            ("YouTube search: Hydrapple ex gameplay", yt("Hydrapple ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 蜜集大蛇 ex", bili("宝可梦卡牌 蜜集大蛇 ex 卡组")),
        ],
        "extra_refs": [("Limitless Hydrapple decks", "https://limitlesstcg.com/decks/352")],
    },
    {
        "slug": "rocket_honchkrow",
        "title": "Team Rocket's Honchkrow / 火箭队乌鸦头头",
        "local_file": "decks/pool_356_rocket_s_honchkrow.csv",
        "kaggle_archetype": "",
        "cn": "火箭队乌鸦头头",
        "key_cards": ["Team Rocket's Murkrow", "Team Rocket's Honchkrow", "Team Rocket support", "disruption package"],
        "plan": "Team Rocket disruption/tempo。与 TR Mewtwo 共享部分 support，但 payoff 不同。",
        "combos": [
            "先判断是打干扰还是打奖赏 race。",
            "不能把它混入 TR Mewtwo 模型；二者需要不同 plan 标签。",
            "对控制局要追踪对手手牌/公开 reveal 信息。",
        ],
        "training_notes": "缺少 Kaggle 高分样本；先作为 opponent coverage。",
        "videos": [
            ("YouTube search: Team Rocket's Honchkrow gameplay", yt("Team Rocket's Honchkrow Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 火箭队 乌鸦头头", bili("宝可梦卡牌 火箭队 乌鸦头头 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "metagross_metal_maker",
        "title": "Metagross Metal Maker / 巨金怪 Metal Maker",
        "local_file": "decks/pool_361_metagross_metal_maker.csv",
        "kaggle_archetype": "",
        "cn": "巨金怪 Metal Maker",
        "key_cards": ["Beldum", "Metang/Metagross", "Metal Maker", "metal energy", "evolution search"],
        "plan": "Metal acceleration Stage-2。核心是通过 Metal Maker/金属能加速建立连续攻击。",
        "combos": [
            "Metal Maker 要在能量和可贴目标都合适时优先使用。",
            "进化链和能量加速要一起监控，不能只看 attack available。",
            "和 Archaludon 共享金属能主题，但决策节奏不同。",
        ],
        "training_notes": "Azul 有 Pitch Black decklist，可作为 strategy seed；Kaggle evidence 需要另行确认。",
        "videos": [
            ("AzulGG: Metagross - Pitch Black", "https://www.youtube.com/watch?v=zbI4zAlObZ4"),
            ("YouTube search: Metagross Metal Maker gameplay", yt("Metagross Metal Maker Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "hop_trevenant",
        "title": "Hop's Trevenant / 赫普朽木妖",
        "local_file": "decks/pool_363_hop_s_trevenant.csv",
        "kaggle_archetype": "Hop Trevenant",
        "cn": "赫普朽木妖",
        "key_cards": ["Hop's Phantump", "Hop's Trevenant", "Hop support", "disruption/control package"],
        "plan": "Stage-1/2 control variant。0812 高分 band 只出现极少行，不能直接当主流。",
        "combos": [
            "控制卡组要把对手资源消耗、手牌公开信息和自身攻击窗口一起看。",
            "如果模型没有 history/reveal memory，很容易只做当下最大分动作。",
            "先用 trace 观察真实高分局，再写规则。",
        ],
        "training_notes": "Kaggle evidence 极弱；作为低权重环境覆盖。",
        "videos": [
            ("YouTube search: Hop's Trevenant gameplay", yt("Hop's Trevenant Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 赫普 朽木妖", bili("宝可梦卡牌 赫普 朽木妖 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "mega_greninja",
        "title": "Mega Greninja ex / 超级甲贺忍蛙 ex",
        "local_file": "decks/pool_370_mega_greninja_ex.csv",
        "kaggle_archetype": "",
        "cn": "超级甲贺忍蛙 ex",
        "key_cards": ["Froakie", "Frogadier", "Mega Greninja ex", "water/search package", "switching"],
        "plan": "Water Mega attacker。需要兼顾进化链、切换和 bench pressure。",
        "combos": [
            "水系进化线不能只看当回合攻击；要保证下回合还能继续输出。",
            "若有 bench damage/狙击效果，需要加入 target map 诊断。",
            "当前 Kaggle evidence 不足，先作为扩展 deck pool。",
        ],
        "training_notes": "待新增 deck_plans entry 和高分样本抽取。",
        "videos": [
            ("YouTube search: Mega Greninja ex gameplay", yt("Mega Greninja ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 超级甲贺忍蛙 ex", bili("宝可梦卡牌 超级甲贺忍蛙 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "beedrill",
        "title": "Beedrill ex / 大针蜂 ex",
        "local_file": "decks/pool_371_beedrill_ex.csv",
        "kaggle_archetype": "",
        "cn": "大针蜂 ex",
        "key_cards": ["Weedle", "Kakuna", "Beedrill ex", "grass/search package", "poison/disruption tools"],
        "plan": "Grass evolution tempo/disruption。具体要看 decklist 是否走纯进化攻击或异常状态/干扰路线。",
        "combos": [
            "先识别 evolution chain 是否完整，再决定是否强行进攻。",
            "对低 HP basic 可以抢奖；对高 HP ex 要计算是否需要多回合铺伤害。",
            "如果包含 poison/disruption，训练指标要单独监控这些非攻击收益。",
        ],
        "training_notes": "Azul 有 Pitch Black decklist，可作为公开策略入口；Kaggle 样本需补。",
        "videos": [
            ("AzulGG: Beedrill - Pitch Black", "https://www.youtube.com/watch?v=k-UkIhDA75A"),
            ("YouTube search: Beedrill ex gameplay", yt("Beedrill ex Pokemon TCG gameplay deck profile")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "sylveon_safeguard",
        "title": "Sylveon Safeguard / 仙子伊布 Safeguard",
        "local_file": "decks/pool_373_sylveon_safeguard.csv",
        "kaggle_archetype": "",
        "cn": "仙子伊布 Safeguard",
        "key_cards": ["Eevee", "Sylveon", "Safeguard effect", "control/support package"],
        "plan": "Safeguard/control。核心是让特定防护效果对上 ex-heavy 攻击，而不是盲目 race。",
        "combos": [
            "模型必须知道哪些 opponent attacker 被 Safeguard 阻止。",
            "一旦 active wall 正确，要优先维持 board 和资源，不要无意义切走。",
            "对非 ex 或绕过效果的对手要快速切换计划。",
        ],
        "training_notes": "非常适合 rule-overlay，但需要先验证 cg 引擎对 Safeguard 的具体实现。",
        "videos": [
            ("YouTube search: Sylveon Safeguard gameplay", yt("Sylveon Safeguard Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 仙子伊布 Safeguard", bili("宝可梦卡牌 仙子伊布 Safeguard 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
    {
        "slug": "mega_abomasnow",
        "title": "Mega Abomasnow ex / 超级暴雪王 ex",
        "local_file": "decks/pool_400_mega_abomasnow_ex.csv",
        "kaggle_archetype": "",
        "cn": "超级暴雪王 ex",
        "key_cards": ["Snover", "Mega Abomasnow ex", "grass/water energy package", "healing or tank tools"],
        "plan": "Mega tank/tempo。具体强度取决于 HP、回复和能量节奏。",
        "combos": [
            "先确认能量成本和连续攻击能力。",
            "如果 deck 走 tank/heal，要把伤害阈值和 retreat/switch 作为显式信号。",
            "Kaggle evidence 暂少，适合作为 future pool。",
        ],
        "training_notes": "先补数据和合法性测试，不建议直接进提交候选。",
        "videos": [
            ("YouTube search: Mega Abomasnow ex gameplay", yt("Mega Abomasnow ex Pokemon TCG gameplay deck profile")),
            ("Bilibili search: 超级暴雪王 ex", bili("宝可梦卡牌 超级暴雪王 ex 卡组")),
        ],
        "extra_refs": [("Play Limitless deck search", "https://play.limitlesstcg.com/decks")],
    },
]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_date(rows: list[dict[str, str]]) -> str:
    dates = sorted({r.get("date", "") for r in rows if r.get("date")})
    return dates[-1] if dates else ""


def best_sigs(sig_rows: list[dict[str, str]], arch: str, n: int = 5) -> list[dict[str, str]]:
    if not arch:
        return []
    date = latest_date(sig_rows)
    rows = [r for r in sig_rows if r.get("date") == date and r.get("archetype") == arch]
    for r in rows:
        for k in ("max_score", "rows", "win_rate_no_draw", "avg_score", "wins", "losses"):
            try:
                r[k] = float(r[k]) if "." in str(r[k]) else int(r[k])
            except Exception:
                pass
    rows.sort(key=lambda r: (float(r.get("max_score", 0)), int(r.get("rows", 0)), float(r.get("win_rate_no_draw", 0))), reverse=True)
    return rows[:n]


def band_rows(band_rows_all: list[dict[str, str]], arch: str) -> list[dict[str, str]]:
    if not arch:
        return []
    date = latest_date(band_rows_all)
    rows = [r for r in band_rows_all if r.get("date") == date and r.get("archetype") == arch]
    order = {"1200+": 0, "1100-1199": 1, "1000-1099": 2, "900-999": 3, "800-899": 4, "700-799": 5, "600-699": 6}
    rows.sort(key=lambda r: order.get(r.get("score_band"), 99))
    return rows[:8]


def matchup_rows(matchup_all: list[dict[str, str]], arch: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not arch:
        return [], []
    rows = []
    for r in matchup_all:
        if r.get("archetype") != arch:
            continue
        if r.get("opponent_archetype") == arch:
            continue
        try:
            games = int(float(r.get("games", 0)))
            wr = float(r.get("win_rate", 0))
        except Exception:
            continue
        if games < 100:
            continue
        rr = dict(r)
        rr["_games"] = games
        rr["_wr"] = wr
        rows.append(rr)
    weak = sorted(rows, key=lambda r: (r["_wr"], -r["_games"]))[:5]
    strong = sorted(rows, key=lambda r: (r["_wr"], r["_games"]), reverse=True)[:5]
    return strong, weak


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["|" + "|".join(str(x) for x in row) + "|" for row in rows]
    return "\n".join(out)


def render_sig_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前本地 0812/0813 ladder 分段统计没有稳定签名。先不要把该 archetype 作为高权重 RR 结论来源。"
    body = []
    for r in rows:
        body.append([
            str(r.get("deck_sig", "")),
            str(r.get("max_score", "")),
            str(r.get("rows", "")),
            f"{float(r.get('win_rate_no_draw', 0)):.3f}",
            str(r.get("top_teams", ""))[:90],
        ])
    return md_table(["deck_sig", "max_score", "rows", "WR(no draw)", "top teams"], body)


def render_band_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "无稳定分段统计。"
    body = []
    for r in rows:
        body.append([
            r.get("score_band", ""),
            r.get("rows", ""),
            f"{float(r.get('win_rate_no_draw', 0)):.3f}",
            r.get("max_score", ""),
            r.get("top_sigs", "")[:80],
        ])
    return md_table(["band", "rows", "WR", "max_score", "top sigs"], body)


def render_matchups(strong: list[dict[str, str]], weak: list[dict[str, str]]) -> str:
    if not strong and not weak:
        return "本地 matchup 先验不足。需要从最新 episode 重算 `archetype_matchups.csv`。"
    rows = []
    seen: set[str] = set()
    for r in strong:
        opp = r.get("opponent_archetype", "")
        seen.add(opp)
        rows.append(["高胜率", opp, str(r["_games"]), f"{r['_wr']:.3f}"])
    low_added = False
    for r in weak:
        opp = r.get("opponent_archetype", "")
        if opp in seen:
            continue
        low_added = True
        rows.append(["低胜率/接近五五", opp, str(r["_games"]), f"{r['_wr']:.3f}"])
    if weak and not low_added:
        rows.append(["低胜率/接近五五", "无足够非重复样本", "-", "-"])
    return md_table(["类型", "对手 archetype", "games", "WR"], rows)


def shell_quote_single(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def best_sig_for_commands(sig_rows: list[dict[str, str]], arch: str) -> str:
    rows = best_sigs(sig_rows, arch, 1)
    return str(rows[0].get("deck_sig", "")) if rows else "REPLACE_WITH_DECK_SIG"


def render_workflow(item: dict[str, object], sig_rows: list[dict[str, str]], weak: list[dict[str, str]]) -> str:
    slug = str(item["slug"])
    arch = str(item.get("kaggle_archetype", ""))
    deck = str(item["local_file"])
    sig = best_sig_for_commands(sig_rows, arch)
    arch_arg = f"  --archetype {shell_quote_single(arch)} \\\n" if arch else ""
    deck_sig_note = (
        f"当前自动填入的 `DECK_SIG={sig}` 来自 2026-08-13 ladder 强签名表。"
        if sig != "REPLACE_WITH_DECK_SIG"
        else "当前没有稳定 ladder 强签名。先从最新 `pool_manifest.csv` / episode replay 中确认 `DECK_SIG`，或补 archetype classifier 后再训练。"
    )
    weak_opps = [r.get("opponent_archetype", "") for r in weak if r.get("opponent_archetype")]
    weak_text = "、".join(weak_opps[:4]) if weak_opps else "暂无足够样本；先用 balanced RR 池覆盖所有主流 archetype"
    return f"""## 单独训练

目标是先训练一个 deck-sig specialist，而不是把多个不同 game plan 混在一起。{deck_sig_note}

```bash
export CORPUS=${{CORPUS:-data/bc_corpus_banded_latest}}
export ARCHETYPE={shell_quote_single(arch or str(item['title']))}
export DECK_SIG={sig}
export DECK={shell_quote_single(deck)}
export OUT_DIR=checkpoints/decks
export LOG_DIR=logs/deck_train
mkdir -p "$OUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} python3 -u tools/bc2_train.py \\
  --corpus "$CORPUS" \\
{arch_arg}  --deck-sig "$DECK_SIG" \\
  --score-bands 900-999 1000-1099 1100-1199 1200+ \\
  --date-from 2026-08-01 \\
  --date-to 2026-08-15 \\
  --epochs 8 \\
  --batch-size 1024 \\
  --width 512 \\
  --arch pointer \\
  --win-weight 1.5 \\
  --loss-weight 0.4 \\
  --draw-weight 0.8 \\
  --split-by-game \\
  --load-progress-every 200000 \\
  --checkpoint-every 1 \\
  --save "$OUT_DIR/bc2_{slug}_${{DECK_SIG}}_single.npz" \\
  2>&1 | tee "$LOG_DIR/bc2_{slug}_${{DECK_SIG}}_single.log"
```

如果该 deck 是 Stage-2、control 或需要明显全局视角的卡组，可以追加一组对照，不覆盖上面的 baseline:

```bash
CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-1}} python3 -u tools/bc2_train.py \\
  --corpus "$CORPUS" \\
{arch_arg}  --deck-sig "$DECK_SIG" \\
  --score-bands 900-999 1000-1099 1100-1199 1200+ \\
  --date-from 2026-08-01 \\
  --date-to 2026-08-15 \\
  --epochs 8 \\
  --batch-size 768 \\
  --width 768 \\
  --arch cross_attn \\
  --state-layers 2 \\
  --step-plan \\
  --step-plan-loss-weight 0.2 \\
  --win-weight 1.5 \\
  --loss-weight 0.4 \\
  --draw-weight 0.8 \\
  --split-by-game \\
  --load-progress-every 200000 \\
  --checkpoint-every 1 \\
  --save "$OUT_DIR/bc2_{slug}_${{DECK_SIG}}_cross_stepplan.npz" \\
  2>&1 | tee "$LOG_DIR/bc2_{slug}_${{DECK_SIG}}_cross_stepplan.log"
```

训练日志里要重点看: train/val 是否同步下降、best epoch 是否不是过早停止、`first_action`/`policy_raw` 是否恶化、样本数是否足够、是否过滤到目标 `deck_sig`。

## Random 测试

先用 300 局快速 gate，候选提交前再跑 500 或 1000 局。random 不能代表 Kaggle 强度，但如果这里明显不稳，通常说明基础启动、进化或攻击流程没学好。

```bash
export POLICY="$OUT_DIR/bc2_{slug}_${{DECK_SIG}}_single.npz"
mkdir -p logs/deck_eval/{slug}

python3 tools/eval_bc.py "$POLICY" \\
  --deck "$DECK" \\
  --games 300 \\
  --workers 16 \\
  --progress-every 50 \\
  --max-turns 700 \\
  2>&1 | tee logs/deck_eval/{slug}/random_g300.log
```

候选提交前:

```bash
python3 tools/eval_bc.py "$POLICY" \\
  --deck "$DECK" \\
  --games 500 \\
  --workers 32 \\
  --progress-every 50 \\
  --max-turns 700 \\
  2>&1 | tee logs/deck_eval/{slug}/random_g500.log
```

## 推荐 RR 测试

优先测两类池:

- `balanced` 池: 每个主流 archetype 至少 1-2 个高质量 shadow，避免低质量卡组拉高平均胜率。
- `latest ladder` 池: 按最新 Kaggle 分段权重保留环境主流签名，模拟从 600 分往上爬时可能遇到的对手。

该 deck 当前优先关注的低胜率/接近五五对手: {weak_text}。

先构建一个每类 top2 的轻量 RR 池:

```bash
export RR_MANIFEST=${{RR_MANIFEST:-logs/rr_pool_latest/filtered_balanced.csv}}
export FOCUS_MANIFEST=logs/deck_eval/{slug}/rr_pool_top2_per_arch.csv
mkdir -p logs/deck_eval/{slug}

python3 tools/select_manifest_top_per_archetype.py \\
  --manifest "$RR_MANIFEST" \\
  --max-per-arch 2 \\
  --out "$FOCUS_MANIFEST"
```

candidate-only RR:

```bash
python3 tools/eval_round_robin.py \\
  --entry {slug}="$POLICY:$DECK" \\
  --manifest "$FOCUS_MANIFEST" \\
  --candidate-only \\
  --skip-bad-entries \\
  --games 100 \\
  --workers 32 \\
  --max-turns 700 \\
  --progress-every 20 \\
  --out-csv logs/deck_eval/{slug}/rr_top2_per_arch_g100.csv \\
  2>&1 | tee logs/deck_eval/{slug}/rr_top2_per_arch_g100.log
```

如果 RR 暴露出具体坏 matchup，再用 fixed-seed trace 复现。`OPP_ENTRY` 可以直接从 manifest 的 `eval_entry` 列复制:

```bash
export OPP_ENTRY='opponent_name=checkpoints/opponent.npz:logs/opponent_deck.csv'

python3 tools/trace_matchup_decisions.py \\
  --candidate {slug}="$POLICY:$DECK" \\
  --opponent "$OPP_ENTRY" \\
  --games 20 \\
  --seed 20260817 \\
  --max-turns 700 \\
  --progress-every 1 \\
  --out-prefix logs/deck_eval/{slug}/trace_vs_bad_opp
```

## Kaggle episode 动态回放

可以引入，但不要把它当成可离线复现的唯一记录。Kaggle 网页动态 replay 依赖登录态、网页脚本和当前 UI，静态 Markdown 里通常只能放 episode/submission 链接；真正可复现的材料应下载 replay JSON 并写入本地日志。

人工查看流程:

1. 打开 Kaggle 比赛页面的 Submissions 或 Episodes。
2. 找到目标 `submission_id` 的 episode。
3. 打开 episode detail / replay 页面，逐回合看 setup、attach、evolve、ability、attack、target 和最终 status。
4. 把关键 episode id 写回本文件或 `AGENT_HANDOFF.md`，并下载 replay JSON 做本地 trace。

本地 replay 分析:

```bash
export SUB_ID=REPLACE_WITH_KAGGLE_SUBMISSION_ID
export REPLAY_DIR=logs/kaggle_replay_${{SUB_ID}}_{slug}
mkdir -p "$REPLAY_DIR"

python3 tools/analyze_kaggle_replays.py "$SUB_ID" \\
  --deck "$DECK" \\
  --known-decks-dir logs/ladder_pool_0805_all/decks \\
  --cache-dir "$REPLAY_DIR/cache" \\
  --out "$REPLAY_DIR/episodes.csv" \\
  --summary-out "$REPLAY_DIR/summary_by_opponent_deck_sig.csv" \\
  --group-by opponent_deck_sig \\
  --max-episodes 200 \\
  --write-opponent-decks \\
  --opponent-decks-dir "$REPLAY_DIR/opponent_decks" \\
  --progress-every 10
```

把 live replay 中遇到的新对手转成本地 RR:

```bash
python3 tools/make_kaggle_opp_round_robin_cmd.py \\
  --policy-name {slug} \\
  --policy "$POLICY" \\
  --deck "$DECK" \\
  --opp-dir "$REPLAY_DIR/opponent_decks" \\
  --games 100 \\
  --progress-every 20 \\
  --out-csv "$REPLAY_DIR/local_vs_live_opponents_g100.csv" \\
  > "$REPLAY_DIR/run_live_opponent_rr.sh"

bash "$REPLAY_DIR/run_live_opponent_rr.sh"
```

这样动态 replay 的结论会落到 `episodes.csv`、`summary_by_opponent_deck_sig.csv`、opponent deck CSV 和本地 RR 结果里，后续才能比较“线上输在哪里”和“本地是否能复现”。
"""


def render_deck_doc(item: dict[str, object], sig_rows: list[dict[str, str]], band_all: list[dict[str, str]], matchup_all: list[dict[str, str]]) -> str:
    arch = str(item.get("kaggle_archetype", ""))
    strong, weak = matchup_rows(matchup_all, arch)
    sig_table = render_sig_table(best_sigs(sig_rows, arch))
    band_table = render_band_table(band_rows(band_all, arch))
    matchup_table = render_matchups(strong, weak)
    workflow = render_workflow(item, sig_rows, weak)

    videos = item["videos"]
    video_lines = "\n".join(f"- [{name}]({url})" for name, url in videos)
    refs = item.get("extra_refs", [])
    ref_lines = "\n".join(f"- [{name}]({url})" for name, url in refs) if refs else "- 暂无额外资料，优先从 Limitless / Play Limitless / YouTube 搜索补充。"
    key_lines = "\n".join(f"- {x}" for x in item["key_cards"])
    combo_lines = "\n".join(f"- {x}" for x in item["combos"])

    return f"""# {item['title']}

资料更新时间: 2026-08-17  
本地 deck 模板: `{item['local_file']}`  
Kaggle 统计 archetype: `{arch or '暂未单独识别'}`

## 当前 Kaggle 强签名

依据: `logs/ladder_distribution_0812_0813_20260814/deck_sig_summary.csv`，优先看 2026-08-13 的最高分、行数和无平局胜率。

{sig_table}

## 分段分布

{band_table}

## 打法摘要

{item['plan']}

## 关键牌

{key_lines}

## 关键 combo / 决策点

{combo_lines}

## 优劣势对局先验

依据: `logs/matchup_notes_20260805/0724_0804_score900/archetype_matchups.csv`。这是 Kaggle episode 先验，不等同于真实 PTCG 线下胜率。

{matchup_table}

## 训练和评测注意事项

{item['training_notes']}

{workflow}

评测时至少要覆盖:

- random gate: 确认基础操作不会输给 legal random；
- latest ladder RR: 用最新分段中的主流 deck-sig shadow；
- historical strong RR: 与历史高光模型/强 archetype 做横向比较；
- fixed-seed loss trace: 对每个明显输局保留可复现 seed，逐回合看 setup、attach、evolve、ability、attack 和 target。

## 视频 / 实战

{video_lines}

如果链接是搜索入口，需要优先选择 2026 轮换后、与当前 Kaggle 可用卡池接近的视频；不要直接把旧环境打法写成强规则。

## 卡面素材

推荐只把卡图作为本地研究缓存或文档阅读辅助，不要把下载的卡图提交进仓库或 submission 包。

{md_table(['来源', '链接', '用途'], [[name, f'[{name}]({url})', note] for name, url, note in IMAGE_SOURCES])}

## 后续可转成规则/trace 的问题

- 当前强签名是否真的覆盖了 600 -> 1100 的爬分阶段，而不只是高分段幸存局？
- 失败 trace 中是否存在明确 miss: setup/evolve/attach/ability/attack/target/reveal memory？
- 该 archetype 的强胜局是稳定策略，还是对手事故/抽牌运气？
- 如果要加 rule overlay，触发条件必须能从 observation/legal options 中稳定判断，且 fixed-seed replay 应证明行为改变。
"""


def render_image_doc() -> str:
    return f"""# 卡面图片与外部素材

宝可梦卡面、卡文和商标通常受版权保护。这里的目标是为研究、trace 审阅和人工理解提供素材入口，不是把卡图打包进仓库或 submission。

## 推荐来源

{md_table(['来源', '链接', '适合用途'], [[name, f'[{name}]({url})', note] for name, url, note in IMAGE_SOURCES])}

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
"""


def render_index(sig_rows: list[dict[str, str]]) -> str:
    rows = []
    for item in ARCHETYPES:
        arch = str(item.get("kaggle_archetype", ""))
        best = best_sigs(sig_rows, arch, 1)
        if best:
            b = best[0]
            best_txt = f"{b.get('deck_sig')} / max {b.get('max_score')} / rows {b.get('rows')} / WR {float(b.get('win_rate_no_draw', 0)):.3f}"
        else:
            best_txt = "暂无稳定 Kaggle 分段签名"
        rows.append([
            f"[{item['title']}]({item['slug']}.md)",
            f"`{item['local_file']}`",
            arch or "待识别",
            best_txt,
        ])
    return f"""# 卡组资料索引

这个目录把本地 `decks/` 模板、Kaggle ladder 分段、真实 PTCG 资料和视频入口放在一起。阅读顺序建议:

1. 先看本页的“强签名总表”，决定 RR/shadow 池要覆盖哪些 archetype。
2. 再看具体卡组文件里的打法、关键 combo、视频和 matchup 先验。
3. 按每个卡组文件里的“单独训练 / Random 测试 / 推荐 RR 测试 / Kaggle episode 动态回放”跑完整验证链。
4. 最后回到 `tools/trace_matchup_decisions.py`、`ptcg_rl/deck_plans.py`、`ptcg_rl/rule_overlay.py` 把策略变成可验证规则或训练信号。

## 强签名总表

数据主依据是 2026-08-13 ladder 分段统计。对于没有稳定样本的本地模板，只记录为待补充，不直接当作强环境结论。

{md_table(['卡组文档', '本地 deck', 'Kaggle archetype', '当前强签名'], rows)}

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
"""


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    sig_rows = load_csv(SIG_SUMMARY)
    band_all = load_csv(BAND_SUMMARY)
    matchup_all = load_csv(MATCHUPS)

    (DOC_DIR / "00_deck_index.md").write_text(render_index(sig_rows), encoding="utf-8")
    (DOC_DIR / "card_image_sources.md").write_text(render_image_doc(), encoding="utf-8")
    for item in ARCHETYPES:
        slug = str(item["slug"])
        (DOC_DIR / f"{slug}.md").write_text(render_deck_doc(item, sig_rows, band_all, matchup_all), encoding="utf-8")

    print(f"wrote {len(ARCHETYPES) + 2} files under {DOC_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
