from __future__ import annotations

FEATURE_VERSION = "v15_turn_block_plan_v1"

DEFAULT_HISTORY_K = 48
DEFAULT_PLAN_STEPS = 4
DEFAULT_MAX_OPTIONS = 64
MAX_SELECT_COUNT = 10
KNOWN_OPP_CARDS = 16

# Engine option type ids as exported in observations.
TYPE_NUMBER = 0
TYPE_YES = 1
TYPE_NO = 2
TYPE_CARD = 3
TYPE_TOOL_CARD = 4
TYPE_ENERGY_CARD = 5
TYPE_ENERGY = 6
TYPE_PLAY = 7
TYPE_ATTACH = 8
TYPE_EVOLVE = 9
TYPE_ABILITY = 10
TYPE_DISCARD = 11
TYPE_RETREAT = 12
TYPE_ATTACK = 13
TYPE_END = 14
TYPE_SKILL = 15
TYPE_SPECIAL_CONDITION = 16
N_ACTION_TYPES = 18

TYPE_NAMES = {
    TYPE_NUMBER: "NUMBER",
    TYPE_YES: "YES",
    TYPE_NO: "NO",
    TYPE_CARD: "CARD",
    TYPE_TOOL_CARD: "TOOL_CARD",
    TYPE_ENERGY_CARD: "ENERGY_CARD",
    TYPE_ENERGY: "ENERGY",
    TYPE_PLAY: "PLAY",
    TYPE_ATTACH: "ATTACH",
    TYPE_EVOLVE: "EVOLVE",
    TYPE_ABILITY: "ABILITY",
    TYPE_DISCARD: "DISCARD",
    TYPE_RETREAT: "RETREAT",
    TYPE_ATTACK: "ATTACK",
    TYPE_END: "END",
    TYPE_SKILL: "SKILL",
    TYPE_SPECIAL_CONDITION: "SPECIAL_CONDITION",
}

KEY_ACTION_TYPES = (
    TYPE_PLAY,
    TYPE_ATTACH,
    TYPE_EVOLVE,
    TYPE_ABILITY,
    TYPE_RETREAT,
    TYPE_ATTACK,
    TYPE_END,
)

# SelectContext::DamageCounterAny after ToJson export.
DAMAGE_COUNTER_ANY_CONTEXT = 14

# Canonical event fields stored per history token.
EVENT_FIELDS = (
    "event_type",
    "source",
    "owner",
    "card",
    "card2",
    "attack",
    "context",
    "select_type",
    "from_area",
    "to_area",
    "value",
    "turn_delta",
    "step_delta",
    "same_turn",
    "mask",
)

EVENT_SOURCE_NONE = 0
EVENT_SOURCE_OWN_ACTION = 1
EVENT_SOURCE_PUBLIC_LOG = 2

# Coarse block/plan modes. These are not final strategy labels; they are
# train-time probes that tell us whether the model can represent turn intent.
MODE_NONE = 0
MODE_SETUP = 1
MODE_RESOURCE = 2
MODE_ATTACK = 3
MODE_DISRUPT = 4
MODE_DAMAGE_PLAN = 5
MODE_SWITCH = 6
MODE_END = 7
N_PLAN_MODES = 8

MODE_NAMES = {
    MODE_NONE: "none",
    MODE_SETUP: "setup",
    MODE_RESOURCE: "resource",
    MODE_ATTACK: "attack",
    MODE_DISRUPT: "disrupt",
    MODE_DAMAGE_PLAN: "damage_plan",
    MODE_SWITCH: "switch",
    MODE_END: "end",
}

# Public card-movement area ids from engine Json logs.
AREA_DECK = 1
AREA_HAND = 2
AREA_TRASH = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_LOOKING = 12
AREA_PLAYING = 13
AREA_DECK_BOTTOM = 14
AREA_TEMPORARY = 24

PUBLIC_TO_HAND_AREAS = {AREA_DECK, AREA_TRASH, AREA_LOOKING, AREA_DECK_BOTTOM, AREA_TEMPORARY}
HAND_RESET_TO_AREAS = {AREA_DECK, AREA_DECK_BOTTOM}


def type_name(type_id: int) -> str:
    return TYPE_NAMES.get(int(type_id), f"TYPE_{int(type_id)}")


def mode_name(mode_id: int) -> str:
    return MODE_NAMES.get(int(mode_id), f"MODE_{int(mode_id)}")

