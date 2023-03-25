ATK_RANGE = 3
N_TEAM = 16
N_PROF = 8
N_TILE_TYPE = 16
MAX_STEP = 1024
TERRAIN_SIZE = 160
HALF_TERRAIN_SIZE = TERRAIN_SIZE // 2
MAP_LEFT = 16
MAP_RIGHT = 144
MAP_SIZE = MAP_RIGHT - MAP_LEFT + 1
WINDOW = 15
WINDOW_CENTER = 7
PASSABLE_TILES = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
OBSTACLE_TILES = [t for t in range(N_TILE_TYPE) if t not in PASSABLE_TILES]

NEGATIVE_POP = -1
NEUTRAL_POP = -2
HOSTILE_POP = -3

EQUIPMENT = [
    'HatLevel',
    'BottomLevel',
    'TopLevel',
    'HeldLevel',
    'AmmunitionLevel',
]
ATK_TYPE = [
    'Melee',
    'Range',
    'Mage',
]
N_ATK_TYPE = len(ATK_TYPE)



N_ITEM_LVL = 10
N_ITEM_SLOT = 12
N_ITEM_LIMIT = 11  # if surpass the limit, automatically sell one


