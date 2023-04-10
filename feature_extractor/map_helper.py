
import nmmo
import numpy as np
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material
from nmmo.systems.item import ItemState
from feature_extractor.entity_helper import EntityHelper
from nmmo.io import action

from feature_extractor.game_state import GameState
from team_helper import TeamHelper


EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

DEFOGGING_VALUE = 16
VISITATION_MEMORY = 100

N_CH = 7
IMG_SIZE = 25
DUMMY_IMG_FEAT = np.zeros((N_CH, IMG_SIZE, IMG_SIZE))

TEAMMATE_REPR = 1 / 5.
ENEMY_REPR = 2 / 5.
NEGATIVE_REPR = 3 / 5.
NEUTRAL_REPR = 4 / 5.
HOSTILE_REPR = 1.

DEPLETION_MAP = {
  material.Forest.index: material.Scrub.index,
  material.Tree.index: material.Stump.index,
  material.Ore.index: material.Slag.index,
  material.Crystal.index: material.Fragment.index,
  material.Herb.index: material.Weeds.index,
  material.Fish.index: material.Ocean.index,
}

class MapHelper:
  def __init__(self, config: nmmo.config.Config, team_id: int, team_size: int) -> None:
    self.config = config

    self.MAP_SIZE = self.config.MAP_SIZE
    self.TEAM_SIZE = team_size
    self.team_id = team_id

    self.tile_map = None
    self.fog_map = None
    self.visit_map = None
    self.poison_map = None
    self.entity_map = None

    self.X_IMG = np.arange(self.MAP_SIZE+1).repeat(self.MAP_SIZE+1)\
      .reshape(self.MAP_SIZE+1, self.MAP_SIZE+1)
    self.Y_IMG = self.X_IMG.transpose(1, 0)

  def reset(self):
    self.tile_map = self._get_init_tile_map()
    self.fog_map = np.zeros((self.MAP_SIZE+1, self.MAP_SIZE+1))
    self.visit_map = np.zeros((self.TEAM_SIZE, self.MAP_SIZE+1, self.MAP_SIZE+1))
    self.poison_map = self._get_init_poison_map()
    self.entity_map = None

  def update(self, obs, game_state: GameState, entity_helper: EntityHelper):
    if game_state.curr_step % 16 == 15:
      self.poison_map += 1  # poison shrinking

    self.fog_map = np.clip(self.fog_map - 1, 0, DEFOGGING_VALUE)  # decay
    self.visit_map = np.clip(self.visit_map - 1, 0, VISITATION_MEMORY)  # decay

    entity_map = np.zeros((5, self.MAP_SIZE+1, self.MAP_SIZE+1))

    for player_id, player_obs in obs.items():
      # mark tile
      tile_obs = player_obs['Tile']
      tile_pos = tile_obs[:, TileAttr["row"]:TileAttr["col"]+1].astype(int)
      tile_type = tile_obs[:, TileAttr["material_id"]].astype(int)
      self._mark_point(self.fog_map, tile_pos, DEFOGGING_VALUE)
      x, y = tile_pos[0]
      self.tile_map[
         x:x+self.config.PLAYER_VISION_DIAMETER,
         y:y+self.config.PLAYER_VISION_DIAMETER
      ] = tile_type.reshape(
         self.config.PLAYER_VISION_DIAMETER,
         self.config.PLAYER_VISION_DIAMETER
      )

      # mark team/enemy/npc
      entity_obs = player_obs['Entity']
      entity_positions = entity_obs[:, EntityAttr["row"]:EntityAttr["col"]+1].astype(int)
      entity_populations = entity_obs[:, EntityAttr["population_id"]].astype(int)

      # xcxc rewrite to not use population
      self._mark_point(entity_map[0], entity_positions, entity_populations == self.team_id)  # team
      self._mark_point(entity_map[1], entity_positions, np.logical_and(entity_populations != self.team_id, entity_populations > 0))  # enemy
      self._mark_point(entity_map[2], entity_positions, entity_populations == -1)  # negative
      self._mark_point(entity_map[3], entity_positions, entity_populations == -2)  # neutral
      self._mark_point(entity_map[4], entity_positions, entity_populations == -3)  # hostile

      # update visit map
      # xcxc
      # self._mark_point(self.visit_map[player_id], entity_helper._member_location[player_id], VISITATION_MEMORY)

      # change tile in advance
      for pos, pop in zip(entity_positions, entity_populations):
        if pop >= 0:  # is player
          new_tile = DEPLETION_MAP.get(self.tile_map[pos[0], pos[1]])
          if new_tile is not None:
            self.tile_map[pos[0], pos[1]] = new_tile

            for row_offset in range(-1, 2):
              for col_offset in range(-1, 2):
                if self.tile_map[pos[0]+row_offset, pos[1]+col_offset] == material.Fish.index:
                  self.tile_map[pos[0]+row_offset, pos[1]+col_offset] = material.Ocean.index

    self.entity_map = entity_map[0] * TEAMMATE_REPR + entity_map[1] * ENEMY_REPR + \
      entity_map[2] * NEGATIVE_REPR + entity_map[3] * NEUTRAL_REPR + entity_map[4] * HOSTILE_REPR

  def extract_tile_feature(self, obs, entity_helper: EntityHelper):
    imgs = []
    for i in range(self.TEAM_SIZE):
      # replace with dummy feature if dead
      if i-1 not in obs:
        imgs.append(DUMMY_IMG_FEAT)
        continue

      curr_pos = entity_helper._member_location[i]
      l, r = curr_pos[0] - IMG_SIZE // 2, curr_pos[0] + IMG_SIZE // 2 + 1
      u, d = curr_pos[1] - IMG_SIZE // 2, curr_pos[1] + IMG_SIZE // 2 + 1
      tile_img = self.tile_map[l:r, u:d] / (1 + max(material.All.indices))
      # obstacle_img = np.sum([self.tile_map[l:r, u:d] == t for t in material.Impassible.indicies], axis=0)
      entity_img = self.entity_map[l:r, u:d]
      poison_img = np.clip(self.poison_map[l:r, u:d], 0, np.inf) / 20.
      fog_img = self.fog_map[l:r, u:d] / DEFOGGING_VALUE
      # view_img = (fog_img == 1.).astype(np.float32)
      visit_img = self.visit_map[i][l:r, u:d] / VISITATION_MEMORY
      coord_imgs = [self.X_IMG[l:r, u:d] / self.MAP_SIZE, self.Y_IMG[l:r, u:d] / self.MAP_SIZE]
      img = np.stack([tile_img, entity_img, poison_img, fog_img, visit_img, *coord_imgs])
      imgs.append(img)
    imgs = np.stack(imgs)
    return imgs

  def nearby_features(self, row: int, col: int):
    near_tile_map = self.tile_map[row-4:row+5, col-4:col+5]
    food_arr = water_arr = herb_arr = fish_arr = obstacle_arr = []
    for i in range(9):
      for j in range(9):
        if abs(i-4) + abs(j-4) <= 4:
          food_arr.append(near_tile_map[i, j] == material.Forest.index)
          water_arr.append(near_tile_map[i, j] == material.Water.index)
          herb_arr.append(near_tile_map[i, j] == material.Herb.index)
          fish_arr.append(near_tile_map[i, j] == material.Fish.index)
          obstacle_arr.append(near_tile_map[i, j] in material.Impassible.indices)
    food_arr[-1] = max(0, self.poison_map[row, col]) / 20.  # patch after getting trained
    water_arr[-1] = max(0, self.poison_map[row+1, col]) / 20.  # patch after getting trained
    herb_arr[-1] = max(0, self.poison_map[row, col+1]) / 20.  # patch after getting trained
    fish_arr[-1] = max(0, self.poison_map[row-1, col]) / 20.  # patch after getting trained
    obstacle_arr[-1] = max(0, self.poison_map[row, col-1]) / 20.  # patch after getting trained
    # xcxc
    return np.zeros(206)
    return np.concatenate([
        food_arr, water_arr, herb_arr, fish_arr, obstacle_arr,
    ])

  def legal_moves(self, obs):
    moves = np.zeros((self.team_size, len(action.Direction.edges) + 1))
    for i in range(self.team_size):
      if i in obs:
        moves[i,:-1] = obs[i]["ActionTargets"][action.Move][action.Direction]
      if sum(moves[i]) == 0:
        moves[i][-1] = 1
    return moves


  def _get_init_tile_map(self):
    arr = np.zeros((self.MAP_SIZE+1, self.MAP_SIZE+1))
    # mark the most outside circle of grass
    map_left = self.config.MAP_BORDER
    map_right = self.MAP_SIZE - self.config.MAP_BORDER
    arr[map_left:map_right+1, map_left:map_right+1] = 2
    # mark the unseen tiles
    arr[map_left+1:map_right, map_left+1:map_right] = max(material.All.indices) + 1
    return arr

  def _get_init_poison_map(self):
    arr = np.ones((self.MAP_SIZE + 1, self.MAP_SIZE + 1))
    for i in range(self.MAP_SIZE // 2):
      l, r = i + 1, self.MAP_SIZE - i
      arr[l:r, l:r] = -i
    # positive value represents the poison strength
    # negative value represents the shortest distance to poison area
    return arr

  def _mark_point(self, arr_2d, index_arr, value, clip=False):
      arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
          np.clip(value, 0., 1.) if clip else value
