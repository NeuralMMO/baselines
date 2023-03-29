import collections
import copy
from types import SimpleNamespace

import nmmo
import nmmo.io.action as nmmo_act
import numpy as np
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.io import action
from nmmo.lib import material
from nmmo.systems import item
from nmmo.systems.item import ItemState
from feature_extractor.map_helper import MapHelper
from feature_extractor.target_tracker import TargetTracker

from model.util import multi_hot_generator, one_hot_generator
from team_helper import TeamHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

N_SELF_FEATURE = 262
PER_ENTITY_FEATURE = 30
N_NPC_CONSIDERED = 9
N_ENEMY_CONSIDERED = 9
AWARE_RANGE = 15
ATK_RANGE = 3

NEGATIVE_POP = -1
NEUTRAL_POP = -2
HOSTILE_POP = -3
ATK_TYPE = [
  'Melee',
  'Range',
  'Mage',
]
N_ATK_TYPE = len(ATK_TYPE)

class EntityHelper:
  def __init__(self, config: nmmo.config.Config,
               team_id: int, team_helper: TeamHelper,
               target_tracker: TargetTracker, map_helper: MapHelper) -> None:

    self._config = config
    self._team_helper = team_helper
    self._target_tracker = target_tracker
    self._map_helper = map_helper

    self._team_id = team_id
    self._team_agent_ids = team_helper._teams[team_id]
    self.TEAM_SIZE = len(self._team_agent_ids)

    self._member_location = {}
    self._team_members = {}
    self._professions = None

    self._curr_obs = None

    self.npc_tgt = None
    self.enemy_tgt = None

    self._profession = None, # set in reset()

    self.features = SimpleNamespace(
      team = one_hot_generator(self._team_helper.num_teams, int(self._team_id)),

      # team position of each team member [0..TEAM_SIZE)
      team_position = {
        i: one_hot_generator(self.TEAM_SIZE, i) for i in range(self.TEAM_SIZE)
      },

      # profession of each team member
      professions = None,

      # near-by npcs arr[TEAM_SIZE, N_NPC_CONSIDERED, PER_ENTITY_FEATURE]
      npcs = None,

      # near-by enemies arr[TEAM_SIZE, N_ENEMY_CONSIDERED, PER_ENTITY_FEATURE]
      enemies = None,

      # mask for team members that are dead. arr[TEAM_SIZE]
      team_mask = None,
      # npcs that should be ignored. arr[TEAM_SIZE, N_NPC_CONSIDERED]
      npc_mask = None,
      # enemies that should be ignored. arr[TEAM_SIZE, N_NPC_CONSIDERED]
      enemy_mask = None,

      entities = None,

      # team member features.
      # arr[TEAM_SIZE, N_SELF_FEATURE] = [
      team_member = lambda i: np.concatenate([
        self.entities[self._team_agent_ids[i]],
        self.features.team,
        self.features.team_position[i],
        self.features.professions[i],
        self._map_helper.nearby_features(self._member_location[i]),
      ]),
    )

    return team_members_arr, npcs_arrs, enemies_arrs, team_mask, entity_mask[0], entity_mask[1], target_npcs_ids, target_enemies_ids


  def reset(self, init_obs):
    self._choose_professions()
    self.update(init_obs)

  def update(self, obs):
    self._curr_obs = obs

    self._entities = {}
    self._member_location = {}

    entity_features = {} # id -> entity_features

    for member_obs in obs.values():
      for entity_ob in member_obs['Entity']:
        id = entity_ob[EntityAttr["id"]]
        if id == 0:
          continue
        if id in self._entity_features:
          continue
        self._entities[id] = entity_ob
      entity_features[id] = self._extract_entity_features(entity_ob)

    team_members_features = np.zeros((self.TEAM_SIZE, N_SELF_FEATURE))
    team_mask = np.ones(self.TEAM_SIZE)

    for member_id in range(self.TEAM_SIZE):
      if member_id in obs:
        agent_id = self._team_agent_ids[member_id]
        location = self._entities[agent_id][EntityAttr["row"]:EntityAttr["col"]+1]
        self._member_location[member_id] = location
        team_mask[member_id] = 0

    for i in range(self.TEAM_SIZE):
      team_members_arr[i] = self._extract_per_entity_feature(
        self._team_members.get(i, None), i)


    for member_id, player_obs in obs.items():
       entity_id = self._team_helper.agent_for_team_and_position[self._team_id, member_id]
       self._team_members[member_id] = self._entity_row(player_obs['Entity'], entity_id)
       self._member_location[member_id] = self._team_members[member_id][EntityAttr["row"]:EntityAttr["col"]+1]

    # arr = np.zeros(PER_ENTITY_FEATURE)
        # food_arr = water_arr = herb_arr = fish_arr = obstacle_arr = np.zeros(41)
    arr = np.concatenate([
        arr, population_arr, index_arr, prof_arr,
        food_arr, water_arr, herb_arr, fish_arr, obstacle_arr,
    ])
    return arr

  def member_position(self, member_id):
    return self._member_location[member_id]

  def member(self, member_id):
    return self._team_members[member_id]

  def extract_entity_feature(self, obs):
    # assign the features of npcs and enemies to each member
    others_arrs = [np.zeros((self.TEAM_SIZE, n, PER_ENTITY_FEATURE))
                    for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    entity_mask = [np.ones((self.TEAM_SIZE, n))
                    for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    ids_as_target = [np.zeros((self.TEAM_SIZE, n))
                      for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    for k in range(2):
      n_considered = (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)[k]
      entities = (npcs, enemies)[k]
      # first extract all the features along with entity's idx & position
      features = [{
        'idx': raw_arr[EntityAttr["id"]],
        'row': raw_arr[EntityAttr["row"]],
        'col': raw_arr[EntityAttr["col"]],
        'pop': raw_arr[EntityAttr["population_id"]],
        'arr': self._extract_per_entity_feature(raw_arr),
      } for idx, raw_arr in entities.items()]

      for i, (my_row, my_col) in self._member_location.items():
        def l1_to_me(f):
            return max(abs(f['row'] - my_row), abs(f['col'] - my_col))

        nearests = sorted(features, key=l1_to_me)[:n_considered]
        for j, feat in enumerate(nearests):
          if l1_to_me(feat) <= ATK_RANGE and feat['pop'] != NEUTRAL_POP:  # as target
            ids_as_target[k][i][j] = feat['idx']
          if l1_to_me(feat) <= AWARE_RANGE:  # as visible entity
            others_arrs[k][i][j] = feat['arr']
            entity_mask[k][i][j] = 0

    npcs_arrs, enemies_arrs = others_arrs
    target_npcs_ids, target_enemies_ids = ids_as_target
    return team_members_arr, npcs_arrs, enemies_arrs, team_mask, entity_mask[0], entity_mask[1], target_npcs_ids, target_enemies_ids

  def legal_target(self, obs):
    # 'target_t': {i: obs[i]["ActionTargets"][action.Attack][action.Target] for i in range(self.TEAM_SIZE)},
    target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)  # first npc, then enemy
    no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
    return np.concatenate([target_attackable, no_target], axis=-1)

  def _extract_entity_features(self, entity_observation):
    play_area = (self._config.MAP_SIZE - 2*self._config.MAP_BORDER)
    o = entity_observation
    return np.array([
        1.,  # alive mark
        o[EntityAttr["id"]] in self._target_entity_id,  # attacked by my team
        o[EntityAttr["attacker_id"]] < 0,  # attacked by npc
        o[EntityAttr["attacker_id"]] > 0,  # attacked by player
        o[EntityAttr["item_level"]] / 20.,
        (o[EntityAttr["row"]] - self._config.MAP_SIZE // 2) / self._config.MAP_SIZE // 2,
        (o[EntityAttr["col"]] - self._config.MAP_SIZE // 2) / self._config.MAP_SIZE // 2,
        o[EntityAttr["time_alive"]] / self._config.HORIZON,
        (o[EntityAttr["row"]] - self._config.MAP_BORDER) / play_area,
        (o[EntityAttr["col"]] - self._config.MAP_BORDER) / play_area,
        o[EntityAttr["id"]] >= 0,  # player
        o[EntityAttr["id"]] in self._team_agent_ids,  # my team
        o[EntityAttr["population_id"]] == -1,  # passive npc
        o[EntityAttr["population_id"]] == -2,  # neutral npc
        o[EntityAttr["population_id"]] == -3,  # hostile npc
        o[EntityAttr["damage"]] / 10.,
        o[EntityAttr["time_alive"]] / self._config.HORIZON,
        o[EntityAttr["gold"]] / 100.,
        o[EntityAttr["health"]] / 100.,
        o[EntityAttr["food"]] / 100.,
        o[EntityAttr["water"]] / 100.,
        o[EntityAttr["melee_level"]] / 10.,
        o[EntityAttr["range_level"]] / 10.,
        o[EntityAttr["mage_level"]] / 10.,
        o[EntityAttr["fishing_level"]] / 10.,
        o[EntityAttr["herbalism_level"]] / 10.,
        o[EntityAttr["prospecting_level"]] / 10.,
        o[EntityAttr["carving_level"]] / 10.,
        o[EntityAttr["alchemy_level"]] / 10.,
    ])

  def _entity_row(self, entity_obs, member_id):
    return entity_obs[entity_obs[:,EntityAttr["id"]] == self._entity_id(member_id)][0]

  def _choose_professions(self):
    seed = np.random.randint(N_ATK_TYPE)
    profs = [ATK_TYPE[(seed + i) % N_ATK_TYPE]
              for i in range(self.TEAM_SIZE)]
    np.random.shuffle(profs)
    self._professions = profs
    for prof in profs:
      self._features_proffessions = one_hot_generator(N_ATK_TYPE, ATK_TYPE.index(prof))


