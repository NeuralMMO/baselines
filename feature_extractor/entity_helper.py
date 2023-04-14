
from typing import Callable, Dict, Tuple
import nmmo
import numpy as np
from nmmo.entity.entity import EntityState

from feature_extractor.target_tracker import TargetTracker
from model.util import one_hot_generator
from team_helper import TeamHelper

EntityAttr = EntityState.State.attr_name_to_col

N_SELF_FEATURE = 262
PER_ENTITY_FEATURE = 29
N_NPC_CONSIDERED = 9
N_ENEMY_CONSIDERED = 9
AWARE_RANGE = 15
ATK_RANGE = 3

ATK_TYPE = [
  'Melee',
  'Range',
  'Mage',
]
N_ATK_TYPE = len(ATK_TYPE)

class EntityHelper:
  def __init__(self, config: nmmo.config.Config,
               team_helper: TeamHelper, team_id: int,
               target_tracker: TargetTracker, map_helper) -> None:

    self._config = config
    self._target_tracker = target_tracker
    self._map_helper = map_helper
    self._team_helper = team_helper

    self._team_id = team_id
    self.TEAM_SIZE = self._team_helper.team_size[team_id]
    self._team_agent_ids = set(self._team_helper.teams[team_id])

    self.member_professions = None

    self._entities = {}
    self.member_location = {}
    self._entity_features = {}

    self._team_feature = one_hot_generator(
      self._team_helper.num_teams, int(self._team_id))

    # team position of each team member [0..TEAM_SIZE)
    self._team_position_feature = {
      i: one_hot_generator(self.TEAM_SIZE, i) for i in range(self.TEAM_SIZE)
    }

  def reset(self, init_obs: Dict) -> None:
    self._choose_professions()
    self.update(init_obs)

  # merge all the member observations and denormalize them into
  # self._entities, self._member_location, self._entity_features
  def update(self, obs: Dict) -> None:
    self._entities = {} # id -> entity_ob
    self.member_location = {} # id -> (row, col)
    self._entity_features = {} # id -> entity_features

    for agent_id, agent_obs in obs.items():
      entity_rows = agent_obs['Entity'][:,EntityAttr["id"]] != 0
      for entity_ob in agent_obs['Entity'][entity_rows]:
        id = int(entity_ob[EntityAttr["id"]])
        if id in self._entity_features:
          continue
        self._entities[id] = entity_ob
        self._entity_features[id] = self._extract_entity_features(entity_ob)

      # update the location of each team member
      agent_pos = self._team_helper.agent_position(agent_id)
      if agent_id in self._entities:
        row, col = self._entities[agent_id][EntityAttr["row"]:EntityAttr["col"]+1]
        self.member_location[agent_pos] = (int(row), int(col))

  def team_features_and_mask(self, obs):
    team_members_features = np.zeros((self.TEAM_SIZE, N_SELF_FEATURE))
    team_mask = np.zeros(self.TEAM_SIZE)
    for idx in range(self.TEAM_SIZE):
      agent_id = self._team_helper.agent_id(self._team_id, idx)

      if agent_id not in self._entity_features:
        team_mask[idx] = 1
        continue

      if idx in self.member_location:
        (row, col) = self.member_location[idx]
        nearby_features = self._map_helper.nearby_features(row, col)
      else:
        nearby_features = self._map_helper.dummy_nearby_features()

      team_members_features[idx] = np.concatenate([
        self._entity_features[agent_id],
        self._team_feature,
        self._team_position_feature[idx],
        self._professions_feature[idx],
        nearby_features

      ])
    return team_members_features, team_mask

  def npcs_features_and_mask(self, obs):
    npc_features = np.zeros((self.TEAM_SIZE, N_NPC_CONSIDERED, PER_ENTITY_FEATURE))
    npc_mask = np.ones((self.TEAM_SIZE, N_NPC_CONSIDERED))
    for idx in range(self.TEAM_SIZE):
      npc_features[idx], npc_mask[idx] = self._nearby_entity_features(
        idx, N_NPC_CONSIDERED,
        lambda id: id < 0
      )
    return npc_features, npc_mask

  def enemies_features_and_mask(self, obs):
    enemy_features = np.zeros((self.TEAM_SIZE, N_ENEMY_CONSIDERED, PER_ENTITY_FEATURE))
    enemy_mask = np.ones((self.TEAM_SIZE, N_ENEMY_CONSIDERED))

    for idx in range(self.TEAM_SIZE):
      enemy_features[idx], enemy_mask[idx] = self._nearby_entity_features(
        idx, N_ENEMY_CONSIDERED,
        lambda id: id not in self._team_agent_ids
      )
    return enemy_features, enemy_mask

  # find closest entities matching filter_func
  def _nearby_entity_features(self, member_pos, max_entities: int, filter_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    features = np.zeros((max_entities, PER_ENTITY_FEATURE))
    mask = np.ones(max_entities)

    if member_pos not in self.member_location:
      return features, mask

    (row, col) = self.member_location[member_pos]
    nearby_entities = []
    for id, entity_ob in self._entities.items():
      if filter_func(id):
        dist_and_id = (max(
            abs(entity_ob[EntityAttr['row']] - row),
            abs(entity_ob[EntityAttr['col']] - col)), id)
        nearby_entities.append(dist_and_id)

    nearby_entities = sorted(nearby_entities)[:max_entities]
    for idx, (dist, id) in enumerate(nearby_entities):
      if dist < AWARE_RANGE:
        features[idx] = self._entity_features[id]
        mask[idx] = 0

    return features, mask

  def legal_target(self, obs):
    pass
    # xcxc
    # # 'target_t': {i: obs[i]["ActionTargets"][action.Attack][action.Target] for i in range(self.TEAM_SIZE)},
    # target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)  # first npc, then enemy
    # no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
    # return np.concatenate([target_attackable, no_target], axis=-1)

  def _extract_entity_features(self, entity_observation: np.ndarray) -> np.ndarray:
    play_area = (self._config.MAP_SIZE - 2*self._config.MAP_BORDER)
    o = entity_observation
    return np.array([
      1.,  # alive mark
      o[EntityAttr["id"]] in self._target_tracker.target_entity_id,  # attacked by my team
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

  def _choose_professions(self):
    seed = np.random.randint(N_ATK_TYPE)
    profs = [ATK_TYPE[(seed + i) % N_ATK_TYPE]
              for i in range(self.TEAM_SIZE)]

    np.random.shuffle(profs)

    self.member_professions = profs
    self._professions_feature = [
      one_hot_generator(N_ATK_TYPE, ATK_TYPE.index(prof)) for prof in profs
    ]
  
  def entity_by_id(self, id):
    return self._entities[id]


