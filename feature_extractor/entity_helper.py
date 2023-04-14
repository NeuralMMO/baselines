from typing import Callable, Dict, Tuple

import numpy as np

import nmmo
from nmmo.entity.entity import EntityState

from feature_extractor.target_tracker import TargetTracker

from team_helper import TeamHelper

from model.util import one_hot_generator
from model.model import ModelArchitecture

EntityAttr = EntityState.State.attr_name_to_col

PER_ENTITY_FEATURE = ModelArchitecture.n_ent_feat
N_NPC_CONSIDERED = ModelArchitecture.n_npc_considered
N_ENEMY_CONSIDERED = ModelArchitecture.n_enemy_considered
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
    self.team_size = self._team_helper.team_size[team_id]
    self._team_agent_ids = set(self._team_helper.teams[team_id])

    self.member_professions = None
    self._professions_feature = None

    self._entities = {}
    self.member_location = {}
    self._entity_features = {}

    self._team_feature = one_hot_generator(
      self._team_helper.num_teams, int(self._team_id))

    # team position of each team member [0..team_size)
    self._team_position_feature = {
      i: one_hot_generator(self.team_size, i) for i in range(self.team_size)
    }

    # calculate the # of player features
    self.n_self_feat = ModelArchitecture.n_ent_feat + \
                       self._team_helper.num_teams + \
                       self._team_helper.team_size[team_id] + \
                       ModelArchitecture.n_atk_type + \
                       ModelArchitecture.n_nearby_feat

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
        ent_id = int(entity_ob[EntityAttr["id"]])
        if ent_id in self._entity_features:
          continue
        self._entities[ent_id] = entity_ob
        self._entity_features[ent_id] = self._extract_entity_features(entity_ob)

      # update the location of each team member
      agent_pos = self._team_helper.agent_position(agent_id)
      if agent_id in self._entities:
        row, col = self._entities[agent_id][EntityAttr["row"]:EntityAttr["col"]+1]
        self.member_location[agent_pos] = (int(row), int(col))

  def team_features_and_mask(self):
    team_members_features = np.zeros((self.team_size, self.n_self_feat))
    team_mask = np.zeros(self.team_size)
    for idx in range(self.team_size):
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

    return np.array(team_members_features), team_mask

  def npcs_features_and_mask(self):
    npc_features = np.zeros((self.team_size, N_NPC_CONSIDERED, PER_ENTITY_FEATURE))
    npc_mask = np.ones((self.team_size, N_NPC_CONSIDERED))
    for idx in range(self.team_size):
      npc_features[idx], npc_mask[idx] = self._nearby_entity_features(
        idx, N_NPC_CONSIDERED,
        lambda id: id < 0
      )
    return npc_features, npc_mask

  def enemies_features_and_mask(self):
    enemy_features = np.zeros((self.team_size, N_ENEMY_CONSIDERED, PER_ENTITY_FEATURE))
    enemy_mask = np.ones((self.team_size, N_ENEMY_CONSIDERED))

    for idx in range(self.team_size):
      enemy_features[idx], enemy_mask[idx] = self._nearby_entity_features(
        idx, N_ENEMY_CONSIDERED,
        lambda id: id not in self._team_agent_ids
      )
    return enemy_features, enemy_mask

  # find closest entities matching filter_func
  def _nearby_entity_features(self, member_pos,
                              max_entities: int,
                              filter_func: Callable)-> Tuple[np.ndarray, np.ndarray]:
    features = np.zeros((max_entities, PER_ENTITY_FEATURE))
    mask = np.ones(max_entities)

    if member_pos not in self.member_location:
      return features, mask

    (row, col) = self.member_location[member_pos]
    nearby_entities = []
    for ent_id, entity_ob in self._entities.items():
      if filter_func(ent_id):
        dist_and_id = (max(
            abs(entity_ob[EntityAttr['row']] - row),
            abs(entity_ob[EntityAttr['col']] - col)), ent_id)
        nearby_entities.append(dist_and_id)

    nearby_entities = sorted(nearby_entities)[:max_entities]
    for idx, (dist, ent_id) in enumerate(nearby_entities):
      if dist < AWARE_RANGE:
        features[idx] = self._entity_features[ent_id]
        mask[idx] = 0

    return features, mask

  # CHECK ME: legal_target seems to be relevant to some code in action.py and/or target_tracker.py
  #   Does this belong here?
  #   N_NPC_CONSIDERED and N_EMENY_CONSIDERED also seem relevant
  # def legal_target(self, obs):
  #   pass
  #   # 'target_t': {i: obs[i]["ActionTargets"][action.Attack][action.Target]
  #                 for i in range(self.team_size)},
  #   first npc, then enemy
  #   target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)
  #   no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
  #   return np.concatenate([target_attackable, no_target], axis=-1)

  def _extract_entity_features(self, entity_observation: np.ndarray) -> np.ndarray:
    play_area = self._config.MAP_SIZE - 2*self._config.MAP_BORDER
    o = entity_observation
    attack_level = max(o[[EntityAttr["melee_level"], EntityAttr["range_level"],
                          EntityAttr["mage_level"]]])
    return np.array([
      1.,  # alive mark
      o[EntityAttr["id"]] in self._target_tracker.target_entity_id,  # attacked by my team
      o[EntityAttr["attacker_id"]] < 0,  # attacked by npc
      o[EntityAttr["attacker_id"]] > 0,  # attacked by player
      attack_level / 10., # added the missing feature: o[IDX_ENT_LVL] / 10.
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
              for i in range(self.team_size)]

    np.random.shuffle(profs)

    self.member_professions = profs
    self._professions_feature = [
      one_hot_generator(N_ATK_TYPE, ATK_TYPE.index(prof)) for prof in profs
    ]
