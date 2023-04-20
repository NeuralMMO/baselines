from typing import Dict, Tuple

import numpy as np

import nmmo
from nmmo.entity.entity import EntityState

from team_helper import TeamHelper

from model.util import one_hot_generator
from model.model import ModelArchitecture

EntityAttr = EntityState.State.attr_name_to_col

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
               team_helper: TeamHelper, team_id: int) -> None:
    self._config = config
    self._team_helper = team_helper

    self._team_id = team_id
    self.team_size = self._team_helper.team_size[team_id]
    self._team_agent_ids = set(self._team_helper.teams[team_id])

    self.member_professions = None
    self._professions_feature = None

    self._entities = {}
    self.member_location = {}
    self._entity_features = {}

    # CHECK ME: target_tracker merged to entity_helper
    # NOTE: attack_target is updated in _trans_attack()
    self.attack_target = None

    self._team_feature = one_hot_generator(
      self._team_helper.num_teams, int(self._team_id))

    # team position of each team member [0..team_size)
    self._team_position_feature = {
      i: one_hot_generator(self.team_size, i) for i in range(self.team_size)
    }

    # calculate the # of player features
    self.n_self_feat = ModelArchitecture.ENTITY_NUM_FEATURES + \
                       self._team_helper.num_teams + \
                       self._team_helper.team_size[team_id] + \
                       ModelArchitecture.NUM_PROFESSIONS + \
                       ModelArchitecture.NEARBY_NUM_FEATURES

  def reset(self, init_obs: Dict) -> None:
    self.attack_target = [None] * self.team_size
    self._choose_professions()
    self.update(init_obs)

  # merge all the member observations and denormalize them into
  # self._entities, self._member_location, self._entity_features
  def update(self, obs: Dict) -> None:
    self._entities = {} # id -> entity_ob (1 row)
    self.member_location = {} # id -> (row, col)
    self._entity_features = {} # id -> entity_features

    for agent_id, agent_obs in obs.items():
      entity_rows = agent_obs['Entity'][:, EntityAttr["id"]] != 0
      for entity_ob in agent_obs['Entity'][entity_rows]:
        ent_id = int(entity_ob[EntityAttr["id"]])
        if ent_id in self._entity_features:
          continue
        self._entities[ent_id] = entity_ob
        self._entity_features[ent_id] = self._extract_entity_features(entity_ob)

      # update the location of each team member
      agent_pos = self.agent_id_to_pos(agent_id)
      if agent_id in self._entities:
        row, col = self._entities[agent_id][EntityAttr["row"]:EntityAttr["col"]+1]
        self.member_location[agent_pos] = (int(row), int(col))

  def team_features_and_mask(self, map_helper):
    team_members_features = np.zeros((self.team_size, self.n_self_feat))
    team_mask = np.zeros(self.team_size, dtype=np.float32)
    for member_pos in range(self.team_size):
      agent_id = self.pos_to_agent_id(member_pos)

      if agent_id not in self._entity_features:
        team_mask[member_pos] = 1
        continue

      if member_pos in self.member_location:
        (row, col) = self.member_location[member_pos]
        nearby_features = map_helper.nearby_features(row, col)
      else:
        nearby_features = map_helper.dummy_nearby_features()

      team_members_features[member_pos] = np.concatenate([
        self._entity_features[agent_id],
        self._team_feature,
        self._team_position_feature[member_pos],
        self._professions_feature[member_pos],
        nearby_features
      ])

    return np.array(team_members_features, dtype=np.float32), team_mask

  def npcs_features_and_mask(self):
    n_npc_considered = ModelArchitecture.ENTITY_NUM_NPCS_CONSIDERED
    npc_features = np.zeros((self.team_size, n_npc_considered,
                            ModelArchitecture.ENTITY_NUM_FEATURES))
    npc_mask = np.ones((self.team_size, n_npc_considered))
    npc_target = np.zeros((self.team_size, n_npc_considered))

    for member_pos in range(self.team_size):
      # pylint: disable=unbalanced-tuple-unpacking
      npc_features[member_pos], npc_mask[member_pos], npc_target[member_pos] = \
        self._nearby_entity_features(member_pos, n_npc_considered, lambda id: id < 0)

    return npc_features.astype(np.float32), \
           npc_mask.astype(np.float32), \
           npc_target.astype(np.float32)

  def enemies_features_and_mask(self):
    n_enemy_considered = ModelArchitecture.ENTITY_NUM_ENEMIES_CONSIDERED
    enemy_features = np.zeros((self.team_size, n_enemy_considered,
                              ModelArchitecture.ENTITY_NUM_FEATURES))
    enemy_mask = np.ones((self.team_size, n_enemy_considered))
    enemy_target = np.zeros((self.team_size, n_enemy_considered))

    for member_pos in range(self.team_size):
      # pylint: disable=unbalanced-tuple-unpacking
      enemy_features[member_pos], enemy_mask[member_pos], enemy_target[member_pos] = \
        self._nearby_entity_features(member_pos, n_enemy_considered,
                lambda id: (id > 0) and (id not in self._team_agent_ids))

    return enemy_features.astype(np.float32), \
           enemy_mask.astype(np.float32), \
           enemy_target.astype(np.float32)

  # find closest entities matching filter_func
  def _nearby_entity_features(self, member_pos,
                              max_entities: int,
                              filter_func) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((max_entities, ModelArchitecture.ENTITY_NUM_FEATURES))
    # NOTE: mask=1 indicates out-of-visual-range
    mask = np.ones(max_entities)
    attack_target = np.zeros(max_entities)

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
      if dist <= ATK_RANGE: # NOTE: realikun did not attack neutral npcs
        attack_target[idx] = ent_id
      if dist < AWARE_RANGE:
        features[idx] = self._entity_features[ent_id]
        mask[idx] = 0

    return features, mask, attack_target

  @staticmethod
  def legal_target(npc_target, enemy_target):
    target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)
    no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
    return np.concatenate([target_attackable, no_target], axis=-1).astype(np.float32)

  def _extract_entity_features(self, entity_observation: np.ndarray) -> np.ndarray:
    play_area = self._config.MAP_SIZE - 2*self._config.MAP_BORDER
    o = entity_observation
    attack_level = max(o[[EntityAttr["melee_level"], EntityAttr["range_level"],
                          EntityAttr["mage_level"]]])

    # CHECK ME: revisit entity feature scalers
    return np.array([
      1.,  # alive mark
      o[EntityAttr["id"]] in self.attack_target,  # attacked by my team
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
    # NOTE: realikun treats only melee, range, mage as professions
    #   harvesting ammos don't seem to considered seriously
    seed = np.random.randint(N_ATK_TYPE)
    profs = [ATK_TYPE[(seed + i) % N_ATK_TYPE]
              for i in range(self.team_size)]

    np.random.shuffle(profs)

    self.member_professions = profs
    self._professions_feature = [
      one_hot_generator(N_ATK_TYPE, ATK_TYPE.index(prof)) for prof in profs
    ]

  #####################################
  # team-related helper functions
  def pos_to_agent_id(self, member_pos):
    return self._team_helper.agent_id(self._team_id, member_pos)

  def is_pos_alive(self, member_pos):
    return member_pos in self.member_location

  def is_agent_in_team(self, agent_id):
    return agent_id in self._team_agent_ids

  def agent_id_to_pos(self, agent_id):
    return self._team_helper.agent_position(agent_id)

  def agent_team(self, agent_id):
    return self._team_helper.team_and_position_for_agent[agent_id][0]

  def agent(self, agent_id):
    if agent_id not in self._entities:
      return None

    info = EntityState.parse_array(self._entities[agent_id].astype(np.int32))
    # add level for using armors
    info.level = max(getattr(info, skill + '_level')
                     for skill in ['melee', 'range', 'mage', 'fishing', 'herbalism',
                                   'prospecting', 'carving', 'alchemy'])
    return info
