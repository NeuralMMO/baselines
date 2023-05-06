import unittest
import numpy as np

import nmmo

from feature_extractor.feature_extractor import FeatureExtractor

from tests.feature_extractor.testhelpers import FeaturizerTestTemplate

RANDOM_SEED = 0 # random.randint(0, 10000)


class TestFeatureExtractor(FeaturizerTestTemplate):

  feature_extractors = None

  def _create_test_env(self):
    feature_extractors = {
        team_id: FeatureExtractor(self.team_helper.teams, team_id, self.config)
        for team_id in self.team_helper.teams }

    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()

    for team_id, feat_ext in feature_extractors.items():
      team_obs = self._filter_obs(init_obs, team_id)
      feat_ext.reset(team_obs)

    return env, feature_extractors

  def test_init_reset_call(self):
    env, feature_extractors = self._create_test_env()

    # step, get team obs, update the feature extractor
    obs, _, _, _ = env.step({})
    for team_id, feat_ext in feature_extractors.items():
      team_obs = self._filter_obs(obs, team_id)
      feat_ext(team_obs)

  @staticmethod
  def _get_idx(arr, get_zero=True):
    if len(arr) > 0:
      indices = np.where((arr == 0) == get_zero)[0]
      if len(indices) > 0:
        return np.random.choice(indices)

    # if not found, return 0 -> the rest of the code should handle this
    return 0

  def test_trans_action_attack_targets(self):
    env, feature_extractors = self._create_test_env()

    # just look at the team 1
    team_id = 1
    team_size = len(self.team_helper.teams[team_id])
    featurizer = feature_extractors[team_id]

    # peeking featurizer.entity_helper._entity_targets
    entity_targets = featurizer.entity_helper._entity_targets
    # the first four get the index of zeros, 
    # the next four get the index of non-zeros
    targets = [self._get_idx(entity_targets[member_pos]) if member_pos < 4
               else self._get_idx(entity_targets[member_pos], get_zero=False)
               for member_pos in range(team_size)]
    
    # input actions are all zeros
    input_actions = {
      'move': np.zeros(team_size, dtype=np.int32), # idx for action.Direction.edges
      'style': np.zeros(team_size, dtype=np.int32), # idx for action.Style.edges
      # idx for entity_helper._entity_targets[member_pos] -> entity id
      'target': np.array(targets, dtype=np.int32),
    }

    trans_actions = featurizer.translate_actions(input_actions)

    for member_pos in range(team_size):
      target_ent = entity_targets[member_pos][targets[member_pos]]
      if target_ent > 0:
        self.assertEqual(target_ent,
                         trans_actions[member_pos][nmmo.action.Attack][nmmo.action.Target])
      else:
        self.assertTrue(nmmo.action.Attack not in trans_actions[member_pos])

    print()

    pass


if __name__ == '__main__':
  unittest.main()
