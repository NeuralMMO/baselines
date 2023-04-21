import unittest

import nmmo

from feature_extractor.feature_extractor import FeatureExtractor

from tests.feature_extractor.testhelpers import FeaturizerTestTemplate

RANDOM_SEED = 0 # random.randint(0, 10000)


class TestFeatureExtractor(FeaturizerTestTemplate):

  feature_extractors = None

  def test_init_reset_call(self):
    self.feature_extractors = {
        team_id: FeatureExtractor(self.team_helper.teams, team_id, self.config)
        for team_id in self.team_helper.teams }

    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()

    for team_id, feat_ext in self.feature_extractors.items():
      team_obs = self._filter_obs(init_obs, team_id)
      feat_ext.reset(team_obs)

    # step, get team obs, update the feature extractor
    obs, _, _, _ = env.step({})
    for team_id, feat_ext in self.feature_extractors.items():
      team_obs = self._filter_obs(obs, team_id)
      feat_ext(team_obs)

  def test_trans_action(self):
    pass


if __name__ == '__main__':
  unittest.main()
