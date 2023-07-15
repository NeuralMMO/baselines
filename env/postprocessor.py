from collections import defaultdict

import pufferlib
import pufferlib.emulation

# pylint: disable=import-error
from feature_extractor.feature_extractor import FeatureExtractor


class Postprocessor(pufferlib.emulation.Postprocessor):
  def __init__(self, env, teams, team_id):
    super().__init__(env, teams, team_id)
    # self._feature_extractor = FeatureExtractor(teams, team_id, env.config)

  def reset(self, team_obs):
    super().reset(team_obs)
    # self._feature_extractor.reset(team_obs)

  def rewards(self, team_rewards, team_dones, team_infos, step):
    agents = list(set(team_rewards.keys()).union(set(team_dones.keys())))

    team_reward = sum(team_rewards.values())
    team_info = {"stats": defaultdict(float)}

    for agent_id in agents:
      agent = self.env.realm.players.dead_this_tick.get(
          agent_id, self.env.realm.players.get(agent_id)
      )

      if agent is None:
        continue

    return team_reward, team_info

  # def features(self, obs, step):
  #   # for ob in obs.values():
  #   #   ob["featurized"] = self._feature_extractor(obs)
  #   return obs

  # def actions(self, actions, step):
  #   return self._feature_extractor.translate_actions(actions)
