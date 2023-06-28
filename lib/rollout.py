# pylint: disable=bad-builtin, no-member, protected-access
from collections import defaultdict

import random
from typing import List
import numpy as np

import torch

import nmmo
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import pufferlib.vectorization.serial
import tqdm
from env.nmmo_env import NMMOEnv, RewardsConfig
from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_helper import TeamHelper
from lib.agent.baseline_agent import BaselineAgent
from nmmo.render.replay_helper import ReplayHelper

class Rollout():
  def __init__(self,
    config: nmmo.config.Config,
    team_helper: TeamHelper,
    rewards_config: RewardsConfig,
    model_checkpoints: List[str],
    replay_helper: ReplayHelper = None,
  ) -> None:

    self._config = config
    self._binding = pufferlib.emulation.Binding(
      # env_creator=lambda: NMMOTeamEnv(config, team_helper, rewards_config),
      env_creator=lambda: NMMOEnv(config, rewards_config),
      env_name="Neural Team MMO",
      suppress_env_prints=False,
    )
    self._replay_helper = replay_helper

    self._agent_to_model = {}
    self._num_agents_per_model = defaultdict(float)
    self._agents = []
    for i in range(team_helper.num_teams):
      model = model_checkpoints[i % len(model_checkpoints)]
      self._agent_to_model[i] = model
      self._num_agents_per_model[model] += 1
      self._agents.append(BaselineAgent(self._binding, weights_path=model))

  def run_episode(self, seed, max_steps=1024):
    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    env = self._binding.raw_env_creator()

    #  xcxc
    # env._env.realm.record_replay(self._replay_helper)
    env.realm.record_replay(self._replay_helper)

    agent_rewards = defaultdict(float)

    actions = {}
    obs = env.reset(seed=seed)

    for step in tqdm.tqdm(range(max_steps)):
      obs, rewards, dones, infos = env.step(actions)

      for agent_id, reward in rewards.items():
        agent_rewards[agent_id - 1] += reward # xcxc

      if len(obs) == 0:
        break

      actions = {
        agent_id: self._agents[agent_id - 1].act(obs)
        for agent_id, obs in obs.items() }

    model_rewards = defaultdict(float)
    for agent_id, reward in agent_rewards.items():
      model = self._agent_to_model[agent_id - 1]
      model_rewards[model] += reward / self._num_agents_per_model[model]

    return agent_rewards, model_rewards
