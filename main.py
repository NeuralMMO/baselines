import torch
from baseline_env import BaselineEnv
import cleanrl_ppo_lstm
import pufferlib.emulation
import pufferlib.registry.nmmo
import pufferlib.frameworks.cleanrl
import nmmo
from team_env import TeamEnv
from model.policy import Policy
from team_helper import TeamHelper


teams = [[player for player in range(team * 8 + 1, (team + 1) * 8 + 1)] for team in range(16)]
team_helper = TeamHelper(teams)

def create_env():
  nmmo_env = nmmo.Env()
  env = BaselineEnv(nmmo_env, team_helper)
  return env

if __name__ == "__main__":
  num_cores = 1

  binding = pufferlib.emulation.Binding(
    env_creator=create_env,
    env_name="Neural MMO",
  )

  agent = pufferlib.frameworks.cleanrl.make_policy(Policy, lstm_layers=1)(
    binding
  )

  assert binding is not None
  cleanrl_ppo_lstm.train(
    binding,
    agent,
    cuda=torch.cuda.is_available(),
    total_timesteps=10_000_000,
    track=True,
    num_envs=num_cores,
    num_cores=num_cores,
    num_buffers=4,
    num_minibatches=4,
    num_agents=team_helper.num_teams,
    wandb_project_name="pufferlib",
    wandb_entity="platypus",
  )
