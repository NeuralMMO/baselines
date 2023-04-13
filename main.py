import torch
import cleanrl_ppo_lstm
import pufferlib.emulation
import pufferlib.registry.nmmo
import pufferlib.frameworks.cleanrl
import nmmo
from model.policy import Policy
from model.model import MemoryBlock
from feature_extractor.feature_extractor import FeatureExtractor


if __name__ == "__main__":
  num_cores = 4

  config = nmmo.config.MediumConfig()

  def make_env():
    return nmmo.Env(config)

  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    teams = {i: [i*8+j+1 for j in range(8)] for i in range(16)},
    featurizer_cls=FeatureExtractor,
    featurizer_args=[config],
  )

  agent = pufferlib.frameworks.cleanrl.make_policy(
      Policy,
      recurrent_cls=MemoryBlock,
      recurrent_args=[2048, 4096],
      recurrent_kwargs={'num_layers': 1},
      )(
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
    num_agents=16,
    wandb_project_name="nmmo",
    wandb_entity="daveey",
  )
