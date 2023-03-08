import torch
import cleanrl_ppo_lstm
import pufferlib.emulation
import pufferlib.registry.nmmo
import pufferlib.frameworks.cleanrl
import nmmo
from model.policy import Policy

if __name__ == "__main__":
    num_cores = 1

    binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name="Neural MMO",
    )

    agent = pufferlib.frameworks.cleanrl.make_cleanrl_policy(Policy, lstm_layers=1)(
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
        num_agents=128,
        wandb_project_name="pufferlib",
        wandb_entity="platypus",
    )
