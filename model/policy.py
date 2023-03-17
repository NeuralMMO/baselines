import torch
import pufferlib

from model.model import NMMONet
from model.translator import Translator

class Policy(pufferlib.models.Policy):
    def __init__(self, binding, input_size=512, hidden_size=512):
        super().__init__(input_size, hidden_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state_handler_dict = {}
        torch.set_num_threads(1)

        self.net = NMMONet().to(self.device)
        self.feature_parser = Translator()
        self.reset = True

        self.raw_single_observation_space = binding.raw_single_observation_space

        # A dumb example encoder that applies a linear layer to agent self features
        observation_size = binding.raw_single_observation_space["Entity"].shape[1]

        self.encoder = torch.nn.Linear(observation_size, hidden_size)
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, n) for n in binding.single_action_space.nvec]
        )
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            self.raw_single_observation_space, env_outputs
        )
        if self.reset:
            self.feature_parser.reset(env_outputs)
            self.reset = False
        trans_obs = self.feature_parser.trans_obs(env_outputs)
        print(trans_obs.keys())
        env_outputs = env_outputs["Entity"][:, 0, :]
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions
