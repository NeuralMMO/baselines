import torch
import torch.nn as nn

from .const import N_PLAYER_PER_TEAM
from .policy import NMMONet
from .util import tensorize_state, legal_mask


class Agent:
    def __init__(self, use_gpu, *args, **kwargs):
        self.use_gpu = use_gpu

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.state_handler_dict = {}

        torch.set_num_threads(1)
        self.training_iter = 0

    def register_model(self, name, model):
        assert isinstance(model, nn.Module)
        if name in self.state_handler_dict:
            raise KeyError(f"model named with {name} reassigned.")
        self.state_handler_dict[name] = model

    def loads(self, agent_dict):
        self.training_iter = agent_dict['training_iter']

        for name, np_dict in agent_dict['model_dict'].items():
            model = self.state_handler_dict[name]  # alias
            state_dict = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in zip(model.state_dict().keys(), np_dict.values())
            }
            model.load_state_dict(state_dict)


class NMMOAgent(Agent):
    def __init__(self, use_gpu):
        super().__init__(use_gpu)
        self.net = NMMONet().to(self.device)
        self.register_model('net', self.net)

        self.hx = torch.zeros((N_PLAYER_PER_TEAM, self.net.n_lstm_hidden))
        self.cx = torch.zeros((N_PLAYER_PER_TEAM, self.net.n_lstm_hidden))

    @tensorize_state
    def infer(self, state, train=True):
        with torch.no_grad():
            logits, value, self.hx, self.cx = self.net.infer(state, self.hx, self.cx)
            logits = {k: legal_mask(logits[k], state['legal'][k]) for k in logits}

            dists = {k: torch.distributions.Categorical(logits=logits[k]) for k in logits}
            if train:
                actions = {k: dists[k].sample() for k in logits}
            else:
                actions = {k: dists[k].probs.argmax(dim=-1) for k in logits}
            actions = [a.numpy() for a in actions.values()]

        return actions
