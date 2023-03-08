import os

import torch
from neurips2022nmmo import Team

from .agent import NMMOAgent
from .translator import Translator


class PvPAI(Team):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = NMMOAgent(False)
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'model.pth')
        agent_dict = torch.load(model_path)
        self.agent.loads(agent_dict)

        self.translator = Translator()
        self.step = 0

    def reset(self):
        self.step = 0

    def act(self, observations):
        if self.step == 0:
            self.translator.reset(observations)
        state = self.translator.trans_obs(observations)
        actions = self.agent.infer(state)
        actions = self.translator.trans_action(actions)
        self.step += 1
        return actions
