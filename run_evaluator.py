from pdb import set_trace as T

import torch

import nmmo

import evaluate
from scripted import baselines

from neural import policy

import cleanrl_lstm_wrapper

if __name__ == '__main__':
    model = 'model_4xt4_96vcpu_1b.pt'

    config = cleanrl_lstm_wrapper.Config
    config.AGENTS = [baselines.Forage, baselines.Combat, nmmo.Agent]

    evaluator  = evaluate.Evaluator(config, cleanrl_lstm_wrapper.Agent)
    state_dict = torch.load(model)
    evaluator.load_model(state_dict)

    #evaluator.render()
    for i in range(50):
        evaluator.evaluate()
        print(evaluator)
