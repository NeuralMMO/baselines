from pdb import set_trace as T

import nmmo

import evaluate
from scripted import baselines

from neural import policy

import cleanrl_lstm_wrapper

if __name__ == '__main__':
    model = 'model_1xt4_32vcpu_500m.pt'

    config = cleanrl_lstm_wrapper.Config
    config.AGENTS = [baselines.Forage, baselines.Combat, nmmo.Agent]

    evaluator  = evaluate.Evaluator(config, cleanrl_lstm_wrapper.Agent)
    state_dict = torch.load(model, map_location=device) 
    evaluator.load_policy(state_dict)

    #evaluator.render()
    for i in range(50):
        evaluator.evaluate()
        print(evaluator)
