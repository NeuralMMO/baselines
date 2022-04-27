from pdb import set_trace as T
import torch

import nmmo

import evaluate
from scripted import baselines

from neural import policy
import cleanrl_lstm_wrapper

if __name__ == '__main__':
    class EvalConfig(cleanrl_lstm_wrapper.Config):
        PLAYERS   = [
                baselines.Fisher, baselines.Herbalist,
                baselines.Prospector, baselines.Carver, baselines.Alchemist,
                baselines.Melee, baselines.Range, baselines.Mage, nmmo.Agent]

        SPECIALIZE = 'specialist'

        #MAP_GENERATE_PREVIEWS = True
        #MAP_FORCE_GENERATION  = True

    #nmmo.MapGenerator(EvalConfig()).generate_all_maps()

    model = 'model_2xt4-32vcpu_1-6_137m.pt'
    state_dict = torch.load(model, map_location='cpu')
    state_dict = {k.lstrip('module')[1:]: v for k, v in state_dict.items()}

    evaluator = evaluate.Evaluator(EvalConfig, cleanrl_lstm_wrapper.Agent, num_cpus=2, device='cpu')
    evaluator.load_model(state_dict)
    #evaluator.render()
    for i in range(10):
        evaluator.evaluate()
    #async_handles = evaluator.ray_evaluate()
    #evaluator.ray_sync(async_handles)


    print(evaluator)
