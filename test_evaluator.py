from pdb import set_trace as T

import nmmo

import evaluate
from scripted import baselines

from neural import policy

if __name__ == '__main__':
    config = nmmo.config.Default
    config.FORCE_MAP_GENERATION = True
    config.HORIZON = 32
    config.HIDDEN  = 16
    config.EMBED   = 16
    
    config.AGENTS = [baselines.Forage, baselines.Combat, nmmo.Agent]
    #policy = policy.Baseline(config)
    import cleanrl_lstm_wrapper
    policy = cleanrl_lstm_wrapper.Agent

    evaluator = evaluate.Evaluator(config, policy)
    #evaluator.render()
    #evaluator.evaluate()
    #async_handles = evaluator.ray_evaluate()
    #evaluator.ray_sync(async_handles)

    from concurrent.futures import ThreadPoolExecutor
    eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")


    future = eval_executor.submit(run_evaluation, path)
    future.add_done_callback(openskill_writer.on_evaluation_done)




    print(evaluator)
