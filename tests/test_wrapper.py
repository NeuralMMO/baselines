import nmmo

import rllib_wrapper
import config

def test_env():
    conf = {'config': config.baselines.Medium()}
    env = rllib_wrapper.RLlibEnv(conf)
    env.reset()

    for i in range(32):
        env.step({})

if __name__ == '__main__':
    test_env()
