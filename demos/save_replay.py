'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import json

import nmmo
from nmmo.websocket import Application

from scripted import baselines
from demos import minimal


class Config(minimal.Config):
    SAVE_REPLAY           = 'replay'

class FakeRealm:
    def __init__(self):
        with open('replay.json', 'r') as inp:
            self.replay = json.load(inp)
        self.idx = 0
        
    @property
    def packet(self):
        data = self.replay['packets'][self.idx]

        data['environment']    = self.replay['map']

        self.idx += 1

        return data

def load_replay():
    realm  = FakeRealm()
    client = Application(realm)
    for _ in range(128):
        pos, cmd = client.update(realm.packet) 

if __name__ == '__main__':
    print('Running simulation...')
    minimal.simulate(nmmo.Env, Config, horizon=128)
    print('Loading replay. Open the client.')
    load_replay()
