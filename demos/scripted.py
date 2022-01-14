'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo
from nmmo.lib import colors, material

from scripted import baselines

from demos import minimal


class LavaAgent(nmmo.Agent):
    '''Likes to jump in lava'''
    scripted = True
    color = colors.Neon.RED

    def __call__(self, obs):
        ob    = nmmo.scripting.Observation(self.config, obs)
        agent = ob.agent

        attribute = nmmo.scripting.Observation.attribute
        Tile      = nmmo.Serialized.Tile
        lava      = material.Lava.index

        direction = None
        if attribute(ob.tile(-1, 0), Tile.Index) == lava:
            direction = nmmo.action.North

        if attribute(ob.tile(1, 0), Tile.Index) == lava:
            direction = nmmo.action.South

        if attribute(ob.tile(0, 1), Tile.Index) == lava:
            direction = nmmo.action.East

        if attribute(ob.tile(0, -1), Tile.Index) == lava:
            direction = nmmo.action.West

        if not direction:
            return {}

        return {nmmo.action.Move: {nmmo.action.Direction: direction}}

class LavaAgentConfig(minimal.Config):
    AGENTS    = [LavaAgent]

if __name__ == '__main__':
    minimal.simulate(nmmo.Env, LavaAgentConfig, render=True)
