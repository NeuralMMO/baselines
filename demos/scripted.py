'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo
from nmmo.lib import colors, material

from scripted import baselines

from demos import minimal

#Subclass nmmo.Agent
class LavaAgent(nmmo.Agent):
    '''Likes to jump in lava'''

    # Color of rendered agents
    color    = colors.Neon.RED

    # Required flag indicates scripted/neural
    scripted = True

    def __call__(self, obs):
        '''Override __call__ to specify agent behavior'''

        # Wrapper that extracts data from tensorized obs
        ob    = nmmo.scripting.Observation(self.config, obs)

        # Vector representation of the current agent
        agent = ob.agent

        # Static fn extracts attributes from vector objects
        attribute = nmmo.scripting.Observation.attribute

        Tile      = nmmo.Serialized.Tile
        lava      = material.Lava.index

        # ob.tile: extract single tile from observation
        # attribute: extract Index attribute from vector
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

        # Actions and arg names specified in nmmo.action
        return {nmmo.action.Move: {nmmo.action.Direction: direction}}

class LavaAgentConfig(minimal.Config):
    AGENTS    = [LavaAgent]

if __name__ == '__main__':
    minimal.simulate(nmmo.Env, LavaAgentConfig, render=True)
