'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo

from scripted import baselines
from demos import minimal


class Lifetime(nmmo.overlay.Overlay):
    '''Renders tiles that have been occupied by long-lived
    agents in green and tiles that have not in red'''
    def update(self, obs):
        '''Called once per env step to update overlay data'''
        for player in self.realm.realm.players.values():
            lifetime = player.history.timeAlive.val
            r, c     = player.base.pos
            self.values[r, c] = -lifetime

    def register(self, obs):
        # Postprocessing function that converts to a color map
        overlay = nmmo.lib.overlay.twoTone(
                self.values, preprocess='clip', invert='True')

        # Tells the environment to send the overlay to the client
        self.realm.register(overlay)


class LifetimeOverlayEnv(nmmo.Env):
    def __init__(self, config):
        super().__init__(config)

        # Add Lifetime overlay to the overlay registry
        self.registry.overlays['lifetime'] = Lifetime


if __name__ == '__main__':
    minimal.simulate(LifetimeOverlayEnv, minimal.Config, render=True)
