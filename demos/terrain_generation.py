'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T
import numpy as np

import nmmo

from demos import minimal


class CustomMapGenerator(nmmo.MapGenerator):
    '''Subclass the base NMMO Map Generator'''
    def generate_map(self, idx):
        '''Override the default per-map generation method'''
        config  = self.config
        size    = config.TERRAIN_SIZE

        # Create fractal and material placeholders
        fractal = np.zeros((size, size)) #Unused in demo
        matl    = np.zeros((size, size), dtype=object)

        for r in range(size):
            for c in range(size):
                linf = max(abs(r - size//2), abs(c - size // 2))

                # Set per-tile materials
                if linf < 4:
                    matl[r, c] = nmmo.Terrain.STONE
                elif linf < 8:
                    matl[r, c] = nmmo.Terrain.WATER
                elif linf < 12:
                    matl[r, c] = nmmo.Terrain.FOREST
                elif linf <= size//2 - config.TERRAIN_BORDER:
                    matl[r, c] = nmmo.Terrain.GRASS
                else:
                    matl[r, c] = nmmo.Terrain.LAVA

        # Return signature includes fractal and material
        # Pass a zero array if fractal is not relevant
        return fractal, matl


class CustomTerrainGeneration(minimal.Config):
    # Enable custom generator
    MAP_GENERATOR         = CustomMapGenerator

    # Render preview to maps/custom
    GENERATE_MAP_PREVIEWS = True


if __name__ == '__main__':
    minimal.simulate(nmmo.Env, CustomTerrainGeneration, render=True)
