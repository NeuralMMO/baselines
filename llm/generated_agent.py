from nmmo import material
import nmmo.systems.item as item_system

from scripted.baselines import Forage


class Agent(Forage):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.tool = item_system.Rod  # For fishing, a resource that replenishes health, food, and water

    @property
    def supplies(self):
        return {
            item_system.Ration.ITEM_TYPE_ID: 1,  # Just one ration to start with
            item_system.Potion.ITEM_TYPE_ID: 1   # Just one potion to start with
        }

    @property
    def wishlist(self):
        return {
            item_system.Hat.ITEM_TYPE_ID,     # Basic armor
            item_system.Top.ITEM_TYPE_ID,     # Basic armor
            item_system.Bottom.ITEM_TYPE_ID,  # Basic armor
            self.tool.ITEM_TYPE_ID            # Tool for fishing
        }

    def __call__(self, obs):
        super().__call__(obs)
        
        # Prioritize usage of available resources for survival
        self.use()

        # Exchange items in the market if possible
        self.exchange()

        # Gather resources when possible
        if self.forage_criterion:
            self.forage()
        # Explore in the absence of resources or the encroaching fog
        elif self.fog_criterion or not self.gather([material.Fish]):
            self.explore()
        
        return self.actions