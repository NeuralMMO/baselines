import nmmo
from nmmo import material
from nmmo.systems import skill
import nmmo.systems.item as item_system
from nmmo.lib import colors
from nmmo.core import action
from nmmo.core.observation import Observation

from scripted import attack, move
from scripted import Scripted


class Sleeper(Scripted):
  '''Do Nothing'''
  def __call__(self, obs):
    super().__call__(obs)
    return {}
class Random(Scripted):
  '''Moves randomly'''
  def __call__(self, obs):
    super().__call__(obs)

    move.rand(self.config, self.ob, self.actions, self._np_random)
    return self.actions

class Meander(Scripted):
  '''Moves randomly on safe terrain'''
  def __call__(self, obs):
    super().__call__(obs)

    move.meander(self.config, self.ob, self.actions, self._np_random)
    return self.actions

class Explore(Scripted):
  '''Actively explores towards the center'''
  def __call__(self, obs):
    super().__call__(obs)

    self.explore()

    return self.actions

class Forage(Scripted):
  '''Forages using Dijkstra's algorithm and actively explores'''
  def __call__(self, obs):
    super().__call__(obs)

    if self.forage_criterion:
        self.forage()
    else:
        self.explore()

    return self.actions

class Combat(Scripted):
  '''Forages, fights, and explores'''
  def __init__(self, config, idx):
    super().__init__(config, idx)
    self.style  = [action.Melee, action.Range, action.Mage]

  @property
  def supplies(self):
    return {
      item_system.Ration.ITEM_TYPE_ID: 2,
      item_system.Potion.ITEM_TYPE_ID: 2,
      self.ammo.ITEM_TYPE_ID: 10
    }

  @property
  def wishlist(self):
    return {
      item_system.Hat.ITEM_TYPE_ID,
      item_system.Top.ITEM_TYPE_ID,
      item_system.Bottom.ITEM_TYPE_ID,
      self.weapon.ITEM_TYPE_ID,
      self.ammo.ITEM_TYPE_ID
    }

  def __call__(self, obs):
    super().__call__(obs)
    self.use()
    self.exchange()

    self.adaptive_control_and_targeting()
    self.attack()

    return self.actions

class Gather(Scripted):
  '''Forages, fights, and explores'''
  def __init__(self, config, idx):
    super().__init__(config, idx)
    self.resource = [material.Fish, material.Herb, material.Ore, material.Tree, material.Crystal]

  @property
  def supplies(self):
    return {
      item_system.Ration.ITEM_TYPE_ID: 1,
      item_system.Potion.ITEM_TYPE_ID: 1
    }

  @property
  def wishlist(self):
    return {
      item_system.Hat.ITEM_TYPE_ID,
      item_system.Top.ITEM_TYPE_ID,
      item_system.Bottom.ITEM_TYPE_ID,
      self.tool.ITEM_TYPE_ID
    }

  def __call__(self, obs):
    super().__call__(obs)
    self.use()
    self.exchange()

    if self.forage_criterion:
      self.forage()
    elif self.fog_criterion or not self.gather(self.resource):
      self.explore()

    return self.actions

class Fisher(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.resource = [material.Fish]
    self.tool     = item_system.Rod

class Herbalist(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.resource = [material.Herb]
    self.tool     = item_system.Gloves

class Prospector(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.resource = [material.Ore]
    self.tool     = item_system.Pickaxe

class Carver(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.resource = [material.Tree]
    self.tool     = item_system.Axe

class Alchemist(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.resource = [material.Crystal]
    self.tool     = item_system.Chisel

class Melee(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.style  = [action.Melee]
    self.weapon = item_system.Spear
    self.ammo   = item_system.Whetstone

class Range(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.style  = [action.Range]
    self.weapon = item_system.Bow
    self.ammo   = item_system.Arrow

class Mage(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.SPECIALIZE:
      self.style  = [action.Mage]
    self.weapon = item_system.Wand
    self.ammo   = item_system.Runes