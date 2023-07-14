src_mapping = {
    "norm": """def norm(progress):
  return max(min(progress, 1.0), 0.0)""",

    "Success":"""def Success(gs: GameState, subject: Group):
  ''' Returns True. For debugging.
  '''
  return True""",

    "TickGE":"""def TickGE(gs: GameState, subject: Group, num_tick: int):
  '''True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  '''
  return norm(gs.current_tick / num_tick)""",

    "CanSeeTile":"""def CanSeeTile(gs: GameState, subject: Group, tile_type: type[Material]):
  ''' True if any agent in subject can see a tile of tile_type
  '''
  return any(tile_type.index in t for t in subject.obs.tile.material_id)""",

    "StayAlive": '''def StayAlive(gs: GameState, subject: Group):
  """True if all subjects are alive.
  """
  return count(subject.health > 0) == len(subject)''',

    "AllDead": '''def AllDead(gs: GameState, subject: Group):
  """True if all subjects are dead.
  """
  return norm(1.0 - count(subject.health) / len(subject))''',

    "OccupyTile": '''def OccupyTile(gs: GameState, subject: Group, row: int, col: int):
  """True if any subject agent is on the desginated tile.
  """
  return np.any((subject.row == row) & (subject.col == col))''',

    "AllMembersWithinRange": '''def AllMembersWithinRange(gs: GameState, subject: Group, dist: int):
  """True if the max l-inf distance of teammates is
         less than or equal to dist
  """
  current_dist = max(subject.row.max()-subject.row.min(),
      subject.col.max()-subject.col.min())
  if current_dist <= 0:
    return 1.0
  return norm(dist / current_dist)''',

  "CanSeeAgent": '''def CanSeeAgent(gs: GameState, subject: Group, target: int):
  """True if obj_agent is present in the subjects' entities obs.
  """
  return any(target in e.ids for e in subject.obs.entities)''',

  "CanSeeGroup": '''def CanSeeGroup(gs: GameState, subject: Group, target: Iterable[int]):
  """ Returns True if subject can see any of target
  """
  return any(CanSeeAgent(gs, subject, agent) for agent in target)''',

  "DistanceTraveled": '''def DistanceTraveled(gs: GameState, subject: Group, dist: int):
  """True if the summed l-inf distance between each agent's current pos and spawn pos
        is greater than or equal to the specified _dist.
  """
  if not any(subject.health > 0):
    return False
  r = subject.row
  c = subject.col
  dists = utils.linf(list(zip(r,c)),[gs.spawn_pos[id_] for id_ in subject.entity.id])
  return norm(dists.sum() / dist)''',

  "AttainSkill": '''def AttainSkill(gs: GameState, subject: Group, skill: Skill, level: int, num_agent: int):
  """True if the number of agents having skill level GE level
        is greather than or equal to num_agent
  """
  skill_level = getattr(subject,skill.__name__.lower() + '_level')
  return norm(sum(skill_level >= level) / num_agent)''',

  "CountEvent": '''def CountEvent(gs: GameState, subject: Group, event: str, N: int):
  """True if the number of events occured in subject corresponding
      to event >= N
  """
  return norm(len(getattr(subject.event, event)) / N)''',

  "ScoreHit": '''def ScoreHit(gs: GameState, subject: Group, combat_style: type[Skill], N: int):
  """True if the number of hits scored in style
  combat_style >= count
  """
  hits = subject.event.SCORE_HIT.combat_style == combat_style.SKILL_ID
  return norm(count(hits) / N)''',

  "DefeatEntity": '''def DefeatEntity(gs: GameState, subject: Group, agent_type: str, level: int, num_agent: int):
  """True if the number of agents (agent_type, >= level) defeated
        is greater than or equal to num_agent
  """
  # NOTE: there is no way to tell if an agent is a teammate or an enemy
  #   so agents can get rewarded for killing their own teammates
  defeated_type = subject.event.PLAYER_KILL.target_ent > 0 if agent_type == 'player' \
                    else subject.event.PLAYER_KILL.target_ent < 0
  defeated = defeated_type & (subject.event.PLAYER_KILL.level >= level)
  if num_agent > 0:
    return norm(count(defeated) / num_agent)
  return 1.0''',

  "HoardGold": '''def HoardGold(gs: GameState, subject: Group, amount: int):
  """True iff the summed gold of all teammate is greater than or equal to amount.
  """
  return norm(subject.gold.sum() / amount)''',

  "EarnGold": '''def EarnGold(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold earned is greater than or equal to amount.
  """
  return norm(subject.event.EARN_GOLD.gold.sum() / amount)''',

    "SpendGold": '''def SpendGold(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold spent is greater than or equal to amount.
  """
  return norm(subject.event.BUY_ITEM.gold.sum() / amount)''',

    "MakeProfit": '''def MakeProfit(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold earned-spent is greater than or equal to amount.
  """
  profits = subject.event.EARN_GOLD.gold.sum()
  costs = subject.event.BUY_ITEM.gold.sum()
  return  norm((profits-costs) / amount)''',

    "InventorySpaceGE": '''def InventorySpaceGE(gs: GameState, subject: Group, space: int):
  """True if the inventory space of every subjects is greater than or equal to
       the space. Otherwise false.
  """
  max_space = gs.config.ITEM_INVENTORY_CAPACITY
  return all(max_space - inv.len >= space for inv in subject.obs.inventory)''',

    "OwnItem": '''def OwnItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if the number of items owned (_item_type, >= level)
     is greater than or equal to quantity.
  """
  owned = (subject.item.type_id == item.ITEM_TYPE_ID) & (subject.item.level >= level)
  return norm(sum(subject.item.quantity[owned]) / quantity)''',

    "EquipItem": '''def EquipItem(gs: GameState, subject: Group, item: type[Item], level: int, num_agent: int):
  """True if the number of agents that equip the item (_item_type, >=_level)
     is greater than or equal to _num_agent.
  """
  equipped = (subject.item.type_id == item.ITEM_TYPE_ID) & \
             (subject.item.level >= level) & \
             (subject.item.equipped > 0)
  if num_agent > 0:
    return norm(count(equipped) / num_agent)
  return 1.0''',

    "FullyArmed": '''def FullyArmed(gs: GameState, subject: Group,
               combat_style: type[Skill], level: int, num_agent: int):
  """True if the number of fully equipped agents is greater than or equal to _num_agent
       Otherwise false.
       To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively,
       and see whether these are equipped and has level greater than or equal to _level.
  """
  WEAPON_IDS = {
    nmmo_skill.Melee: {'weapon':5, 'ammo':13}, # Spear, Whetstone
    nmmo_skill.Range: {'weapon':6, 'ammo':14}, # Bow, Arrow
    nmmo_skill.Mage: {'weapon':7, 'ammo':15} # Wand, Runes
  }
  item_ids = { 'hat':2, 'top':3, 'bottom':4 }
  item_ids.update(WEAPON_IDS[combat_style])

  lvl_flt = (subject.item.level >= level) & \
            (subject.item.equipped > 0)
  type_flt = np.isin(subject.item.type_id,list(item_ids.values()))
  _, equipment_numbers = np.unique(subject.item.owner_id[lvl_flt & type_flt],
                                   return_counts=True)
  if num_agent > 0:
    return norm((equipment_numbers >= len(item_ids.items())).sum() / num_agent)
  return 1.0''',

  "ConsumeItem": '''def ConsumeItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity consumed of item type above level is >= quantity
  """
  type_flt = subject.event.CONSUME_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.CONSUME_ITEM.level >= level
  return norm(subject.event.CONSUME_ITEM.number[type_flt & lvl_flt].sum() / quantity)''',

  "HarvestItem": '''def HarvestItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity harvested of item type above level is >= quantity
  """
  type_flt = subject.event.HARVEST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.HARVEST_ITEM.level >= level
  return norm(subject.event.HARVEST_ITEM.number[type_flt & lvl_flt].sum() / quantity)''',

  "ListItem": '''def ListItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity listed of item type above level is >= quantity
  """
  type_flt = subject.event.LIST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.LIST_ITEM.level >= level
  return norm(subject.event.LIST_ITEM.number[type_flt & lvl_flt].sum() / quantity)''',

  "BuyItem": '''def BuyItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity purchased of item type above level is >= quantity
  """
  type_flt = subject.event.BUY_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.BUY_ITEM.level >= level
  return norm(subject.event.BUY_ITEM.number[type_flt & lvl_flt].sum() / quantity)''',

}