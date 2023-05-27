uniq_predicates = ["TickGE","StayAlive","AllDead","EatFood","DrinkWater","CanSeeTile","CanSeeAgent","OccupyTile","DistanceTraveled",
                   "AllMembersWithinRange","ScoreHit","ScoreKill","AttainSkill","InventorySpaceGE","OwnItem","EquipItem","FullyArmed",
                   "ConsumeItem","GiveItem","DestroyItem","HarvestItem","HoardGold","GiveGold","ListItem","EarnGold","BuyItem",
                   "SpendGold","MakeProfit"]

import_str = {"short_import": """from predicates import TickGE,StayAlive,AllDead,EatFood,DrinkWater,CanSeeTile,CanSeeAgent,OccupyTile
from predicates import DistanceTraveled,AllMembersWithinRange,ScoreHit,ScoreKill,AttainSkill,InventorySpaceGE,OwnItem
from predicates import EquipItem,FullyArmed,ConsumeItem,GiveItem,DestroyItem,HarvestItem,HoardGold,GiveGold,ListItem,EarnGold,BuyItem,SpendGold,MakeProfit
# Armour
item.Hat, item.Top, item.Bottom, 
# Weapon
item.Sword, item.Bow, item.Wand
# Tool
item.Rod, item.Gloves, item.Pickaxe, item.Chisel, item.Arcane
# Ammunition
item.Scrap, item.Shaving, item.Shard, 
# Consumable
item.Ration, item.Poultice
# Materials
tile_type.Lava, tile_type.Water,tile_type.Grass, tile_type.Scrub, 
tile_type.Forest, tile_type.Stone, tile_type.Slag,  tile_type.Ore
tile_type.Stump, tile_type.Tree, tile_type.Fragment, tile_type.Crystal,
tile_type.Weeds, tile_type.Ocean, tile_type.Fish""", "long_import":

"""
Base Predicates to use in tasks:
TickGE(gs, num_tick):True if the current tick is greater than or equal to the specified num_tick.Is progress counter.
CanSeeTile(gs, subject,tile_type):True if any agent in subject can see a tile of tile_type
StayAlive(gs,subject): True if all subjects are alive.
AllDead(gs, subject):True if all subjects are dead.
OccupyTile(gs,subject,row, col):True if any subject agent is on the desginated tile.
AllMembersWithinRange(gs,subject,dist):True if the max l-inf distance of teammates is less than or equal to dist
CanSeeAgent(gs,subject,target): True if obj_agent is present in the subjects' entities obs.
CanSeeGroup(gs,subject,target): Returns True if subject can see any of target
DistanceTraveled(gs, subject, dist): True if the summed l-inf distance between each agent's current pos and spawn pos is greater than or equal to the specified _dist.
AttainSkill(gs,subject,skill,level,num_agent):True if the number of agents having skill level GE level is greather than or equal to num_agent
CountEvent(gs,subject,event,N): True if the number of events occured in subject corresponding to event >= N
ScoreHit(gs, subject, combat_style, N):True if the number of hits scored in style combat_style >= count
HoardGold(gs, subject, amount): True iff the summed gold of all teammate is greater than or equal to amount.
EarnGold(gs, subject, amount): True if the total amount of gold earned is greater than or equal to amount.
SpendGold(gs, subject, amount): True if the total amount of gold spent is greater than or equal to amount.
MakeProfit(gs, subject, amount) True if the total amount of gold earned-spent is greater than or equal to amount.
InventorySpaceGE(gs, subject, space): True if the inventory space of every subjects is greater than or equal to the space. Otherwise false.
OwnItem(gs, subject, item, level, quantity): True if the number of items owned (_item, >= level) is greater than or equal to quantity.
EquipItem(gs, subject, item, level, num_agent): True if the number of agents that equip the item (_item_type, >=_level) is greater than or equal to _num_agent.
FullyArmed(GameState, subject, combat_style, level, num_agent): True if the number of fully equipped agents is greater than or equal to _num_agent Otherwise false. To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively, these are equipped and has level greater than or equal to _level.
ConsumeItem(GameState, subject, item, level, quantity): True if total quantity consumed of item type above level is >= quantity
HarvestItem(GameState, Group, Item, int, int): True if total quantity harvested of item type above level is >= quantity
ListItem(GameState,subject,item,level, quantity): True if total quantity listed of item type above level is >= quantity
BuyItem(GameState, subject, item, level, quantity) :  True if total quantity purchased of item type above level is >= quantity
# Armour
item.Hat, item.Top, item.Bottom, 
# Weapon
item.Sword, item.Bow, item.Wand
# Tool
item.Rod, item.Gloves, item.Pickaxe, item.Chisel, item.Arcane
# Ammunition
item.Scrap, item.Shaving, item.Shard, 
# Consumable
item.Ration, item.Poultice
# Materials
tile_type.Lava, tile_type.Water,tile_type.Grass, tile_type.Scrub, 
tile_type.Forest, tile_type.Stone, tile_type.Slag,  tile_type.Ore
tile_type.Stump, tile_type.Tree, tile_type.Fragment, tile_type.Crystal,
tile_type.Weeds, tile_type.Ocean, tile_type.Fish
"""}


tasks = ["""def task_BandOfGuildsman(gs: GameState,
                      subject: Group):
    sub_predicates = []
    for skill in [Fishing, Prospecting, Carving, Alchemy]:
      sub_predicates.append(AttainSkill(subject=subject,
                                        skill=skill,
                                        level=4,
                                        num_agent=1))
    return AND(*sub_predicates)""",
    """def task_headhunt(gs: GameState, vip: Group, target: Group):
    # True when vip is alive and target is dead
    return StayAlive(vip) & AllDead(target)""",

    """def task_ScoreHitAnyStyle(gs: GameState,
                       subject: Group,
                       N: int):
    #True if the number of hits scored in any combat style >= N
    mage = ScoreHit(subject=subject, combat_style=Mage, N=N)
    melee = ScoreHit(subject=subject, combat_style=Melee, N=N)
    range = ScoreHit(subject=subject, combat_style=Range, N=N)
    return mage(gs)+melee(gs)+range(gs)""",
    """def task_EquipmentLevel(gs: GameState,
                        subject: Group,
                        number: int):
        equipped = (subject.item.equipped>0)
        levels = subject.item.level[equipped]
        return levels.sum() >= number""",
    """def task_CombatSkill(gs, agent, lvl):
        return OR(AttainSkill(agent, nmmo_skill.Melee, lvl, 1),
                AttainSkill(agent, nmmo_skill.Range, lvl, 1),
                AttainSkill(agent, nmmo_skill.Mage, lvl, 1))""",
    """def task_ForageSkill(gs, agent, lvl):
         return OR(AttainSkill(agent, nmmo_skill.Fishing, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Herbalism, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Prospecting, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Carving, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Alchemy, lvl, 1))""", 
    """def task_gold(entity: Entity):
    # Task to hoard 30 gold
    return HoardGold(gold=30)""",

    """def task_gold_pickaxe(entity: Entity):
        # Task to hoard 30 gold, have health above 50 and own a level 3 pickaxe
        return AND(HoardGold(gold=30), OwnItem(item.Pickaxe, min_level=3))""",

    """def task_travel(destination: Tile):
        # Task to travel to a given destination tile
        return DistanceTraveled(destination=destination)""",

    """def task_hunt(target: Agent):
        # Task to hunt a specific target agent
        return AND(AllMembersWithinRange(agent=target, distance=10), ScoreKill(target=target))""",

    """def task_gather_food():
        # Task to gather food
        return AND(CanSeeTile(tile_type=TileType.Food), EatFood())""",

    """def task_gather_water():
        # Task to gather water
        return AND(CanSeeTile(tile_type=TileType.Water), DrinkWater())""",

    """def task_craft_item(item_type: Item):
        # Task to craft a specific item
        return AND(AttainSkill(skill_type=SkillType.Crafting, min_level=item.level_required), InventorySpaceGE(item.space_required), OwnItem(item.required_items), GiveItem(item.required_items), EquipItem(item))""",

    """def task_sell_item(item: Item):
        # Task to sell a specific item and earn gold
        return AND(OwnItem(item), ListItem(item), SellItem(item), EarnGold(gold=item.sell_value))""",

    """def task_buy_item(item: Item):
        # Task to buy a specific item using gold
        return AND(ListItem(item), BuyItem(item), SpendGold(gold=item.buy_value), OwnItem(item))""",

    """def task_upgrade_weapon(weapon_type: WeaponType):
        # Task to upgrade a specific weapon type
        return AND(AttainSkill(skill_type=SkillType.WeaponUpgrading, min_level=weapon_type.level_required), InventorySpaceGE(weapon_type.space_required), OwnItem(weapon_type.required_items), GiveItem(weapon_type.required_items), EquipItem(weapon_type), MakeProfit())""",

    """def task_kill_enemies():
        # Task to kill at least 5 enemies
        return AND(ScoreHit(combat_style=CombatStyle.Melee, count=5), AllMembersWithinRange(dist=5))""",

    """def task_mine_ore():
        # Task to mine at least 10 ore using a level 2 pickaxe
        return AND(OwnItem(item.Pickaxe, min_level=2), HarvestItem(item.Ore, min_level=1, quantity=10))""",

    """def task_craft_items():
        # Task to craft at least 3 level 2 swords and level 1 armor for all teammates
        return AND(EarnGold(amount=100), AttainSkill(skill=Skill.Crafting, level=2, num_agent=len(entity.teammates))
        OwnItem(item.Gold, quantity=50),
        ConsumeItem(item.Ore, min_level=2, quantity=9),
        CraftItem(item.Sword, level=2, quantity=3),
        CraftItem(item.Top, level=1, num_agent=len(entity.teammates))
        CraftItem(item.Bottom, level=1, num_agent=len(entity.teammates))
        CraftItem(item.Hat, level=1, num_agent=len(entity.teammates))""",

    """def task_collect_ammo():
        # Task to collect at least 50 shards and 30 scrap for ammo
        return AND(HarvestItem(item.Shard, min_level=1, quantity=50),
        HarvestItem(item.Scrap, min_level=1, quantity=30),
        OwnItem(item.Scrap, quantity=30),
        OwnItem(item.Shard, quantity=50))""",

    """def task_heal_teammates():
        # Task to heal all teammates to full health using poultices
        return AND(CountEvent(event=Event.TookDamage, N=len(entity.teammates))
        OwnItem(item.Poultice, min_level=1, quantity=len(entity.teammates))
        ConsumeItem(item.Poultice, min_level=1, quantity=len(entity.teammates))
        HealTeammates())""",

    """def task_explore_map():
        # Task to explore the entire map and return to base
        return AND(DistanceTraveled(dist=1000),
        OccupyTile(row=0, col=0),
        AllMembersWithinRange(dist=5),
        CanSeeTile(tile_type=TileType.Exit))""",

    """def task_mine_iron(entity: Entity):
        # Task to mine iron ore
        return AND(CanSeeTile(tile_type=TileType.Ore), OccupyTile(row=entity.row, col=entity.col), OwnItem(item.Pickaxe, min_level=2), EarnGold(gold=10))""",

     """def task_farm_crop(entity: Entity):
        # Task to farm crops and harvest at least 20 units of food
        return AND(CanSeeTile(tile_type=TileType.Soil), OccupyTile(row=entity.row, col=entity.col), OwnItem(item.Tool, min_level=1), HarvestItem(tile_type=TileType.Food, min_level=1, quantity=20))""",

    """def task_fish(entity: Entity):
        # Task to fish and catch at least 10 fish
        return AND(CanSeeTile(tile_type=TileType.Water), OccupyTile(row=entity.row, col=entity.col), OwnItem(item.Rod, min_level=1), HarvestItem(tile_type=TileType.Fish, min_level=1, quantity=10))""",

    """def task_craft_item(entity: Entity):
        # Task to craft an item, spending at most 20 gold and owning all necessary materials
        return AND(SpendGold(gold=20), OwnItem(item.Material, min_level=2, quantity=3), CanSeeTile(tile_type=TileType.Workbench), CraftItem(item=Item.Tool, min_level=2))""",

    """def task_sell_item(entity: Entity):
        # Task to sell an item for at least 15 gold
        return AND(OwnItem(item=Item.Tool), ListItem(item=Item.Tool, min_level=2, quantity=1), BuyItem(item=Item.Tool, min_level=2, quantity=1), EarnGold(gold=15))""",

    """def task_explore(entity: Entity):
        # Task to explore and travel at least 100 units from spawn point
        return AND(DistanceTraveled(dist=100))""",

    """def task_train_combat(entity: Entity):
        # Task to train combat, attaining at least level 3 in melee combat for 2 agents
        return AND(EquipItem(item.Sword, min_level=2, num_agent=2), AttainSkill(skill=SkillType.Combat, level=3, num_agent=2))""",

    """def task_mine_gem(entity: Entity):
        # Task to mine a gem and earn at least 50 gold
        return AND(CanSeeTile(tile_type=TileType.Ore), OccupyTile(row=entity.row, col=entity.col), OwnItem(item.Pickaxe, min_level=4), EarnGold(gold=50), HarvestItem(tile_type=TileType.Crystal, min_level=1, quantity=1))""",

    """def task_heal(entity: Entity):
        # Task to heal to full health using poultices
        return AND(OwnItem(item.Poultice, min_level=1, quantity=5), Heal())""",

    """def task_sell_fish(entity: Entity):
        # Task to sell fish for at least 10 gold
        return AND(OwnItem(item=Item.Fish), ListItem(item=Item.Fish, min_level=1, quantity=10), BuyItem(item=Item.Fish, min_level=1, quantity=10), EarnGold(gold=10))""",


    """def task_mine_stone(entity: Entity):
        # Task to mine 50 stone and own a level 2 pickaxe
        return AND(CountEvent(event=Event.MineStone, N=50), OwnItem(item.Pickaxe, min_level=2))""",

    """def task_hunt_rabbit(entity: Entity):
        # Task to hunt 10 rabbits and own a level 3 bow
        return AND(CountEvent(event=Event.HuntRabbit, N=10), OwnItem(item.Bow, min_level=3))""",

    """def task_fish(entity: Entity):
        # Task to fish 20 times and own a level 2 rod
        return AND(CountEvent(event=Event.Fish, N=20), OwnItem(item.Rod, min_level=2))""",

    """def task_harvest_crystal(entity: Entity):
        # Task to harvest 25 crystals and own a level 3 chisel
        return AND(CountEvent(event=Event.HarvestCrystal, N=25), OwnItem(item.Chisel, min_level=3))""",

    """def task_explore(entity: Entity):
        # Task to explore the map and stay alive
        return AND(TickGE(num_tick=500), StayAlive(), DistanceTraveled(dist=100))""",

    """def task_buy_armor(entity: Entity):
        # Task to buy a level 3 top and a level 2 bottom
        return AND(BuyItem(item.Top, min_level=3, quantity=1), BuyItem(item.Bottom, min_level=2, quantity=1))""",

    """def task_sell_items(entity: Entity):
        # Task to make a profit of 50 gold by selling items
        return AND(MakeProfit(amount=50), SellItems())""",

    """def task_craft_item(entity: Entity):
        # Task to craft 3 level 2 swords
        return AND(CraftItem(item.Sword, level=2, quantity=3))""",

    """def task_equip_item(entity: Entity):
        # Task to equip a level 3 wand with at least 2 agents
        return AND(EquipItem(item.Wand, min_level=3, num_agent=2))""",

    """def task_consume_food(entity: Entity):
        # Task to consume 10 rations
        return AND(ConsumeItem(item.Ration, min_level=1, quantity=10))""",

    """def task_build_structure(entity: Entity):
        # Task to build a structure using 100 stone and 50 wood
        return AND(CountEvent(event=Event.GatherWood, N=50), CountEvent(event=Event.MineStone, N=100), BuildStructure())""",

    """def task_harvest_wood(entity: Entity):
        # Task to harvest 50 wood and own a level 2 axe
        return AND(CountEvent(event=Event.HarvestWood, N=50), OwnItem(item.Axe, min_level=2))""",

    """def task_equip_armor(entity: Entity):
        # Task to fully equip 4 agents with level 3 hat, top, and bottom
        return AND(FullyArmed(combat_style=CombatStyle.defense, level=3, num_agent=4))""",

    """def task_sell_materials(entity: Entity):
        # Task to make a profit of 100 gold by selling materials
        return AND(MakeProfit(amount=100), SellMaterials())""",

    """def task_craft_ammunition(entity: Entity):
        # Task to craft 10 level 2 shards
        return AND(CraftItem(item.Shard, level=2, quantity=10))"""]