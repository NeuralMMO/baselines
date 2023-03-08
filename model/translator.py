import collections
import copy

import nmmo.io.action as nmmo_act
import numpy as np

from .const import *
from .util import one_hot_generator, multi_hot_generator


DEFOGGING_VALUE = 16
VISITATION_MEMORY = 100

N_CH = 7
IMG_SIZE = 25
DUMMY_IMG_FEAT = np.zeros((N_CH, IMG_SIZE, IMG_SIZE))
X_IMG = np.arange(TERRAIN_SIZE+1).repeat(TERRAIN_SIZE+1).reshape(TERRAIN_SIZE+1, TERRAIN_SIZE+1)
Y_IMG = X_IMG.transpose(1, 0)
TEAMMATE_REPR = 1 / 5.
ENEMY_REPR = 2 / 5.
NEGATIVE_REPR = 3 / 5.
NEUTRAL_REPR = 4 / 5.
HOSTILE_REPR = 1.

N_NPC_CONSIDERED = 9
N_ENEMY_CONSIDERED = 9
AWARE_RANGE = 15
PER_ITEM_FEATURE = 11
N_SELF_FEATURE = 262
PER_ENTITY_FEATURE = 30
DUMMY_ITEM_TYPES = np.zeros(N_ITEM_SLOT)
DUMMY_ITEM_ARRS = np.zeros((N_ITEM_SLOT, PER_ITEM_FEATURE))

N_MOVE = 4
N_ATK_TARGET = N_NPC_CONSIDERED + N_ENEMY_CONSIDERED
N_ATK_TYPE = 3
N_USE = 2
N_SELL = 2

USE_POULTICE = 0
USE_RATION = 1
SELL_POULTICE = 0
SELL_RATION = 1


def get_init_tile_map():
    arr = np.zeros((TERRAIN_SIZE+1, TERRAIN_SIZE+1))
    # mark the most outside circle of grass
    arr[MAP_LEFT:MAP_RIGHT+1, MAP_LEFT:MAP_RIGHT+1] = 2
    # mark the unseen tiles
    arr[MAP_LEFT+1:MAP_RIGHT, MAP_LEFT+1:MAP_RIGHT] = N_TILE_TYPE
    return arr


def get_init_poison_map():
    arr = np.ones((TERRAIN_SIZE + 1, TERRAIN_SIZE + 1))
    for i in range(TERRAIN_SIZE // 2):
        l, r = i + 1, TERRAIN_SIZE - i
        arr[l:r, l:r] = -i
    # positive value represents the poison strength
    # negative value represents the shortest distance to poison area
    return arr


def mark_point(arr_2d, index_arr, value, clip=False):
    arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
        np.clip(value, 0., 1.) if clip else value


class Translator:
    def __init__(self):
        self.curr_step = None
        self.curr_obs = None
        self.prev_obs = None

        self.tile_map = None
        self.fog_map = None
        self.visit_map = None
        self.poison_map = None
        self.entity_map = None

        self.npc_tgt = None
        self.enemy_tgt = None

        self.target_entity_id = None
        self.target_entity_pop = None
        self.player_kill_num = None  # for comparing with playerDefeat stat only
        self.npc_kill_num = None  # statistics for reward only
        self.step_onto_herb_cnt = None  # statistics for reward only

        self.prof = None
        self.best_hats = None
        self.best_tops = None
        self.best_bottoms = None
        self.best_weapons = None
        self.best_tools = None
        self.force_use_idx = None
        self.force_sell_idx = None
        self.force_sell_price = None
        self.force_buy_idx = None
        self.rescue_cooldown = None  # cooldown rounds of sell poultice to teammates

        self.prev_actions = None

    def reset(self, init_obs):
        self.curr_step = 0
        self.prev_obs = init_obs
        self.rescue_cooldown = 0

        self.tile_map = get_init_tile_map()
        self.fog_map = np.zeros((TERRAIN_SIZE+1, TERRAIN_SIZE+1))
        self.visit_map = np.zeros((N_PLAYER_PER_TEAM, TERRAIN_SIZE+1, TERRAIN_SIZE+1))
        self.poison_map = get_init_poison_map()
        self.entity_map = None

        self.target_entity_id = [None] * N_PLAYER_PER_TEAM
        self.target_entity_pop = [None] * N_PLAYER_PER_TEAM
        self.player_kill_num = [0] * N_PLAYER_PER_TEAM
        self.npc_kill_num = {kind: [0] * N_PLAYER_PER_TEAM for kind in 'pnh'}  # p: passive, n: neutral, h: hostile
        self.step_onto_herb_cnt = [0] * N_PLAYER_PER_TEAM

        self.prof = self._choose_prof()

        self.prev_actions = np.array([N_MOVE, N_ATK_TARGET, N_USE, N_SELL])[None, :] \
                              .repeat(N_PLAYER_PER_TEAM, axis=0)  # init as idle

    def trans_obs(self, obs):
        self.curr_obs = obs
        self.rescue_cooldown = max(0, self.rescue_cooldown - 1)

        self._update_global_maps(obs)
        self._update_kill_num(obs)
        tile = self._extract_tile_feature(obs)
        self._process_items(obs)  # use & sell
        self._process_market(obs)  # buy
        item_type, item = self._extract_item_feature(obs)
        team, npc, enemy, *masks, self.npc_tgt, self.enemy_tgt = self._extract_entity_feature(obs)
        game = self._extract_game_feature(obs)
        legal = self._extract_legal_action(obs, self.npc_tgt, self.enemy_tgt)
        reset_flag = np.array([self.curr_step == 0])  # for resetting RNN hidden

        state = {
            'tile': tile,
            'item_type': item_type,
            'item': item,
            'team': team,
            'npc': npc,
            'enemy': enemy,
            'team_mask': masks[0],
            'npc_mask': masks[1],
            'enemy_mask': masks[2],
            'game': game,
            'legal': legal,
            'prev_act': self.prev_actions,
            'reset': reset_flag,
        }
        return state

    @staticmethod
    def _choose_prof():
        seed = np.random.randint(N_ATK_TYPE)
        profs = [ATK_TYPE[(seed + i) % N_ATK_TYPE]
                 for i in range(N_PLAYER_PER_TEAM)]
        np.random.shuffle(profs)
        return profs

    def _update_global_maps(self, obs):
        if self.curr_step % 16 == 15:
            self.poison_map += 1  # poison shrinking
        self.fog_map = np.clip(self.fog_map - 1, 0, DEFOGGING_VALUE)  # decay
        self.visit_map = np.clip(self.visit_map - 1, 0, VISITATION_MEMORY)  # decay
        entity_map = np.zeros((5, TERRAIN_SIZE+1, TERRAIN_SIZE+1))
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:  # dead
                continue
            # mark tile
            tile_obs = obs[i]['Tile']['Continuous']
            tile_pos = tile_obs[:, -2:].astype(int)
            tile_type = tile_obs[:, 1]
            mark_point(self.fog_map, tile_pos, DEFOGGING_VALUE)
            x, y = tile_pos[0]
            self.tile_map[x:x+WINDOW, y:y+WINDOW] = tile_type.reshape(WINDOW, WINDOW)
            # mark team/enemy/npc
            n_entity = obs[i]['Entity']['N'][0]
            entity_obs = obs[i]['Entity']['Continuous']
            entity_pos = entity_obs[:n_entity, 7:9].astype(int)
            entity_pop = entity_obs[:n_entity, 6]
            my_pop = entity_obs[0, 6]
            mark_point(entity_map[0], entity_pos, entity_pop == my_pop)  # team
            mark_point(entity_map[1], entity_pos, np.logical_and(entity_pop != my_pop, entity_pop > 0))  # enemy
            mark_point(entity_map[2], entity_pos, entity_pop == -1)  # negative
            mark_point(entity_map[3], entity_pos, entity_pop == -2)  # neutral
            mark_point(entity_map[4], entity_pos, entity_pop == -3)  # hostile
            # update visit map
            mark_point(self.visit_map[i], entity_pos[:1], VISITATION_MEMORY)
            # update herb gathering count
            my_curr_pos = entity_obs[0, 7:9].astype(int)
            my_prev_pos = self.prev_obs[i]['Entity']['Continuous'][0, 7:9].astype(int)
            if self.tile_map[my_curr_pos[0], my_curr_pos[1]] == TILE_HERB and \
                    (my_curr_pos[0] != my_prev_pos[0] or my_curr_pos[1] != my_prev_pos[1]):
                self.step_onto_herb_cnt[i] += 1
            # change tile in advance
            for pos, pop in zip(entity_pos, entity_pop):
                if pop >= 0:  # is player
                    if self.tile_map[pos[0], pos[1]] == TILE_FOREST:
                        self.tile_map[pos[0], pos[1]] = TILE_SCRUB
                    if self.tile_map[pos[0], pos[1]] == TILE_ORE:
                        self.tile_map[pos[0], pos[1]] = TILE_SLAG
                    if self.tile_map[pos[0], pos[1]] == TILE_TREE:
                        self.tile_map[pos[0], pos[1]] = TILE_STUMP
                    if self.tile_map[pos[0], pos[1]] == TILE_CRYSTAL:
                        self.tile_map[pos[0], pos[1]] = TILE_FRAGMENT
                    if self.tile_map[pos[0], pos[1]] == TILE_HERB:
                        self.tile_map[pos[0], pos[1]] = TILE_WEEDS
                    if self.tile_map[pos[0], pos[1]+1] == TILE_FISH:
                        self.tile_map[pos[0], pos[1]+1] = TILE_OCEAN
                    if self.tile_map[pos[0], pos[1]-1] == TILE_FISH:
                        self.tile_map[pos[0], pos[1]-1] = TILE_OCEAN
                    if self.tile_map[pos[0]+1, pos[1]] == TILE_FISH:
                        self.tile_map[pos[0]+1, pos[1]] = TILE_OCEAN
                    if self.tile_map[pos[0]-1, pos[1]] == TILE_FISH:
                        self.tile_map[pos[0]-1, pos[1]] = TILE_OCEAN
        self.entity_map = entity_map[0] * TEAMMATE_REPR + entity_map[1] * ENEMY_REPR + \
            entity_map[2] * NEGATIVE_REPR + entity_map[3] * NEUTRAL_REPR + entity_map[4] * HOSTILE_REPR

    def _update_kill_num(self, obs):
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:  # dead
                continue
            if self.target_entity_id[i] is None:  # no target
                continue
            entity_obs = obs[i]['Entity']['Continuous']
            entity_in_sight = entity_obs[:, 1]
            if self.target_entity_id[i] not in entity_in_sight:
                if self.target_entity_id[i] > 0:
                    self.player_kill_num[i] += 1
                elif self.target_entity_id[i] < 0:
                    if self.target_entity_pop[i] == -1:
                        self.npc_kill_num['p'][i] += 1
                    elif self.target_entity_pop[i] == -2:
                        self.npc_kill_num['n'][i] += 1
                    elif self.target_entity_pop[i] == -3:
                        self.npc_kill_num['h'][i] += 1
                    else:
                        raise ValueError('Unknown npc pop:', self.target_entity_pop[i])

    def _extract_tile_feature(self, obs):
        imgs = []
        for i in range(N_PLAYER_PER_TEAM):
            # replace with dummy feature if dead
            if i not in obs:
                imgs.append(DUMMY_IMG_FEAT)
                continue

            curr_pos = obs[i]['Entity']['Continuous'][0, 7:9].astype(int)
            l, r = curr_pos[0] - IMG_SIZE // 2, curr_pos[0] + IMG_SIZE // 2 + 1
            u, d = curr_pos[1] - IMG_SIZE // 2, curr_pos[1] + IMG_SIZE // 2 + 1
            tile_img = self.tile_map[l:r, u:d] / N_TILE_TYPE
            # obstacle_img = np.sum([self.tile_map[l:r, u:d] == t for t in OBSTACLE_TILES], axis=0)
            entity_img = self.entity_map[l:r, u:d]
            poison_img = np.clip(self.poison_map[l:r, u:d], 0, np.inf) / 20.
            fog_img = self.fog_map[l:r, u:d] / DEFOGGING_VALUE
            # view_img = (fog_img == 1.).astype(np.float32)
            visit_img = self.visit_map[i][l:r, u:d] / VISITATION_MEMORY
            coord_imgs = [X_IMG[l:r, u:d] / TERRAIN_SIZE, Y_IMG[l:r, u:d] / TERRAIN_SIZE]
            img = np.stack([tile_img, entity_img, poison_img, fog_img, visit_img, *coord_imgs])
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs

    def _process_items(self, obs):
        # reset
        self.best_hats = [None] * N_PLAYER_PER_TEAM
        self.best_tops = [None] * N_PLAYER_PER_TEAM
        self.best_bottoms = [None] * N_PLAYER_PER_TEAM
        self.best_weapons = [None] * N_PLAYER_PER_TEAM
        self.best_tools = [None] * N_PLAYER_PER_TEAM
        self.force_use_idx = [None] * N_PLAYER_PER_TEAM
        self.force_sell_idx = [None] * N_PLAYER_PER_TEAM
        self.force_sell_price = [None] * N_PLAYER_PER_TEAM

        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:
                continue
            n = obs[i]['Item']['N'][0]
            item_obs = obs[i]['Item']['Continuous'][:n]
            my_obs = obs[i]['Entity']['Continuous'][0]

            # force use weapons and armors
            usable_types = [  # reflect priority
                ATK_TO_WEAPON[self.prof[i]],
                ITEM_HAT,
                ITEM_TOP,
                ITEM_BOTTOM,
            ]
            best_savers = [
                self.best_weapons,
                self.best_hats,
                self.best_tops,
                self.best_bottoms,
            ]
            for item_type, saver in zip(usable_types, best_savers):
                if item_type in ARMORS:
                    max_equipable_lvl = max(my_obs[-N_PROF:])  # maximum of all levels
                else:
                    max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[item_type]]
                items = item_obs[item_obs[:, IDX_ITEM_INDEX] == item_type]  # those with the target type
                items = items[items[:, IDX_ITEM_LEVEL] <= max_equipable_lvl]  # those within the equipable level
                if len(items) > 0:
                    max_item_lvl = max(items[:, IDX_ITEM_LEVEL])
                    items = items[items[:, IDX_ITEM_LEVEL] == max_item_lvl]  # those with highest level
                    min_id = min(items[:, IDX_ITEM_ID])  # always choose the one with the minimal item id as the best
                    idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()
                    saver[i] = item_obs[idx]
                    if not item_obs[idx][IDX_ITEM_EQUIPPED] and self.force_use_idx[i] is None:
                        self.force_use_idx[i] = idx  # save for later translation

            # force use tools
            if self.best_weapons[i] is None:
                tools = []
                for tool_type in TOOLS:
                    tools_ = item_obs[item_obs[:, IDX_ITEM_INDEX] == tool_type]
                    max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[tool_type]]
                    tools_ = tools_[tools_[:, IDX_ITEM_LEVEL] <= max_equipable_lvl]
                    tools.append(tools_)
                tools = np.concatenate(tools)
                if len(tools) > 0:
                    max_tool_lvl = max(tools[:, IDX_ITEM_LEVEL])
                    tools = tools[tools[:, IDX_ITEM_LEVEL] == max_tool_lvl]  # those with highest level
                    min_id = min(tools[:, IDX_ITEM_ID])  # always choose the one with the minimal item id as the best
                    idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()
                    self.best_tools[i] = item_obs[idx]
                    if not item_obs[idx][IDX_ITEM_EQUIPPED] and self.force_use_idx[i] is None:
                        self.force_use_idx[i] = idx  # save for later translation

            # directly sell ammo
            items = np.concatenate([
                item_obs[item_obs[:, IDX_ITEM_INDEX] == ammo_type]
                for ammo_type in AMMOS
            ], axis=0)
            if len(items) > 0:
                item_id = items[0][IDX_ITEM_ID]
                item_lvl = items[0][IDX_ITEM_LEVEL]
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == item_id).item()
                self.force_sell_idx[i] = idx
                self.force_sell_price[i] = max(1, int(item_lvl) - 1)
                continue

            # directly sell weapons not belong to my profession
            other_weapon_types = [w for w in WEAPONS
                                  if w != ATK_TO_WEAPON[self.prof[i]]]
            items = np.concatenate([
                item_obs[item_obs[:, IDX_ITEM_INDEX] == weapon_type]
                for weapon_type in other_weapon_types
            ], axis=0)
            if len(items) > 0:
                item_id = items[0][IDX_ITEM_ID]
                item_lvl = items[0][IDX_ITEM_LEVEL]
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == item_id).item()
                self.force_sell_idx[i] = idx
                self.force_sell_price[i] = min(99, np.random.randint(10) + int(item_lvl) * 25 - 10)
                continue

            # sell tools
            items = np.concatenate([
                item_obs[item_obs[:, IDX_ITEM_INDEX] == tool_type]
                for tool_type in TOOLS
            ], axis=0)
            if self.best_weapons[i] is None and self.best_tools[i] is not None:
                best_idx = self.best_tools[i][IDX_ITEM_ID]
                items = items[items[:, IDX_ITEM_ID] != best_idx]  # filter out the best
            if len(items) > 0:
                to_sell = sorted(items, key=lambda x: x[IDX_ITEM_LEVEL])[0]  # sell the worst first
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == to_sell[IDX_ITEM_ID]).item()
                lvl = to_sell[IDX_ITEM_LEVEL]
                self.force_sell_idx[i] = idx
                self.force_sell_price[i] = int(lvl) * 6 + np.random.randint(3)
                continue

            # sell armors and weapons of my profession
            sell_not_best_types = [
                ITEM_HAT,
                ITEM_TOP,
                ITEM_BOTTOM,
                ATK_TO_WEAPON[self.prof[i]],
            ]  # reflect priority
            best_savers = [
                self.best_hats,
                self.best_tops,
                self.best_bottoms,
                self.best_weapons,
            ]
            for item_type, saver in zip(sell_not_best_types, best_savers):
                items = item_obs[item_obs[:, IDX_ITEM_INDEX] == item_type]  # those with the target type
                if saver[i] is not None:
                    items = items[items[:, IDX_ITEM_ID] != saver[i][IDX_ITEM_ID]]  # filter out the best
                    best_lvl = saver[i][IDX_ITEM_LEVEL]
                    if best_lvl < 6:
                        # reserve items no more than level 6 for future use
                        reserves = items[items[:, IDX_ITEM_LEVEL] > best_lvl]
                        reserves = reserves[reserves[:, IDX_ITEM_LEVEL] <= 6]
                        if len(reserves) > 0:
                            reserve = sorted(reserves, key=lambda x: x[IDX_ITEM_LEVEL])[-1]  # the best one to reserve
                            items = items[items[:, IDX_ITEM_ID] != reserve[IDX_ITEM_ID]]  # filter out the reserved
                if len(items) > 0:
                    to_sell = sorted(items, key=lambda x: x[IDX_ITEM_LEVEL])[0]  # sell the worst first
                    idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == to_sell[IDX_ITEM_ID]).item()
                    lvl = to_sell[IDX_ITEM_LEVEL]
                    self.force_sell_idx[i] = idx
                    if item_type in WEAPONS:
                        self.force_sell_price[i] = min(99, np.random.randint(10) + int(lvl) * 25 - 10)
                    else:  # ARMORS
                        self.force_sell_price[i] = 3 + np.random.randint(3) + int(lvl - 1) * 4
                    break

    def _process_market(self, obs):
        # reset
        self.force_buy_idx = [None] * N_PLAYER_PER_TEAM

        alive_ids = np.array([i for i in range(N_PLAYER_PER_TEAM) if i in obs])
        n = obs[alive_ids[0]]['Market']['N'][0]
        raw_market_obs = obs[alive_ids[0]]['Market']['Continuous'][:n]
        market_obs = copy.deepcopy(raw_market_obs)  # will be modified later

        # combat rating
        ratings = []
        for i in alive_ids:
            rating = 0
            if self.best_weapons[i] is not None:
                rating += self.best_weapons[i][IDX_ITEM_LEVEL] * 10
            elif self.best_tools[i] is not None:
                rating += self.best_tools[i][IDX_ITEM_LEVEL] * 4
            if self.best_hats[i] is not None:
                rating += self.best_hats[i][IDX_ITEM_LEVEL] * 4
            if self.best_tops[i] is not None:
                rating += self.best_tops[i][IDX_ITEM_LEVEL] * 4
            if self.best_bottoms[i] is not None:
                rating += self.best_bottoms[i][IDX_ITEM_LEVEL] * 4
            ratings.append(rating)
        ratings = np.array(ratings)

        # buy weapons & armors
        agent_order = np.argsort(ratings)
        alive_ids = alive_ids[agent_order]  # reorder, low rating buy first
        for i in alive_ids:
            my_obs = obs[i]['Entity']['Continuous'][0]
            n_item = obs[i]['Item']['N'][0]
            my_items = obs[i]['Item']['Continuous'][:n_item]
            try:
                my_gold = my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_GOLD][0][IDX_ITEM_QUANTITY]
            except IndexError:
                my_gold = 0

            care_types = [ATK_TO_WEAPON[self.prof[i]], *ARMORS]
            savers = [
                self.best_weapons,
                self.best_hats,
                self.best_tops,
                self.best_bottoms,
            ]

            wishlist = []
            enhancements = []
            for typ, saver in zip(care_types, savers):
                if typ in ARMORS:
                    max_equipable_lvl = max(my_obs[-N_PROF:])  # maximum of all levels
                else:
                    max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[typ]]
                curr_best_lvl = 0
                if saver[i] is not None:
                    curr_best_lvl = saver[i][IDX_ITEM_LEVEL]
                mkt_comds = market_obs[market_obs[:, IDX_ITEM_INDEX] == typ]
                mkt_comds = mkt_comds[mkt_comds[:, IDX_ITEM_PRICE] <= my_gold]
                mkt_comds = mkt_comds[mkt_comds[:, IDX_ITEM_LEVEL] <= max_equipable_lvl]
                mkt_comds = mkt_comds[mkt_comds[:, IDX_ITEM_LEVEL] > curr_best_lvl]
                if len(mkt_comds) > 0:
                    best_comd = sorted(mkt_comds, key=lambda x: x[IDX_ITEM_LEVEL])[-1]
                    wishlist.append(best_comd)
                    best_lvl = best_comd[IDX_ITEM_LEVEL]
                    delta_per_lvl = 4 if typ not in WEAPONS else 10
                    delta = delta_per_lvl * (best_lvl - curr_best_lvl)
                    enhancements.append(delta)
                else:
                    wishlist.append(None)
                    enhancements.append(0)

            if max(enhancements) > 0:
                to_buy = wishlist[enhancements.index(max(enhancements))]
                self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                idx = np.argwhere(market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                market_obs[idx][IDX_ITEM_QUANTITY] -= 1
                if market_obs[idx][IDX_ITEM_QUANTITY] == 0:
                    # remove from market obs to prevent competition among teammates
                    market_obs = np.concatenate([market_obs[:idx], market_obs[idx+1:]])

        # survival rating
        alive_ids = np.array([i for i in range(N_PLAYER_PER_TEAM) if i in obs])
        ratings = []
        for i in alive_ids:
            rating = 0
            n_item = obs[i]['Item']['N'][0]
            my_items = obs[i]['Item']['Continuous'][:n_item]
            n_poultice = len(my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_POULTICE])
            n_ration = len(my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_RATION])
            if n_poultice == 1:
                rating += 4
            elif n_poultice >= 2:
                rating += 8
            if n_ration >= 1:
                rating += 2
            ratings.append(rating)
        ratings = np.array(ratings)

        # emergent case to buy poultice
        will_die_id = None
        will_die_gold = None
        team_n_poultice = [None] * N_PLAYER_PER_TEAM
        for i in alive_ids:
            my_obs = obs[i]['Entity']['Continuous'][0]
            n_item = obs[i]['Item']['N'][0]
            my_items = obs[i]['Item']['Continuous'][:n_item]
            my_poultices = my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
            team_n_poultice[i] = len(my_poultices)
            if len(my_poultices) > 0:  # not emergent
                continue
            if my_obs[IDX_ENT_HEALTH] > 35:  # not emergent
                continue
            # if my_obs[IDX_ENT_FOOD] >= 50 and my_obs[IDX_ENT_WATER] >= 50:
            #     my_pop = my_obs[IDX_ENT_POPULATION]
            #     entity_obs = obs[i]['Entity']['Continuous']
            #     n_ent_observed = obs[i]['Entity']['N'][0]
            #     other_entities = entity_obs[1:n_ent_observed]
            #     other_enemies =
            try:
                my_gold = my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_GOLD][0][IDX_ITEM_QUANTITY]
            except IndexError:
                my_gold = 0
            mkt_ps = market_obs[market_obs[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
            mkt_ps = mkt_ps[mkt_ps[:, IDX_ITEM_PRICE] <= my_gold]
            if len(mkt_ps) > 0:
                to_buy = sorted(mkt_ps, key=lambda x: x[IDX_ITEM_PRICE])[0]  # cheapest
                self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                idx = np.argwhere(market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                market_obs[idx][IDX_ITEM_QUANTITY] -= 1
                if market_obs[idx][IDX_ITEM_QUANTITY] == 0:
                    # remove from market obs to prevent repeatedly buying among teammates
                    market_obs = np.concatenate([market_obs[:idx], market_obs[idx+1:]])
            else:
                will_die_id = i
                will_die_gold = my_gold

        # sell poultice for emergent rescue
        if will_die_id is not None and self.rescue_cooldown == 0 and will_die_gold > 0:
            for i in reversed(alive_ids):
                if team_n_poultice[i] > 0 and self.force_sell_idx[i] is not None:
                    n_item = obs[i]['Item']['N'][0]
                    my_items = obs[i]['Item']['Continuous'][:n_item]
                    my_poultices = my_items[my_items[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
                    team_pop = next(iter(obs.values()))['Entity']['Continuous'][0][IDX_ENT_POPULATION]
                    to_sell = sorted(my_poultices, key=lambda x: x[IDX_ITEM_LEVEL])[-1]  # sell the best
                    idx = np.argwhere(my_items[:, IDX_ITEM_ID] == to_sell[IDX_ITEM_ID]).item()
                    self.force_sell_idx[i] = idx
                    self.force_sell_price[i] = max(int(will_die_gold // 2), 1) if team_pop > 0 else 1
                    self.rescue_cooldown = 3
                    break

        # normal case to buy at least two poultices, at least one ration
        agent_order = np.argsort(ratings)
        alive_ids = alive_ids[agent_order]  # reorder, low rating buy first
        for cons_type in [ITEM_POULTICE, ITEM_RATION]:
            for i in alive_ids:
                if self.force_buy_idx[i] is not None:
                    continue
                n_item = obs[i]['Item']['N'][0]
                my_items = obs[i]['Item']['Continuous'][:n_item]
                if cons_type == ITEM_RATION and len(my_items[my_items[:, IDX_ITEM_INDEX] == cons_type]) >= 1:
                    continue
                if cons_type == ITEM_POULTICE and len(my_items[my_items[:, IDX_ITEM_INDEX] == cons_type]) >= 2:
                    continue
                mkt_cons = market_obs[market_obs[:, IDX_ITEM_INDEX] == cons_type]
                acceptable_price = 2 + self.curr_step // 300
                mkt_cons = mkt_cons[mkt_cons[:, IDX_ITEM_PRICE] <= acceptable_price]
                if len(mkt_cons) > 0:
                    to_buy = sorted(mkt_cons, key=lambda x: x[IDX_ITEM_PRICE])[0]  # cheapest
                    self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                    idx = np.argwhere(market_obs[:, IDX_ITEM_ID] == to_buy[IDX_ITEM_ID]).item()
                    market_obs[idx][IDX_ITEM_QUANTITY] -= 1
                    if market_obs[idx][IDX_ITEM_QUANTITY] == 0:
                        # remove from market obs to prevent repeatedly buying among teammates
                        market_obs = np.concatenate([market_obs[:idx], market_obs[idx + 1:]])

    def _extract_item_feature(self, obs):
        items_arrs = []
        items_types = []
        for i in range(N_PLAYER_PER_TEAM):
            # replace with dummy feature if dead
            if i not in obs:
                items_types.append(DUMMY_ITEM_TYPES)
                items_arrs.append(DUMMY_ITEM_ARRS)
                continue

            item_obs = obs[i]['Item']['Continuous']
            n = obs[i]['Item']['N'][0]
            item_arrs = []
            item_types = []
            for j in range(N_ITEM_SLOT):
                o = item_obs[j]
                item_types.append(o[IDX_ITEM_INDEX])  # type is 0 if j < n
                arr = self._extract_per_item_feature(o if j < n else None)
                item_arrs.append(arr)
            item_types = np.array(item_types)
            item_arrs = np.stack(item_arrs)

            items_types.append(item_types)
            items_arrs.append(item_arrs)
        items_types = np.stack(items_types)
        items_arrs = np.stack(items_arrs)

        return items_types, items_arrs

    @staticmethod
    def _extract_per_item_feature(o):
        if o is not None:
            arr = np.array([
                o[IDX_ITEM_LEVEL] / 10.,
                o[IDX_ITEM_QUANTITY] / 200. if o[IDX_ITEM_INDEX] == ITEM_GOLD else o[IDX_ITEM_QUANTITY] / 10.,
                o[IDX_ITEM_MELEE_ATK] / 100.,
                o[IDX_ITEM_RANGE_ATK] / 100.,
                o[IDX_ITEM_MAGE_ATK] / 100.,
                o[IDX_ITEM_MELEE_DEF] / 40.,
                o[IDX_ITEM_RANGE_DEF] / 40.,
                o[IDX_ITEM_MAGE_DEF] / 40.,
                o[IDX_ITEM_HP_RST] / 100.,
                o[IDX_ITEM_PRICE] / 100.,
                o[IDX_ITEM_EQUIPPED],
            ])
        else:
            arr = np.zeros(PER_ITEM_FEATURE)
        return arr

    def _extract_entity_feature(self, obs):
        team_pop = next(iter(obs.values()))['Entity']['Continuous'][0][IDX_ENT_POPULATION]
        team_members_idx = np.arange(8) + team_pop * N_PLAYER_PER_TEAM + 1

        # merge obs from all the 8 agents
        team_members = {}  # 0~7 -> raw_arr
        enemies = {}  # entity_id -> raw_arr
        npcs = {}  # entity_id -> raw_arr
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:
                continue
            entity_obs = obs[i]['Entity']['Continuous']
            n = obs[i]['Entity']['N'][0]
            team_members[i] = entity_obs[0]
            for j in range(1, n):
                if entity_obs[j][IDX_ENT_ENTITY_ID] < 0:
                    npcs[entity_obs[j][IDX_ENT_ENTITY_ID]] = entity_obs[j]
                elif entity_obs[j][IDX_ENT_ENTITY_ID] not in team_members_idx:
                    enemies[entity_obs[j][IDX_ENT_ENTITY_ID]] = entity_obs[j]

        # extract feature of each team member itself
        team_members_arr = np.zeros((N_PLAYER_PER_TEAM, N_SELF_FEATURE))
        team_mask = np.array([i not in obs for i in range(N_PLAYER_PER_TEAM)])
        for i in range(N_PLAYER_PER_TEAM):
            team_members_arr[i] = self._extract_per_entity_feature(team_members.get(i, None), team_pop, i)

        # assign the features of npcs and enemies to each member
        others_arrs = [np.zeros((N_PLAYER_PER_TEAM, n, PER_ENTITY_FEATURE))
                       for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
        entity_mask = [np.ones((N_PLAYER_PER_TEAM, n))
                       for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
        ids_as_target = [np.zeros((N_PLAYER_PER_TEAM, n))
                         for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
        for k in range(2):
            n_considered = (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)[k]
            entities = (npcs, enemies)[k]
            # first extract all the features along with entity's idx & position
            features = [{
                'idx': idx,
                'row': raw_arr[IDX_ENT_ROW_INDEX],
                'col': raw_arr[IDX_ENT_COL_INDEX],
                'pop': raw_arr[IDX_ENT_POPULATION],
                'arr': self._extract_per_entity_feature(raw_arr, team_pop),
            } for idx, raw_arr in entities.items()]

            for i in range(N_PLAYER_PER_TEAM):
                if i not in team_members:  # dead
                    continue
                my_row = team_members[i][IDX_ENT_ROW_INDEX]
                my_col = team_members[i][IDX_ENT_COL_INDEX]

                def l1_to_me(f):
                    return max(abs(f['row'] - my_row), abs(f['col'] - my_col))

                nearests = sorted(features, key=l1_to_me)[:n_considered]
                for j, feat in enumerate(nearests):
                    if l1_to_me(feat) <= ATK_RANGE and feat['pop'] != NEUTRAL_POP:  # as target
                        ids_as_target[k][i][j] = feat['idx']
                    if l1_to_me(feat) <= AWARE_RANGE:  # as visible entity
                        others_arrs[k][i][j] = feat['arr']
                        entity_mask[k][i][j] = 0

        npcs_arrs, enemies_arrs = others_arrs
        target_npcs_ids, target_enemies_ids = ids_as_target
        return team_members_arr, npcs_arrs, enemies_arrs, team_mask, entity_mask[0], entity_mask[1], target_npcs_ids, target_enemies_ids

    def _extract_per_entity_feature(self, o, team_pop=None, my_index=None):
        if o is not None:
            arr = np.array([
                1.,  # alive mark
                o[IDX_ENT_ENTITY_ID] in self.target_entity_id,  # attacked by my team
                o[IDX_ENT_ATTACKER_ID] < 0,  # attacked by npc
                o[IDX_ENT_ATTACKER_ID] > 0,  # attacked by player
                o[IDX_ENT_LVL] / 10.,
                o[IDX_ENT_ITEM_LVL] / 20.,
                (o[IDX_ENT_ROW_INDEX] - HALF_TERRAIN_SIZE) / HALF_TERRAIN_SIZE,
                (o[IDX_ENT_COL_INDEX] - HALF_TERRAIN_SIZE) / HALF_TERRAIN_SIZE,
                o[IDX_ENT_TIME_ALIVE] / MAX_STEP,
                (o[IDX_ENT_ROW_INDEX] - MAP_LEFT) / MAP_SIZE,
                (o[IDX_ENT_COL_INDEX] - MAP_LEFT) / MAP_SIZE,
                o[IDX_ENT_POPULATION] >= 0,  # player
                o[IDX_ENT_POPULATION] == team_pop,  # is teammate
                o[IDX_ENT_POPULATION] == -1,  # passive npc
                o[IDX_ENT_POPULATION] == -2,  # neutral npc
                o[IDX_ENT_POPULATION] == -3,  # hostile npc
                o[IDX_ENT_DAMAGE] / 10.,
                o[IDX_ENT_TIME_ALIVE] / MAX_STEP,
                o[IDX_ENT_GOLD] / 100.,
                o[IDX_ENT_HEALTH] / 100.,
                o[IDX_ENT_FOOD] / 100.,
                o[IDX_ENT_WATER] / 100.,
                o[IDX_ENT_MELEE_LVL] / 10.,
                o[IDX_ENT_RANGE_LVL] / 10.,
                o[IDX_ENT_MAGE_LVL] / 10.,
                o[IDX_ENT_FISHING_LVL] / 10.,
                o[IDX_ENT_HERBALISM_LVL] / 10.,
                o[IDX_ENT_PROSPECTING_LVL] / 10.,
                o[IDX_ENT_CARVING_LVL] / 10.,
                o[IDX_ENT_ALCHEMY_LVL] / 10.,
            ])
        else:
            arr = np.zeros(PER_ENTITY_FEATURE)

        if my_index is not None:
            population_arr = one_hot_generator(N_TEAM, int(team_pop))
            index_arr = one_hot_generator(N_PLAYER_PER_TEAM, my_index)
            prof_idx = ATK_TYPE.index(self.prof[my_index])
            prof_arr = one_hot_generator(N_ATK_TYPE, prof_idx)
            if o is not None:
                row = o[IDX_ENT_ROW_INDEX].astype(int)
                col = o[IDX_ENT_COL_INDEX].astype(int)
                near_tile_map = self.tile_map[row-4:row+5, col-4:col+5]
                food_arr = []
                water_arr = []
                herb_arr = []
                fish_arr = []
                obstacle_arr = []
                for i in range(9):
                    for j in range(9):
                        if abs(i-4) + abs(j-4) <= 4:
                            food_arr.append(near_tile_map[i, j] == TILE_FOREST)
                            water_arr.append(near_tile_map[i, j] == TILE_WATER)
                            herb_arr.append(near_tile_map[i, j] == TILE_HERB)
                            fish_arr.append(near_tile_map[i, j] == TILE_FISH)
                            obstacle_arr.append(near_tile_map[i, j] in OBSTACLE_TILES)
                food_arr[-1] = max(0, self.poison_map[row, col]) / 20.  # patch after getting trained
                water_arr[-1] = max(0, self.poison_map[row+1, col]) / 20.  # patch after getting trained
                herb_arr[-1] = max(0, self.poison_map[row, col+1]) / 20.  # patch after getting trained
                fish_arr[-1] = max(0, self.poison_map[row-1, col]) / 20.  # patch after getting trained
                obstacle_arr[-1] = max(0, self.poison_map[row, col-1]) / 20.  # patch after getting trained
            else:
                food_arr = water_arr = herb_arr = fish_arr = obstacle_arr = np.zeros(41)
            arr = np.concatenate([
                arr, population_arr, index_arr, prof_arr,
                food_arr, water_arr, herb_arr, fish_arr, obstacle_arr,
            ])
        return arr

    def _extract_game_feature(self, obs):
        game_progress = self.curr_step / MAX_STEP
        n_alive = sum([i in obs for i in range(N_PLAYER_PER_TEAM)])
        arr = np.array([
            game_progress,
            n_alive / N_PLAYER_PER_TEAM,
            *multi_hot_generator(n_feature=16, index=int(game_progress*16)+1),
            *multi_hot_generator(n_feature=N_PLAYER_PER_TEAM, index=n_alive),
        ])
        return arr

    def _extract_legal_action(self, obs, npc_target, enemy_target):
        # --- move ---
        team_pos = np.zeros((N_PLAYER_PER_TEAM, 2), dtype=int)
        team_food = np.ones(N_PLAYER_PER_TEAM) * 100
        team_stuck = [False] * N_PLAYER_PER_TEAM

        # first filter out obstacles
        legal_move = np.zeros((N_PLAYER_PER_TEAM, N_MOVE + 1))
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:
                legal_move[i][-1] = 1
                continue
            tiles_type = obs[i]['Tile']['Continuous'][:, 1].reshape((WINDOW, WINDOW))
            n_ent_observed = obs[i]['Entity']['N'][0]
            entity_pos = obs[i]['Entity']['Continuous'][1:n_ent_observed, 7:9].astype(int).tolist()
            center = np.array([WINDOW_CENTER, WINDOW_CENTER])
            for j, e in enumerate(nmmo_act.Direction.edges):
                next_pos = center + e.delta
                if tiles_type[next_pos[0]][next_pos[1]] in PASSABLE_TILES:
                    if next_pos.tolist() not in entity_pos:
                        legal_move[i][j] = 1
                    else:
                        ent_on_next_pos_can_move = False
                        for ee in nmmo_act.Direction.edges:  # a rough secondary judgement
                            next_next_pos = next_pos + ee.delta
                            if tiles_type[next_next_pos[0]][next_next_pos[1]] in PASSABLE_TILES:
                                if next_next_pos.tolist() not in entity_pos:
                                    ent_on_next_pos_can_move = True
                                    break
                        if ent_on_next_pos_can_move:
                            legal_move[i][j] = 1
            # save something for later use, and detect whether it is stuck
            my_obs = obs[i]['Entity']['Continuous'][0]
            my_pos = my_obs[7:9].astype(int)
            team_pos[i] = my_pos
            team_food[i] = my_obs[IDX_ENT_FOOD]
            stuck = []
            for e in nmmo_act.Direction.edges:
                d = np.array(e.delta).astype(int)
                near_pos = my_pos + d
                tile_type = self.tile_map[near_pos[0], near_pos[1]]
                entity_type = self.entity_map[near_pos[0], near_pos[1]]
                st = tile_type in OBSTACLE_TILES or entity_type == TEAMMATE_REPR
                stuck.append(st)
            if sum(stuck) == 4:
                team_stuck[i] = True

        # then prevent blocking out from teammates
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:
                continue
            for j, e in enumerate(nmmo_act.Direction.edges):  # [North, South, East, West]
                d = np.array(e.delta).astype(int)
                near_pos = (team_pos[i] + d).tolist()
                if near_pos in team_pos.tolist():
                    teammate_idx = team_pos.tolist().index(near_pos)
                    counter_dir = [1, 0, 3, 2][j]
                    if team_stuck[i]:
                        legal_move[teammate_idx][counter_dir] = 0
                    else:
                        my_food = team_food[i]
                        teammate_food = team_food[teammate_idx]
                        if my_food < teammate_food:
                            legal_move[teammate_idx][counter_dir] = 0

        for i in range(N_PLAYER_PER_TEAM):
            if sum(legal_move[i][:N_MOVE]) == 0:
                legal_move[i][-1] = 1

        # --- attack ---
        target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)  # first npc, then enemy
        no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
        legal_target = np.concatenate([target_attackable, no_target], axis=-1)

        # --- use & sell ---
        legal_use = np.zeros((N_PLAYER_PER_TEAM, N_USE + 1))
        legal_sell = np.zeros((N_PLAYER_PER_TEAM, N_SELL + 1))
        legal_use[:, -1] = 1
        legal_sell[:, -1] = 1
        for i in range(N_PLAYER_PER_TEAM):
            if i not in obs:
                continue
            n = obs[i]['Item']['N'][0]
            item_obs = obs[i]['Item']['Continuous'][:n]
            my_obs = obs[i]['Entity']['Continuous'][0]

            if self.force_use_idx[i] is None:
                poultices = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
                if my_obs[IDX_ENT_HEALTH] <= 60 and len(poultices) > 0:
                    legal_use[i][USE_POULTICE] = 1
                rations = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_RATION]
                if (my_obs[IDX_ENT_FOOD] < 50 or my_obs[IDX_ENT_WATER] < 50) and len(rations) > 0:
                    legal_use[i][USE_RATION] = 1

            if n > N_ITEM_LIMIT and self.force_sell_idx[i] is None:
                poultices = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
                rations = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_RATION]
                if len(poultices) > 1:
                    legal_sell[i][SELL_POULTICE] = 1
                    legal_sell[i][-1] = 0
                if len(rations) > 1:
                    legal_sell[i][SELL_RATION] = 1
                    legal_sell[i][-1] = 0

        legal = {
            'move': legal_move,
            'target': legal_target,
            'use': legal_use,
            'sell': legal_sell,
        }
        return legal

    def trans_action(self, actions):
        actions = np.array(actions)
        self.prev_actions = actions.T.copy()

        raw_actions = collections.defaultdict(dict)
        for i in range(N_PLAYER_PER_TEAM):
            move, target, use, sell = actions[:, i]
            self._trans_move(i, raw_actions, move)
            self._trans_attack(i, raw_actions, target)
            self._trans_use(i, raw_actions, use)
            self._trans_sell(i, raw_actions, sell)
            self._trans_buy(i, raw_actions)

        self.curr_step += 1
        self.prev_obs = self.curr_obs
        return raw_actions

    @staticmethod
    def _trans_move(i, raw_actions, move):
        if move != N_MOVE:  # is not idle
            raw_actions[i][nmmo_act.Move] = {nmmo_act.Direction: move}

    def _trans_attack(self, i, raw_actions, target):
        if target != N_ATK_TARGET:  # exist some target to attack
            if target < N_NPC_CONSIDERED:
                self.target_entity_id[i] = int(self.npc_tgt[i][target])
                self.target_entity_pop[i] = 1
            else:
                self.target_entity_id[i] = int(self.enemy_tgt[i][target - N_NPC_CONSIDERED])
            # change the id from entity_id to index in obs
            entity_obs = self.curr_obs[i]['Entity']['Continuous']
            target_row_id = np.argwhere(
                entity_obs[:, IDX_ENT_ENTITY_ID] == self.target_entity_id[i]).item()
            self.target_entity_pop[i] = entity_obs[target_row_id, IDX_ENT_POPULATION]
            atk_type = PROF_TO_ATK_TYPE[self.prof[i]]
            raw_actions[i][nmmo_act.Attack] = {
                nmmo_act.Style: atk_type,
                nmmo_act.Target: target_row_id,
            }
        else:
            self.target_entity_id[i] = None
            self.target_entity_pop[i] = None

    def _trans_use(self, i, raw_actions, use):
        if i not in self.curr_obs:  # dead
            return

        if self.force_use_idx[i] is not None:
            raw_actions[i][nmmo_act.Use] = {nmmo_act.Item: self.force_use_idx[i]}
        elif use != N_USE:
            n = self.curr_obs[i]['Item']['N'][0]
            item_obs = self.curr_obs[i]['Item']['Continuous'][:n]
            if use == USE_POULTICE:
                poultices = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
                min_lvl = min(poultices[:, IDX_ITEM_LEVEL])
                poultices = poultices[poultices[:, IDX_ITEM_LEVEL] == min_lvl]  # those with lowest level
                min_id = min(poultices[:, IDX_ITEM_ID])
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()
            else:  # USE_RATION
                rations = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_RATION]
                min_lvl = min(rations[:, IDX_ITEM_LEVEL])
                rations = rations[rations[:, IDX_ITEM_LEVEL] == min_lvl]  # those with lowest level
                min_id = min(rations[:, IDX_ITEM_ID])
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()

            raw_actions[i][nmmo_act.Use] = {nmmo_act.Item: idx}

    def _trans_sell(self, i, raw_actions, sell):
        if i not in self.curr_obs:  # dead
            return

        if self.force_sell_idx[i] is not None:
            raw_actions[i][nmmo_act.Sell] = {
                nmmo_act.Item: self.force_sell_idx[i],
                nmmo_act.Price: int(self.force_sell_price[i]),
            }
        elif sell != N_SELL:
            n = self.curr_obs[i]['Item']['N'][0]
            item_obs = self.curr_obs[i]['Item']['Continuous'][:n]
            if sell == SELL_POULTICE:
                poultices = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_POULTICE]
                min_lvl = min(poultices[:, IDX_ITEM_LEVEL])
                poultices = poultices[poultices[:, IDX_ITEM_LEVEL] == min_lvl]  # those with lowest level
                min_id = min(poultices[:, IDX_ITEM_ID])
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()
            else:  # SELL_RATION
                rations = item_obs[item_obs[:, IDX_ITEM_INDEX] == ITEM_RATION]
                min_lvl = min(rations[:, IDX_ITEM_LEVEL])
                rations = rations[rations[:, IDX_ITEM_LEVEL] == min_lvl]  # those with lowest level
                min_id = min(rations[:, IDX_ITEM_ID])
                idx = np.argwhere(item_obs[:, IDX_ITEM_ID] == min_id).item()

            raw_actions[i][nmmo_act.Sell] = {
                nmmo_act.Item: idx,
                nmmo_act.Price: int(2 + self.curr_step // 300),
            }

    def _trans_buy(self, i, raw_actions):
        if i not in self.curr_obs:  # dead
            return

        if self.force_buy_idx[i] is not None:
            raw_actions[i][nmmo_act.Buy] = {
                nmmo_act.Item: self.force_buy_idx[i],
            }
