from pdb import set_trace as TT

from collections import defaultdict

import numpy as np
from numpy import ndarray
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.algorithms.callbacks import DefaultCallbacks

import nmmo
import pufferlib

from config.cleanrl import Train
from neural import policy, io, subnets

from typing import Dict


from typing import Dict, Tuple


class CompetitionConfig(nmmo.config.Medium, nmmo.config.AllGameSystems):
    TASKS = None
    MAP_N = 40
    PATH_MAPS = "maps"

    COMBAT_FRIENDLY_FIRE = False
    PLAYER_N = 128
    PLAYER_LOADER = nmmo.spawn.TeamLoader
    PLAYERS = 16 * [nmmo.Agent]

    PLAYER_DEATH_FOG = 240
    PLAYER_DEATH_FOG_FINAL_SIZE = 15
    PLAYER_DEATH_FOG_SPEED = 1 / 16

    SPECIALIZE = True

    # Progession
    PROGRESSION_MELEE_BASE_DAMAGE = 7
    PROGRESSION_RANGE_BASE_DAMAGE = 7
    PROGRESSION_MAGE_BASE_DAMAGE = 7

    # NPC
    NPC_BASE_DEFENSE = 0
    NPC_LEVEL_DEFENSE = 15
    NPC_BASE_DAMAGE = 10
    NPC_LEVEL_DAMAGE = 15

    # Equipment
    EQUIPMENT_WEAPON_BASE_DAMAGE = 0
    EQUIPMENT_WEAPON_LEVEL_DAMAGE = 10
    EQUIPMENT_AMMUNITION_BASE_DAMAGE = 0
    EQUIPMENT_AMMUNITION_LEVEL_DAMAGE = 10
    EQUIPMENT_TOOL_BASE_DEFENSE = 0
    EQUIPMENT_TOOL_LEVEL_DEFENSE = 4
    EQUIPMENT_ARMOR_BASE_DEFENSE = 0
    EQUIPMENT_ARMOR_LEVEL_DEFENSE = 4

    # Terrain
    TERRAIN_WATER = 0.29

    @property
    def PLAYER_SPAWN_FUNCTION(self):
        return nmmo.spawn.spawn_concurrent


class FeatureParser:
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5, 14, 15)  # lava, water, stone,
    spec = spaces.Dict({
        "terrain":
        spaces.Box(low=0, high=15, shape=(15, 15), dtype=np.int64),
        "reachable":
        spaces.Box(low=0, high=1, shape=(15, 15), dtype=np.float32),
        "death_fog_damage":
        spaces.Box(low=0, high=1, shape=(15, 15), dtype=np.float32),
        "entity_population":
        spaces.Box(low=0, high=5, shape=(15, 15), dtype=np.int64),
        "self_entity":
        spaces.Box(low=0, high=1, shape=(1, 26), dtype=np.float32),
        "other_entity":
        spaces.Box(low=0, high=1, shape=(15, 26), dtype=np.float32),
        "va_move":
        spaces.Box(low=0, high=1, shape=(5, ), dtype=np.float32),
        "va_attack_target":
        spaces.Box(low=0, high=1, shape=(16, ), dtype=np.float32),
    })

    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        observations: Dict[int, Dict[str, ndarray]],
        step: int,
    ) -> Dict[str, ndarray]:
        ret = {}
        for agent_id in observations:
            terrain, death_fog_damage, population, reachable, va_move = self.parse_local_map(
                observations[agent_id], step)
            entity, va_target = self.parse_entity(observations[agent_id])
            self_entity = entity[:1, :]
            other_entity = entity[1:, :]
            ret[agent_id] = {
                "terrain": terrain,
                "death_fog_damage": death_fog_damage,
                "reachable": reachable,
                "entity_population": population,
                "self_entity": self_entity,
                "other_entity": other_entity,
                "va_move": va_move,
                "va_attack_target": va_target,
            }
        return ret

    def parse_local_map(
        self,
        observation: Dict[str, ndarray],
        step: int,
    ) -> Tuple[ndarray, ndarray]:
        tiles = observation["Tile"]["Continuous"]
        entities = observation["Entity"]["Continuous"]
        terrain = np.zeros(shape=self.spec["terrain"].shape,
                           dtype=self.spec["terrain"].dtype)
        death_fog_damage = np.zeros(shape=self.spec["death_fog_damage"].shape,
                                    dtype=self.spec["death_fog_damage"].dtype)
        population = np.zeros(shape=self.spec["entity_population"].shape,
                              dtype=self.spec["entity_population"].dtype)
        va = np.ones(shape=self.spec["va_move"].shape,
                     dtype=self.spec["va_move"].dtype)

        # terrain, death_fog
        R, C = tiles[0, 2:4]
        for tile in tiles:
            absolute_r, absolute_c = tile[2:4]
            relative_r, relative_c = int(absolute_r - R), int(absolute_c - C)
            terrain[relative_r, relative_c] = int(tile[1])
            dmg = self.compute_death_fog_damage(absolute_r, absolute_c, step)
            death_fog_damage[relative_r, relative_c] = dmg / 100.0

        # entity population map
        P = entities[0, 6]
        for e in entities:
            if e[0] == 0: break
            absolute_r, absolute_c = e[7:9]
            relative_r, relative_c = int(absolute_r - R), int(absolute_c - C)
            if e[6] == P:
                p = 1
            elif e[6] >= 0:
                p = 2
            elif e[6] < 0:
                p = abs(e[6]) + 2
            population[relative_r, relative_c] = p

        # reachable area
        reachable = self.gen_reachable_map(terrain)

        # valid move
        for i, (r, c) in enumerate(self.NEIGHBOR):
            if terrain[r, c] in self.OBSTACLE:
                va[i + 1] = 0

        return terrain, death_fog_damage, population, reachable, va

    def parse_entity(
        self,
        observation: Dict[str, ndarray],
        max_size: int = 16,
    ) -> Tuple[ndarray, ndarray]:
        cent = CompetitionConfig.MAP_CENTER // 2
        entities = observation["Entity"]["Continuous"]
        va = np.zeros(shape=self.spec["va_attack_target"].shape,
                      dtype=self.spec["va_attack_target"].dtype)
        va[0] = 1.0

        entities_list = []
        P, R, C = entities[0, 6:9]
        for i, e in enumerate(entities[:max_size]):
            if e[0] == 0: break
            # attack range
            p, r, c = e[6:9]
            if p != P and abs(R - r) <= 3 and abs(C - c) <= 3:
                va[i] = 1
            # population
            population = [0 for _ in range(5)]
            if p == P:
                population[0] = 1
            elif p >= 0:
                population[1] = 1
            elif p < 0:
                population[int(abs(p)) + 1] = 1
            entities_list.append(
                np.array(
                    [
                        float(e[2] == 0),  # attacked
                        e[3] / 10.0,  # level
                        e[4] / 10.0,  # item_level
                        (r - 16) / 128.0,  # r
                        (c - 16) / 128.0,  # c
                        (r - 16 - cent) / 128.0,  # delta_r
                        (c - 16 - cent) / 128.0,  # delta_c
                        e[9] / 100.0,  # damage
                        e[10] / 1024.0,  # alive_time
                        e[12] / 100.0,  # gold
                        e[13] / 100.0,  # health
                        e[14] / 100.0,  # food
                        e[15] / 100.0,  # water
                        e[16] / 10.0,  # melee
                        e[17] / 10.0,  # range
                        e[18] / 10.0,  # mage
                        e[19] / 10.0,  # fishing
                        e[20] / 10.0,  # herbalism
                        e[21] / 10.0,  # prospecting
                        e[22] / 10.0,  # carving
                        e[23] / 10.0,  # alchmy
                        *population,
                    ],
                    dtype=np.float32))
        if len(entities_list) < max_size:
            entities_list.extend([
                np.zeros(26)
                for _ in range(max_size - len(entities_list))
            ])
        return np.asarray(entities_list), va

    @staticmethod
    def compute_death_fog_damage(r: int, c: int, step: int) -> float:
        C = CompetitionConfig
        if step < C.PLAYER_DEATH_FOG:
            return 0
        r, c = r - 16, c - 16
        cent = C.MAP_CENTER // 2
        # Distance from center of the map
        dist = max(abs(r - cent), abs(c - cent))
        if dist > C.PLAYER_DEATH_FOG_FINAL_SIZE:
            time_dmg = C.PLAYER_DEATH_FOG_SPEED * (step - C.PLAYER_DEATH_FOG +
                                                   1)
            dist_dmg = dist - cent
            dmg = max(0, dist_dmg + time_dmg)
        else:
            dmg = 0
        return dmg

    def gen_reachable_map(self, terrain: ndarray) -> ndarray:
        """
        grid: M * N
            1: passable
            0: unpassable
        """
        from collections import deque
        M, N = terrain.shape
        passable = ~np.isin(terrain, self.OBSTACLE)
        reachable = np.zeros_like(passable)
        visited = np.zeros_like(passable)
        q = deque()
        start = M // 2, N // 2
        q.append(start)
        visited[start[0], start[1]] = 1
        while q:
            cur_r, cur_c = q.popleft()
            reachable[cur_r, cur_c] = 1
            for (dr, dc) in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
                r, c = cur_r + dr, cur_c + dc
                if not (0 <= r < M and 0 <= c < N):
                    continue
                if not visited[r, c] and passable[r, c]:
                    q.append((r, c))
                visited[r, c] = 1
        return reachable


class ActionHead(nn.Module):
    name2dim = {"move": 5, "attack_target": 16}

    def __init__(self, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(input_dim, output_dim)
            for name, output_dim in self.name2dim.items()
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = {name: self.heads[name](x) for name in self.name2dim}
        return out


def make_policy(config, observation_space):
    class NMMONet(TorchModelV2, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nn.Module.__init__(self)
            self.config = config

            self.local_map_cnn = nn.Sequential(
                nn.Conv2d(24, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
            )
            self.local_map_fc = nn.Linear(32 * 4 * 4, 64)

            self.self_entity_fc1 = nn.Linear(26, 32)
            self.self_entity_fc2 = nn.Linear(32, 32)

            self.other_entity_fc1 = nn.Linear(26, 32)
            self.other_entity_fc2 = nn.Linear(15 * 32, 32)

            self.fc = nn.Linear(64 + 32 + 32, 64)
            self.action_head = ActionHead(64)
            self.value_head = nn.Linear(64, 1)

        def value_function(self):
            return self.value.view(-1)

        def local_map_embedding(self, input_dict):
            terrain = input_dict["terrain"].long().unsqueeze(1)
            death_fog_damage = input_dict["death_fog_damage"].unsqueeze(1)
            reachable = input_dict["reachable"].unsqueeze(1)
            population = input_dict["entity_population"].long().unsqueeze(1)

            T, B, *_ = terrain.shape

            terrain = F.one_hot(terrain, num_classes=16).permute(0, 1, 4, 2, 3)
            population = F.one_hot(population,
                                num_classes=6).permute(0, 1, 4, 2, 3)
            death_fog_damage = death_fog_damage.unsqueeze(dim=2)
            reachable = reachable.unsqueeze(dim=2)
            local_map = torch.cat(
                [terrain, reachable, population, death_fog_damage], dim=2)

            local_map = torch.flatten(local_map, 0, 1).to(torch.float32)
            local_map_emb = self.local_map_cnn(local_map)
            local_map_emb = local_map_emb.view(T * B, -1).view(T, B, -1)
            local_map_emb = F.relu(self.local_map_fc(local_map_emb))

            return local_map_emb

        def entity_embedding(self, input_dict):
            self_entity = input_dict["self_entity"].unsqueeze(1)
            other_entity = input_dict["other_entity"].unsqueeze(1)

            T, B, *_ = self_entity.shape

            self_entity_emb = F.relu(self.self_entity_fc1(self_entity))
            self_entity_emb = self_entity_emb.view(T, B, -1)
            self_entity_emb = F.relu(self.self_entity_fc2(self_entity_emb))

            other_entity_emb = F.relu(self.other_entity_fc1(other_entity))
            other_entity_emb = other_entity_emb.view(T, B, -1)
            other_entity_emb = F.relu(self.other_entity_fc2(other_entity_emb))

            return self_entity_emb, other_entity_emb

        def forward(self, input_dict, state, seq_lens):
            training = True

            #Is this the orig obs space or the modified one?
            input_dict = pufferlib.emulation.unpack_batched_obs(
                    observation_space, input_dict['obs'])
                    
            T, B, *_ = input_dict["terrain"].shape
            local_map_emb = self.local_map_embedding(input_dict)
            self_entity_emb, other_entity_emb = self.entity_embedding(input_dict)

            x = torch.cat([local_map_emb, self_entity_emb, other_entity_emb],
                        dim=-1)
            x = F.relu(self.fc(x))

            logits = self.action_head(x)
            self.value = self.value_head(x)#.view(T, B)

            output = []
            for key, val in logits.items():
                output.append(val.squeeze(1))
            output = torch.cat(output, 1)

            return output, state

    return NMMONet

def make_old_policy(config):
    class Policy(RecurrentNetwork, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nn.Module.__init__(self)
            self.config = config
            nmmo.io.action.Action.hook(config)

            self.input  = io.Input(config,
                    embeddings=io.MixedEmbedding,
                    attributes=subnets.SelfAttention)
            self.output = io.Output(config)
            self.value  = nn.Linear(config.HIDDEN, 1)
            self.policy = policy.Simple(config)
            self.lstm = pufferlib.torch.BatchFirstLSTM(config.HIDDEN, config.HIDDEN)

        def get_initial_state(self):
            return [self.value.weight.new(1, self.config.HIDDEN).zero_(),
                    self.value.weight.new(1, self.config.HIDDEN).zero_()]

        def forward_rnn(self, x, state, seq_lens):
            B, TT, _  = x.shape
            x         = x.reshape(B*TT, -1)

            x         = nmmo.emulation.unpack_obs(self.config, x)
            lookup    = self.input(x)
            hidden, _ = self.policy(lookup)

            hidden        = hidden.view(B, TT, self.config.HIDDEN)
            hidden, state = self.lstm(hidden, state)
            hidden        = hidden.reshape(B*TT, self.config.HIDDEN)

            self.val = self.value(hidden).squeeze(-1)
            logits   = self.output(hidden, lookup)

            flat_logits = []
            for atn in nmmo.Action.edges(self.config):
                for arg in atn.edges:
                    flat_logits.append(logits[atn][arg])

            flat_logits = torch.cat(flat_logits, 1)
            return flat_logits, state

        def value_function(self):
            return self.val.view(-1)

    return Policy

class NMMOLogger(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        assert len(base_env.envs) == 1, 'One env per worker'
        env = base_env.envs[0].par_env

        inv_map = {agent.policyID: agent for agent in env.config.PLAYERS}

        stats = env.terminal()
        stats = {**stats['Player'], **stats['Env']}
        policy_ids = stats.pop('PolicyID')

        for key, vals in stats.items():
            policy_stat = defaultdict(list)

            # Per-population metrics
            for policy_id, v in zip(policy_ids, vals):
                policy_stat[policy_id].append(v)

            for policy_id, vals in policy_stat.items():
                policy = inv_map[policy_id].__name__

                k = f'{policy}_{policy_id}_{key}'
                episode.custom_metrics[k] = np.mean(vals)

        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )


class CompetitionEnv(nmmo.Env):
    def action_space(self, agent):
        #Note: is alph sorted
        return spaces.Dict({
            'attack': spaces.Discrete(16),
            'move': spaces.Discrete(5),
        })

    def step(self, actions):
        for k, atns in actions.items():
            move = atns['move']
            target = atns['attack']

            actions[k] = {}

            if move != 4: #Don't move
                actions[k][nmmo.action.Move] = {
                        nmmo.action.Direction: move}
            actions[k][nmmo.action.Attack] = {
                nmmo.action.Style: 2, #Mage
                nmmo.action.Target: target}

        return super().step(actions)


# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)

config = CompetitionConfig()

env_cls = pufferlib.emulation.wrap(CompetitionEnv,
        feature_parser=FeatureParser(config))
env_creator = lambda: env_cls(config)

pufferlib.rllib.register_env('nmmo', env_cls)
test_env = env_creator()
observation_space = test_env.structured_observation_space(1)
obs = test_env.reset()

ModelCatalog.register_custom_model('custom', make_policy(config, observation_space)) 

trainer = RLTrainer(
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    algorithm="PPO",
    config={
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 32,
        "train_batch_size": 2**10,
        #"train_batch_size": 2**19,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 1,
        "framework": "torch",
        "env": "nmmo",
        "multiagent": {
            "count_steps_by": "agent_steps"
        },
        "model": {
            "custom_model": "custom",
            'custom_model_config': {'config': config},
            "max_seq_len": 16
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
    run_config=RunConfig(
        local_dir='results',
        verbose=1,
        stop={"training_iteration": 5},
        checkpoint_config=CheckpointConfig(
            num_to_keep=5,
            checkpoint_frequency=1,
        ),
        callbacks=[
            WandbLoggerCallback(
                project='NeuralMMO',
                api_key_file='wandb_api_key',
                log_config=False,
            )
        ]
    ),
    param_space={
        'callbacks': NMMOLogger,
    }
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

#policy = RLCheckpoint.from_checkpoint(result.checkpoint).get_policy()

'''
def multiagent_self_play(trainer: Type[Trainer]):
    new_weights = trainer.get_policy("player1").get_weights()
    for opp in Config.OPPONENT_POLICIES:
        prev_weights = trainer.get_policy(opp).get_weights()
        trainer.get_policy(opp).set_weights(new_weights)
        new_weights = prev_weights

local_weights = trainer.workers.local_worker().get_weights()
trainer.workers.foreach_worker(lambda worker: worker.set_weights(local_weights))
'''