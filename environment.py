import os
from argparse import Namespace
from collections import defaultdict

import pufferlib
import pufferlib.emulation

import nmmo
from nmmo.render.replay_helper import FileReplayHelper

from leader_board import get_team_result

# @daveey can we use nmmo.config.Default as a base here?
class Config(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Progression,
    nmmo.config.Profession,
    nmmo.config.Equipment,
    nmmo.config.Item,
    nmmo.config.Exchange,
    nmmo.config.Combat,
    nmmo.config.NPC,
):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.MAP_FORCE_GENERATION = False
        self.PLAYER_N = args.num_agents
        self.HORIZON = args.max_episode_length
        self.MAP_N = args.num_maps
        self.PLAYER_DEATH_FOG = args.death_fog_tick
        self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
        self.MAP_CENTER = args.map_size
        self.NPC_N = args.num_npcs
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size


class Postprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO."""

    def __init__(self, env, teams, team_id, replay_save_dir=None):
        super().__init__(env, teams, team_id)
        self._num_replays_saved = 0
        self._replay_save_dir = None
        if replay_save_dir is not None and self.team_id == 1:
            self._replay_save_dir = replay_save_dir
        self._replay_helper = None
        self._reset_episode_stats()

    def reset(self, team_obs, dummy=False):
        super().reset(team_obs)
        if not dummy:
            if self._replay_helper is None and self._replay_save_dir is not None:
                self._replay_helper = FileReplayHelper()
                self.env.realm.record_replay(self._replay_helper)
            if self._replay_helper is not None:
                self._replay_helper.reset()

        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._cod_attacked = 0
        self._cod_starved = 0
        self._cod_dehydrated = 0
        self._task_completed = 0
        self._curriculum = defaultdict(list)

        # for team results
        # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
        self._time_alive = 0
        self._gold_owned = 0
        self._damage_received = 0
        self._damage_inflicted = 0
        self._ration_consumed = 0
        self._potion_consumed = 0
        self._melee_level = 0
        self._range_level = 0
        self._mage_level = 0
        self._fishing_level = 0
        self._herbalism_level = 0
        self._prospecting_level = 0
        self._carving_level = 0
        self._alchemy_level = 0

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task._reward_count))
        if task.completed:
            self._task_completed += 1.0 / self.team_size

        if agent.damage.val > 0:
            self._cod_attacked += 1.0 / self.team_size
        elif agent.food.val == 0:
            self._cod_starved += 1.0 / self.team_size
        elif agent.water.val == 0:
            self._cod_dehydrated += 1.0 / self.team_size

        # For TeamResult
        self._time_alive += agent.history.time_alive.val
        self._gold_owned += agent.gold.val
        self._damage_received += agent.history.damage_received
        self._damage_inflicted += agent.history.damage_inflicted
        self._ration_consumed += agent.ration_consumed
        self._potion_consumed += agent.poultice_consumed
        self._melee_level += agent.melee_level.val
        self._range_level += agent.range_level.val
        self._mage_level += agent.mage_level.val
        self._fishing_level += agent.fishing_level.val
        self._herbalism_level += agent.herbalism_level.val
        self._prospecting_level += agent.prospecting_level.val
        self._carving_level += agent.carving_level.val
        self._alchemy_level += agent.alchemy_level.val

    def rewards(self, team_rewards, team_dones, team_infos, step):
        """Calculate team rewards and update stats."""

        team_reward = sum(team_rewards.values())
        team_info = {"stats": defaultdict(float)}

        for agent_id in team_dones:
            if team_dones[agent_id] is True:
                agent = self.env.realm.players.dead_this_tick.get(
                    agent_id, self.env.realm.players.get(agent_id)
                )
                if agent is None:
                    continue
                self._update_stats(agent)

        return team_reward, team_info

    def infos(self, team_reward, env_done, team_done, team_infos, step):
        """Update team infos and save replays."""

        team_infos = super().infos(team_reward, env_done, team_done, team_infos, step)

        if env_done:
            team_infos["stats"]["cod/attacked"] = self._cod_attacked
            team_infos["stats"]["cod/starved"] = self._cod_starved
            team_infos["stats"]["cod/dehydrated"] = self._cod_dehydrated
            team_infos["stats"]["task/completed"] = self._task_completed
            team_infos["curriculum"] = self._curriculum

            team_result, achieved, performed, _ = get_team_result(
                self.env.realm, self.teams, self.team_id
            )
            for key, val in list(achieved.items()) + list(performed.items()):
                team_infos["stats"][key] = float(val)

            # Fill in the TeamResult
            team_result.time_alive = self._time_alive
            team_result.gold_owned = self._gold_owned
            team_result.completed_task_count = round(self._task_completed * self.team_size)
            team_result.damage_received = self._damage_received
            team_result.damage_inflicted = self._damage_inflicted
            team_result.ration_consumed = self._ration_consumed
            team_result.potion_consumed = self._potion_consumed
            team_result.melee_level = self._melee_level
            team_result.range_level = self._range_level
            team_result.mage_level = self._mage_level
            team_result.fishing_level = self._fishing_level
            team_result.herbalism_level = self._herbalism_level
            team_result.prospecting_level = self._prospecting_level
            team_result.carving_level = self._carving_level
            team_result.alchemy_level = self._alchemy_level

            team_infos["team_results"] = (self.team_id, team_result)

            if self._replay_save_dir is not None:
                self._replay_helper.save(
                    os.path.join(
                        self._replay_save_dir, f"replay_{self._num_replays_saved}.json"), compress=False)
                self._num_replays_saved += 1

        return team_infos


def create_binding(args: Namespace):
    """Create an environment binding."""

    return pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        default_args=[Config(args)],
        env_name="Neural MMO",
        suppress_env_prints=False,
        emulate_const_horizon=args.max_episode_length,
        postprocessor_cls=Postprocessor,
        postprocessor_kwargs={
            'replay_save_dir': args.replay_save_dir
        },
    )
