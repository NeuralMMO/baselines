from argparse import Namespace
from collections import defaultdict

import nmmo
import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, score_unique_events

class Config(nmmo.config.Default):
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

        self.COMMUNICATION_SYSTEM_ENABLED = False

        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity

class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
      replay_save_dir=None,
      sqrt_achievement_rewards=False,
      heal_bonus_weight=0,
      explore_bonus_weight=0,
      ):
        super().__init__(env, agent_id)
        self.sqrt_achievement_rewards = sqrt_achievement_rewards
        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight

    def reset(self, obs):
        '''Called at the start of each episode'''
        super().reset(obs)

    def action(self, action):
        '''Called before actions are passed from the model to the environment'''
        return action

    @property
    def observation_space(self):
        '''If you modify the shape of features, you need to specify the new obs space'''
        return super().observation_space

    def observation(self, obs):
        '''Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        '''
        return obs

    def reward_done_info(self, reward, done, info):
        '''Called on reward, done, and info before they are returned from the environment'''
        reward, done, info = super().reward_done_info(reward, done, info)

        # The below lines update the stats and do NOT affect the reward.
        #infos = {"stats": defaultdict(float)}  # DO NOT REMOVE
        agent_id = self.agent_id

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.

        # Add "Healing" score based on health increase and decrease due to food and water
        health_restore = 0
        if agent_id in self.env.realm.players:
            health_restore += self.env.realm.players[agent_id].resources.health_restore
        healing_bonus = self.heal_bonus_weight if health_restore > 0 else 0

        # Unique event-based rewards, similar to exploration bonus
        # NOTE: this gets slower as the size of event log increases
        log = self.env.realm.event_log.get_data(agents=[agent_id])
        explore_bonus = self.explore_bonus_weight * score_unique_events(self.env.realm, log,
          sqrt_rewards=self.sqrt_achievement_rewards)

        reward = reward + explore_bonus + healing_bonus

        return reward, done, info


def make_env_creator(args: Namespace):
    # TODO: Max episode length
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                'replay_save_dir': args.replay_save_dir,
                'sqrt_achievement_rewards': args.sqrt_achievement_rewards,
                'heal_bonus_weight': args.heal_bonus_weight,
                'explore_bonus_weight': args.explore_bonus_weight,
            },
        )
        return env
    return env_creator
