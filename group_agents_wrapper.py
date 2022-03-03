from pdb import set_trace as T

from collections import defaultdict

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# info key for the individual rewards of an agent, for example:
# info: {
#   group_1: {
#      _group_rewards: [5, -1, 1],  # 3 agents in this group
#   }
# }
GROUP_REWARDS = "_group_rewards"

# info key for the individual infos of an agent, for example:
# info: {
#   group_1: {
#      _group_infos: [{"foo": ...}, {}],  # 2 agents in this group
#   }
# }
GROUP_INFO = "_group_info"


class GroupAgentsWrapper(MultiAgentEnv):
    """Wraps a MultiAgentEnv environment with agents grouped as specified.

    See multi_agent_env.py for the specification of groups.

    This API is experimental.
    """

    def __init__(self, env, groups, obs_space=None, act_space=None):
        """Wrap an existing multi-agent env to group agents together.

        See MultiAgentEnv.with_agent_groups() for usage info.

        Args:
            env (MultiAgentEnv): env to wrap
            groups (dict): Grouping spec as documented in MultiAgentEnv.
            obs_space (Space): Optional observation space for the grouped
                env. Must be a tuple space.
            act_space (Space): Optional action space for the grouped env.
                Must be a tuple space.
        """
        super().__init__()
        self.env = env
        self.groups = groups
        self.agent_id_to_group = {}
        self.agent_id_order = defaultdict(list)

        if obs_space is not None:
            self.observation_space = obs_space
        if act_space is not None:
            self.action_space = act_space

        # Groups can be a function or a dictionary
        if callable(groups):
            return

        for group_id, agent_ids in groups.items():
            for agent_id in agent_ids:
                if agent_id in self.agent_id_to_group:
                    raise ValueError(
                        "Agent id {} is in multiple groups".format(agent_id)
                    )
                self.agent_id_to_group[agent_id] = group_id

    def seed(self, seed=None):
        if not hasattr(self.env, "seed"):
            # This is a silent fail. However, OpenAI gyms also silently fail
            # here.
            return

        self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        return self._group_items(obs)

    def step(self, action_dict):
        # Ungroup and send actions
        action_dict = self._ungroup_items(action_dict)
        obs, rewards, dones, infos = self.env.step(action_dict)

        # Apply grouping transforms to the env outputs
        obs = self._group_items(obs)
        rewards = self._group_items(rewards)
        all_done = dones['__all__']
        dones = self._group_items(dones, agg_fn=lambda gvals: all(gvals))
        dones['__all__'] = all_done
        infos = self._group_items(
            infos, agg_fn=lambda gvals: {GROUP_INFO: gvals}
        )

        # Aggregate rewards, but preserve the original values in infos
        for agent_id, rew in rewards.items():
            if isinstance(rew, list):
                rewards[agent_id] = sum(rew)
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id][GROUP_REWARDS] = rew

        return obs, rewards, dones, infos

    def _ungroup_items(self, items):
        out = {}
        for group, itms in items.items():
            group_idxs = self.agent_id_order[group]
            for (idx, agent_id), itm in zip(group_idxs, itms):
                assert idx not in out, 'Duplicate item index'
                out[idx] = (agent_id, itm)

        out = dict([e[1] for e in sorted(out.items())])
        return out

    def _group_items(self, items, agg_fn=lambda e: e):
        grouped_items = {}
        for idx, (agent_id, item) in enumerate(items.items()):
            if agent_id in self.agent_id_to_group:
                group_id = self.agent_id_to_group[agent_id]
            elif callable(self.groups):
                group_id = self.groups(agent_id)
            else:
                group_id = agent_id

            if group_id not in grouped_items:
                grouped_items[group_id] = []

            grouped_items[group_id].append(item)
            self.agent_id_order[group_id].append((idx, agent_id))

        for key in list(grouped_items.keys()):
            grouped_items[key] = agg_fn(grouped_items[key]) 

        return grouped_items
