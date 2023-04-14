import torch
import pufferlib

from model.model import EntityEncoder, InteractionBlock, MemoryBlock, NMMONet, PolicyHead, SelfEncoder

class Policy(pufferlib.models.Policy):
    def __init__(self, binding, input_size=2048, hidden_size=4096):
        super().__init__(binding, input_size, hidden_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state_handler_dict = {}
        torch.set_num_threads(1)

        in_ch = 7
        in_size = [25, 25]
        n_player_feat = 262
        n_game_feat = 26
        n_legal = {
            'move': 5,
            'target': 19,
            'use': 3,
            'sell': 3,
        }
        n_self_feat = n_player_feat + n_game_feat + sum(n_legal.values())
        n_npc_feat = n_enemy_feat = 29

        self.n_attn_hidden = n_attn_hidden = n_ally_hidden = 256
        self.n_lstm_hidden = n_lstm_hidden = n_self_hidden = 512

        self.self_net = SelfEncoder(in_ch, in_size, n_self_feat,
                                    n_legal, n_self_hidden)
        self.ally_net = EntityEncoder('ally', n_ally_hidden, n_attn_hidden)
        self.npc_net = EntityEncoder('npc', n_npc_feat, n_attn_hidden)
        self.enemy_net = EntityEncoder('enemy', n_enemy_feat, n_attn_hidden)

        self.interact_net = InteractionBlock(n_attn_hidden)

        self.value_head = torch.nn.Linear(n_lstm_hidden, 1)

        self.featurized_single_observation_space = binding.featurized_single_observation_space

        self.policy_head = PolicyHead(n_lstm_hidden, n_legal)
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(512, n) for n in binding.single_action_space.nvec[:13]]
        )

    def critic(self, hidden):
        hidden = hidden.view(-1, 8, 512)
        return self.value_head(hidden.mean(dim=1))

    def encode_observations(self, env_outputs):
        x = pufferlib.emulation.unpack_batched_obs(
            self.featurized_single_observation_space, env_outputs
        )
        batch_size = x['tile'].shape[0]
        num_agents = x['tile'].shape[1]

        # x = self._preprocess(x)

        # h_self = self.self_net(x) # (batch_size, num_agents, 512)

        # h_ally = self.ally_net(self._self_as_ally_feature(h_self), h_self) # (batch_size, num_agent, 256)
        # h_npc = self.npc_net(x, h_self) # (batch_size, num_agents, 9, 256)
        # h_enemy = self.enemy_net(x, h_self) # (batch_size, num_agents, 9, 256)

        # h_inter = self.interact_net(x, h_self, h_ally, h_npc, h_enemy) # (batch_size, 2048)

        # self.recurrent_policy.h_self = h_self
        # self.recurrent_policy.h_inter = h_inter

        # num_features = h_inter.shape[2]
        # h_inter = h_inter.view(batch_size, num_agents*num_features)

        return torch.zeros((batch_size, num_agents*num_features), dtype=torch.float32, device=self.device), None
        # return h_inter, None # (batch_size, num_agents * num_feature), None

    def decode_actions(self, hidden, lookup, concat=True):
        hidden = hidden.view(-1, 8, 512)

        actions = [0 * dec(hidden) for dec in self.decoders]

        action_list = []
        for team_id in range(8):
            action_list += [a[:, team_id] for a in actions]
        actions = action_list

        # TODO: Remove this after eliminating NaNs
        actions = [torch.zeros_like(e) for e in actions]

        if concat:
            return torch.cat(actions, dim=-1)

        return actions

    @staticmethod
    def _preprocess(x):
        bs, na = x['tile'].shape[:2]
        team_mask_self = x['team_mask'][:, :, None]
        team_mask_ally = x['team_mask'].repeat(1, 2) \
            .unfold(dimension=1, size=na-1, step=1)[:, 1:-1]
        x['team_mask'] = torch.cat([team_mask_self, team_mask_ally], dim=-1)
        return x

    def _self_as_ally_feature(self, h_self):
        bs, na = h_self.shape[:2]
        tmp_x = dict()
        tmp_x['ally'] = h_self[:, :, :self.n_attn_hidden].repeat(1, 2, 1) \
            .unfold(dimension=1, size=na-1, step=1)[:, 1:-1].transpose(2, 3)
        return tmp_x
