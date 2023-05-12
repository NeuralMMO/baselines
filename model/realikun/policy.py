import torch
import pufferlib

from model.realikun.model import EntityEncoder, InteractionBlock, MemoryBlock,  ModelArchitecture,  PolicyHead, SelfEncoder
from env.nmmo_team_env import NMMOTeamEnv

class BaselinePolicy(pufferlib.models.Policy):
  def __init__(self, binding, input_size=2048, hidden_size=4096):
    super().__init__(binding, input_size, hidden_size)

    # xcxc Do these belong here?
    self.state_handler_dict = {}
    torch.set_num_threads(1)

    self.self_net = SelfEncoder(
      ModelArchitecture.TILE_NUM_CHANNELS,
      ModelArchitecture.TILE_IMG_SIZE,
      ModelArchitecture.SELF_NUM_FEATURES,
      ModelArchitecture.ACTION_NUM_DIM,
      ModelArchitecture.SELF_EMBED
    )

    self.ally_net = EntityEncoder(
      'ally',
      ModelArchitecture.SELF_AS_ALLY_EMBED,
      ModelArchitecture.ATTENTION_HIDDEN)

    self.npc_net = EntityEncoder(
      'npc',
      ModelArchitecture.ENTITY_NUM_FEATURES,
      ModelArchitecture.ATTENTION_HIDDEN)

    self.enemy_net = EntityEncoder(
      'enemy',
      ModelArchitecture.ENTITY_NUM_FEATURES,
      ModelArchitecture.ATTENTION_HIDDEN)

    self.interact_net = InteractionBlock(ModelArchitecture.ATTENTION_HIDDEN)

    self.value_head = torch.nn.Linear(ModelArchitecture.LSTM_HIDDEN, 1)

    self.observation_space = binding.featurized_single_observation_space
    self.single_observation_space = binding.single_observation_space


    self.policy_head = PolicyHead(
        ModelArchitecture.LSTM_HIDDEN, ModelArchitecture.ACTION_NUM_DIM)

  def critic(self, hidden):
      # xcxc 512 is hardcoded here
      hidden = hidden.view(-1, ModelArchitecture.NUM_PLAYERS_PER_TEAM, 512)
      return self.value_head(hidden.mean(dim=1))

  def encode_observations(self, env_outputs):
    x = env_outputs
    if isinstance(env_outputs, torch.Tensor):
      x = pufferlib.emulation.unpack_batched_obs(
        self.observation_space,
        env_outputs
    )

    batch_size = x['tile'].shape[0]
    num_agents = x['tile'].shape[1]

    x = self._preprocess(x)

    h_self = self.self_net(x) # (batch_size, num_agents, SELF_EMBED)

    h_ally = self.ally_net(self._self_as_ally_feature(h_self), h_self)
    h_npc = self.npc_net(x, h_self) # (batch_size, num_agents, 9, 256)
    h_enemy = self.enemy_net(x, h_self) # (batch_size, num_agents, 9, 256)

    h_inter = self.interact_net(x, h_self, h_ally, h_npc, h_enemy) # (batch_size, 2048)

    self.recurrent_policy.h_self = h_self
    self.recurrent_policy.h_inter = h_inter
    self.recurrent_policy.reset = x["reset"]

    batch_size, num_agents, num_features = h_inter.shape
    h_inter = h_inter.view(batch_size, num_agents*num_features)

    return h_inter, x["legal"] # (batch_size, num_agents * num_feature)

  def decode_actions(self, hidden, lookup, concat=True):
    batch_size = hidden.shape[0]

    # reshape the batch so that we compute actions per-agent
    hidden = hidden.view(-1, 512)
    action_logits = self.policy_head(hidden)
    if concat:
      action_logits = torch.cat(action_logits, dim=-1)

    action_logits = [
      a.view(batch_size, ModelArchitecture.NUM_PLAYERS_PER_TEAM, -1)
      for a in action_logits]

    team_actions = []
    # action ordering fixed. see ModelArchitecture.ACTION_NUM_DIM
    for logits, at in zip(action_logits, lookup.keys()):
      mask = lookup[at]
      masked_action = logits + (1-mask)*torch.finfo(logits.dtype).min
      for player in range(ModelArchitecture.NUM_PLAYERS_PER_TEAM):
        team_actions.append(masked_action[:, player, :])

    return team_actions

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
    tmp_x['ally'] = h_self[:, :, :ModelArchitecture.ATTENTION_HIDDEN].repeat(1, 2, 1) \
        .unfold(dimension=1, size=na-1, step=1)[:, 1:-1].transpose(2, 3)
    return tmp_x

  @staticmethod
  def create_policy():
    return pufferlib.frameworks.cleanrl.make_policy(
      BaselinePolicy,
      recurrent_cls=MemoryBlock,
      recurrent_args=[2048, 4096],
      recurrent_kwargs={'num_layers': 1},
    )

  @staticmethod
  def env_creator(config, team_helper):
    return lambda: NMMOTeamEnv(config, team_helper)
