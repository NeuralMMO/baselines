import pufferlib
import torch
from model.basic.policy import BasicPolicy
from model.basic_teams.policy import BasicTeamsPolicy
from model.improved.policy import ImprovedPolicy
from model.random.policy import RandomPolicy
from model.realikun.policy import RealikunPolicy
from model.realikun_simple.policy import RealikunSimplifiedPolicy
import lib.agent.util
import pufferlib.emulation

class PolicyRegistry(pufferlib.policy_store.PolicyRegistry):
  def __init__(self, binding: pufferlib.emulation.Binding):
    self._binding = binding

  def load_policy(self, policy_record: PolicyRecord):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = torch.load(policy_record.model_path, map_location=device)

    policy = self.policy_class(model["model_type"])(self._binding)
    lib.agent.util.load_matching_state_dict(
      policy,
      model["agent_state_dict"]
    )
    return policy.to(device)

  @staticmethod
  def policy_class(model_type: str):
      policy_dict = {
          "realikun": RealikunPolicy.create_policy,
          "realikun-simplified": RealikunSimplifiedPolicy.create_policy,
          "random": RandomPolicy.create_policy,
          "basic": lambda: BasicPolicy.create_policy(num_lstm_layers=0),
          "basic-lstm": lambda: BasicPolicy.create_policy(num_lstm_layers=1),
          "improved": lambda: ImprovedPolicy.create_policy(num_lstm_layers=0),
          "improved-lstm": lambda: ImprovedPolicy.create_policy(num_lstm_layers=1),
          "basic-teams": lambda: BasicTeamsPolicy.create_policy(num_lstm_layers=0),
          "basic-teams-lstm": lambda: BasicTeamsPolicy.create_policy(num_lstm_layers=0)
      }

      if model_type not in policy_dict:
          raise ValueError(f"Unsupported model type: {model_type}")

      return policy_dict[model_type]()
