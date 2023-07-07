import torch
from lib.agent.agent import Agent
from lib.agent.util import load_matching_state_dict
from model.basic.policy import BasicPolicy
from model.basic_teams.policy import BasicTeamsPolicy
from model.improved.policy import ImprovedPolicy
from model.random.policy import RandomPolicy
from model.realikun.policy import RealikunPolicy
from model.realikun_simple.policy import RealikunSimplifiedPolicy
import lib.agent.util
import pufferlib.emulation

def policy_class(model_type: str):
  if model_type == "realikun":
    return RealikunPolicy.create_policy()
  if model_type == "realikun-simplified":
    return RealikunSimplifiedPolicy.create_policy()
  elif model_type == "random":
    return RandomPolicy.create_policy()
  elif model_type == "basic":
    return BasicPolicy.create_policy(num_lstm_layers=0)
  elif model_type == "basic-lstm":
    return BasicPolicy.create_policy(num_lstm_layers=1)
  elif model_type == "improved":
    return ImprovedPolicy.create_policy(num_lstm_layers=0)
  elif model_type == "improved-lstm":
    return ImprovedPolicy.create_policy(num_lstm_layers=1)
  elif model_type == "basic-teams":
    return BasicTeamsPolicy.create_policy(num_lstm_layers=0)
  elif model_type == "basic-teams-lstm":
    return BasicTeamsPolicy.create_policy(num_lstm_layers=0)
  else:
    raise ValueError(f"Unsupported model type: {model_type}")

def load_policy(model_init_from_path: str, binding: pufferlib.emulation.Binding):
  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
  model = torch.load(model_init_from_path, map_location=device)
  policy = policy_class(
    model.get("model_type", "realikun"))(binding)
  lib.agent.util.load_matching_state_dict(
    policy,
    model["agent_state_dict"]
  )
  return policy.to(device)

def create_policy(model_type: str, binding: pufferlib.emulation.Binding):
  return policy_class(model_type)(binding)
