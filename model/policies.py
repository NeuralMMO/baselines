from lib.agent.agent import Agent
from lib.agent.util import load_matching_state_dict
from model.basic.policy import BasicPolicy
from model.basic_teams.policy import BasicTeamsPolicy
from model.improved.policy import ImprovedPolicy
from model.random.policy import RandomPolicy
from model.realikun.policy import RealikunPolicy
from model.realikun_simple.policy import RealikunSimplifiedPolicy

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
