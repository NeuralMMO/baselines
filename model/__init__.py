import pufferlib
import pufferlib.emulation
import torch

from model.decode.policy import Policy as DecodePolicy
from model.improved.policy import ImprovedPolicy
from model.random.policy import RandomPolicy
from model.realikun.policy import RealikunPolicy

policy_dict = {
    "realikun": lambda: RealikunPolicy.create_policy(num_lstm_layers=0),
    "realikun-lstm": lambda: RealikunPolicy.create_policy(num_lstm_layers=1),
    "random": lambda: RandomPolicy.create_policy,
    "improved": lambda: ImprovedPolicy.create_policy(num_lstm_layers=0),
    "decode": lambda: DecodePolicy.create_policy(num_lstm_layers=0),
    "improved-lstm": lambda: ImprovedPolicy.create_policy(num_lstm_layers=1),
}


def create_policy(name: str, binding: pufferlib.emulation.Binding):
  # pylint: disable=E1101
  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
  return policy_dict[name]()(binding).to(device)
