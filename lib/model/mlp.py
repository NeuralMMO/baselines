import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
  def __init__(self, n_input, n_hiddens):
    super().__init__()

    assert len(n_hiddens) > 0

    self._linear = nn.ModuleList()
    n_prev = n_input
    for n_curr in n_hiddens:
      self._linear.append(nn.Linear(n_prev, n_curr))
      self._linear.append(nn.ReLU(inplace=True))
      n_prev = n_curr

  def forward(self, x: torch.Tensor):
    for layer in self._linear:
      x = layer(x)
    return x
