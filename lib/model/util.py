import copy
import functools
from collections import Sequence

import numpy as np
import torch


def one_hot_generator(n_feature, index):
    arr = np.zeros(n_feature,)
    arr[index] = 1
    return arr


def multi_hot_generator(n_feature, index):
    arr = np.zeros(n_feature,)
    arr[:index] = 1
    return arr


def legal_mask(logit, legal):
    mask = torch.ones_like(legal) * -np.inf
    logit = torch.where(legal == 1., logit, mask)
    return logit


def single_as_batch(func):
    def _recursive_processing(x, squeeze=False):
        if isinstance(x, Sequence):
            return (_recursive_processing(_, squeeze) for _ in x)
        elif isinstance(x, dict):
            return {k: _recursive_processing(v, squeeze) for k, v in x.items()}
        else:
            return x.squeeze(0) if squeeze else x.unsqueeze(0)

    @functools.wraps(func)
    def wrap(self, *tensors):
        tensors = _recursive_processing(tensors)
        result = func(self, *tensors)
        return _recursive_processing(result, squeeze=True)

    return wrap


def tensorize_state(func):
    def _recursive_processing(state, device):
        if not isinstance(state, torch.Tensor):
            if isinstance(state, dict):
                for k, v in state.items():
                    state[k] = _recursive_processing(state[k], device)
            else:
                state = torch.FloatTensor(state).to(device)
        return state

    @functools.wraps(func)
    def wrap(self, state, *arg, **kwargs):
        state = copy.deepcopy(state)
        state = _recursive_processing(state, self.device)
        return func(self, state, *arg, **kwargs)

    return wrap
