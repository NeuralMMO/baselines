from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

def modelSize(net):
   '''Print model size'''
   params = 0
   for e in net.parameters():
      params += np.prod(e.size())
   params = int(params/1000)
   print("Network has ", params, "K params")

def ModuleList(module, *args, n=1):
   '''Repeat module n times'''
   return nn.ModuleList([module(*args) for i in range(n)])
