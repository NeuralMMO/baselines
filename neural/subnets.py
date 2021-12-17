from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from collections import defaultdict

from neural import utils

def Conv2d(fIn, fOut, k, stride=1):
   '''torch Conv2d with same padding'''
   assert k % 2 == 0
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, 
      k, stride=stride, padding=pad)

def Pool(k, stride=1, pad=0):
   return torch.nn.MaxPool2d(
      k, stride=stride, padding=pad)

class ConvReluPool(nn.Module):
   def __init__(self, fIn, fOut, 
         k, stride=1, pool=2):
      super().__init__()
      self.conv = Conv2d(
         fIn, fOut, k, stride)
      self.pool = Pool(k)

   def forward(self, x):
      x = self.conv(x)
      x = F.relu(x)
      x = self.pool(x)
      return x

class ReluBlock(nn.Module):
   def __init__(self, h, layers=2, postRelu=True):
      super().__init__()
      self.postRelu = postRelu

      self.layers = utils.ModuleList(
         nn.Linear, h, h, n=layers)

   def forward(self, x):
      for idx, fc in enumerate(self.layers):
         if idx != 0:
            x = torch.relu(x)
         x = fc(x)

      if self.postRelu:
         x = torch.relu(x)

      return x

class DotReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.key = ReluBlock(
         h, layers, postRelu=False)

      self.val = ReluBlock(
         h, layers, postRelu=False)
   
   def forward(self, k, v):
      k = self.key(k).unsqueeze(-2)
      v = self.val(v)
      x = torch.sum(k * v, -1)
      return x

class ScaledDotProductAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = np.sqrt(h)

   def forward(self, Q, K, V):
      Kt  = K.transpose(-2, -1)
      QK  = torch.matmul(Q, Kt)

      #Original is attending over hidden dims?
      #QK  = torch.softmax(QK / self.scale, dim=-2)
      QK    = torch.softmax(QK / self.scale, dim=-1)
      score = torch.sum(QK, -2)
      QKV = torch.matmul(QK, V)
      return QKV, score

class SelfAttention(nn.Module):
   def __init__(self, xDim, yDim, flat=True):
      super().__init__()

      self.Q = torch.nn.Linear(xDim, yDim)
      self.K = torch.nn.Linear(xDim, yDim)
      self.V = torch.nn.Linear(xDim, yDim)

      self.attention = ScaledDotProductAttention(yDim)
      self.flat = flat

   def forward(self, q):
      Q = self.Q(q)
      K = self.K(q)
      V = self.V(q)

      attn, scores = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn, scores

class BatchFirstLSTM(nn.LSTM):
   def __init__(self, *args, **kwargs):
      super().__init__(*args, batch_first=True, **kwargs)

   def forward(self, input, hx):
      h, c       = hx
      h          = h.transpose(0, 1)
      c          = c.transpose(0, 1)
      hidden, hx = super().forward(input, [h, c])
      h, c       = hx
      h          = h.transpose(0, 1)
      c          = c.transpose(0, 1)
      return hidden, [h, c]
