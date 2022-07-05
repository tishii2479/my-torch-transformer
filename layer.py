from math import sqrt
import numpy as np
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from function import attention

# PyTorchにおけるカスタムレイヤーの実装
# https://www.bigdata-navi.com/aidrops/2890/


class SubLayer(nn.Module):
    def __init__(
        self,
        subLayer: nn.Module,
        d_model: int
    ):
        super().__init__()
        self.subLayer = subLayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        '''
        x = (batch_size, sequence_length, d_model)
        '''
        return self.layer_norm.forward(self.subLayer.forward(x) + x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        d_k: int = None,
        d_v: int = None
    ):
        super().__init__()
        self.h = h
        self.d_k = d_model // h if d_k is None else d_k
        self.d_v = d_model // h if d_v is None else d_v

        self.lq = nn.Linear(d_model, self.d_k * h, bias=False)
        self.lk = nn.Linear(d_model, self.d_k * h, bias=False)
        self.lv = nn.Linear(d_model, self.d_v * h, bias=False)

        self.lo = nn.Linear(h * self.d_v, d_model, bias=False)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor
    ):
        '''
        Q = (batch_size, sequence_length, d_model)
        K = (batch_size, sequence_length, d_model)
        V = (batch_size, sequence_length, d_model)

        return (batch_size, sequence_length, d_model)
        '''
        Q = self.lq.forward(Q)
        K = self.lk.forward(K)
        V = self.lv.forward(V)

        heads = []

        for i in range(self.h):
            head_i = attention(
                Q[:, :, i * self.d_k: (i+1) * self.d_k],
                K[:, :, i * self.d_k: (i+1) * self.d_k],
                V[:, :, i * self.d_v: (i+1) * self.d_v]
            )
            heads.append(head_i)

        return self.lo.forward(torch.cat(heads, dim=2))


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: lambda x: x = F.relu
    ):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x: Tensor):
        '''
        x = (batch_size, sequence_length, d_model)
        '''
        return self.l2.forward(self.activation(self.l1.forward(x)))
