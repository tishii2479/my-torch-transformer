import numpy as np
from torch import Tensor, nn
import torch
import torch.nn.functional as F

# PyTorchにおけるカスタムレイヤーの実装
# https://www.bigdata-navi.com/aidrops/2890/


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class PostionWiseFeedForwardNetwork(nn.Module):
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
        return self.l2.forward(self.activation(self.l1.forward(x)))
