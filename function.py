from math import sqrt
import torch
from torch import Tensor
import torch.nn.functional as F


def attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor
):
    '''
    Q = (batch_size, sequence_length, d_k)
    K = (batch_size, sequence_length, d_k)
    V = (batch_size, sequence_length, d_v)

    return (batch_size, sequence_length, d_v)
    '''
    d_k = K.size(2)
    a = F.softmax(torch.div(torch.matmul(
        Q, K.transpose(1, 2)), sqrt(d_k)), dim=2)
    return torch.matmul(a, V)
