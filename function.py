from math import sqrt
import torch
from torch import Tensor
import torch.nn.functional as F


def attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_masked: bool = False
):
    '''
    Q = (batch_size, sequence_length, d_k)
    K = (batch_size, sequence_length, d_k)
    V = (batch_size, sequence_length, d_v)

    return (batch_size, sequence_length, d_v)
    '''
    d_k = K.size(2)
    l = Q.size(1)
    qk = torch.div(torch.matmul(Q, K.transpose(1, 2)), sqrt(d_k))
    if is_masked:
        mask = torch.nan_to_num(torch.triu(
            torch.ones((l, l)), 1) * (-torch.inf))
        qk = qk + mask
    a = F.softmax(qk, dim=2)
    return torch.matmul(a, V)
