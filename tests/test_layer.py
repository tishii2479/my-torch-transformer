from math import sqrt
from sys import stderr
import unittest
import torch
import torch.nn.functional as F

from layer import *
from function import *


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.batch_size = 64
        self.sequence_length = 16

    def test_feed_forward_network(self):
        shape = (self.batch_size, self.sequence_length, self.d_model)
        d_ff = 2048
        ffn = FeedForwardNetwork(self.d_model, d_ff)

        x = torch.rand(shape)
        y = ffn.forward(x)
        self.assertEqual(y.shape, shape)

    def test_attention(self):
        shape = (self.batch_size, self.sequence_length, self.d_model)
        Q = torch.rand(shape)
        K = torch.rand(shape)
        V = torch.rand(shape)

        y = attention(Q, K, V)
        self.assertEqual(y.shape, shape)

    def test_attention_value(self):
        Q = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        K = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        V = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        d_k = 3

        y = attention(Q, K, V)
        expected = torch.matmul(
            F.softmax(torch.div(torch.matmul(Q, K.transpose(1, 2)), sqrt(d_k)), dim=2), V.reshape(3, 3))
        # print('expected is:', expected)
        self.assertTrue(torch.equal(y, expected.reshape(Q.shape)))

    def test_multi_head_attention(self):
        shape = (self.batch_size, self.sequence_length, self.d_model)
        xs = torch.rand(shape)
        h = 8

        multiHeadAttention = MultiHeadAttention(h, self.d_model)
        y = multiHeadAttention.forward(xs, xs, xs)
        self.assertEqual(y.shape, shape)

    def test_sub_feed_forward_network(self):
        shape = (self.batch_size, self.sequence_length, self.d_model)
        d_ff = 2048
        ffn = FeedForwardNetwork(self.d_model, d_ff)
        sublayer_ffn = SubLayer(ffn, self.d_model)

        x = torch.rand(shape)
        y = sublayer_ffn.forward(x)
        self.assertEqual(y.shape, shape)


if __name__ == '__main__':
    unittest.main()
