from math import sqrt
import unittest
import torch
import torch.nn.functional as F

from layer import *
from function import *


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.embedding_size = 512
        self.batch_size = 64
        self.sequence_length = 16

    def test_feed_forward_network(self):
        shape = (self.batch_size, self.sequence_length, self.embedding_size)
        d_ff = 2048
        ffn = FeedForwardNetwork(self.embedding_size, d_ff)

        x = torch.rand(shape)
        y = ffn.forward(x)
        self.assertEqual(y.shape, shape)

    def test_attention(self):
        shape = (self.batch_size, self.sequence_length, self.embedding_size)
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
        self.assertTrue(torch.equal(y, expected.reshape(Q.shape)),
                        f'actual={y}, expected={expected}')

    def test_masked_attention_value(self):
        Q = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        K = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        V = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])

        y = attention(Q, K, V, is_masked=True)
        self.assertTrue(torch.equal(
            y[0][0][1], torch.tensor(0.)), f'actual={y}')
        self.assertTrue(torch.equal(
            y[0][0][2], torch.tensor(0.)), f'actual={y}')
        self.assertTrue(torch.equal(
            y[0][1][2], torch.tensor(0.)), f'actual={y}')

    def test_multi_head_attention(self):
        shape = (self.batch_size, self.sequence_length, self.embedding_size)
        x = torch.rand(shape)
        h = 8

        multiHeadAttention = MultiHeadAttention(h, self.embedding_size)
        y = multiHeadAttention.forward(x, x, x)
        self.assertEqual(y.shape, shape)

    def test_sub_feed_forward_network(self):
        shape = (self.batch_size, self.sequence_length, self.embedding_size)
        d_ff = 2048
        ffn = FeedForwardNetwork(self.embedding_size, d_ff)

        x = torch.rand(shape)
        y = ffn.forward(x)
        y = add_norm(x, y, self.embedding_size)
        self.assertEqual(y.shape, shape)

    def test_sub_multi_head_attention(self):
        shape = (self.batch_size, self.sequence_length, self.embedding_size)
        x = torch.rand(shape)
        h = 8

        mh_attention = MultiHeadAttention(h, self.embedding_size)

        y = mh_attention.forward(x, x, x)
        y = add_norm(x, y, self.embedding_size)
        self.assertEqual(y.shape, shape)


if __name__ == '__main__':
    unittest.main()
