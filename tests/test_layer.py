import unittest
import torch
from layer import *


class TestLayer(unittest.TestCase):
    def test_feed_forward_network(self):
        d_model = 512
        d_ff = 2048
        batch_size = 64
        ffn = PostionWiseFeedForwardNetwork(d_model, d_ff)

        x = torch.rand((batch_size, d_model))
        y = ffn.forward(x)
        self.assertEqual(y.shape, (batch_size, d_model))


if __name__ == '__main__':
    unittest.main()
