import unittest
import torch.nn.functional as F
import torch

from transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.params = {
            'N': 6,
            'h': 8,
            'd_model': 512,
            'd_ff': 4096,
            'activation': F.relu
        }

    def test_transformer(self):
        batch_size = 64
        source_length = 16
        target_length = 8

        transformer = Transformer(self.params)
        source = torch.rand(
            (batch_size, source_length, self.params['d_model']))
        target = torch.rand(
            (batch_size, target_length, self.params['d_model']))

        transformer.forward(source, target)


if __name__ == '__main__':
    unittest.main()
