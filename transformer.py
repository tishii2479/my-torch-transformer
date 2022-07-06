from function import add_norm
from torch import nn
from layer import MultiHeadAttention, FeedForwardNetwork


class Transformer(nn.Module):
    def __init__(
        self,
        params: dict
    ):
        self.params = params
        self.encoders = [TransformerEncoderBlock(
            params) for _ in range(params['N'])]
        self.decoders = [TransformerDecoderBlock(
            params) for _ in range(params['N'])]

    def forward(self, source, target):
        encoder_output = source
        for block in self.encoders:
            encoder_output = block.forward(encoder_output)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        params: dict,
    ):
        self.params = params
        self.mh_attention = MultiHeadAttention(params['h'], params['d_model'])
        self.ffn = FeedForwardNetwork(
            params['d_model'], params['d_ff'], params['activation'])

    def forward(self, x):
        '''
        x = (batch_size, sequence_length, embedding_size)
        '''
        y = add_norm(
            self.mh_attention.forward(x, x, x), x)
        return add_norm(self.ffn.forward(y), y, self.params['d_model'])


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        params: dict
    ):
        self.params = params
        self.mh_attention = MultiHeadAttention(
            params['h'], params['d_model'], is_masked=True)
        self.masked_mh_attention = MultiHeadAttention(
            params['h'], params['d_model'], is_masked=True)
        self.ffn = FeedForwardNetwork(
            params['d_model'], params['d_ff'], params['activation'])

    def forward(self, x, encoder_output):
        '''
        encoder_output = (batch_size, sequence_length, embedding_size)
        x = (batch_size, sequence_length, embedding_size)
        '''
        y = add_norm(
            self.mh_attention.forward(x, x, x), x)
        y = add_norm(
            self.masked_mh_attention.forward(x, encoder_output, encoder_output), x)
        return add_norm(self.ffn.forward(y), y, self.params['d_model'])
