from tensorflow.keras.layers import Layer
import copy
c = lambda x: copy.deepcopy(x)

from attention import *


class Transformer(Layer):
    def __init__(self, mask, N):
        super().__init__()
        self.transformer_layers = [ TransformerLayer(MHAttention, FeedForward, PreNorm, mask) for _ in range(N) ]

    def call(self, x):
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x


class TransformerLayer(Layer):
    def __init__(self, attn, ff, norm, mask):
        super().__init__()
        self.attn, self.ff = attn, ff
        self.norm1, self.norm2 = c(norm), c(norm)
        self.mask = mask

    def call(self, x):
        out = self.norm1( self.attn(x, self.mask) )
        return self.norm2( self.ff(out) )
