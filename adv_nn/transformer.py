from tensorflow.keras.layers import Layer
import copy
c = lambda x: copy.deepcopy(x)

from attention import *


class Transformer(Layer):
    def __init__(self, d_model, heads, mask, N):
        super().__init__()
        self.transformer_layers = [ TransformerLayer( MHAttention(d_model, heads, mask_length=mask.shape[0]),
                                                      FeedForward(d_model),
                                                      PreNorm() ) for _ in range(N) ]
        self.mask = mask

    def call(self, x):
        for transformer in self.transformer_layers:
            x = transformer(x, self.mask)
        return x


class TransformerLayer(Layer):
    def __init__(self, attn, ff, norm):
        super().__init__()
        self.attn, self.ff = attn, ff
        self.norm1, self.norm2 = c(norm), c(norm)

    def call(self, x, mask):
        out = self.norm1( self.attn(x, mask) )
        return self.norm2( self.ff(out) )
