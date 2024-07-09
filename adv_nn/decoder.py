from attention import *
import copy
c = copy.deepcopy()


class Transformer(tf.keras.layers.Layer):
    def init(self, mask, N):
      self.transformer_layers = [ TransformerLayer(MHAttention, FeedForward, PreNorm, mask) for _ in range(N) ]

    def call(self, x):
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x


class TransformerLayer(tf.keras.layers.Layer):
    def init(self, attn, ff, norm, mask):
        self.attn, self.ff = attn, ff, 
        self.norm1, self.norm2 = c(norm), c(norm)

    def call(self, x):
        out = self.norm1(self.attn(x))
        return self.norm2(self.ff(out))
