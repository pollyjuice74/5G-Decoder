import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, dims, heads, linear=True, k=256):
        super().__init__()
        assert (dims % heads) == 0, 'dimension must be divisible by the number of heads'
        self.linear = linear
        self.dims = dims
        self.heads = heads
        self.dim_head = dims // heads
        self.k = k
        
        self.to_q, self.to_k, self.to_v = [ Dense(self.dim_head * heads, use_bias=False) for _ in range(3) ]
        self.to_out = Dense(dims)
        self.dropout = Dropout(dims) # to d-dimentional embeddings
            
    def call(self, x, mask=None):
        out_att = self.lin_attention(x, mask) if self.linear else self.attention(x, mask)
        return out_attn

    def n_to_k(self):
        pass

    def lin_attention(self, x, mask): # O(n)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)
        pass

    def attention(self, mask): # O(n^2)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)
        pass
