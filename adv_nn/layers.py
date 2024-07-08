import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, dims, heads, linear=True):
        super().__init__()
        assert (dims % heads) == 0, 'dimension must be divisible by the number of heads'
        self.linear = linear
            
    def call(self):
        out_att = self.lin_attention(k, q, v) if self.linear else self.attention(k, q, v)
        return 

    def n_to_k(self):
        pass

    def lin_attention(self): # O(n)
        pass

    def attention(self): # O(n^2)
        pass
