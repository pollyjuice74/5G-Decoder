import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, dims, heads):
        super().__init__()
        assert (dims % heads) == 0, 'dimension must be divisible by the number of heads'

    def call(self):
        self.lin_attention(k, q, v)
        return 

    def n_to_k(self):
        pass

    def lin_attention(self):
        pass
        
