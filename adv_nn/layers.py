import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform


class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, dims, heads, linear=True, k=256):
        super().__init__()
        assert (dims % heads) == 0, 'dimension must be divisible by the number of heads'
        self.linear = linear
        self.dims = dims
        self.heads = heads
        self.dim_head = dims // heads
        self.k = k

        if linear:
            self.proj_k = self.add_weight("proj_k", shape=[dims, k], initializer=GlorotUniform())
            self.proj_v = self.add_weight("proj_v", shape=[dims, k], initializer=GlorotUniform())
        
        self.to_q, self.to_k, self.to_v = [ Dense(self.dim_head * heads, use_bias=False) for _ in range(3) ]
        self.to_out = Dense(dims)
        self.dropout = Dropout(dims) # to d-dimentional embeddings
            
    def call(self, x, mask=None):
        out_att = self.lin_attention(x, mask) if self.linear else self.attention(x, mask)
        return out_attn

    def lin_attention(self, x, mask): # O(n)
        b,n,_ = tf.shape(x)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)

        # Project key and val into k-dimentional space
        key = tf.einsum('bnd,nk->bkd', key, self.proj_k)
        val = tf.einsum('bnd,nk->bkd', val, self.proj_v)

        # Reshape splitting for heads
        query = tf.reshape(query, (b, n, self.heads, self.dim_head))
        key = tf.reshape(key, (b, self.k, self.heads, self.dim_head))
        val = tf.reshape(val, (b, self.k, self.heads, self.dim_head))
        query, key, value = [ tf.transpose(x, [0, 2, 1, 3]) for x in [query, key, value] ]

        # Main attn logic: sftmx( q@k / d**-0.5 ) @ v 
        scores = tf.einsum('bhnd,bhkd->bhnk', query, key) / (tf.sqrt(self.dim_head))
        scores += (mask * -1e9) if mask is not None else 0.
        attn = tf.nn.softmax(scores, axis=0) #-1
        attn = self.dropout(attn)
        out = tf.einsum('bhnk,bhkd->bhnd', attn, val)

        # Reshape and pass through out layer
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (b, n, -1))
        return self.to_out(out)

    def attention(self, mask): # O(n^2)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)
        pass
