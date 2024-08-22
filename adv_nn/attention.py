import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Layer
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.activations import gelu
from tensorflow.keras import saving


class FeedForward(Layer):
    def __init__(self, dim, mult=4, activation=None, dropout=0.01):
        super().__init__()
        self.w1 = Dense(dim * mult)
        self.w2 = Dense(dim)
        self.dropout = Dropout(dropout) 
        self.act = activation if activation else gelu

    def call(self, x):
        x = self.act(self.w1(x))
        return self.w2(self.dropout(x))


@saving.register_keras_serializable()
class PreNorm(Layer):
    def __init__(self, dropout=0.01, **kwargs):
        super().__init__(**kwargs)
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout)
        
    def call(self, x):
        return self.norm(self.dropout(x))

    def get_config(self):
        config = super(PreNorm, self).get_config()
        config.update({"dropout": self.dropout.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MHAttention(Layer):
    def __init__(self, dims, heads, mask_length, linear=True, dropout=0.01):
        super().__init__()
        assert (dims % heads) == 0, 'dimension must be divisible by the number of heads'
        self.linear = linear
        self.dims = dims
        self.heads = heads
        self.dim_head = dims // heads

        if linear:
            self.k_proj = self.get_k_proj(mask_length) # n+m
            self.proj_k = None
            self.proj_v = None              

        self.to_q, self.to_k, self.to_v = [ Dense(self.dims, use_bias=False) for _ in range(3) ]
        self.to_out = Dense(dims)
        self.dropout = Dropout(dropout) # to d-dimentional embeddings

    def call(self, x, mask=None):
        out_att = self.lin_attention(x, mask) if self.linear else self.attention(x, mask)
        return out_att
            
    def get_k_proj(self, mask_length):
        # gets dimention for linear tranformer vector projection
        for k_proj in range(mask_length // 2, 0, -1): # starts at half the mask length TO 0
            if mask_length % k_proj == 0:
                return k_proj

    def lin_attention(self, x, mask): # O(n)
        b,n,_ = tf.shape(x)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)

        # Creates shape (n,k_proj) proj matrices for key and val 
        if self.proj_k is None or self.proj_v is None: 
            # TODO: Could make projs dense layers #
            self.proj_k = self.add_weight("proj_k", shape=[n, self.k_proj], initializer=GlorotUniform())
            self.proj_v = self.add_weight("proj_v", shape=[n, self.k_proj], initializer=GlorotUniform())

        # Project key and val into k-dimentional space
        key = tf.einsum('bnd,nk->bkd', key, self.proj_k)
        val = tf.einsum('bnd,nk->bkd', val, self.proj_v)

        # Reshape splitting for heads
        query = tf.reshape(query, (b, n, self.heads, self.dim_head))
        key = tf.reshape(key, (b, self.k_proj, self.heads, self.dim_head))
        val = tf.reshape(val, (b, self.k_proj, self.heads, self.dim_head))
        query, key, val = [ tf.transpose(x, [0, 2, 1, 3]) for x in [query, key, val] ]

        # Low-rank mask (n,k_proj)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, [n, self.k_proj], method='nearest')
        mask = tf.reshape(mask, (1, 1, n, self.k_proj))

        # Main attn logic: sftmx( q@k / d**0.5 ) @ v 
        scores = tf.einsum('bhnd,bhkd->bhnk', query, key) / (tf.sqrt( tf.cast(self.dim_head, dtype=tf.float32) ))
        scores += (mask * -1e9) if mask is not None else 0.
        attn = tf.nn.softmax(scores, axis=-1) # (b,h,n,k_proj)
        attn = self.dropout(attn)
        out = tf.einsum('bhnk,bhkd->bhnd', attn, val)

        # Reshape and pass through out layer
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (b, n, -1))
        return self.to_out(out)

    def attention(self, mask): # O(n^2)
        b,n,_ = tf.shape(x)
        query, key, val = self.to_q(x), self.to_k(x), self.to_v(x)
        query, key, val = [ tf.reshape(x, (b, n, self.heads, self.dim_head)) for x in [query, key, val] ]
        query, key, val = [ tf.transpose(x, [0, 2, 1, 3]) for x in [query, key, val] ]

        scores = tf.einsum('bhnd,bhnd->bhnn', query, key) / (tf.sqrt(self.dim_head))
        scores += (mask * -1e9) if mask is not None else 0. # apply mask non-edge connections
        attn = tf.nn.softmax(scores, axis=-1) #-1 
        attn = self.dropout(attn)
        out = tf.einsum('bhnn,bhnm->bhnd', attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (b, n, -1))
        return self.to_out(out)



