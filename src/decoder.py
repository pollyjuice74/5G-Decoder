
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Layer
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from .utils import llr_to_bin, logits_to_bin, bin_to_llr, logits_to_llr


class LinearMHAttention( Layer ):
    """
    LinearMHAttention
    -----------------
    A layer that performs efficient attention computation by projecting the key and value tensors
    into a lower-dimensional space.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.

    key_dim : int
        Dimensionality of the key, value, and query vectors.

    mask_shape : tuple
        Shape of the attention mask, typically derived from the parity-check matrix (PCM).

    mask_division_shape : int
        Parameter used to divide and reduce the size of the projection space.

    dropout : float, optional
        Dropout rate applied to attention weights. Defaults to 0.1.

    Input
    -----
    query : tf.Tensor
        Tensor of shape [batch_size, sequence_length, key_dim] representing the query vectors.

    value : tf.Tensor
        Tensor of shape [batch_size, sequence_length, key_dim] representing the value vectors.

    key : tf.Tensor, optional
        Tensor of shape [batch_size, sequence_length, key_dim] representing the key vectors.
        Defaults to `value` if not provided.

    attention_mask : tf.Tensor, optional
        Mask tensor used to exclude certain positions from attention computation.

    Output
    ------
    out : tf.Tensor
        Output tensor of shape [batch_size, sequence_length, key_dim].
    """
    def __init__(self, num_heads, key_dim, mask_shape, mask_division_shape, dropout=0.1):
        super().__init__()
        assert (key_dim % num_heads) == 0, 'dimension must be divisible by the number of heads'
        self.dims = key_dim
        self.heads = num_heads
        self.dim_head = self.dims // self.heads

        self.k_proj = self.get_k_proj(mask_shape, mask_division_shape) # n+m
        self.proj_k = None
        self.proj_v = None

        self.to_q, self.to_k, self.to_v = [ Dense(self.dims, use_bias=False, activation='gelu') for _ in range(3) ]
        self.to_out = Dense(self.dims, activation='gelu')
        self.dropout = Dropout(dropout) # to d-dimentional embeddings

    def build(self, input_shape):
        """Build the projection weights for key and value tensors."""
        n_value = input_shape[1] # (b, n, d)
        # Creates shape (n,k_proj) proj matrices for key and value
        self.proj_k = self.add_weight("proj_k", shape=[n_value, self.k_proj], initializer=GlorotUniform())
        self.proj_v = self.add_weight("proj_v", shape=[n_value, self.k_proj], initializer=GlorotUniform())

    def get_k_proj(self, mask_shape, mask_division_shape):
        """Determine the key projection dimension based on mask shape and division parameter."""
        mask_length = mask_shape[1] # mask_shape (b, n+m, n+m)
        # gets dimention for linear tranformer vector projection
        for k_proj in range(mask_length // mask_division_shape, 0, -1): # starts at half the mask length to 0
            if mask_length % k_proj == 0:
                return tf.cast(k_proj, tf.int32)

    def call(self, query, value, key=None, attention_mask=None, training=False): # O(n)
        """Perform forward pass with linear multi-head attention mechanism."""
        shape = tf.shape(query) # (b, n, d)
        b = tf.cast(shape[0], tf.int32)
        n = tf.cast(shape[1], tf.int32)

        key = value if key is None else key

        assert query.shape[-1] is not None, "The last dimension of x is undefined."

        query, key, val = self.to_q(query), self.to_k(key), self.to_v(value)

        # Project key and val into k-dimentional space
        key = tf.einsum('bnd,nk->bkd', key, self.proj_k)
        val = tf.einsum('bnd,nk->bkd', val, self.proj_v)

        # Reshape splitting for heads
        query = tf.reshape(query, (b, n, self.heads, self.dim_head))
        key = tf.reshape(key, (b, self.k_proj, self.heads, self.dim_head))
        val = tf.reshape(val, (b, self.k_proj, self.heads, self.dim_head))
        query, key, val = [ tf.transpose(x, [0, 2, 1, 3]) for x in [query, key, val] ]

        # Low-rank mask (n,k_proj)
        mask = tf.expand_dims(attention_mask, axis=-1)
        mask = tf.image.resize(mask, [n, self.k_proj], method='nearest')
        mask = tf.reshape(mask, (1, 1, n, self.k_proj))

        # Main attn logic: sftmx( q@k / d**0.5 ) @ v
        scores = tf.einsum('bhnd,bhkd->bhnk', query, key) / (tf.sqrt( tf.cast(self.dim_head, dtype=tf.float32) ))
        scores = scores * mask  if mask is not None else scores
        attn = tf.nn.softmax(scores, axis=-1) # (b,h,n,k_proj)
        attn = self.dropout(attn) if training else attn
        out = tf.einsum('bhnk,bhkd->bhnd', attn, val)

        # Reshape and pass through out layer
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (b, n, -1))
        return self.to_out(out)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    TransformerEncoderBlock
    -----------------------
    A layer implementing a single Transformer Encoder Block with Multi-Head Attention,
    Feedforward Networks (FFN), and Layer Normalization.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output embeddings.

    num_heads : int
        Number of attention heads.

    d_ff : int
        Dimensionality of the Feedforward Network (FFN).

    linear : bool
        Whether to use Linear Multi-Head Attention instead of regular attention.

    mask_shape : tuple
        Shape of the attention mask.

    mask_division_shape : int
        Parameter used to divide and reduce the size of the projection space.

    dropout_rate : float, optional
        Dropout rate applied to the attention and FFN layers. Defaults to 0.1.

    Input
    -----
    x : tf.Tensor
        Input tensor of shape [batch_size, sequence_length, d_model].

    mask : tf.Tensor
        Attention mask tensor.

    training : bool
        Indicates whether the layer is in training mode.

    Output
    ------
    out2 : tf.Tensor
        Output tensor of shape [batch_size, sequence_length, d_model].
    """
    def __init__(self, d_model, num_heads, d_ff, linear, mask_shape, mask_division_shape, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = (
            LinearMHAttention(num_heads=num_heads,
                              key_dim=d_model,
                              mask_shape=mask_shape,
                              mask_division_shape=mask_division_shape,
                              dropout=dropout_rate)
            if linear
            else MultiHeadAttention(num_heads=num_heads,
                                    key_dim=d_model,
                                    dropout=dropout_rate)
        )
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='gelu'),
            Dense(d_model),
        ])

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, mask, training):
        """Apply attention and feedforward layers with residual connections and normalization."""
        # Multi-Head Attention
        attn_output = self.mha(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Add & Normalize

        # Feedforward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Add & Normalize
        return out2


class Decoder( Layer ):
    """
    Decoder
    -------
    The main class for decoding LDPC codes using transformer layers.

    Parameters
    ----------
    args : Namespace
        Contains configuration parameters, such as `code`, `d_model`, `heads`, and `t_layers`.

    linear : bool, optional
        Whether to use Linear Multi-Head Attention. Defaults to True.

    verbose : bool, optional
        Whether to display PCM and mask matrices. Defaults to True.

    dropout_rate : float, optional
        Dropout rate for all layers. Defaults to 0.1.

    rate_matching : bool, optional
        Indicates whether to enable rate matching. Defaults to False.

    Methods
    -------
    create_mask(H : tf.Tensor) -> tf.Tensor
        Creates the attention mask based on the parity-check matrix (PCM).

    get_syndrome(vn_vector : tf.Tensor, from_llr : bool, from_logits : bool) -> tf.Tensor
        Computes the syndrome (PCM @ r.T = 0) to verify correctness of codeword in binary format.

    call(x_nodes : tf.Tensor, training : bool) -> tf.Tensor
        Executes the forward pass of the decoder.

    Input
    -----
    x_nodes : tf.Tensor
        Input tensor containing variable node (VN) and check node (CN) embeddings
        of shape [batch_size, num_nodes, hidden_dims].

    training : bool
        Indicates whether the model is in training mode.

    Output
    ------
    vn_logits : tf.Tensor
        Decoded logits for variable nodes of shape [batch_size, num_variable_nodes].
    """
    def __init__(self,
                 args,
                 linear=True,
                 verbose=True,
                 dropout_rate=0.1,
                 rate_matching=False):
        super().__init__()
        code = args.code
        self.pcm = tf.cast(code.H, dtype=tf.int32)

        # shapes
        self._m, self._n = self.pcm.shape
        self._k = self._n - self._m
        self.dims = args.d_model
        self.t_layers = args.t_layers

        self.rate_matching = rate_matching

        # mask
        self.mask = self.create_mask(self.pcm)
        if verbose:
            for matrix, title in zip([self.pcm, tf.squeeze(self.mask, axis=0)], ["PCM Matrix", "Mask Matrix"]):
                plt.imshow(matrix, cmap='viridis'); plt.colorbar(); plt.title(title); plt.show()

        # layers
        self.node_embeddings = Dense(self.dims, activation='gelu')
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model=args.d_model,
                num_heads=args.heads,
                d_ff=args.d_model * 4,
                linear=linear,
                mask_shape=self.mask.shape,
                mask_division_shape=args.mask_division_shape,
                dropout_rate=dropout_rate,
            )
            for _ in range(self.t_layers)
        ]
        self.forward_channel = Dense(1)
        self.dropout = Dropout(dropout_rate)
        self.to_n = Dense(self._n, activation='gelu')

    def create_mask(self, H):
        """Create an attention mask based on the parity-check matrix (PCM)."""
        # Initialize diagonal identity mask
        mask = tf.eye(2 * self._n - self._k, dtype=tf.float32).numpy()

        for i in range(self._m):
            # Get indices where H[i] == 1
            indices = np.where(H[i].numpy() == 1)[0]  # Convert TensorFlow tensor to NumPy array

            # Nested loop over indices
            for j in indices:
                for k in indices:
                    if j != k:
                        mask[j, k] = 1
                        mask[k, j] = 1
                        mask[self._n + i, j] = 1
                        mask[j, self._n + i] = 1

        # Convert the mask back to a TensorFlow tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        # Expand mask's batch size for tf MHA
        mask = tf.expand_dims(mask, axis=0)  # Shape: (1, n+m, n+m)
        return mask

    def get_syndrome(self, vn_vector, from_llr=True, from_logits=False):
        """ Calculate syndrome (pcm @ r.T = 0) if r is correct in binary """
        assert not (from_llr and from_logits), "Cannot convert from llr and logits at the same time."
        vn_vector = tf.transpose(vn_vector) # (n,b)
        if from_llr: 
            bin_vector = llr_to_bin(vn_vector) 
        elif from_logits:
            bin_vector = logits_to_bin(vn_vector)
        else:
            bin_vector = vn_vector
        return tf.cast( (self.pcm @ bin_vector) % 2, dtype=tf.float32) # (m,n)@(n,b)->(m,b)

    def call(self, x_nodes, training=False):
        """Perform a forward pass through the decoder to generate variable node logits."""
        # Embed cn/vn nodes vector
        x_nodes = self.node_embeddings( x_nodes ) # (b, n+m, hidden_dims)
        # Pass through each encoder block
        for block in self.encoder_blocks:
            x_nodes = block(x_nodes,
                            mask=self.mask,
                            training=training)
        x_nodes = tf.squeeze( self.forward_channel(x_nodes), axis=-1 ) # (b, n+m, hidden_dims)->(b, n+m)
        x_nodes = self.dropout(x_nodes) if training else x_nodes
        vn_logits = self.to_n(x_nodes) # (b, n+m)->(b,n)

        return vn_logits
