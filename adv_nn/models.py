import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
import numpy as np

import scipy as sp
from scipy.sparse import issparse, csr_matrix, coo_matrix

from dataset import sign_to_bin, llr_to_bin
from transformer import * 

## Generative Adversarial Network approach

# Should  use specifically 5G standard LDPC/Polar codes
    # Main algorithms for,
        # LDPC: Belief Propagation
        # Polar: Successive Cancellation Listing, Successive Cancellation

    # Could store different weights for generator and discriminator, 
    # make a super noise generator that will always attack really good at the discriminator

# Base model for the Generator and Discriminator models, 
    # contains diffusion computations, line search and creates mask of pcm.

class TransformerDiffusion( Layer ):
    def __init__(self, args):
        super().__init__()
        self.model_type = args.model_type
        self.n_steps = args.n_steps

        code = args.code
        assert issparse(code.H), "Code's pcm must be sparse."
        self.pcm = code.H
        # shapes
        self.m, self.n = self.pcm.shape
        self.k = self.n - self.m

        self.mask = self.create_mask(self.pcm)
        # layers
        self.src_embed = tf.Variable( tf.random.uniform([1, self.n + self.m, args.d_model]), trainable=True )
        self.decoder = Transformer(args.d_model, args.heads, self.mask, args.t_layers)
        self.fc = Dense(1)
        self.to_n = Dense(1)
        self.time_embed = Embedding(args.n_steps, args.d_model)

        self.betas = tf.constant( tf.linspace(1e-3, 1e-2, args.n_steps)*0 + args.sigma )
        self.betas_bar = tf.constant( tf.math.cumsum(self.betas, 0) )

        self.split_diff = False#args.split_diff
        self.ls_active = args.ls_active

    def create_mask(self, H):
        m,n = H.shape
        mask = tf.eye(n+m, dtype=tf.float32) # (n+m, n+m)
        cn_con, vn_con, _ = sp.sparse.find(H)

        for cn, vn_i in zip(cn_con, vn_con):
            # cn to vn connections in the mask
            mask = tf.tensor_scatter_nd_update(mask, [[n+cn, vn_i],[vn_i, n+cn]], [1.0,1.0])

            # distance 2 vn neighbors of vn_i
            related_vns = vn_con[cn_con==cn]
            for vn_j in related_vns:
                mask = tf.tensor_scatter_nd_update(mask, [[vn_i, vn_j],[vn_j, vn_i]], [1.0,1.0])

        # -infinity where mask is not set
        mask = tf.cast( tf.math.logical_not(mask > 0), dtype=tf.float32) # not(mask > 0) for setting non connections to -1e9
        return mask

    def get_sigma(self, t):
        # make sure t is a positive int
        t = tf.cast( tf.abs(t), tf.int32 )
        # gather betas
        betas_t = tf.gather(self.betas, t)
        betas_bar_t = tf.gather(self.betas_bar, t)

        return betas_bar_t * betas_t / (betas_bar_t + betas_t)

    def get_syndrome(self, r_t):
        # Calculate syndrome (pcm @ r = 0) if r is correct in binary
        r_t = tf.reshape(r_t, (self.n, -1)) # (n,b)
        return self.pcm.dot( llr_to_bin( r_t ).numpy() ) % 2 # (m,n)@(n,b)->(m,b)

    # Extracts noise estimate z_hat from r
    def tran_call(self, r_t, t):
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.n, -1)) # (n,b)
        t = tf.cast(t, dtype=tf.int32)

        # Compute synd and magn
        syndrome = tf.reshape( self.get_syndrome(llr_to_bin(r_t)), (self.pcm.shape[0], -1) ) # (m,n)@(n,b)->(m,b) check nodes
        magnitude = tf.reshape( tf.abs(r_t), (self.n, -1) ) #(n,b) variable nodes
        # make sure their the same dtype
        magnitude, syndrome = [ tf.cast(tensor, dtype=tf.float32) for tensor in [magnitude, syndrome] ]

        # Concatenate synd and magn
        nodes = tf.concat([magnitude, syndrome], axis=0) # data for vertices
        nodes = tf.reshape(nodes, (1, self.pcm.shape[0]+self.n, -1)) # (1, n+m, b)
        print(nodes.shape)

        # Embedding nodes w/ attn and 'time' (sum syn errs) dims
        nodes_emb = tf.reshape( self.src_embed * nodes, (self.src_embed.shape[-1], self.pcm.shape[0]+self.n, -1) ) # (d,n+m,b)
        time_emb = tf.reshape( self.time_embed(t), (self.src_embed.shape[-1], 1, -1) ) # (d,1,b)

        # Applying embeds
        emb_t = time_emb * nodes_emb # (d, n+m, b)
        logits = self.decoder(emb_t) # (d, n+m, d) # TODO: missing batch dims b
        print(emb_t, logits)

        # Reduce (d,n+m,d)->(d,n+m)
        logits = tf.squeeze( self.fc(logits), axis=-1 )
        node_logits = tf.reshape( logits[:, :self.n], (self.n, -1) ) # (n,d) take the first n logits from the concatenation
        # (n,d)->(n,)
        z_hat = self.to_n(node_logits)
        print(logits.shape, z_hat.shape)
        return z_hat

    # optimal lambda l for theoretical and for error prediction
    def line_search(self, r_t, sigma, err_hat, lin_splits=20):
        l_values =  tf.reshape( tf.linspace(1., 20., lin_splits), (1, 1, lin_splits) )
        r_t, sigma, err_hat = [ tf.expand_dims(tensor, axis=-1) for tensor in [r_t, sigma, err_hat] ]# (n,b, 1)
        print(f"sigma: {sigma}, err_hat: {err_hat}")

        # Compute theoretical step size w/ ls splits
        z_hat_values = l_values*(sigma*err_hat) # (n,b, l), l is lin_splits
        r_values = llr_to_bin(r_t - z_hat_values) # (n,b, l)
        # sum of synds (m,n)@(n,b*l)->(m,b*l)->(b*l, 1)
        sum_synds = tf.reduce_sum( tf.abs( self.pcm.dot( tf.squeeze(r_values, axis=1) ) % 2 ),
                                   axis=0 )[:, tf.newaxis]
        print(sum_synds.shape)

        # Pick optimal ls value
        if self.model_type=='dis':
             ixs = tf.math.argmin(sum_synds, axis=0, output_type=tf.int32) # (b,1) w/ ixs of optimal line search for batch b
        elif self.model_type=='gen':
             ixs = tf.math.argmax(sum_synds, axis=0, output_type=tf.int32) # (b,1)

        print(r_values.shape, z_hat_values.shape, ixs.shape)
        # (b, l, n) for indexing on l
        r_values, z_hat_values = [ tf.transpose(tensor, perm=[1,2,0])
                                            for tensor in [r_values, z_hat_values] ]

        # concat range of batch ixs [0,...,n-1] and optimal line search ixs in gather_nd
        indices = tf.concat( [tf.range(ixs.shape[0]), ixs], axis=-1) # (b,2)

        # print(r_values, z_hat_values, indices)
        # ix on lin_splits w/ gather_nd st. ix,(b, l, n)->(n,b)
        r_t1, z_hat = [ tf.reshape( tf.gather_nd(tensor, indices), (self.n, -1) )
                                             for tensor in [r_values, z_hat_values] ]
        print(r_t1, z_hat_values)
        return r_t1, z_hat # r at t-1

    # def train(self, r_t, struct_noise=0, sim_ampl=True):
    #     # t = tf.random.uniform( (c_0.shape[0] // 2 + 1,), minval=0,maxval=self.n_steps, dtype=tf.int32 )
    #     # t = tf.concat([t, self.n_steps - t - 1], axis=0)[:c_0.shape[0]] # reshapes t to size x_0
    #     # t = tf.cast(t, dtype=tf.int32)

    #     # noise_factor = tf.math.sqrt( tf.gather(self.betas_bar, t) )
    #     # noise_factor = tf.reshape(noise_factor, (-1, 1))
    #     # z = tf.random.normal(c_0.shape)
    #     # h = np.random.rayleigh(size=c_0.shape)if sim_ampl else 1.

    #     # added noise to codeword
    #     # c_t = tf.transpose(h * c_0 + struct_noise + (z*noise_factor))
    #     # calculate sum of syndrome
    #     t = tf.math.reduce_sum( self.get_syndrome( llr_to_bin(tf.sign(c_t)) ), axis=0 ) # (batch_size, 1)

    #     z_hat = self.tran_call(c_t, t) # model prediction

    #     if self.model_type=='dis':
    #         z_mul = c_t * tf.transpose(c_0) # actual noise added through the channel

    #     elif self.model_type=='gen':
    #         c_t += z_hat # could contain positive or negative values
    #         z_mul = c_t * tf.transpose(c_0) # moidfied channel noise st. it will fool the discriminator

    #     z_mul = tf.reshape(z_mul, (z_hat.shape[0], -1))
    #     return c_hat, synd #z_hat, llr_to_bin(z_mul), c_t

# Construct discriminator (decoder using reverse diffusion)
    # Will have to come up with ways to try to decode the noised codeword against specific noise
    # that will be trying to fool it. 
    
    # For optimization:
        # use Linformer having a O(n) on top of already improved complexity using the pcm mask
        # use split diffusion to improve accuracy and efficiency by guiding model rather than EMA

class Decoder( TransformerDiffusion ):
    def __init__(self, args):
        super().__init__(args)

    # 'test' function
    def call(self, r_t):
        for i in range(self.m):
            print(r_t.shape)
            # both (n,)
            r_t, z_hat = self.rev_diff_call(r_t) if not self.split_diff else self.split_rdiff_call(r_t)

            # Check if synd is 0 return r_t
            if tf.reduce_sum( self.get_syndrome(r_t) ) == 0:
                return r_t, z_hat, i

        return r_t, z_hat, i

    # Refines recieved codeword r at time t
    def rev_diff_call(self, r_t):
        print("Rev def call...")
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.n, -1)) # (n,b)
        # 'time step' of diffusion is really ix of abs(sum synd errors)
        t = tf.reduce_sum( self.get_syndrome(llr_to_bin(r_t)), axis=0 ) # (m,n)@(n,b)->(m,b)->(1,b)
        t = tf.cast(tf.abs(t), dtype=tf.int32)

        # Transformer error prediction
        z_hat_crude = self.tran_call(r_t, t) # (n,1)
        print("z_hat_crude: ", z_hat_crude)

        # Compute diffusion vars
        sigma = self.get_sigma(t) # theoretical step size
        print("sigma: ", sigma)
        err_hat = r_t - tf.sign(z_hat_crude * r_t) # (n,1)

        # Refined estimate of the codeword for the ls diffusion step
        r_t1, z_hat = self.line_search(r_t, sigma, err_hat) if self.ls_active else 1.
        # r_t1[t==0] = r_t[t==0] # if cw has 0 synd. keep as is

        return r_t1, z_hat # r at t-1, both (n,1)

    def split_rdiff_call(self, r_t):
        print("Rev diff call with split diffusion...")
        # Ensure r_t is correctly shaped
        r_t = tf.reshape(r_t, (self.n, -1))  # (n,b)
        t = tf.reduce_sum(self.get_syndrome(llr_to_bin(r_t)), axis=0)  # (m,n)@(n,b)->(m,b)->(1,b)
        t = tf.cast(tf.abs(t), dtype=tf.int32)
        
        # First half-step condition subproblem
        z_hat_crude = self.tran_call(r_t, t)
        r_t_half = r_t - 0.5 * self.fc(z_hat_crude * self.get_sigma(t))
        
        # Full-step diffusion subproblem
        r_t1 = r_t_half + tf.random.normal(r_t_half.shape) * tf.sqrt(self.get_sigma(t))
        
        # Second half-step condition subproblem
        z_hat_crude_half = self.tran_call(r_t1, t)  # Reuse the second `tran_call`
        r_t1 = r_t1 - 0.5 * self.fc(z_hat_crude_half * self.get_sigma(t))
        
        return r_t1, z_hat_crude_half  # r at t-1, both (n,1)
        
# Construct generator (encoder using forward diffusion to simulate channel)
    # By simulating channel it will try to come up with ways to fool discriminator/decoder
    # through noising the original codeword

    # Could make a general generator for GNN, ECCT, DDECCT, and AECC(Adversarial Error Correcting Code) 
    # to compare how different models learn or have weaknesses in decoding. 

class Generator( TransformerDiffusion ):
    def __init__(self):
        super().__init__()

    # 'test' function
    def call(self, c_0, z):
        c_t = c_0
        
        for i in range(self.pcm.shape[0]):
            c_t, z_G = self.fwd_diff_call(c_t)   
           
        assert z_G==(c_t-c_0), "Cumulative z_G should be the same as c_t-c_0" 
        return z_G

    def fwd_diff_call(self, c_t):
        t = ( self.pcm @ to_bin(r_t) ).sum()
        z_G = self.tran_call(c_t, t)
        
        sigma = self.get_sigma(t)
        noise = c_t - tf.sign(c_t*z_G)
        l = self.line_search(sigma, noise, model_type='gen') if self.ls_active else 1.
        
        c_t =  c_t + (l * sigma * noise)
        return c_t, z_G


   














        
