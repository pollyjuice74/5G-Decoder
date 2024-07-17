import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
import numpy as np

from dataset import sign_to_bin
from decoder import * 

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
        self.pcm = tf.cast(code.H, dtype=tf.float32)

        self.mask = self.create_mask(code.H, args.n_rings)
        self.src_embed = tf.Variable( tf.random.uniform([1, code.n + code.m, args.d_model]), trainable=True )
        self.decoder = Transformer(args.d_model, args.heads, self.mask, args.t_layers)
        self.fc = Dense(1)
        self.to_n = Dense(1)
        self.time_embed = Embedding(args.n_steps, args.d_model)

        self.betas = tf.constant( tf.linspace(1e-3, 1e-2, args.n_steps)*0 + args.sigma )
        self.betas_bar = tf.constant( tf.math.cumsum(self.betas, 0) )
        self.ls_active = args.ls_active

    def get_sigma(self, t):
        # make sure t is a positive int
        t = tf.cast( tf.abs(t), tf.int32 ) 
        # gather betas
        betas_t = tf.gather(self.betas, t)
        betas_bar_t = tf.gather(self.betas_bar, t)

        return betas_bar_t * betas_t / (betas_bar_t + betas_t)

    def get_syndrome(self, r_t):
        # Calculate syndrome (pcm @ r = 0) if r is correct in binary
        r_t = tf.reshape(r_t, (self.pcm.shape[1], -1)) # (n,b)
        return tf.einsum('mn,nb->mb', self.pcm, sign_to_bin(r_t)) % 2

    # optimal lambda l for theoretical and for error prediction
    def line_search(self, r_t, sigma, err_hat, lin_splits=20):
        l_values =  tf.reshape( tf.linspace(1., 20., lin_splits), (1, 1, lin_splits) )
        r_t, sigma, err_hat = [ tf.expand_dims(tensor, axis=-1) for tensor in [r_t, sigma, err_hat] ]# (n,b, 1)

        # Compute theoretical step size w/ ls splits
        z_hat_values = l_values*(sigma*err_hat) # (n,b, l), l is lin_splits
        r_values = sign_to_bin(r_t - z_hat_values) # (n,b, l)

        # Compute sum of synds
        sum_synds = tf.reduce_sum( tf.einsum('mn,nbl->mbl', self.pcm, r_values) % 2, axis=0 ) # (m,n)@(n,b, l)->(m,b, l)->(1,b, l)

        # Pick optimal ls value
        if self.model_type=='dis':
             ixs = tf.math.argmin(sum_synds, axis=-1, output_type=tf.int32)[:, tf.newaxis] # (b,1) w/ ixs of optimal line search for batch b
        elif self.model_type=='gen':
             ixs = tf.math.argmax(sum_synds, axis=-1, output_type=tf.int32)[:, tf.newaxis] # (b,1)
        
        # (b, l, n) for indexing on l
        r_values, z_hat_values = [ tf.transpose(tensor, perm=[1,2,0]) 
                                            for tensor in [r_values, z_hat_values] ]

        # concat range of batch ixs [0,...,n-1] and optimal line search ixs in gather_nd
        indices = tf.concat([ tf.range(ixs.shape[0])[:, tf.newaxis], ixs], axis=-1) # (b,2)

        # print(r_values, z_hat_values, indices)
        # ix on lin_splits w/ gather_nd st. ix,(b, l, n)->(n,b)
        r_t1, z_hat = [ tf.reshape( tf.gather_nd(tensor, indices), (self.pcm.shape[1], -1) )
                                             for tensor in [r_values, z_hat_values] ]
        return r_t1, z_hat # r at t-1

    def create_mask(self, H, n_rings=1):
        m,n = H.shape
        mask = tf.eye(n+m, dtype=tf.float32)
        init_H = True

        for _ in range(n_rings):
            mask = tf.identity(mask)
            if init_H:
                mask = self._extend_connectivity(mask, H, init_H=init_H)
                init_H = False
            else:
                mask = self._extend_connectivity(mask, init_H=init_H, m=m,n=n)

        src_mask = tf.cast( tf.math.logical_not(mask > 0), dtype=tf.float32) # not(mask > 0) for setting non connections to -1e9
        return src_mask

    def _extend_connectivity(self, mask, H=None, init_H=False, m=None, n=None):
        if init_H:
            m, n = H.shape
        else:
            m, n = int(m), int(n)  # Ensure m and n are integers

        loop_len = int(m) if init_H else int(n + m)

        for i in range(loop_len):
            indices = tf.where(H[i] > 0) if init_H else tf.where(mask[i] > 0)
            for j in indices:
                j = j[0]
                ixs = [ [j,n+i],[n+i,j] ] if init_H else [ [i,j],[j,i] ]
                mask = tf.tensor_scatter_nd_update(mask, ixs, [1.0, 1.0])

        return mask

    def train(self, c_0, struct_noise=0, sim_ampl=True):
        t = tf.random.uniform( (c_0.shape[0] // 2 + 1,), minval=0,maxval=self.n_steps, dtype=tf.int32 )
        t = tf.concat([t, self.n_steps - t - 1], axis=0)[:c_0.shape[0]] # reshapes t to size x_0
        t = tf.cast(t, dtype=tf.int32)

        noise_factor = tf.math.sqrt( tf.gather(self.betas_bar, t) )
        noise_factor = tf.reshape(noise_factor, (-1, 1))
        z = tf.random.normal(c_0.shape)
        h = np.random.rayleigh(size=c_0.shape)if sim_ampl else 1.

        # added noise to codeword
        c_t = tf.transpose(h * c_0 + struct_noise + (z*noise_factor)) 
        # calculate sum of syndrome
        t = tf.math.reduce_sum( self.get_syndrome( sign_to_bin(tf.sign(c_t)) ), axis=0 ) # (batch_size, 1)

        z_hat = self.tran_call(c_t, t) # model prediction

        if self.model_type=='dis':
            z_mul = c_t * tf.transpose(c_0) # actual noise added through the channel

        elif self.model_type=='gen':
            c_t += z_hat # could contain positive or negative values
            z_mul = c_t * tf.transpose(c_0) # moidfied channel noise st. it will fool the discriminator

        z_mul = tf.reshape(z_mul, (z_hat.shape[0], -1))
        return z_hat, sign_to_bin(z_mul), c_t

    # Extracts noise estimate z_hat from r
    def tran_call(self, r_t, t):
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.pcm.shape[1], -1)) # (n,b)
        t = tf.cast(t, dtype=tf.int32)

        # Compute synd and magn 
        syndrome = tf.reshape( self.get_syndrome(sign_to_bin(r_t)), (self.pcm.shape[0], -1) ) # (m,n)@(n,b)->(m,b) check nodes
        magnitude = tf.reshape( tf.abs(r_t), (self.pcm.shape[1], -1) ) #(n,b) variable nodes

        # Concatenate synd and magn 
        nodes = tf.concat([magnitude, syndrome], axis=0) # data for vertices
        nodes = tf.reshape(nodes, (1, self.pcm.shape[0]+self.pcm.shape[1], -1)) # (1, n+m, b)

        # Embedding nodes w/ attn and 'time' (sum syn errs) dims
        nodes_emb = tf.reshape( self.src_embed * nodes, (self.src_embed.shape[-1], self.pcm.shape[0]+self.pcm.shape[1], -1) ) # (d,n+m,b)
        time_emb = tf.reshape( self.time_embed(t), (self.src_embed.shape[-1], 1, -1) ) # (d,1,b)

        # Applying embeds
        emb_t = time_emb * nodes_emb # (d, n+m, b)
        logits = self.decoder(emb_t) # (d, n+m, d) # TODO: missing batch dims b

        # Reduce (d,n+m,d)->(d,n+m)
        logits = tf.squeeze( self.fc(logits), axis=-1 )
        # (d,n+m)->(n,) take the first n logits from the concatenation
        z_hat = self.to_n( logits[:self.pcm.shape[1]] ) 
        return z_hat

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
       for i in range(self.pcm.shape[0]):
           print(r_t.shape)
           r_t, z_hat = self.rev_diff_call(r_t) # both (n,)

           # Check if synd is 0 return r_t
           if tf.reduce_sum( self.get_syndrome(r_t) ) == 0:
               return r_t, z_hat, i

       return r_t, z_hat, i

    # Refines recieved codeword r at time t
    def rev_diff_call(self, r_t):
        print("Rev def call...")
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.pcm.shape[1], -1)) # (b, n)
        # 'time step' of diffusion is really ix of abs(sum synd errors)
        t = tf.reduce_sum( self.get_syndrome(sign_to_bin(r_t)), axis=0 ) # (m,n)@(n,b)->(m,b)->(1,b)
        t = tf.cast(tf.abs(t), dtype=tf.int32)

        # Transformer error prediction
        z_hat_crude = self.tran_call(r_t, t) # (n,1)

        # Compute diffusion vars 
        sigma = self.get_sigma(t) # theoretical step size
        err_hat = r_t - tf.sign(z_hat_crude * r_t) # (n,1)

        # Refined estimate of the codeword for the ls diffusion step
        r_t1, z_hat = self.line_search(r_t, sigma, err_hat) if self.ls_active else 1.
        # r_t1[t==0] = r_t[t==0] # if cw has 0 synd. keep as is

        return r_t1, z_hat # r at t-1, both (n,1)
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


   














        
