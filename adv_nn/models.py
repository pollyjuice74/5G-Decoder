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
        # assert isinstance(code.H, tf.sparse.SparseTensor), "Code's pcm must be sparse."
        self.pcm = tf.cast(code.H, dtype=tf.int32)
        # shapes
        self.m, self.n = self.pcm.shape
        self.k = self.n - self.m
        self.dims = args.d_model
        self.batch_size = args.batch_size

        self.mask = self.create_mask(self.pcm)
        # trans_call layers
        self.src_embed = tf.Variable( tf.random.uniform([self.dims, self.n + self.m, 1]), trainable=True )
        self.decoder = Transformer(args.d_model, args.heads, self.mask, args.t_layers)
        self.to_n = Dense(1)
        self.to_m = Dense(1)
        self.time_embed = Embedding(args.n_steps, args.d_model)
        # diff layers
        self.fc = Dense(1)

        self.betas = tf.constant( tf.linspace(1e-3, 1e-2, args.n_steps)*0 + args.sigma )
        self.betas_bar = tf.constant( tf.math.cumsum(self.betas, 0) )

        self.split_diff = False#args.split_diff
        self.ls_active = args.ls_active

        scheduler = tf.keras.optimizers.schedules.CosineDecay( initial_learning_rate=args.lr, decay_steps=args.epochs ) # 1000 is size of trainloader
        self.optimizer =  tf.keras.optimizers.Adam(learning_rate=scheduler)

    def create_mask(self, H):
        m,n = H.shape
        mask = tf.eye(n+m, dtype=tf.float32) # (n+m, n+m)
        indices = tf.where(H != 0)#H.indices
        cn_con, vn_con = indices[:, 0], indices[:, 1]

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
        r_t_bin = tf.cast(llr_to_bin(r_t), dtype=tf.int32)
        return (self.pcm @ r_t_bin) % 2 # (m,n)@(n,b)->(m,b)

    # Extracts noise estimate z_hat from r
    def tran_call(self, r_t, t):
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.n, self.batch_size)) # (n,b)
        tf.print("Inside Tran call:", r_t, t)
        t = tf.cast(t, dtype=tf.int32)

        # Compute synd and magn
        syndrome = tf.reshape( self.get_syndrome(llr_to_bin(r_t)), (self.pcm.shape[0], self.batch_size) ) # (m,n)@(n,b)->(m,b) check nodes
        magnitude = tf.reshape( tf.abs(r_t), (self.n, self.batch_size) ) #(n,b) variable nodes
        # make sure their the same dtype
        magnitude, syndrome = [ tf.cast(tensor, dtype=tf.float32) for tensor in [magnitude, syndrome] ]

        # Concatenate synd and magn
        nodes = tf.concat([magnitude, syndrome], axis=0) # data for vertices
        nodes = tf.reshape(nodes, (1, self.n+self.m, self.batch_size)) # (1, n+m, b)
        # print(nodes.shape)

        print(self.src_embed.shape)
        # Embedding nodes w/ attn and 'time' (sum syn errs) dims
        nodes_emb = tf.reshape( self.src_embed * nodes, (self.src_embed.shape[0], self.pcm.shape[0]+self.n, self.batch_size) ) # (d,n+m,b)
        time_emb = tf.reshape( self.time_embed(t), (self.src_embed.shape[0], 1, self.batch_size) ) # (d,1,b)
        print(nodes_emb.shape, time_emb.shape)

        # Applying embeds
        emb_t = time_emb * nodes_emb # (d, n+m, b)
        emb_t = tf.transpose(emb_t, (2, 1, 0)) # (d, n+m, b)-> (b, n+m, d)
        print(emb_t.shape)
        logits = self.decoder(emb_t) # (b, n+m, d) # TODO: missing batch dims b
        logits = tf.transpose(logits, (2, 1, 0)) # (b, n+m, d)-> (d, n+m, b)
        print("logits: ", logits.shape)

        # Reduce (d,n+m,d)->(d,n+m)
        # logits = tf.squeeze( self.fc(logits), axis=-1 )
        vn_logits = tf.reshape( logits[:, :self.n, :], (self.n, self.batch_size, self.dims) ) # (n,b, d) take the first n logits from the concatenation
        cn_logits = tf.reshape( logits[:, self.n:, :], (self.m, self.batch_size, self.dims) ) # (m,b, d) take the last m logits from the concatenation
        # print(vn_logits, cn_logits)

        z_hat = tf.squeeze( self.to_n(vn_logits), axis=-1 )# (n,b, d)->(n, b)
        synd = tf.squeeze( self.to_m(cn_logits), axis=-1 )# (m,b, d)->(m, b)
        print(z_hat.shape, synd.shape)

        return z_hat, synd

    # optimal lambda l for theoretical and for error prediction
    def line_search(self, r_t, sigma, err_hat, lin_splits=20):
        l_values =  tf.reshape( tf.linspace(1., 20., lin_splits), (1, 1, lin_splits) )
        r_t, sigma, err_hat = [ tf.expand_dims(tensor, axis=-1) for tensor in [r_t, sigma, err_hat] ]# (n,b, 1)
        # print(f"sigma: {sigma}, err_hat: {err_hat}")

        # Compute theoretical step size w/ ls splits
        z_hat_values = l_values*(sigma*err_hat) # (n,b, l), l is lin_splits
        r_values = llr_to_bin(r_t - z_hat_values) # (n,b, l)
        r_values = tf.reshape(r_values, [r_values.shape[0], -1]) # (n,b*l)
        tf.print("r_values", r_values.shape)

        # sum of synds (m,n)@(n,b*l)->(m,b*l)->(b*l, 1)
        sum_synds = tf.reduce_sum( tf.abs( (self.pcm @ r_values) % 2 ),
                                   axis=0 )
        sum_synds = tf.reshape(sum_synds, (-1, lin_splits)) # (b, l)
        tf.print("In linesearch Sum Syndromes: ", sum_synds)

        # Pick optimal ls value
        if self.model_type=='dis':
             ixs = tf.math.argmin(sum_synds, axis=1, output_type=tf.int32)[:, tf.newaxis] # (b,1) w/ ixs of optimal line search for batch b
        elif self.model_type=='gen':
             ixs = tf.math.argmax(sum_synds, axis=1, output_type=tf.int32)[:, tf.newaxis] # (b,1)
        # print(r_values.shape, z_hat_values.shape, ixs.shape)

        r_values = r_t - z_hat_values
        # (b, l, n) for indexing on l
        r_values, z_hat_values = [ tf.reshape(tensor, [-1, lin_splits, r_values.shape[0]])
                                            for tensor in [r_values, z_hat_values] ]

        # concat range of batch ixs [0,...,n-1] and optimal line search ixs for gather_nd
        indices = tf.concat([ tf.range(ixs.shape[0])[:, tf.newaxis], ixs ],
                                                            axis=-1) # (b,2)

        # print(r_values, z_hat_values, indices)
        # ix on lin_splits w/ gather_nd st. ix,(b, l, n)->(n,b)
        r_t1, z_hat = [ tf.reshape( tf.gather_nd(tensor, indices), (self.n, -1) )
                                             for tensor in [r_values, z_hat_values] ]
        # print(r_t1, z_hat_values)
        return r_t1, z_hat # r at t-1

    def loss_fn(self, synd):
        return tf.reduce_mean(tf.square(synd))

    def train_step(self, llr_ch):
        with tf.GradientTape() as tape:
            _, synd = self.tran_call(llr_ch,
                                     tf.reduce_sum( self.get_syndrome(llr_ch) ))
            loss = self.loss_fn(synd)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, r_t, struct_noise=0, sim_ampl=True):
        t = tf.random.uniform( (c_0.shape[0] // 2 + 1,), minval=0,maxval=self.n_steps, dtype=tf.int32 )
        t = tf.concat([t, self.n_steps - t - 1], axis=0)[:c_0.shape[0]] # reshapes t to size x_0
        t = tf.cast(t, dtype=tf.int32)

        noise_factor = tf.math.sqrt( tf.gather(self.betas_bar, t) )
        noise_factor = tf.reshape(noise_factor, (-1, 1))
        z = tf.random.normal(c_0.shape)
        h = np.random.rayleigh(size=c_0.shape)if sim_ampl else 1.

        added noise to codeword
        c_t = tf.transpose(h * c_0 + struct_noise + (z*noise_factor))
        calculate sum of syndrome
        t = tf.math.reduce_sum( self.get_syndrome( llr_to_bin(tf.sign(c_t)) ), axis=0 ) # (batch_size, 1)

        z_hat = self.tran_call(c_t, t) # model prediction

        if self.model_type=='dis':
            z_mul = c_t * tf.transpose(c_0) # actual noise added through the channel

        elif self.model_type=='gen':
            c_t += z_hat # could contain positive or negative values
            z_mul = c_t * tf.transpose(c_0) # moidfied channel noise st. it will fool the discriminator

        z_mul = tf.reshape(z_mul, (z_hat.shape[0], -1))
        return c_hat, synd #z_hat, llr_to_bin(z_mul), c_t

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
        i = tf.constant(0)  # Initialize loop counter
        z_hat = tf.zeros_like(r_t)  # Placeholder for z_hat

        def condition(i, r_t, z_hat):
            # Loop while i < self.m and syndrome sum is not zero
            return tf.logical_and(i < 1, tf.reduce_sum(self.get_syndrome(r_t)) != 0) # CHANGE 5 TO SELF.M

        def body(i, r_t, z_hat):
            # Perform reverse or split diffusion
            r_t, z_hat = tf.cond(
                tf.logical_not(self.split_diff),
                lambda: self.split_rdiff_call(r_t),
                lambda: self.rev_diff_call(r_t),
            )
            return tf.add(i, 1), r_t, z_hat

        # Run tf.while_loop with the loop variables
        i, final_r_t, final_z_hat = tf.while_loop(
            condition,
            body,
            loop_vars=[i, r_t, z_hat],
            maximum_iterations=self.n_steps,
            # shape_invariants=[i.get_shape(), tf.TensorShape([None, None]), z_hat.get_shape()]
        )

        return final_r_t, final_z_hat, i


    # Refines recieved codeword r at time t
    def rev_diff_call(self, r_t):
        tf.print("Rev def call with line-search...")
        # Make sure r_t and t are compatible
        r_t = tf.reshape(r_t, (self.n, -1)) # (n,b)
        # 'time step' of diffusion is really ix of abs(sum synd errors)
        t = tf.reduce_sum( self.get_syndrome(llr_to_bin(r_t)), axis=0 ) # (m,n)@(n,b)->(m,b)->(1,b)
        t = tf.cast(tf.abs(t), dtype=tf.int32)
        tf.print("syndromes t: ", t)

        # Transformer error prediction
        z_hat_crude, synd = self.tran_call(r_t, t) # (n,1), (m,1)
        tf.print("After Transformer call: ", z_hat_crude)

        # Compute diffusion vars
        sigma = self.get_sigma(t) # theoretical step size
        z_hat = r_t - tf.sign(z_hat_crude * r_t) # (n,1)
        tf.print("sigma: ", sigma)
        # tf.print("z_hat_crude: ", z_hat_crude)
        tf.print("z_hat: ", z_hat)

        r_t1 = r_t - z_hat_crude
        # # Refined estimate of the codeword for the ls diffusion step
        # r_t1, z_hat = self.line_search(r_t, sigma, err_hat) if self.ls_active else 1.
        # tf.print("After linesearch: ", r_t1)

        # Cast both outputs to float32 for consistency
        r_t1, z_hat = [ tf.cast(tensor, tf.float32) for tensor in [r_t1, z_hat] ]
        # # reshape to (n,b) for consistency
        r_t1, z_hat = [ tf.reshape( tensor, (self.n, self.batch_size) )
                                             for tensor in [r_t1, z_hat] ]

        return r_t1, z_hat # r at t-1, both (n,1)

    def split_rdiff_call(self, r_t):
        tf.print("Rev diff call with split diffusion...")
        # Ensure r_t is correctly shaped
        r_t = tf.reshape(r_t, (self.n, -1))  # (n,b)
        print(r_t.shape)
        t = tf.reduce_sum(self.get_syndrome(llr_to_bin(r_t)), axis=0)  # (m,n)@(n,b)->(m,b)->(1,b)
        t = tf.cast(tf.abs(t), dtype=tf.int32)

        # First half-step condition subproblem
        print(r_t.shape, t)
        z_hat_crude, synd = self.tran_call(r_t, t)
        print("fc input: ", (z_hat_crude * self.get_sigma(t)).shape)
        r_t_half = r_t - 0.5 * self.fc(z_hat_crude * self.get_sigma(t))
        print(r_t_half.shape)

        # Full-step diffusion subproblem
        r_t1 = r_t_half + tf.random.normal(r_t_half.shape) * tf.sqrt(self.get_sigma(t))

        # Second half-step condition subproblem
        z_hat_crude_half, synd = self.tran_call(r_t1, t)  # Reuse the second `tran_call`
        r_t1 = r_t1 - 0.5 * self.fc(z_hat_crude_half * self.get_sigma(t))

        # Cast both outputs to float32 for consistency
        r_t1, z_hat_crude_half = [ tf.cast(tensor, tf.float32) for tensor in [r_t1, z_hat_crude_half] ]
        # # reshape to (n,b) for consistency
        r_t1, z_hat_crude_half = [ tf.reshape( tensor, (self.n, self.batch_size) )
                                             for tensor in [r_t1, z_hat_crude_half] ]
        print(r_t1.shape, z_hat_crude_half.shape)
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


   














        
