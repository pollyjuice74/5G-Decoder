import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
import numpy as np

from decoder import * 

from DDECC.src.codes import Get_Generator_and_Parity

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
        code = args.code
        self.n_steps = args.n_steps
        
        self.betas = tf.linspace(1e-3, 1e-2, args.beta_steps)*0 + args.sigma 
        self.betas_bar = tf.math.cumsum(self.betas, 0)
        self.ls_active = args.ls_active
        
        self.mask = args.mask
        self.src_embed = tf.Variable( tf.random.uniform([code.n + code.m, args.d_model]), trainable=True )
        self.decoder = Transformer(args.mask, args.t_layers)
        self.fc = tf.keras.Sequential([ Dense(1) ])
        self.to_n = Dense(code.n + code.m) 
        self.time_embed = Embedding(args.beta_steps, args.d_model)

    # Extracts noise estimate z_hat from r
    def tran_call(self, r_t, t):
        syndrome = self.pcm @ to_bin(r_t) #(m,1) check nodes
        magnitude = tf.abs(r_t) #(n,1) variable nodes

        nodes = tf.concat([magnitude, syndrome]) # data for vertices, shape (n+m,1)
        
        emb = self.src_emb * nodes
        time_emb = self.time_embed(t) # t is really error estimate of syndrome ###
        
        emb_t = time_emb * emb
        emb_t = self.decoder(emb_t, self.mask, time_emb)
        z_hat = self.fc(emb_t)
        return z_hat 
        
    # optimal lambda l for theoretical and for error prediction
    def line_search(self, sigma, err_hat, model_type='dis'):
        assert model_type in ['dis','gen'], "Invalid model type."
        l_values = tf.linespace(1., 20., 20).reshape(1,1,20)
        syndromes = to_bin(r_t - l_values*(sigma*err_hat)) @ self.pcm
        
        if model_type=='dis': 
             ix = syndromes.argmin() 
        else: 
             ix = syndromes.argmax() 
            
        return l_values[ix]

    def train(self, x_0, sim_ampl=True, model_type='dis'):
        assert model_type in ['dis','gen'], "Invalid model type."
        
        t = tf.keras.random.randint( (x_0.shape[0] // 2 + 1,), minval=0,maxval=self.n_steps )
        t = tf.concat([t, self.n_steps - t - 1], axis=0)[:x_0.shape[0]] # reshapes t to size x_0
        t = tf.cast(t, dtype=tf.int32)
        
        z = tf.random.normal( (x_0.shape) )
        noise_factor = tf.math.sqrt(self.betas_bar[t])
        h = tf.random.rayleigh( (x_0.shape) ) if sim_ampl else 1.
        
        x_t = h * x_0 + (z*noise_factor)
        sum_syn = tf.math.reduce_sum( (x_t @ self.pcm) % 2 ) # sum syndrome
        
        z_hat = self.tran_call(x_t, sum_syn)
        
        if model_type=='dis':
            z_mul = x_t * x_0
            
        elif model_type=='gen':
            x_t += z_hat # could contain positive or negative values
            z_mul = x_t * x_0 # moidfied channel noise st. it will fool the discriminator
            
        return z_hat, to_bin(z_mul)

    def get_sigma(self, t):
        return self.betas_bar[t]*self.beta[t] / (self.betas_bar[t] + self.beta[t]) 

# Construct discriminator (decoder using reverse diffusion)
    # Will have to come up with ways to try to decode the noised codeword against specific noise
    # that will be trying to fool it. 
    
    # For optimization:
        # use Linformer having a O(n) on top of already improved complexity using the pcm mask
        # use split diffusion to improve accuracy and efficiency by guiding model rather than EMA

class Discriminator( TransformerDiffusion ):
    def __init__(self):
        super().__init__()

    # 'test' function
    def call(self, r_t):
       for i in range(self.pcm.shape[0]):
           r_t, z_hat, t = self.rev_diff_call(r_t)
        
           if (r_t @ self.pcm)==0:
               return r_t, z_hat, t, i            
       return r_t, z_hat, t, i
    
    # Refines recieved codeword r at time t
    def rev_diff_call(self, r_t): 
        t = ( self.pcm @ to_bin(r_t) ).sum() # ix for the 'time step' t in the diffusion # 'time step' t is really a error estimate of the syndrome ###
        z_hat = self.tran_call(r_t, t)
        
        sigma = self.get_sigma(t) # theoretical step size
        err_hat = r_t - tf.sign(z_hat*r_t)
        l = self.line_search(sigma, err_hat) if self.ls_active else 1.
    
        r_t1 = r_t - (l * sigma * err_hat) # refined estimate of the codeword for the diffusion step
        # r_t1[t==0] = r_t[t==0] # if cw has 0 synd. keep as is
        return r_t1, z_hat, t # r at time t-1



        

# Construct generator (encoder using forward diffusion to simulate channel)
    # By simulating channel it will try to come up with ways to fool discriminator/decoder
    # through noising the original codeword

    # Could make a general generator for GNN, ECCT, DDECCT, and AECC(Adversarial Error Correcting Code) 
    # to compare how different models learn or have weaknesses in decoding. 

class Generator( TransformerDiffusion ):
    def __init__(self):
        super().__init__()

    # 'test' function
    def call(self, c_0):
        c_t = c_0
        for i in range(self.pcm.shape[0]):
            c_t, z_G = self.fwd_diff_call(c_t, z_G)   
           
        assert z_G==(c_t-c_0), "Cumulative z_G should be the same as c_t-c_0" 
        return z_G

    def fwd_diff_call(self, c_t, z_G):
        t = ( self.pcm @ to_bin(r_t) ).sum()
        z_G = self.tran_call(c_t, t)
        
        sigma = self.get_sigma(t)
        noise = c_t - tf.sign(c_t*z_G)
        l = self.line_search(sigma, noise, model_type='gen') if self.ls_active else 1.
        
        c_t =  c_t + (l * sigma * noise)
        return c_t, z_G


   














        
