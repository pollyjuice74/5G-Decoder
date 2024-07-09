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

class FEC_Dataset(tf.data.Dataset):
    def __new__(cls, code, sigma=0.1, length=1000, zero_cw=True, ones_m=False, flip_cw=False):
        k, n = code.k, code.n
        specs = [ tf.TensorSpec(shape=(k if i==0 else \
                                       n-k if i==7 else \
                                       n for i in range(8))) ]
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tuple(specs),
            args=(code.G, code.H, k, n, sigma, length, zero_cw, ones_m, flip_cw)
        )
  
    @staticmethod
    def _generator(G,H, k,n, sigma, length, zero_cw, ones_m, flip_cw):
        for _ in range(length):
            # raise print("Zero cw and all ones m are not compatible at the same time")
            if zero_cw is None and not ones_m:
                m = tf.squeeze(tf.random.uniform((1,k), minval=0, maxval=2, dtype=tf.int32))
                x = (m @ G) % 2
            else: # SET TO TRUE
                m = tf.zeros((k,), dtype=tf.int64)
                x = tf.zeros((n,), dtype=tf.int64)

            # make all 1s message
            if ones_m:
                m = tf.squeeze(tf.ones((1,k), dtype=tf.int64))
                x = (m @ G) % 2

            # flip binary cw
            if flip_cw:
                x = 1 - x

            # Make noise
            std_noise = random.choice(sigma)
            z = tf.random.normal((args.code.n,), std_noise)
            # Convert y to sign and add noise
            h=1
            y = h*bin_to_sign(x) + z

            # Sign to LLR conversion
            var = std_noise ** 2

            # x,y to llrs
            x = bin_to_sign(x)
            x_llr = sign_to_llr(x, var)
            y_llr = sign_to_llr(y, var)

            magnitude = tf.abs(y)
            syndrome = ( tf.cast(sign_to_bin(tf.sign(y)), tf.int64) @ H ) % 2
            syndrome = bin_to_sign(syndrome)

            yield cast_to_float32(m, x, z, y, x_llr, y_llr, magnitude, syndrome)

@staticmethod
def cast_to_float32(*args):
    return tuple(tf.cast(arg, tf.float32) for arg in args)

@staticmethod
def sign_to_llr(bpsk_vect, noise_variance):
    return 2 * bpsk_vect / noise_variance

@staticmethod
def bin_to_sign(x):
    return 2 * tf.cast(x, tf.float32) - 1

@staticmethod
def sign_to_bin(x):
    return (x + 1) // 2


class Args():
    def __init__(self, model_type, code_type='LDPC', n=121, k=80,  ls_active=True, sigma=0.1, beta_steps=10, t_layers=1, d_model=8):
        assert model_type in ['gen', 'dis'], "Type must be: 'gen', Generator or 'dis', Discriminator."
        assert code_type in ['POLAR', 'BCH', 'CCSDS', 'LDPC', 'MACKAY'], "Invalid linear code type."
        
        self.model_type = model_type
        self.code_type = code_type
        self.code = self.get_code(n,k)
        self.mask = self.create_mask()
        
        self.ls_active = True
        self.sigma = sigma
        self.beta_steps = beta_steps
        self.t_layers = t_layers # transformer layers
        self.d_model = d_model
        self.N_steps = self.code.H.shape[0]+5 # Number of diffusion steps
    
    def create_mask(self):
        H = self.code.H
        H_mask = H
        return H_mask

    def get_code(self, n, k):
        class Code():
            pass
        code = Code()
        code.n, code.k = n, k
        code.code_type = self.code_type
    
        G, H = Get_Generator_and_Parity(code)
        code.G, code.H = tf.convert_to_tensor(G), tf.convert_to_tensor(H)
        return code
        

# Base model for the Generator and Discriminator models, 
    # contains diffusion computations, line search and creates mask of pcm.

class TransformerDiffusion():
    def __init__(self):
        pass

    # optimal lambda l for theoretical and for error prediction
    def line_search(self, factor, err_hat):
        l_values = tf.linespace(1., 20., 20).reshape(1,1,20)
        syndromes = to_bin(r_t - l_values*(factor*err_hat)) @ self.pcm
        ix = syndromes.argmin() # get the index of the min syndrome
        return l_values[ix]


# Construct discriminator (decoder using reverse diffusion)
    # Will have to come up with ways to try to decode the noised codeword against specific noise
    # that will be trying to fool it. 
    
    # For optimization:
        # use Linformer having a O(n) on top of already improved complexity using the pcm mask
        # use split diffusion to improve accuracy and efficiency by guiding model rather than EMA

class Discriminator(TransformerDiffusion):
    def __init__(self, args):
        self.betas = tf.linespace(1e-3, 1e-2, args.beta_steps)*0 + args.sigma 
        self.betas_bar = tf.math.cumsum(self.betas, 0)
        self.ls_active = args.ls_active
        code = args.code
        
        self.src_embed = tf.Variable( tf.random.uniform(code.n + code.m, args.d_model), trainable=True )
        self.decoder = Transformer(args.mask, args.t_layers)
        self.fc = tf.keras.Sequential([ Dense(args.d_model, 1) ])
        self.to_n = Dense(code.n + code.m, code.n) 
        self.time_embed = Embedding(args.beta_steps, args.d_model)
       
    def call(self, r_t):
       for i in range(self.pcm.shape[0]):
           r_t, z_hat, t = self.rev_diff_call(r_t)
        
           if (r_t @ self.pcm)==0:
               return r_t, z_hat, t, i            
       return r_t, z_hat, t, i
    
    # Refines recieved codeword r at time t
    def rev_diff_call(self, r_t): 
        t = ( self.pcm @ to_bin(r_t) ).sum() # ix for the 'time step' t in the diffusion # 'time step' t is really a error estimate of the syndrome ###
        z_hat = self.rev_tran_call(r_t, t)
        
        factor = ( self.betas_bar[t]*self.beta[t] / (self.betas_bar[t] + self.beta[t]) ) # theoretical step size
        err_hat = r_t - tf.sign(z_hat*rt)
        l = self.line_search() if self.ls_active else 1.
    
        r_t1 = r_t - (l * factor * err_hat) # refined estimate of the codeword for the diffusion step
        # r_t1[t==0] = r_t[t==0] # if cw has 0 synd. keep as is
        return r_t1, z_hat, t # r at time t-1

    # Extracts noise estimate z_hat from r
    def rev_tran_call(self, r_t, t):
        syndrome = self.pcm @ to_bin(r_t) #(m,1) check nodes
        magnitude = tf.abs(r_t) #(n,1) variable nodes

        nodes = tf.concat([magnitude, syndrome]) # data for vertices, shape (n+m,1)
        
        emb = self.src_emb * nodes
        time_emb = self.time_embed(t) # t is really error estimate of syndrome ###
        
        emb_t = time_emb * emb
        emb_t = self.decoder(emb_t, self.mask, time_emb)
        z_hat = self.fc(emb_t)
        return z_hat


# Construct generator (encoder using forward diffusion to simulate channel)
    # By simulating channel it will try to come up with ways to fool discriminator/decoder
    # through noising the original codeword

    # Could make a general generator for GNN, ECCT, DDECCT, and AECC(Adversarial Error Correcting Code) 
    # to compare how different models learn or have weaknesses in decoding. 

class Generator(Layer):
    def __init__(self):
        pass

    def call(self, c):
        pass
        # emb = 
        # return z

    def fwd_diff_call(self,):
        pass
        
    def fwd_tran_call(self,):
        pass

        

   














        
