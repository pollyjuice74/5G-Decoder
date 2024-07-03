import torch
import torch.nn.functional as F
from torch_geometric.nn import GPSConv
from torch_geometric.data import Data

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

## Generative Adversarial Network approach

# Should  use specifically 5G standard LDPC/Polar codes
    # Main algorithms for,
        # LDPC: Belief Propagation
        # Polar: Successive Cancellation Listing, Successive Cancellation

    # Could store different weights for generator and discriminator, 
    # make a super noise generator that will always attack really good at the discriminator 
class DataSet():
    def __init__(self):
        pass
    

# Construct generator (encoder using forward diffusion to simulate channel)
    # By simulating channel it will try to come up with ways to fool discriminator/decoder
    # through noising the original codeword

    # Could make a general generator for GNN, ECCT, DDECCT, and AECC(Adversarial Error Correcting Code) 
    # to compare how different models learn or have weaknesses in decoding. 
class Generator(Layer):
    def __init__(self):
        pass

    def call(self, c):
        emb = 
        return z

    def create_mask(self, H):
        return H_mask
        

# Construct discriminator (decoder using reverse diffusion)
    # Will have to come up with ways to try to decode the noised codeword against specific noise
    # that will be trying to fool it. 

class Discriminator(Layer):
    def __init__(self):
        pass
        
    def call(self, r):
        for i in range(self.pcm.shape[0]):
            c_hat, z_hat,  t = self.diff_call(r)
            
            if (c_hat @ self.pcm)==0:
                return c_hat, z_hat, i            
        return c_hat, z_hat, i

    # Refines r
    def diff_call(self, rt): 
        t = # ix for the 'time step' t in the diffusion
        if self.line_search:
            l = self.line_search()
        return rt_1, z_hat, t # r at time t-1

    # optimal lambda l for theoretical and for error prediction
    def line_search(self):
        pass
        
    # Extracts noise z of r
    def tran_call(self, r):
        return c_hat, z_hat


class E2EModel(tf.keras.Model):
    def __init__(self):
        pass















        
