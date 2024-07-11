import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from models import *


class Channel(tf.keras.Model):
    def __init__(self, generator, discriminator, args):
        self.gen = Generator(args) 
        self.dis = Discriminator(args) # ADD trainable/not-trainable option 
        self.bce = BinaryCrossentropy(from_logits=True) 

    # 'test' function
    def call(self, c, z=0):
        z_G = self.gen(c, z)
        r = c + z + z_G if z else c + z_G # z could be structured noise process or only the generator creates noise
        
        c_hat, z_hat, t, i = self.dis(r)

        ber = BER(c_hat, c)
        fer = BLER(c_hat, c)
        print(f"ber: {ber}, bler: {bler}")

        return c, c_hat, #data #, ber, fer, z_G

    def train(self, c, z):
        z_G, _, r = self.gen.train(c, struct_noise=z)
        z_D, _, c_hat = self.dis.train(r)
        
        loss_dis = self.loss_dis(c_hat, c)
        loss_gen = self.loss_gen(ber, fer, z_G) # performance of the decoder, noise created
        loss = loss_dis + loss_gen


# pytorch or tensorflow?

# Channel should take in a generator G() (n,n), discriminator D() (n,n), codeword c (n,1), noise z (n,1), pcm H (m,n), 
    # such that:
      # c_hat, z_hat = D(G(c, z, H)), 
                                  # where z is optional or 0,
                                  # D can be trainable or have static weights,
                                  # and D recieves H from G.

# TRAIN
  # for {batch_ix, H, (m, c, z, r, c_llr, r_llr, magnitude, syndrome)} in enumerate(test_data):
      # z is either 0 or a structured noise process
      # z_G ~ G(c, H) that could also optionally have knowledge of z such that,
      # z_out = z + z_G

      # r = c + z_out
      # c_hat, z_hat = D(r, H) 


# TEST
  # for {batch_ix, H, (m, c, z, r, c_llr, r_llr, magnitude, syndrome)} in enumerate(test_data):

      # for eb_no in Eb/Nos:
          # sigma calculated using eb_no
          # z ~ N(0, sigma) normally dist.
          # r = c + z
          
          # c_hat, z_hat = D(r, H) 
          
          # print(c_hat, c)
          # print(BER(c_hat, c), FER(c_hat, c))     bit error rate, frame error rate


# Eb = b/s, bits per second transmitted through a channel 
# No = Hz, No=k*T for every T temperature units and k is Boltzmann's constant meaning more energy 

# For context, we are training with eb_no of 4,5,6 
# and cities, suburbs, rural areas, mountain areas have eb_nos 
# of  8-15,   10-18,   12-20,       5-12     respectively.

# Frame Error Rate is the ratio of errored frames to 
# transmitted frames where frames are structured binary blocks 
# of data used for transmission. They have a certain amount of
# bits per frame.

# Bit Error Rate is the ratio of errored bits transmitted 
# to total bits transmitted.


# Game Theory Aproach

# G has a set (S,A,K) containing strategies S, actions A and knowledge K,
# D also has an (S,A,K).

# Strategies:
    # G can create a real vector z_G of shape (n,1).
    # D can create a real vectors c_hat, z_hat of shape (n,1).
        
# Actions:
    # G's z_G is constrained by some ability to fool D (make D have more errors than with standard AWGN), 
        # by some factor and by some standard deviation.
    # D's c_hat, z_hat are constrained by some ability to decode an r = c + z + z_G, 
        # into a very close aproximation of c and z 

# Knowledge: 
    # H is common knowledge to both parties, 
    # c,z are known to G but not D, 
    # r is known to D and G.
