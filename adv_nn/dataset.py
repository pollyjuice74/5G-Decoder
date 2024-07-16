import tensorflow as tf
import numpy as np
import random


class FEC_Dataset(tf.data.Dataset): 
    @staticmethod
    def _generator(G,H, k,n, sigma, length, zero_cw, ones_m, flip_cw, sim_ampl=True):
        for _ in range(length):
            # Random bits
            if zero_cw is None and not ones_m:
                m = tf.squeeze(tf.random.uniform((1,k), minval=0, maxval=2, dtype=tf.int32))
                c = (m @ G) % 2
            else: # SET TO TRUE
                m = tf.zeros((k,), dtype=tf.int64)
                c = tf.zeros((n,), dtype=tf.int64)

            # make all 1s message
            if ones_m:
                m = tf.squeeze(tf.ones((1,k), dtype=tf.int64))
                c = (m @ G) % 2

            # flip binary cw
            if flip_cw:
                c = 1 - c

            # Make noise
            std_noise = random.choice(sigma)
            z = tf.random.normal((n,), std_noise)
            h = np.random.rayleigh(std_noise, (n,)) if sim_ampl else 1.  # simulates signal amplitude of multipath propagation signals
            
            # Convert y to sign and add noise
            y = bin_to_sign(c) * h + z

            c = bin_to_sign(c)
            var = std_noise ** 2
            # x,y to llrs
            c_llr = sign_to_llr(c, var)
            y_llr = sign_to_llr(y, var)

            y_reshaped = tf.reshape(y, [-1, 1])
            syndrome = ( H @ tf.cast(sign_to_bin(tf.sign(y_reshaped)), tf.int64) ) % 2
            syndrome = tf.squeeze( bin_to_sign(syndrome) )
            magnitude = tf.abs(y)

            yield cast_to_float32(m, sign_to_bin(c), z, y, c_llr, y_llr, magnitude, syndrome)
            
    def __new__(cls, code, sigma=0.1, length=250, zero_cw=True, ones_m=False, flip_cw=False):
        m, n = code.H.shape
        k = n-m
        specs = tuple( tf.TensorSpec(shape=(k,), dtype=tf.float32) if i == 0 else
                       tf.TensorSpec(shape=(m,), dtype=tf.float32) if i == 7 else
                       tf.TensorSpec(shape=(n,), dtype=tf.float32) 
                       for i in range(8) )
        
        return tf.data.Dataset.from_generator( cls._generator,
                                               output_signature=specs,
                                               args=(code.G, code.H, k, n, sigma, length, zero_cw, ones_m, flip_cw) )

@staticmethod
def cast_to_float32(*args):
    return tuple(tf.cast(arg, tf.float32) for arg in args)

@staticmethod
def sign_to_llr(bpsk_vect, noise_variance):
    return -2 * bpsk_vect / noise_variance

@staticmethod
def bin_to_sign(c):
    return 2 * tf.cast(c, tf.float32) - 1

@staticmethod
def sign_to_bin(c):
    return (c + 1) // 2
