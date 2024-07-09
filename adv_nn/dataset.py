import tensorflow as tf
import random


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
