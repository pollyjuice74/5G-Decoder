# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GPSConv
# from torch_geometric.data import Data

import tensorflow as tf
import torch
import numpy as np

import sionna as sn
from sionna.utils import BitErrorRate, BinarySource, ebnodb2no
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder



class E2EModelDDECC(tf.keras.Model):
    def __init__(self, model, decoder,
                       batch_size=1,
                       return_infobits=False,
                       es_no=False,
                       decoder_active=False):
        super().__init__()

        self._n = decoder.encoder._n
        self._k = decoder.encoder._k

        self._binary_source = BinarySource()
        self._num_bits_per_symbol = 4 # QAM16

        # Channel
        ############################
        # Encoding
        self._encoder = model.encoder
        self._mapper = Mapper("qam", self._num_bits_per_symbol) #

        # Channel
        self._channel = AWGN() #
        # adversarial channel noise emulator

        # Decoding
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol) #
        # Decoders
        self._decoder = model # DDECCT
        self._decoder5g = decoder # LDPC5GDecoder
        ############################

        self._return_infobits = return_infobits
        self._es_no = es_no

        self._batch_size = batch_size

    def train(self):
      pass

    def test(self):
      pass

    # @tf.function(jit_compile=True)
    def call(self, ebno_db):
        # Noise Variance
        if self._decoder is not None and self._es_no==False: # no rate-adjustment for uncoded transmission or es_no scenario
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._k/self._n) ### LOOK UP EBNODB2NO
        else: #for uncoded transmissions the rate is 1
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, 1) ###
        no = tf.expand_dims(tf.cast(no, tf.float32), axis=-1) # turn to float32, turns shape (9,) -> (9,1)
        print("no, ebno_db: ", no.shape, ebno_db.shape)

        b = self._binary_source([self._batch_size, self._encoder._k]) # (batch_size, k), k information bits
        print("bit: ", b.shape) # print(b.shape[-1]==self._k, b.shape, self._k, self._n - self._k)

        # Turns INFO BITS (batch_size, k) -> (batch_size, n) info and parity bits CODEWORD of rate = k/n
        if self._encoder is not None:
            c = self._encoder(b) ##### c = G @ b.T, (n,k) @ (k,1)
        else:
            c = b

        print("n, c: ", self._n, c.shape)
        # check that rate calculations are correct
        assert self._n == c.shape[-1], "Invalid value of n."

        # zero padding to support odd codeword lengths
        if self._n%2 == 1:
            c_pad = tf.concat([c, tf.zeros([self._batch_size, 1])], axis=1)
        else: # no padding
            c_pad = c
        print("c_pad, c: ", c_pad.shape, c.shape)

        # Channel
        ############################
        x = self._mapper(c_pad)
        y = self._channel([x, no]) ###
        llr = self._demapper([y, no])
        ############################
        print("y, no: ", y.shape, no.shape)

        # remove zero padded bit at the end
        if self._n%2 == 1:
            llr = llr[:,:-1]
        llr = torch.tensor(llr.numpy())
        print("llr: ", llr.shape,)# b, c, x, y)

        # run the decoder
        if self._decoder is not None:
            llr = tf.convert_to_tensor(llr.numpy(), dtype=tf.float32) # Pytorch to Tensorflow
            llr = self._decoder5g(llr) # Gets reshaped (n_ldpc,1) llrs
            print("llr (n_ldpc,): ", llr.shape, " sum positive: ", tf.reduce_sum(tf.boolean_mask(llr, llr > 0)), " n_ldpc: ", self._encoder._n_ldpc)
            print("llr (crude): ", llr[:, 54])

            llr_ddecc = self._decoder(llr, time_step=0) # 9 no values, 100 bits of data, time step 0
            print("llr_ddecc: ", llr_ddecc.shape)

        # codeword, info bits, llr of either cw or info bits
        return c, b, llr_ddecc 

        # if self._return_infobits:
        #     return b, llr_ddecc
        # else:
        #     return c, llr_ddecc
