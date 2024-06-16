# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GPSConv
# from torch_geometric.data import Data

import tensorflow as tf
import numpy as np

import sionna as sn
from sionna.utils import BitErrorRate, BinarySource
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder



class E2EModel(tf.keras.Model):
    """End-to-end model for (GNN-)decoder evaluation.

    Parameters
    ----------
    encoder: Layer or None
        Encoder layer, no encoding applied if None.

    decoder: Layer or None
        Decoder layer, no decoding applied if None.

    k: int
        Number of information bits per codeword.

    n: int
        Codeword lengths.

    return_infobits: Boolean
        Defaults to False. If True, only the ``k`` information bits are
        returned. Must be supported be the decoder as well.

    es_no: Boolean
        Defaults to False. If True, the SNR is not rate-adjusted (i.e., Es/N0).

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (c, llr):
            Tuple:

        c: tf.float32
            A tensor of shape `[batch_size, n] of 0s and 1s containing the
            transmitted codeword bits.

        llr: tf.float32
            A tensor of shape `[batch_size, n] of llrs containing estimated on
            the codeword bits.
    """

    def __init__(self, encoder, decoder, model, 
                       return_infobits=False,
                       es_no=False, 
                       decoder_active=False):        
        super().__init__()

        self._n = encoder._n
        self._k = encoder._k

        self._binary_source = BinarySource()
        self._num_bits_per_symbol = 2
        self._mapper = Mapper("qam", self._num_bits_per_symbol) #
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol) #
        self._channel = AWGN() #
        self._decoder = decoder
        self._encoder = encoder
        # self._return_infobits = return_infobits
        # self._es_no = es_no

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):

        # no rate-adjustment for uncoded transmission or es_no scenario
        if self._decoder is not None and self._es_no==False:
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._k/self._n)
        else: #for uncoded transmissions the rate is 1
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, 1)

        b = self._binary_source([batch_size, self._k])
        if self._encoder is not None:
            c = self._encoder(b)
        else:
            c = b

        # check that rate calculations are correct
        assert self._n==c.shape[-1], "Invalid value of n."

        # zero padding to support odd codeword lengths
        if self._n%2==1:
            c_pad = tf.concat([c, tf.zeros([batch_size, 1])], axis=1)
        else: # no padding
            c_pad = c
        x = self._mapper(c_pad)

        y = self._channel([x, no])
        llr = self._demapper([y, no])

        # remove zero padded bit at the end
        if self._n%2==1:
            llr = llr[:,:-1]

        # and run the decoder
        if self._decoder is not None:
            llr = self._decoder(llr)

        if self._return_infobits:
            return b, llr
        else:
            return c, llr
