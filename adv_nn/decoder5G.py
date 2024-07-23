#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for channel decoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from tensorflow.keras.layers import Layer
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.utils import llr2mi
import matplotlib.pyplot as plt

from sionna.fec.ldpc.decoding import LDPCBPDecoder
from models import Decoder


class LDPC5GDecoder(LDPCBPDecoder):
    def __init__(self,
                 encoder,
                 args,
                 out_llrs=False,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=False,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        # needs the 5G Encoder to access all 5G parameters
        assert isinstance(encoder, LDPC5GEncoder), 'encoder must \
                          be of class LDPC5GEncoder.'
        self._encoder = encoder
        pcm = encoder.pcm

        assert isinstance(return_infobits, bool), 'return_info must be bool.'
        self._return_infobits = return_infobits

        assert isinstance(output_dtype, tf.DType), \
                                'output_dtype must be tf.DType.'
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')
        self._output_dtype = output_dtype

        assert isinstance(stateful, bool), 'stateful must be bool.'
        self._stateful = stateful

        assert isinstance(prune_pcm, bool), 'prune_pcm must be bool.'
        # prune punctured degree-1 VNs and connected CNs. A punctured
        # VN-1 node will always "send" llr=0 to the connected CN. Thus, this
        # CN will only send 0 messages to all other VNs, i.e., does not
        # contribute to the decoding process.
        self._prune_pcm = prune_pcm
        if prune_pcm:
            # find index of first position with only degree-1 VN
            dv = np.sum(pcm, axis=0) # VN degree
            last_pos = encoder._n_ldpc
            for idx in range(encoder._n_ldpc-1, 0, -1):
                if dv[0, idx]==1:
                    last_pos = idx
                else:
                    break
            # number of filler bits
            k_filler = self.encoder.k_ldpc - self.encoder.k
            # number of punctured bits
            nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)
            # effective codeword length after pruning of vn-1 nodes
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned
            # remove last CNs and VNs from pcm
            pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]

            #check for consistency
            assert(self._nb_pruned_nodes>=0), "Internal error: number of \
                        pruned nodes must be positive."
        else:
            self._nb_pruned_nodes = 0
            # no pruning; same length as before
            self._n_pruned = encoder._n_ldpc

        # DECODER
        super().__init__(pcm,
                         trainable,
                         cn_type,
                         hard_out,
                         track_exit,
                         num_iter=num_iter,
                         stateful=stateful,
                         output_dtype=output_dtype,
                         **kwargs)
        
        self.out_llrs = out_llrs
        if not self.out_llrs:
            args.code.H = pcm
            self._decoder = Decoder(args)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build model."""
        if self._stateful:
            assert(len(input_shape)==2), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]

        # check input dimensions for consistency
        assert (input_shape[-1]==self.encoder.n), \
                                'Last dimension must be of length n.'
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

        self._old_shape_5g = input_shape

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` or `[...,k]`
            (``return_infobits`` is True) containing bit-wise soft-estimates
            (or hard-decided bit-values) of all codeword bits (or info
            bits, respectively).

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            ValueError: If ``num_iter`` is not an integer greater (or equal)
                `0`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)


        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
            [tf.zeros([batch_size, 2*self.encoder.z], self._output_dtype),
                          llr_ch_reshaped],
                          1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)


        llr_5g = tf.concat([llr_5g,
                   tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                            self._output_dtype)],
                            1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0,0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -tf.cast(self._llr_max, self._output_dtype) \
            * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        if self.out_llrs:
            return llr_5g
        
        else:
            # DECODER
            ############################################################
            x_hat, _, _ = self._decoder(llr_5g) #super().call(llr_5g)
            x_hat = tf.transpose(x_hat) # (b, n_ldpc)
            ############################################################

            if self._return_infobits: # return only info bits
                # reconstruct u_hat # code is systematic
                u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
                # Reshape u_hat so that it matches the original input dimensions
                output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
                # overwrite first dimension as this could be None (Keras)
                output_shape[0] = -1
                u_reshaped = tf.reshape(u_hat, output_shape)

                # enable other output datatypes than tf.float32
                u_out = tf.cast(u_reshaped, self._output_dtype)

                if not self._stateful:
                    return u_out
                else:
                    return u_out, msg_vn

            else: # return all codeword bits
                # the transmitted CW bits are not the same as used during decoding
                # cf. last parts of 5G encoding function

                # remove last dim
                x = tf.reshape(x_hat, [batch_size, self._n_pruned])

                # remove filler bits at pos (k, k_ldpc)
                x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

                x_no_filler2 = tf.slice(x,
                                        [0, self.encoder.k_ldpc],
                                        [batch_size,
                                        self._n_pruned-self.encoder.k_ldpc])

                x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

                # shorten the first 2*Z positions and end after n bits
                x_short = tf.slice(x_no_filler,
                                  [0, 2*self.encoder.z],
                                  [batch_size, self.encoder.n])

                # if used, apply rate-matching output interleaver again as
                # Sec. 5.4.2.2 in 38.212
                if self._encoder.num_bits_per_symbol is not None:
                    x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

                # Reshape x_short so that it matches the original input dimensions
                # overwrite first dimension as this could be None (Keras)
                llr_ch_shape[0] = -1
                x_short= tf.reshape(x_short, llr_ch_shape)

                # enable other output datatypes than tf.float32
                x_out = tf.cast(x_short, self._output_dtype)

                if not self._stateful:
                    return x_out
                else:
                    return x_out, msg_vn


        
