import tensorflow as tf
import numpy as np
from sionna.fec.ldpc import LDPC5GDecoder

from src.utils5G import generate_pruned_pcm_5g
from src.decoder import Decoder

class Decoder5G( Decoder ):
    """
    Decoder5G
    ---------
    Implements a Transformer-based decoding mechanism for 5G LDPC codes,
    integrating rate-matching and pruning functionalities.

    Parameters
    ----------
    encoder : Encoder
        The LDPC encoder object used for rate-matching and recovery.

    args : Namespace
        Configuration object containing model and code parameters.

    linear : bool, optional
        If True, uses a Linear Transformer. Defaults to False.

    rate_matching : bool, optional
        Enables or disables rate-matching. Defaults to True.

    dropout_rate : float, optional
        Dropout rate applied to the layers. Defaults to 0.1.

    output_all_iter : bool, optional
        If True, outputs results from all iterations. Defaults to False.

    return_infobits : bool, optional
        If True, returns only the information bits. Defaults to False.

    verbose : bool, optional
        Prints additional debugging information. Defaults to True.

    use_bias : bool, optional
        If True, enables bias in the model layers. Defaults to True.

    **kwargs : dict
        Additional arguments for the parent class initialization.

    Input
    -----
    inputs : tf.Tensor
        Tensor of LLR values with shape [batch_size, num_bits].

    training : bool, optional
        Indicates whether the model is in training mode.

    Output
    ------
    decoded_output : tf.Tensor or list of tf.Tensor
        The decoded information or codeword bits depending on `return_infobits`.
        If `output_all_iter` is True, returns a list of outputs for each iteration.
    """
    def __init__(self,
                 encoder,
                 args,
                 linear=False,
                 rate_matching=True,
                 dropout_rate=0.1,
                 output_all_iter=False,
                 return_infobits=False,
                 verbose=True,
                 use_bias=True,
                 **kwargs):

        self._encoder = encoder
        self._return_infobits = return_infobits
        self._llr_max = 20 # internal max value for LLR initialization
        self._output_all_iter = output_all_iter

        # instantiate internal decoder object to access pruned pcm
        # Remark: this object is NOT used for decoding!
        decoder = LDPC5GDecoder(encoder, prune_pcm=True)

        # access pcm and code properties
        self._n_pruned = decoder._n_pruned
        self._num_pruned_nodes = decoder._nb_pruned_nodes
        # prune and remove shortened positions
        self._pcm, self._rm_pattern = generate_pruned_pcm_5g(decoder,
                                                             encoder.n,
                                                             verbose=False)
        # precompute pruned positions
        gather_ind = encoder.n * np.ones(np.size(self._rm_pattern))
        gather_ind_inv = np.zeros(np.size(np.where(self._rm_pattern==1)))
        for idx, pos in enumerate(np.where(self._rm_pattern==1)[0]):
            gather_ind[pos] = idx
            gather_ind_inv[idx] = pos

        self._rm_ind = tf.constant(gather_ind, tf.int32)
        self._rm_inv_ind = tf.constant(gather_ind_inv, tf.int32)

        args.code.H = self._pcm
        args.n, args.m = self._pcm.shape
        args.k = args.n - args.m
        # init Transformer decoder
        super().__init__(args,
                         linear=linear,
                         dropout_rate=dropout_rate,
                         rate_matching=rate_matching,
                         verbose=verbose)

    @property
    def llr_max(self):
        """Max LLR value used for rate-matching."""
        return self._llr_max

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    def call(self, inputs, training=False):
        """Iterative MPNN decoding function."""

        llr_ch = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]

        # add punctured positions
        # append one zero pos
        llr_in = tf.concat([llr_ch, tf.zeros([batch_size, 1], tf.float32)],
                           axis=1)
        llr_rm = tf.gather(llr_in, self._rm_ind, axis=1)

        # Concatenate llrs and syndrome as vn and cn input tensor
        syndrome = tf.reshape( self.get_syndrome(llr_rm),
                                  (batch_size, self._m) ) # (m,n)@(n,b)->(m,b)->(b,m) check nodes
        x_nodes = tf.concat([llr_rm, syndrome], axis=1)[:, :, tf.newaxis] # (b, n+m, 1)

        # and execute the decoder
        x_hat_dec = super().call(x_nodes, training=training) # (2*b, n)

        # we need to de-ratematch for all iterations individually (for training)
        if not self._output_all_iter:
            x_hat_list = [x_hat_dec]
        else:
            x_hat_list = x_hat_dec

        u_out = []
        x_out =[]

        for idx,x_hat in enumerate(x_hat_list):
            if self._return_infobits: # return only info bits
                # reconstruct u_hat # code is systematic
                u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
                u_out.append(u_hat)

            else: # return all codeword bits
                x_short = tf.gather(x_hat, self._rm_inv_ind, axis=1)
                x_out.append(x_short)

        # return no list
        if not self._output_all_iter:
            if self._return_infobits:
                return u_out[-1]
            else:
                return x_out[-1]

        # return list of all iterations
        if self._return_infobits:
            return u_out
        else:
            return x_out