
# for e2e model
from sionna.utils import BinarySource, ebnodb2no
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
import tensorflow as tf

from .utils import logits_to_bin


class E2EModel(tf.keras.Model):
    """
    E2EModel
    --------
    An end-to-end communication model implementing encoding, modulation, AWGN channel, demodulation, and decoding.

    Parameters
    ----------
    encoder : callable
        Encoder function or object to generate codewords from input bits.

    decoder : callable
        Decoder function or object to reconstruct the transmitted codewords.

    k : int
        Number of information bits.

    n : int
        Length of the codeword.

    return_infobits : bool
        Whether to return only the information bits instead of the entire codeword.

    es_no : bool
        Indicates whether Es/No (symbol energy to noise ratio) is used instead of Eb/No (bit energy to noise ratio).

    rate_matching : bool
        Indicates if rate-matching is applied during transmission.

    Input
    -----
    batch_size : int
        Number of samples per batch.

    ebno_db : float
        Signal-to-noise ratio (SNR) in decibels (dB).

    training : bool, optional
        Indicates whether the model is in training mode. Defaults to `False`.

    Output
    ------
    If `return_infobits` is `True`:
        b : tf.Tensor
            Original information bits of shape `[batch_size, k]`.

        c_hat : tf.Tensor
            Decoded information bits of shape `[batch_size, k]`.

        c_hat_logits : tf.Tensor
            Logits of the decoded codewords of shape `[batch_size, n]`.

        llr : tf.Tensor
            Log-likelihood ratios (LLRs) of received symbols of shape `[batch_size, n]`.

    If `return_infobits` is `False`:
        c : tf.Tensor
            Original codeword of shape `[batch_size, n]`.

        c_hat : tf.Tensor
            Decoded codeword of shape `[batch_size, n]`.

        c_hat_logits : tf.Tensor
            Logits of the decoded codewords of shape `[batch_size, n]`.

        llr : tf.Tensor
            Log-likelihood ratios (LLRs) of received symbols of shape `[batch_size, n]`.
    """
    def __init__(self, encoder, decoder, k, n, return_infobits=False, es_no=False, rate_matching=False):
        super().__init__()

        self._n = n
        self._k = k
        self._m = n - k
        self.rate_matching = rate_matching

        self._binary_source = BinarySource(dtype=tf.int32)
        self._num_bits_per_symbol = 2
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._channel = AWGN()
        self._decoder = decoder
        self._encoder = encoder
        self._return_infobits = return_infobits
        self._es_no = es_no

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db, training=False):
        """ Executes the end-to-end encoding, transmission, and decoding process for a batch of codewords. """
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

        # decoder input nodes
        if not self.rate_matching:
            syndrome = tf.reshape( self._decoder.get_syndrome(llr),
                                  (batch_size, self._m) ) # (m,n)@(n,b)->(m,b)->(b,m) check nodes
            x_nodes = tf.concat([llr, syndrome], axis=1)[:, :, tf.newaxis] # (b, n+m, 1)
        else:
           x_nodes = llr # (b, n, 1)

        # and run the decoder
        if self._decoder is not None:
            ############################
            c_hat_logits = self._decoder(x_nodes, training=training) # (2 * batch_size, n)
            ############################

        # final refined codeword prediction
        c_hat = logits_to_bin(c_hat_logits) # for plotting and simulation, requires bits,bits_hat as first two outputs

        if self._return_infobits:
            return b, c_hat, c_hat_logits, llr
        else:
            return c, c_hat, c_hat_logits, llr
