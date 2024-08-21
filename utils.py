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

    def __init__(self, encoder, decoder, k, n, return_infobits=False, es_no=False):
        super().__init__()

        self._n = n
        self._k = k

        self._binary_source = BinarySource()
        self._num_bits_per_symbol = 2
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._channel = AWGN()
        self._decoder = decoder
        self._encoder = encoder
        self._return_infobits = return_infobits
        self._es_no = es_no

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


def export_pgf(ber_plot, col_names):
    """Export results as table for for pgfplots compatible imports.

    Parameters
    ----------
    ber_plot: PlotBER
        An object of PlotBER containing the BER simulations to be exported

    col_names: list of str
        Column names of the exported BER curves
    """
    s = "snr, \t"
    for idx, var_name in enumerate(col_names):
        s += var_name + ", \t"
    s += "\n"

    for idx_snr,snr in enumerate(ber_plot._snrs[0]):
        s += f"{snr:.3f},\t"
        for idx_dec, _ in enumerate(col_names):
            s += f"{ber_plot._bers[idx_dec][idx_snr].numpy():.6E},\t"
        s += "\n"
    print(s)

###################################################################################################

def generate_pruned_pcm_5g(decoder, n, verbose=True):
    """Utility function to get the pruned parity-check matrix of the 5G code.

    Identifies the pruned and shortened positions.
    Hereby, '0' indicates an pruned codeword position
    '1' indicates an codeword position
    '2' indicates a shortened position.

    Parameters
    ---------
    decoder: LDPC5GDecoder
        An instance of the decoder object.

    n: int
        The codeword lengths including rate-matching.

    verbose: Boolean
        Defaults to True. If True, status information during pruning is
        provided.
    """

    enc = decoder._encoder

    # transmitted positions
    pos_tx = np.ones(n)

    # undo puncturing of the first 2*z information bits
    pos_punc = np.concatenate([np.zeros([2*enc.z]),pos_tx], axis=0)

    # puncturing of the last positions
    # total length must be n_ldpc, while pos_tx has length n
    # first 2*z positions are already added
    # -> add n_ldpc - n - 2Z punctured positions
    k_short = enc.k_ldpc - enc.k # number of shortend bits
    num_punc_bits = ((enc.n_ldpc - k_short) - enc.n - 2*enc.z)
    pos_punc2 = np.concatenate(
               [pos_punc, np.zeros([num_punc_bits - decoder._nb_pruned_nodes])])

    # shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
    # the first k positions are the systematic bits
    pos_info = pos_punc2[0:enc.k]

    # parity part
    num_par_bits = (enc.n_ldpc-k_short-enc.k-decoder._nb_pruned_nodes)
    pos_parity = pos_punc2[enc.k:enc.k+num_par_bits]
    pos_short = 2 * np.ones([k_short]) # "2" indicates shortened position

    # and concatenate final pattern
    rm_pattern = np.concatenate([pos_info, pos_short, pos_parity], axis=0)

    # and prune matrix (remove shortend positions from pcm)
    pcm_pruned = np.copy(decoder.pcm.todense())
    idx_short = np.where(rm_pattern==2)
    idx_pruned = np.setdiff1d(np.arange(pcm_pruned.shape[1]), idx_short)
    pcm_pruned = pcm_pruned[:,idx_pruned]
    num_shortened = np.size(idx_short)

    # print information if enabled
    if verbose:
        print("using bg: ", enc._bg)
        print("# information bits:", enc.k)
        print("CW length after rate-matching:", n)
        print("CW length without rm (incl. first 2*Z info bits):",
                                    pcm_pruned.shape[1])
        print("# punctured bits:", num_punc_bits)
        print("# pruned nodes:", decoder._nb_pruned_nodes)
        print("# parity bits", num_par_bits)
        print("# shortened bits", num_shortened)
        print("pruned pcm dimension:", pcm_pruned.shape)
    return pcm_pruned, rm_pattern[idx_pruned]


