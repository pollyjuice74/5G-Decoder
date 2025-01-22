import numpy as np

# Author: Nvidia
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
