import gc
import time
import tensorflow as tf

from sionna.fec.utils import generate_reg_ldpc, LinearEncoder
from sionna.fec.ldpc import LDPCBPDecoder

from .e2e_model import E2EModel
from .decoder import Decoder
from .args import Args

import numpy as np
import matplotlib.pyplot as plt


def decoders_comparison(args, mask_divisions=range(2,3)):
    """
    Compares the performance of various decoders on different PCM matrix sizes.

    This function evaluates the execution time of Belief Propagation (BP), Regular Transformer,
    and Linear Transformer decoders for varying PCM (Parity Check Matrix) sizes.

    Parameters
    ----------
    args : Args
        An instance of the Args class containing configuration parameters for the decoders.

    mask_divisions : range, optional
        Range of mask division values for testing Linear Transformer decoders.
        Defaults to `range(2, 3)`.

    Returns
    -------
    dict
        A dictionary containing execution time data for different decoders and PCM sizes.
        Each key corresponds to a PCM size tuple `(m, n)`, and the value is a dictionary
        of decoder execution times.
    """

    def compare_times(models, eval_iters=100):
        """
        Evaluates the time performance of different decoder models over multiple iterations.

        This function measures the execution time for a batch of input data for each
        decoder model, computes statistics, and cleans up memory.

        Parameters
        ----------
        models : dict
            A dictionary where keys are decoder names (str) and values are the corresponding
            decoder model instances.

        eval_iters : int, optional
            Number of iterations to evaluate each model's performance. Defaults to 100.

        Returns
        -------
        dict
            A dictionary containing execution times for each decoder model. Each key is the
            model's name, and the value is a NumPy array of execution times (in seconds).
        """
        ebno_db = tf.random.uniform([args.batch_size, 1],
                                    minval=args.ebno_db_eval,
                                    maxval=args.ebno_db_eval)
        model_durations = {name: [] for name in models}

        for name, model in models.items():
            model_durations[name] = []

            for _ in range(eval_iters):
                # measure time for call
                time_start = time.time()
                c, c_hat, c_hat_logits, llr_channel = model(args.batch_size, ebno_db)
                duration = time.time() - time_start # in s

                model_durations[name].append(duration)
                
        # Delete models to free memory
        for name, model in models.items():
            del model  # Delete model
            gc.collect()  # Force garbage collection

        # Convert to NumPy array for statistics
        model_durations_np = {name: np.array(times) for name, times in model_durations.items()}
        return model_durations_np

    data = {}
    # Compare time performance between Linear and Regular Transformer decoder on varying pcm sizes
    for n in args.ns_eval:
        print(f"Evaluating on {n}")
        decoders = []
        while True:
            try:
                # generate new code for each length
                pcm, k, n, coderate = generate_reg_ldpc(v=args.v,
                                                        c=args.c,
                                                        n=n,
                                                        allow_flex_len=True,
                                                        verbose=False)
                break
            except Exception as e:
                pass

        encoder = LinearEncoder(pcm, is_pcm=True, dtype=tf.int32)

        # simulate "conventional" BP performance first
        bp_decoder = LDPCBPDecoder(pcm,
                                num_iter=args.num_iter_bp_eval,
                                hard_out=False)
        e2e_bp = E2EModel(encoder, bp_decoder, k, n, rate_matching=True) # Not really rate-matching but just to bypass syndrome computation

        # args for decoder
        args = Args()
        args.t_layers = 8
        args.d_model = 96
        args.batch_size = 10
        args.code.H = pcm
        args.m, args.n = pcm.shape
        args.k = k

        reg_trans_decoder = Decoder(args,
                                    verbose=False,
                                    linear=False) # Regular Transformer Diffusion (LTD) Decoder
        e2e_reg_trans = E2EModel(encoder, reg_trans_decoder, k, n)

        # Store decoders in a dictionary
        decoders = {
            f"BP Decoder (iters={args.num_iter_bp_eval})": e2e_bp,
            "Regular Transformer": e2e_reg_trans
        }
        
        # Iterate over mask divisions, memory is a concern here...
        for mask_div in mask_divisions:
            args.mask_division_shape = mask_div
            lin_trans_decoder = Decoder(args,
                                        verbose=False,
                                        linear=True) # Linear Transformer Diffusion (LTD) Decoder
            e2e_lin_trans = E2EModel(encoder, lin_trans_decoder, k, n)
            
            decoders[f"Linear Transformer (mask_div={args.mask_division_shape})"] = e2e_lin_trans
        print(decoders)

        data[str((n-k, n))] = compare_times(decoders)

    return data

def plot_comparison(data, colors=['blue', 'green', 'red']):
    """
    Visualizes the execution time comparison of decoders across different PCM matrix sizes.

    The function generates a quartile chart to display the execution times of multiple decoders 
    (e.g., BP Decoder, Regular Transformer, Linear Transformer) for varying PCM matrix sizes, 
    with error bars representing the interquartile range.

    Parameters
    ----------
    data : dict
        A dictionary containing execution time data for different decoders and PCM sizes.
        Keys are PCM size tuples `(m, n)`, and values are dictionaries where:
            - Keys are decoder names (e.g., "BP Decoder", "Regular Transformer").
            - Values are lists of execution times for the corresponding decoder.

    colors : list of str, optional
        List of color codes or names for plotting the decoders. The number of colors must
        match the number of decoders being evaluated. Defaults to ['blue', 'green', 'red'].

    Notes
    -----
    - Ensure that the `colors` list matches the number of decoders being evaluated to avoid 
      mismatches in the visualization.
    - The function dynamically calculates quartiles (25th and 75th percentiles) and median 
      execution times for each decoder and PCM size.
    """
    # Extract PCM sizes (x-axis) and ensure execution times are properly structured
    pcm_sizes = list(data.keys())  # List of (m, n) tuples
    decoders = list(data[pcm_sizes[0]].keys())  # Extract decoder names dynamically

    # Convert execution times to NumPy arrays for quartile calculations
    execution_times_np = {
        name: np.array([data[pcm_size][name] for pcm_size in pcm_sizes])
        for name in decoders
    }

    # Plot quartile charts
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, times) in enumerate(execution_times_np.items()):
        medians = np.median(times, axis=1)
        q1 = np.percentile(times, 25, axis=1)
        q3 = np.percentile(times, 75, axis=1)

        ax.errorbar(range(len(pcm_sizes)), medians, yerr=[medians - q1, q3 - medians],
                    fmt='o', label=name, color=colors[i], capsize=5)

    # Set x-axis labels to tuple sizes
    ax.set_xticks(range(len(pcm_sizes)))
    ax.set_xticklabels(pcm_sizes, rotation=45)

    ax.set_xlabel("PCM Matrix Size (m, n)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"Decoder Execution Time Comparison (run on RTX 4070 Super)")
    ax.legend(loc='upper left', title="Decoders")
    ax.grid(True)
    plt.show()

