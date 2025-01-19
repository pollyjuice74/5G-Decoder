from sionna.utils.metrics import compute_ber, compute_bler
import tensorflow as tf
import os
import time


@staticmethod
def bin_to_llr(x):
    """
    Converts binary values (0 or 1) to log-likelihood ratios (LLRs).
    
    Parameters
    ----------
    x : tf.Tensor
        Binary input tensor with values 0 or 1.

    Returns
    -------
    tf.Tensor
        Tensor of LLR values clipped to the range [-20, 20].
    """
    llr_vector = tf.where(x == 0, -20, 20)
    return llr_vector

@staticmethod
def llr_to_bin(c):
    """
    Converts log-likelihood ratios (LLRs) to binary values based on their sign.

    Parameters
    ----------
    c : tf.Tensor
        Tensor of LLR values.

    Returns
    -------
    tf.Tensor
        Binary tensor with values 0 or 1.
    """
    return tf.cast(tf.greater(c, 0), tf.int32)

@staticmethod
def logits_to_bin(c):
    """
    Converts logits to binary values.

    Parameters
    ----------
    c : tf.Tensor
        Tensor of logits.

    Returns
    -------
    tf.Tensor
        Binary tensor with values 0 or 1.
    """
    return tf.cast(tf.greater(c, 0.0), tf.int32)

@staticmethod
def logits_to_llr(c):
    """
    Converts logits to log-likelihood ratios (LLRs).

    Parameters
    ----------
    c : tf.Tensor
        Tensor of logits.

    Returns
    -------
    tf.Tensor
        Tensor of LLR values clipped to the range [-20, 20].
    """
    vn_probs = tf.nn.softmax(c, axis=-1)
    vn_llrs = tf.math.log(vn_probs / (1 - vn_probs + 1e-10))  # Convert to LLR
    return tf.clip_by_value(vn_llrs, clip_value_min=-20.0, clip_value_max=20.0)  # Bound LLRs

def load_weights(model, checkpoint_path):
    """
    Loads weights for a model from a specified checkpoint.

    Parameters
    ----------
    model : tf.keras.Model
        The model to load weights into.

    checkpoint_path : str
        Path to the checkpoint file.
    """
    checkpoint = tf.train.Checkpoint(decoder=model._decoder)
    try:
        checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
        print(f"Successfully restored weights from {checkpoint_path}")
    except AssertionError:
        print("No checkpoint found. Starting training from scratch.")

def save_weights(model, checkpoint_dir, weight_type='last', loss_value=None):
    """
    Saves model weights to a specified directory.

    Parameters
    ----------
    model : tf.keras.Model
        The model whose weights will be saved.

    checkpoint_dir : str
        Directory path for saving the weights.

    weight_type : str, optional
        Type of weights ('best' or 'last'). Default is 'last'.

    ber : float, optional
        BER value to include in the filename if saving the best weights.
    """
    checkpoint = tf.train.Checkpoint(decoder=model._decoder)
    if weight_type == 'best' and loss_value is not None:
        save_path = os.path.join(checkpoint_dir, f'{weight_type}_weights_loss_value_{loss_value:.5e}')
    else:
        save_path = os.path.join(checkpoint_dir, f'{weight_type}_weights')
    checkpoint.save(save_path)
    print(f"Saved {weight_type} weights to {save_path}")

def parse_best_loss_value_from_filename(checkpoint_dir):
    """
    Parses the best BER and its corresponding filename from the checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the checkpoint files.

    Returns
    -------
    tuple
        Best BER value (float) and the corresponding filename (str).
    """
    import re
    import os

    best_loss_value = float('inf')  # Initialize the best BER as infinity
    best_file = None
    
    try:
        for filename in os.listdir(checkpoint_dir):
            match = re.search(r'best_weights_loss_value_(\d+\.\d+e[+-]?\d+)', filename)
            if match:
                loss_value = float(match.group(1))  # Extract BER from filename
                # Update the best ber amd best file
                if loss_value < best_loss_value: 
                    best_loss_value = loss_value
                    best_file = os.path.splitext(filename)[0]  # Remove the extension
    except FileNotFoundError:
        print(f"No checkpoint directory found at {checkpoint_dir}")
    print(f"Best loss_value found: {best_loss_value}")

    return best_loss_value, best_file

def test_step(model, args, loss_fn, learning_rate, epoch):
    """
    Evaluates the model on a batch and logs performance metrics.

    Parameters
    ----------
    model : tf.keras.Model
        Model to evaluate.

    args : Namespace
        Evaluation parameters.

    loss_fn : tf.keras.losses.Loss
        Loss function used for evaluation.

    learning_rate : float
        Current learning rate.

    epoch : int
        Current epoch number.
    """
    ebno_db = tf.random.uniform([args.batch_size, 1],
                                 minval=args.ebno_db_eval,
                                 maxval=args.ebno_db_eval)
    # measure time for call
    time_start = time.time()
    c, c_hat, c_hat_logits, llr_channel = model(args.batch_size, ebno_db)
    duration = time.time() - time_start # in s

    # loss
    loss_value = loss_fn(c, c_hat_logits)
    # ber pred
    ber = compute_ber(c, c_hat).numpy()
    bler = compute_bler(c, c_hat).numpy()
    # ber original
    c_channel = llr_to_bin(llr_channel)
    channel_ber = compute_ber(c, c_channel).numpy()

    print(f'Training epoch {epoch}/{args.epochs}, LR={learning_rate:.2e}, Loss={loss_value.numpy():.5e}, channel_BER={channel_ber:.3e}, BER={ber:.3e}, BLER={bler:.3e} duration per call: {duration:.9f}s')

def train_dec(model, args, file_name, save_path='./Decoder_weights/ECC_weights/', load_decoder_weights=False):
    """
    Trains a model and periodically saves weights and evaluates performance.

    Parameters
    ----------
    model : tf.keras.Model
        Model to train.

    args : Namespace
        Training arguments, including batch size and number of epochs.

    file_name : str
        File name for saving weights.

    save_path : str, optional
        Directory to save weights. Default is './Decoder_weights/ECC_weights/'.

    load_decoder_weights : bool, optional
        Whether to load existing weights before training. Default is False.
    """

    # SGD update iteration
    @tf.function(jit_compile=False)
    def train_step(model, args, loss_fn, optimizer,):
        """
        Performs a single training step.

        Parameters
        ----------
        model : tf.keras.Model
            Model to train.

        args : Namespace
            Training parameters.

        loss_fn : tf.keras.losses.Loss
            Loss function for calculating errors.

        optimizer : tf.keras.optimizers.Optimizer
            Optimizer for updating model weights.

        Returns
        -------
        tuple
            Ground truth labels, predicted logits, and the calculated BER.
        """
        # train for random SNRs within a pre-defined interval
        ebno_db = tf.random.uniform([args.batch_size, 1],
                                    minval=args.ebno_db_min,
                                    maxval=args.ebno_db_max)

        with tf.GradientTape() as tape:
            c, c_hat, c_hat_logits, _ = model(args.batch_size, ebno_db, training=True)
            
            # Calculates the aggregate loss for the first layer, then for the iterative refinement layer
            loss_value = loss_fn(c, c_hat_logits)

        # and apply the SGD updates
        weights = model.trainable_weights
        grads = tape.gradient(loss_value, weights) # variables
        optimizer.apply_gradients(zip(grads, weights))
        return c, c_hat_logits, loss_value

    # loss
    loss_fn =  tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # optimizer
    scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.lr, decay_steps=args.epochs) # 1000 is size of trainloader
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

    # parse best ber and file with that ber
    weights_path = os.path.join(save_path, file_name)
    weights_dir = os.path.dirname(weights_path) # Directory of weights
    best_loss_value, best_file = parse_best_loss_value_from_filename(weights_dir) # Retrieve or initialize best BER

    # Load weights if available
    if load_decoder_weights:
        print(weights_dir, best_file)
        weights_load_path = os.path.join(weights_dir, best_file) 
        load_weights(model, weights_load_path) 

    print("Training Model...")
    for epoch in range(1, args.epochs + 1):
        _, _, loss_value = train_step(model,
                                args,
                                loss_fn,
                                optimizer,)

        # Save the best weights if the current BER is better
        if loss_value < best_loss_value:
            save_weights(model, weights_dir, weight_type='best', loss_value=loss_value)
            best_loss_value = loss_value # Update the best BER

        # eval train iter
        if epoch % args.eval_train_iter == 0:
            test_step(model,
                      args,
                      loss_fn,
                      learning_rate=optimizer.learning_rate.numpy(),
                      epoch=epoch)