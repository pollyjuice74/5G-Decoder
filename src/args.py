
class Args():
    """
    Args
    ----
    A configuration class that holds hyperparameters, model settings, and simulation
    details for training and evaluating the LDPC decoder.

    Attributes
    ----------
    t_layers : int
        Number of Transformer encoder layers.

    d_model : int
        Dimensionality of the Transformer model.

    heads : int
        Number of attention heads in the Transformer.

    mask_division_shape : int
        Division factor for mask projection.

    ns_eval : list
        Evaluation block sizes for time comparison.

    v : int
        Variable node degree for LDPC regular codes.

    c : int
        Check node degree for LDPC regular codes.

    lr : float
        Learning rate for training.

    batch_size : int
        Batch size for training.

    traindata_len : int
        Length of training data.

    testdata_len : int
        Length of test data.

    epochs : int
        Number of training epochs.

    ebno_db_min : float
        Minimum Eb/No value in dB for training.

    ebno_db_max : float
        Maximum Eb/No value in dB for training.

    ebno_db_stepsize : float
        Step size for Eb/No values in dB.

    ebno_db_eval : float
        Eb/No value in dB for evaluation.

    eval_train_iter : int
        Number of iterations for evaluation during training.

    save_weights_iter : int
        Interval for saving model weights during training.

    batch_size_eval : int
        Batch size for evaluation.

    num_iter_bp_eval : int
        Number of iterations for Belief Propagation evaluation.

    mc_batch_size : int
        Batch size for Monte Carlo simulations.

    mc_iters : int
        Number of Monte Carlo iterations.

    code : object
        An object to hold LDPC code parameters.
    """
    def __init__(self, t_layers=4, d_model=128, heads=8, lr=5e-4,
                       batch_size=160, batch_size_eval = 150,
                       eval_train_iter=1000, save_weights_iter=100,
                       c=6, v=3, num_iter_bp_eval=1, mask_division_shape=5,
                       ebno_db_eval=2.5,
                       ebno_db_min=0., ebno_db_max=15., ebno_db_stepsize=1,
                       traindata_len=500, testdata_len=250,
                       mc_batch_size=200, mc_iters=500, 
                       epochs=1000000):
        # model data
        self.t_layers = t_layers
        self.d_model = d_model
        self.heads = heads
        self.mask_division_shape = mask_division_shape
        # time comparison evaluation
        self.ns_eval = list(range(15, 350, 15))
        # LDPC regular code details
        self.v = v
        self.c = c
        # training data
        self.lr = lr
        self.batch_size = batch_size
        self.traindata_len = traindata_len
        self.testdata_len = testdata_len
        self.epochs = epochs
        # dB values
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max
        self.ebno_db_stepsize = ebno_db_stepsize
        # decoder evaluation
        self.ebno_db_eval = ebno_db_eval
        self.eval_train_iter = eval_train_iter
        self.save_weights_iter = save_weights_iter
        self.batch_size_eval = batch_size_eval
        self.num_iter_bp_eval = num_iter_bp_eval
        # simulation
        self.mc_batch_size = mc_batch_size
        self.mc_iters = mc_iters
        # code data
        self.code = type('Code', (), {})()
