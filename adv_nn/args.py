import tensorflow as tf
from DDECC.src.codes import Get_Generator_and_Parity

# from sionna.fec.ldpc.encoding import LDPC5GEncoder
# from decoder5G import LDPC5GDecoder

from scipy.sparse import issparse, csr_matrix


class Args():
    def __init__(self, model_type, code_type='LDPC', n_look_up=121, k_look_up=80, n=400, k=200,
                       n_rings=2, ls_active=True, split_diff=True, sigma=0.1,
                       t_layers=1, d_model=128, heads=8, lr=5e-4, batch_size=128, 
                       traindata_len=500, testdata_len=250, epochs=1000):
        assert model_type in ['gen', 'dis'], "Type must be: 'gen', Generator or 'dis', Discriminator."
        assert code_type in ['POLAR', 'BCH', 'CCSDS', 'LDPC', 'MACKAY', 'LDPC5G', 'POLAR5G'], "Invalid linear code type."
        
        self.model_type = model_type
        self.code_type = code_type

        self.split_diff = split_diff
        self.n_rings = n_rings # ring connectivity of mask
        self.sigma = sigma
        self.t_layers = t_layers
        self.ls_active = ls_active
        
        self.d_model = d_model
        self.heads = heads
                           
        self.lr = lr
        self.batch_size = batch_size
        self.traindata_len = traindata_len
        self.testdata_len = testdata_len
        self.epochs = epochs

        # Ensure that code, m, and n are set properly
        self.code = self.get_code(n_look_up, k_look_up) # n,k look up values in Get_Generator_and_Parity
        
        if self.code_type not in ['LDPC5G', 'POLAR5G']:
            self.n, self.m, self.k = self.code.n, self.code.m, self.code.k
        else: 
            self.n, self.m, self.k = n, n-k, k

        self.n_steps = self.m + 5  # Number of diffusion steps

    def get_code(self, n_look_up, k_look_up):
        code = type('Code', (), {})() # class Code, no base class, no attributes/methods, () instantiate object
        code.n_look_up, code.k_look_up = n_look_up, k_look_up
        code.code_type = self.code_type
    
        if self.code_type not in ['LDPC5G', 'POLAR5G']:
            G, H = Get_Generator_and_Parity(code)
            code.G, code.H = tf.convert_to_tensor(G), csr_matrix( tf.convert_to_tensor(H) )
        
            code.m, code.n = code.H.shape
            code.k = code.n - code.m

        return code
        
