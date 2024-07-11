import tensorflow as tf
from DDECC.src.codes import Get_Generator_and_Parity


class Args():
    def __init__(self, model_type, code_type='LDPC', n=121, k=80, n_rings=2, ls_active=True, sigma=0.1,
                       beta_steps=10, t_layers=1, d_model=8, lr=5e-4, batch_size=128, 
                       traindata_len=500, testdata_len=250, epochs=1000):
        assert model_type in ['gen', 'dis'], "Type must be: 'gen', Generator or 'dis', Discriminator."
        assert code_type in ['POLAR', 'BCH', 'CCSDS', 'LDPC', 'MACKAY'], "Invalid linear code type."
        
        self.model_type = model_type
        self.code_type = code_type
        self.k, self.n = k, n
        self.m = n - k
        self.code = self.get_code(n,k)
        self.n_rings = n_rings # ring connectivity of mask
        
        self.ls_active = True
        self.sigma = sigma
        self.beta_steps = beta_steps
        self.t_layers = t_layers # transformer layers
        self.d_model = d_model
        self.N_steps = self.code.H.shape[0]+5 # Number of diffusion steps
        self.epochs = epochs #
        self.batch_size = batch_size # chunks 
        self.traindata_len = traindata_len
        self.testdata_len = testdata_len

        self.lr = lr

    def get_code(self, n, k):
        class Code():
            pass
        code = Code()
        code.n, code.k = n, k
        code.m = n - k
        code.code_type = self.code_type
    
        G, H = Get_Generator_and_Parity(code)
        code.G, code.H = tf.convert_to_tensor(G), tf.convert_to_tensor(H)
        return code
        
