from DDECC.src.codes import Get_Generator_and_Parity


class Args():
    def __init__(self, model_type, code_type='LDPC', n=121, k=80,  ls_active=True, sigma=0.1, beta_steps=10, t_layers=1, d_model=8):
        assert model_type in ['gen', 'dis'], "Type must be: 'gen', Generator or 'dis', Discriminator."
        assert code_type in ['POLAR', 'BCH', 'CCSDS', 'LDPC', 'MACKAY'], "Invalid linear code type."
        
        self.model_type = model_type
        self.code_type = code_type
        self.code = self.get_code(n,k)
        self.mask = self.create_mask()
        
        self.ls_active = True
        self.sigma = sigma
        self.beta_steps = beta_steps
        self.t_layers = t_layers # transformer layers
        self.d_model = d_model
        self.N_steps = self.code.H.shape[0]+5 # Number of diffusion steps
    
    def create_mask(self):
        H = self.code.H
        H_mask = H
        return H_mask

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
        