import tensorflow as tf
from DDECC.src.codes import Get_Generator_and_Parity


class Args():
    def __init__(self, model_type, code_type='LDPC', n=121, k=80, n_rings=2, ls_active=True, sigma=0.1,
                       beta_steps=10, t_layers=1, d_model=8, lr=5e-4, batch_size=128, 
                       traindata_len=500, testdata_len=250, epochs=1000):
        assert model_type in ['gen', 'dis'], "Type must be: 'gen', Generator or 'dis', Discriminator."
        assert code_type in ['POLAR', 'BCH', 'CCSDS', 'LDPC', 'MACKAY', 'LDPC5G', 'POLAR5G'], "Invalid linear code type."
        
        self.model_type = model_type
        self.code_type = code_type
        self.k, self.n = k, n
        self.m = n - k
        self.n_rings = n_rings # ring connectivity of mask
        self.code = self.get_code(n,k)
        
        self.ls_active = True
        self.sigma = sigma
        self.beta_steps = beta_steps
        self.t_layers = t_layers # transformer layers
        self.d_model = d_model
        self.n_steps = self.code.H.shape[0]+5 # Number of diffusion steps
        self.epochs = epochs #
        self.batch_size = batch_size # chunks 
        self.traindata_len = traindata_len
        self.testdata_len = testdata_len

        self.lr = lr

    def get_code(self, n, k):
        code = type('Code', (), {})() # class Code, no base class, no attributes/methods, () instantiate object
        code.n, code.k = n, k
        code.m = n - k
        code.code_type = self.code_type
    
        if self.code_type=='LDPC5G':
            encoder = LDPC5GEncoder(k,n)
            decoder = LDPC5GDecoder(encoder)
            code.enc = encoder
            code.dec = decoder # contains encoder
            
            H = encoder.pcm
            
        elif self.code_type=='POLAR5G':
            pass
            
        else:
            G, H = Get_Generator_and_Parity(code)
            code.G, code.H = tf.convert_to_tensor(G), tf.convert_to_tensor(H)
        
        code.mask = self.create_mask(H, self.n_rings)
        return code
        
    def create_mask(self, H, n_rings=1):
        m,n = H.shape
        mask = tf.eye(n+m, dtype=tf.float32)
        init_H = True

        for _ in range(n_rings):
            mask = tf.identity(mask)
            if init_H:
                mask = self._extend_connectivity(mask, H, init_H=init_H) 
                init_H = False
            else: 
                mask = self._extend_connectivity(mask, init_H=init_H)

        src_mask = tf.math.logical_not(tf.cast(mask > 0, dtype=tf.bool)) # not(mask > 0)
        return src_mask

    def _extend_connectivity(self, mask, H=None, init_H=False):
        m,n = H.shape
        for i in range(n+m):
            indices = tf.where(H[i] > 0) if init_H else tf.where(mask[i] > 0)
            for j in indices:
                j = j[0]
                ixs = [ [j,n+i],[n+i,j] ] if init_H else [ [i,j],[j,i] ] 
                mask = tf.tensor_scatter_nd_update(mask, ixs, [1.0, 1.0])
                
        return mask
                
        
        






















