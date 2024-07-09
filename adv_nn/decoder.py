class Decoder():
    def init(self, N):
      self.N = N
      self.transformer = # mask

    def call(self, x):
        for _ in range(self.N):
            x = self.transformer(x)
        return x
