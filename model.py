
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, gps_conv_params):
        super(SimpleGNN, self).__init__()
        self.gps_conv = GPSConv(**gps_conv_params)
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gps_conv(x, edge_index, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

# Example usage:
# Assume the input has 16 features and we want 32 output features
in_channels = 16
out_channels = 32
gps_conv_params = {
    'channels': in_channels,
    'conv': None,  # You can pass a specific MessagePassing layer here if needed
    'heads': 1,
    'dropout': 0.0,
    'act': 'relu',
    'norm': 'batch_norm',
    'attn_type': 'multihead',
}

