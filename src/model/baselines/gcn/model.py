"""Local GCN architecture (PyTorch port of tkipf/gcn)."""
from torch import nn
from torch.nn import functional as F
from .layers import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([GCNConv(a, b) for a, b in zip(widths, widths[1:])])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(F.dropout(x, self.dropout, self.training), edge_index)
            if i + 1 < len(self.layers):
                x = F.relu(x)
        return x
