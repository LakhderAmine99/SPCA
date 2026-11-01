"""GAT with independently configurable projection and per-head dimensions."""
from torch import nn
from torch.nn import functional as F
from .layers import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 heads=8, head_channels=None, dropout=0.6):
        super().__init__()
        if num_layers < 1 or heads < 1:
            raise ValueError('Layers and heads must be positive')
        head_channels = head_channels or hidden_channels
        self.projection = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        width = hidden_channels
        for i in range(num_layers):
            last = i == num_layers - 1
            self.layers.append(GATConv(width, out_channels if last else head_channels,
                heads=1 if last else heads, dropout=dropout, concat=not last))
            width = head_channels * heads
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.projection(x)
        for i, layer in enumerate(self.layers):
            x = layer(F.dropout(x, self.dropout, self.training), edge_index)
            if i + 1 < len(self.layers):
                x = F.elu(x)
        return x
