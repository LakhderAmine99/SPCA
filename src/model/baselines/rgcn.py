"""Relation-aware GCN on a packed graph with stable integer relation IDs."""
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, dropout=0.5, num_bases=None):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([
            RGCNConv(a, b, num_relations, num_bases=num_bases)
            for a, b in zip(widths, widths[1:])
        ])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i + 1 < len(self.layers):
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x
