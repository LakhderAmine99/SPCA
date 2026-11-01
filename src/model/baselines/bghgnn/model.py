"""BG-HGNN low-rank feature fusion and a shared homogeneous GAT."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from .layers import BlendFusion


class BGHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 heads=8, num_layers=2, dropout=0.5, rank=4, type_channels=512, combine_channels=64,
                 head_channels=None):
        super().__init__()
        self.register_buffer('type_encoding', torch.rand(1, type_channels))
        self.register_buffer('relation_encoding', torch.rand(num_relations, type_channels))
        self.fusion = BlendFusion(in_channels, type_channels, combine_channels, rank)
        head_channels = head_channels or hidden_channels
        self.projection = nn.Linear(combine_channels, hidden_channels)
        widths = [hidden_channels] + [head_channels * heads] * (num_layers - 1)
        self.layers = nn.ModuleList([GATConv(width, head_channels, heads=heads,
            dropout=dropout) for width in widths])
        self.output = nn.Linear(head_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        relation = x.new_zeros(len(x), self.relation_encoding.size(1))
        counts = x.new_zeros(len(x))
        for i, encoding in enumerate(self.relation_encoding):
            edges = edge_index[:, edge_type == i]
            incoming = torch.bincount(edges[1], minlength=len(x)).to(x.dtype)
            outgoing = torch.bincount(edges[0], minlength=len(x)).to(x.dtype)
            relation += (incoming - outgoing).unsqueeze(-1) * encoding
            counts += incoming + outgoing
        relation /= counts.clamp_min(1).unsqueeze(-1)
        hidden = self.projection(self.fusion(x, self.type_encoding.expand(len(x), -1), relation))
        for layer in self.layers:
            hidden = F.elu(layer(F.dropout(hidden, self.dropout, self.training), edge_index))
        return self.output(hidden)
