"""PyG metapath GCN and schema attention encoders adapted from HeCo."""
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv
from ..common import SemanticAttention, CrossViewContrast


class MetapathEncoder(nn.Module):
    def __init__(self, channels, num_paths, dropout):
        super().__init__()
        self.layers = nn.ModuleList([GCNConv(channels, channels) for _ in range(num_paths)])
        self.activations = nn.ModuleList([nn.PReLU() for _ in range(num_paths)])
        self.semantic = SemanticAttention(channels, channels, dropout)

    def forward(self, x, graphs):
        return self.semantic([act(layer(x, graph)) for layer, act, graph in zip(self.layers, self.activations, graphs)])


class SchemaEncoder(nn.Module):
    def __init__(self, channels, num_paths, dropout, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([GATConv(channels, channels, heads=heads,
            concat=False, dropout=dropout, add_self_loops=False) for _ in range(num_paths)])
        self.semantic = SemanticAttention(channels, channels, dropout)

    def forward(self, x, graphs):
        return self.semantic([F.elu(layer(x, graph)) for layer, graph in zip(self.layers, graphs)])
