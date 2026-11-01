"""GTC Hop2Token transformer and right-normalized relation convolution."""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from ..common import SemanticAttention, CrossViewContrast


class RelationMeanConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__(aggr='mean')
        self.projections = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_relations)])

    def forward(self, x, graphs):
        return sum(layer(self.propagate(graph, x=x, size=(len(x), len(x))))
                   for layer, graph in zip(self.projections, graphs)).relu()


class HopTransformer(nn.Module):
    def __init__(self, in_channels, channels, heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(in_channels, channels)
        from .transformer import EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(channels, 2 * channels, dropout, dropout, heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.Linear(2 * channels, 1)
        self.output = nn.Linear(channels, channels // 2)

    def forward(self, tokens):
        hidden = self.projection(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        root, neighbors = hidden[:, :1], hidden[:, 1:]
        weights = self.attention(torch.cat([root.expand_as(neighbors), neighbors], dim=-1)).softmax(1)
        return self.output((root + (neighbors * weights).sum(1, keepdim=True)).squeeze(1)).relu()
