"""HeCo two-view encoder; schema relations follow the paper's metapath protocol."""
from torch import nn
from torch.nn import functional as F
from .layers import MetapathEncoder, SchemaEncoder, CrossViewContrast


class HeCo(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_metapaths,
                 heads=8, dropout=0.5, temperature=0.8, balance=0.5, contrast_batch_size=512):
        super().__init__()
        self.projection = nn.Linear(in_channels, hidden_channels)
        self.metapath = MetapathEncoder(hidden_channels, num_metapaths, dropout)
        self.schema = SchemaEncoder(hidden_channels, num_metapaths, dropout, heads)
        self.contrast = CrossViewContrast(hidden_channels, temperature, balance, contrast_batch_size)
        self.output = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def views(self, x, graphs):
        hidden = F.elu(F.dropout(self.projection(x), self.dropout, self.training))
        return self.metapath(hidden, graphs), self.schema(hidden, graphs)

    def contrastive_loss(self, x, graphs, positive_edges):
        return self.contrast(*self.views(x, graphs), positive_edges)

    def forward(self, x, graphs, detach_encoder=False):
        hidden = self.metapath(F.elu(self.projection(x)), graphs)
        return self.output(hidden.detach() if detach_encoder else hidden)
