"""GTC transformer/GNN co-contrastive encoder on fixed metapath graphs."""
from torch import nn
from torch.nn import functional as F
from .layers import HopTransformer, RelationMeanConv, SemanticAttention, CrossViewContrast


class GTC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_metapaths,
                 heads=8, dropout=0.5, transformer_layers=6, num_layers=2,
                 temperature=0.7, balance=0.4, contrast_batch_size=512):
        super().__init__()
        self.transformers = nn.ModuleList([HopTransformer(in_channels, hidden_channels,
            heads, transformer_layers, dropout) for _ in range(num_metapaths)])
        self.transformer_projection = nn.Linear(hidden_channels // 2, hidden_channels)
        self.semantic = SemanticAttention(hidden_channels, hidden_channels, dropout)
        self.projection = nn.Linear(in_channels, hidden_channels)
        widths = [hidden_channels] + [2 * hidden_channels] * (num_layers - 1) + [hidden_channels]
        self.gnn = nn.ModuleList([RelationMeanConv(a, b, num_metapaths) for a, b in zip(widths, widths[1:])])
        self.contrast = CrossViewContrast(hidden_channels, temperature, balance, contrast_batch_size)
        self.output = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def transformer_view(self, tokens):
        return self.semantic([self.transformer_projection(model(values)) for model, values in zip(self.transformers, tokens)])

    def contrastive_loss(self, x, graphs, tokens, positive_edges):
        hidden = F.elu(F.dropout(self.projection(x), self.dropout, self.training))
        for layer in self.gnn:
            hidden = layer(hidden, graphs)
        return self.contrast(self.transformer_view(tokens), hidden, positive_edges)

    def forward(self, x, graphs, tokens, detach_encoder=False):
        hidden = self.transformer_view(tokens)
        return self.output(hidden.detach() if detach_encoder else hidden)
