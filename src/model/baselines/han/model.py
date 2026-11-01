"""HAN on precomputed target-to-target metapath graphs, not raw relations."""
from torch import nn
from torch.nn import functional as F
from .layers import HANConv


class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metapath_names,
                 target_type, heads=8, dropout=0.6, head_channels=None):
        super().__init__()
        names = tuple(metapath_names)
        if not names or len(set(names)) != len(names):
            raise ValueError('Provide distinct nonempty metapath names')
        if heads < 1 or hidden_channels % heads:
            raise ValueError('hidden_channels must be divisible by positive heads')
        metadata = ([target_type], [(target_type, name, target_type) for name in names])
        self.projection = nn.Linear(in_channels, hidden_channels)
        attention_width = hidden_channels if head_channels is None else head_channels * heads
        self.conv = HANConv({target_type: hidden_channels}, attention_width,
                            metadata, heads=heads, dropout=dropout)
        self.classifier = nn.Linear(attention_width, out_channels)
        self.target_type = target_type
        self.relations = set(metadata[1])
        self.dropout = dropout

    def forward(self, x, metapath_edge_index_dict):
        if set(metapath_edge_index_dict) != self.relations:
            raise ValueError('Supply exactly the configured target-to-target metapath graphs')
        hidden = self.conv({self.target_type: self.projection(x)}, metapath_edge_index_dict)[self.target_type]
        return self.classifier(F.dropout(F.elu(hidden), p=self.dropout, training=self.training))
