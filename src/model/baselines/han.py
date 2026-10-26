"""HAN on precomputed target-to-target metapath graphs, not raw relations."""
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metapath_names,
                 target_type, heads=8, dropout=0.6):
        super().__init__()
        names = tuple(metapath_names)
        if not names or len(set(names)) != len(names):
            raise ValueError('Provide distinct nonempty metapath names')
        if heads < 1 or hidden_channels % heads:
            raise ValueError('hidden_channels must be divisible by positive heads')
        metadata = ([target_type], [(target_type, name, target_type) for name in names])
        self.conv = HANConv({target_type: in_channels}, hidden_channels,
                            metadata, heads=heads, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.target_type = target_type
        self.relations = set(metadata[1])
        self.dropout = dropout

    def forward(self, x, metapath_edge_index_dict):
        if set(metapath_edge_index_dict) != self.relations:
            raise ValueError('Supply exactly the configured target-to-target metapath graphs')
        hidden = self.conv({self.target_type: x}, metapath_edge_index_dict)[self.target_type]
        return self.classifier(F.dropout(F.elu(hidden), p=self.dropout, training=self.training))
