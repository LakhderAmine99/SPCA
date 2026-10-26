"""HGT classifier on the original heterogeneous graph using PyG HGTConv."""
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata,
                 target_type, heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        if num_layers < 1 or heads < 1 or hidden_channels % heads:
            raise ValueError('Require positive layers/heads and hidden_channels divisible by heads')
        self.projections = nn.ModuleDict({
            key: nn.Linear(width, hidden_channels) for key, width in in_channels_dict.items()
        })
        self.layers = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.target_type = target_type
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        hidden = {key: F.relu(self.projections[key](x)) for key, x in x_dict.items()}
        for layer in self.layers:
            updates = layer(hidden, edge_index_dict)
            hidden = {
                key: F.dropout(F.relu(updates[key]), p=self.dropout, training=self.training)
                if updates.get(key) is not None else value
                for key, value in hidden.items()
            }
        return self.classifier(hidden[self.target_type])
