"""HGB Simple-HGN adapted to heterogeneous target-node classification.

Reference: THUDM/HGB at ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe,
NC/benchmark/methods/baseline/{conv.py,GNN.py,run_new.py}.
https://github.com/THUDM/HGB

Preserves edge-type embeddings, joint incoming-edge attention, node and edge
residuals, typed self-loops, multi-head aggregation, and output L2 normalization.
The graph operations use PyTorch/PyG, so DGL is not required.
"""


import math


import torch


from torch import nn


from torch.nn import functional as F


from torch.utils.checkpoint import checkpoint


from torch_geometric.utils import softmax


from .layers import SimpleHGNConv, _aggregate_dense_messages, _aggregate_message_chunk
from .preprocessing import *

class SimpleHGN(nn.Module):
    """Typed graph attention followed by normalized target-node class scores."""

    def __init__(
        self, x_dict, edge_index_dict, num_classes, hidden_channels=64,
        num_layers=2, heads=8, edge_channels=64, dropout=0.5,
        negative_slope=0.05, attention_residual=0.05, target_type='movie',
        metapaths=None, head_channels=None,
    ):
        super().__init__()
        if num_layers < 1 or target_type not in x_dict:
            raise ValueError('At least one hidden layer and target node features are required')
        self.metapaths = None if metapaths is None else tuple(tuple(path) for path in metapaths)
        self.graph_mode = 'full-typed-graph' if metapaths is None else 'metapath-projected'
        if self.metapaths is not None:
            edge_index_dict = project_metapath_graph(x_dict, edge_index_dict, self.metapaths, target_type)
            x_dict = {target_type: x_dict[target_type]}
        self.relations = tuple(sorted(edge_index_dict))
        node_types, offsets, edges, edge_types = pack_typed_graph(x_dict, edge_index_dict)
        self.node_types = node_types
        self.target_type = target_type
        self.target_start = offsets[target_type]
        self.target_count = x_dict[target_type].shape[0]
        self.register_buffer('edge_index', edges, persistent=False)
        self.register_buffer('edge_types', edge_types, persistent=False)
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(x_dict[node_type].shape[1], hidden_channels)
            for node_type in self.node_types
        })
        for projection in self.projections.values():
            nn.init.xavier_normal_(projection.weight, gain=math.sqrt(2))
        self.layers = nn.ModuleList()
        head_channels = head_channels or hidden_channels
        for layer in range(num_layers):
            self.layers.append(SimpleHGNConv(
                hidden_channels if layer == 0 else head_channels * heads,
                head_channels, len(edge_index_dict) + 1, heads=heads,
                edge_channels=edge_channels, dropout=dropout,
                negative_slope=negative_slope, residual=(layer > 0),
                attention_residual=attention_residual, activation=F.elu,
            ))
        self.output_layer = SimpleHGNConv(
            head_channels * heads, num_classes, len(edge_index_dict) + 1,
            heads=1, edge_channels=edge_channels, dropout=dropout,
            negative_slope=negative_slope, residual=True,
        )

    def forward(self, x_dict, edge_index_dict=None):
        hidden = torch.cat([self.projections[node_type](x_dict[node_type])
                            for node_type in self.node_types], dim=0)
        attention = None
        for layer in self.layers:
            hidden, attention = layer(hidden, self.edge_index, self.edge_types, attention)
            hidden = hidden.flatten(start_dim=1)
        # The single-head output layer starts a fresh attention computation,
        # as in HGB; residual attention is carried between hidden layers only.
        logits, _ = self.output_layer(hidden, self.edge_index, self.edge_types)
        logits = F.normalize(logits.mean(dim=1), p=2, dim=-1, eps=1e-12)
        return logits[self.target_start:self.target_start + self.target_count]


