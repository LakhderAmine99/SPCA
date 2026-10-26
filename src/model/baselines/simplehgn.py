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


def project_metapath_graph(x_dict, edge_index_dict, metapaths, target_type):
    """Create a binary target-to-target relation for each requested metapath."""
    import numpy as np
    import scipy.sparse as sp

    paths = tuple(tuple(path) for path in metapaths)
    if not paths or any(len(path) < 2 or path[0] != target_type or path[-1] != target_type
                        for path in paths):
        raise ValueError('Metapaths must start and end at the target type and contain an edge')
    needed = {(source, destination) for path in paths for source, destination in zip(path, path[1:])}
    matrices = {}
    for (source, _, destination), edges in edge_index_dict.items():
        key = (source, destination)
        if key not in needed:
            continue
        row, column = edges.detach().cpu().numpy()
        adjacency = sp.csr_matrix(
            (np.ones(len(row), dtype=bool), (row, column)),
            shape=(x_dict[source].shape[0], x_dict[destination].shape[0]),
        )
        matrices[key] = adjacency if key not in matrices else matrices[key] + adjacency
    missing = needed - matrices.keys()
    if missing:
        raise ValueError(f'Missing metapath relations: {sorted(missing)}')

    cache = {}

    def compose(path):
        if path not in cache:
            if len(path) == 2:
                cache[path] = matrices[path]
            else:
                middle = len(path) // 2
                cache[path] = (compose(path[:middle + 1]) @ compose(path[middle:])).tocsr()
        return cache[path]

    projected = {}
    device = x_dict[target_type].device
    for path in paths:
        adjacency = compose(path).copy()
        # Simple-HGN adds its dedicated self-loop relation when packing.
        adjacency.setdiag(False)
        adjacency.eliminate_zeros()
        edges = adjacency.tocoo()
        edge_index = torch.from_numpy(np.stack([edges.row, edges.col]).astype(np.int64)).to(device)
        projected[target_type, '_'.join(path), target_type] = edge_index
    return projected


def _aggregate_message_chunk(projected, attention, source, destination):
    messages = projected[source] * attention.unsqueeze(-1)
    output = torch.zeros_like(projected)
    return output.index_add_(0, destination, messages)


def _aggregate_dense_messages(projected, attention, source, destination):
    num_nodes, heads, _ = projected.shape
    weights = attention.new_zeros(heads, num_nodes * num_nodes)
    # Sum parallel relation weights after the joint attention softmax.
    weights.index_add_(1, destination * num_nodes + source, attention.t())
    output = torch.bmm(weights.view(heads, num_nodes, num_nodes), projected.transpose(0, 1))
    return output.transpose(0, 1)


def pack_typed_graph(x_dict, edge_index_dict):
    """Pack node types into one graph, preserving relation IDs and adding loops."""
    node_types = tuple(sorted(x_dict))
    offsets = {}
    num_nodes = 0
    for node_type in node_types:
        offsets[node_type] = num_nodes
        num_nodes += x_dict[node_type].shape[0]
    device = x_dict[node_types[0]].device
    relations = tuple(sorted(edge_index_dict))
    all_edges, all_types = [], []
    for relation_id, (src, relation, dst) in enumerate(relations):
        edge_index = edge_index_dict[(src, relation, dst)].to(device)
        shifted = edge_index + edge_index.new_tensor([[offsets[src]], [offsets[dst]]])
        # All self-loops get the dedicated self-loop relation below.
        shifted = shifted[:, shifted[0] != shifted[1]]
        all_edges.append(shifted)
        all_types.append(torch.full((shifted.shape[1],), relation_id, dtype=torch.long, device=device))
    nodes = torch.arange(num_nodes, device=device)
    all_edges.append(torch.stack([nodes, nodes]))
    all_types.append(torch.full((num_nodes,), len(relations), dtype=torch.long, device=device))
    return node_types, offsets, torch.cat(all_edges, dim=1), torch.cat(all_types)


class SimpleHGNConv(nn.Module):
    """HGB attention layer with a learned edge type term and residual attention."""

    def __init__(
        self, in_channels, out_channels, num_relations, heads=8, edge_channels=64,
        dropout=0.5, negative_slope=0.05, residual=True,
        attention_residual=0.05, activation=None,
    ):
        super().__init__()
        if heads < 1 or not 0 <= attention_residual <= 1:
            raise ValueError('heads must be positive and attention_residual must be in [0, 1]')
        self.heads = heads
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.attention_residual = attention_residual
        self.negative_slope = negative_slope
        self.activation = activation
        self.message_chunk_size = 65536
        self.max_dense_elements = 256 * 1024 * 1024
        self.node_projection = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.edge_embedding = nn.Embedding(num_relations, edge_channels)
        self.edge_projection = nn.Linear(edge_channels, heads * edge_channels, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_channels))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_channels))
        self.attn_edge = nn.Parameter(torch.empty(heads, edge_channels))
        self.feature_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual = None
        if residual:
            self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(
                in_channels, heads * out_channels, bias=False,
            )
        gain = math.sqrt(2)
        for projection in [self.node_projection, self.edge_projection]:
            nn.init.xavier_normal_(projection.weight, gain=gain)
        for attention in [self.attn_src, self.attn_dst, self.attn_edge]:
            # The singleton dimension matches the reference attention tensors.
            nn.init.xavier_normal_(attention.unsqueeze(0), gain=gain)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight, gain=gain)

    def forward(self, features, edge_index, edge_types, previous_attention=None):
        dropped = self.feature_dropout(features)
        projected = self.node_projection(dropped).view(-1, self.heads, self.out_channels)
        src, dst = edge_index
        # Project the small relation table once, avoiding an E x heads x
        # edge_channels tensor on dense metapath graphs.
        edge_features = self.edge_projection(self.edge_embedding.weight)
        edge_features = edge_features.view(-1, self.heads, self.edge_channels)
        node_src = (projected * self.attn_src).sum(-1)
        node_dst = (projected * self.attn_dst).sum(-1)
        edge_scores = (edge_features * self.attn_edge).sum(-1)[edge_types]
        scores = F.leaky_relu(node_src[src] + node_dst[dst] + edge_scores, self.negative_slope)
        # One softmax across ALL edge types entering each destination node.
        attention = softmax(scores, dst, num_nodes=features.shape[0])
        attention = self.attention_dropout(attention)
        if previous_attention is not None:
            if previous_attention.shape != attention.shape:
                raise ValueError('Residual attention must have the same edge and head dimensions')
            attention = ((1 - self.attention_residual) * attention
                         + self.attention_residual * previous_attention.detach())
        output = projected.new_zeros(features.shape[0], self.heads, self.out_channels)
        matrix_size = features.shape[0] ** 2
        if src.numel() > matrix_size / 2 and matrix_size * self.heads <= self.max_dense_elements:
            arguments = (projected, attention, src, dst)
            if torch.is_grad_enabled():
                output = checkpoint(_aggregate_dense_messages, *arguments, use_reentrant=False)
            else:
                output = _aggregate_dense_messages(*arguments)
        else:
            for start in range(0, src.numel(), self.message_chunk_size):
                stop = start + self.message_chunk_size
                arguments = (projected, attention[start:stop], src[start:stop], dst[start:stop])
                if torch.is_grad_enabled() and src.numel() > self.message_chunk_size:
                    # Recompute large gathered messages in backward instead of
                    # retaining one feature vector per metapath edge.
                    contribution = checkpoint(_aggregate_message_chunk, *arguments, use_reentrant=False)
                else:
                    contribution = _aggregate_message_chunk(*arguments)
                output = output + contribution
        if self.residual is not None:
            output = output + self.residual(dropped).view(features.shape[0], -1, self.out_channels)
        if self.activation is not None:
            output = self.activation(output)
        return output, attention.detach()


class SimpleHGN(nn.Module):
    """Typed graph attention followed by normalized target-node class scores."""

    def __init__(
        self, x_dict, edge_index_dict, num_classes, hidden_channels=64,
        num_layers=2, heads=8, edge_channels=64, dropout=0.5,
        negative_slope=0.05, attention_residual=0.05, target_type='movie',
        metapaths=None,
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
        for layer in range(num_layers):
            self.layers.append(SimpleHGNConv(
                hidden_channels if layer == 0 else hidden_channels * heads,
                hidden_channels, len(edge_index_dict) + 1, heads=heads,
                edge_channels=edge_channels, dropout=dropout,
                negative_slope=negative_slope, residual=(layer > 0),
                attention_residual=attention_residual, activation=F.elu,
            ))
        self.output_layer = SimpleHGNConv(
            hidden_channels * heads, num_classes, len(edge_index_dict) + 1,
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
