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


