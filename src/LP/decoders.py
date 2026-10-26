"""Decode endpoint embeddings into raw binary link logits."""
import torch
from torch import nn


def endpoints(source, destination, edge_label_index):
    if source.ndim != 2 or destination.ndim != 2 or source.shape[1] != destination.shape[1]:
        raise ValueError('Endpoint embeddings must be matrices with equal widths')
    if edge_label_index.dtype != torch.long or edge_label_index.ndim != 2 or edge_label_index.shape[0] != 2:
        raise ValueError('edge_label_index must be a long tensor of shape [2, num_edges]')
    if edge_label_index.numel() and (
        (edge_label_index < 0).any()
        or (edge_label_index[0] >= source.shape[0]).any()
        or (edge_label_index[1] >= destination.shape[0]).any()
    ):
        raise ValueError('edge_label_index contains an out-of-range endpoint')
    return source[edge_label_index[0]], destination[edge_label_index[1]]


class DotProductDecoder(nn.Module):
    """Symmetric scores for same-type links; no sigmoid during training."""

    def forward(self, source, destination, edge_label_index):
        left, right = endpoints(source, destination, edge_label_index)
        return (left * right).sum(dim=-1)


class MLPDecoder(nn.Module):
    """Ordered endpoint concatenation for directed or cross-type links."""

    def __init__(self, embedding_channels, hidden_channels=None, dropout=0.0):
        super().__init__()
        hidden_channels = embedding_channels if hidden_channels is None else hidden_channels
        if embedding_channels < 1 or hidden_channels < 1:
            raise ValueError('Channel counts must be positive')
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_channels, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, 1),
        )

    def forward(self, source, destination, edge_label_index):
        left, right = endpoints(source, destination, edge_label_index)
        return self.network(torch.cat((left, right), dim=-1)).squeeze(-1)
