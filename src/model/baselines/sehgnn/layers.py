"""Feature-only SeHGNN for heterogeneous target-node classification.

Architecture reference: ICT-GIMLab/SeHGNN, revision
e92bd37d0b803457339555684f139b4c8f3e160d, hgb/model.py.
https://github.com/ICT-GIMLab/SeHGNN

Uses mean feature propagation, per-metapath MLPs, transformer semantic fusion,
and a residual task MLP. Label propagation and multi-stage training are disabled.
Only PyTorch is required for propagation; DGL and torch_sparse are not needed.
"""


import math


import torch


from torch import nn


class MetapathLinear(nn.Module):
    """Independent linear weights for each semantic feature channel."""

    def __init__(self, num_paths, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_paths, channels, channels))
        self.bias = nn.Parameter(torch.zeros(num_paths, channels))
        for weight in self.weight:
            nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, features):
        return torch.einsum('bpc,pcd->bpd', features, self.weight) + self.bias


class SemanticFusion(nn.Module):
    """SeHGNN's transformer attention over paths, with a gated residual."""

    def __init__(self, channels, num_heads=1, dropout=0.0):
        super().__init__()
        if channels % (4 * num_heads):
            raise ValueError('hidden_channels must be divisible by 4 * num_heads')
        self.num_heads = num_heads
        self.query = nn.Linear(channels, channels // 4)
        self.key = nn.Linear(channels, channels // 4)
        self.value = nn.Linear(channels, channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        batch, paths, channels = features.shape
        query = self.query(features).reshape(batch, paths, self.num_heads, -1).transpose(1, 2)
        key = self.key(features).reshape(batch, paths, self.num_heads, -1).permute(0, 2, 3, 1)
        value = self.value(features).reshape(batch, paths, self.num_heads, -1).transpose(1, 2)
        attention = (query @ key / math.sqrt(query.shape[-1])).softmax(dim=-1)
        fused = self.attention_dropout(attention) @ value
        fused = fused.transpose(1, 2).reshape(batch, paths, channels)
        return features + self.gate * fused


