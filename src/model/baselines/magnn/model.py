"""MAGNN with typed projection, instance encoding and semantic attention."""
import torch
from torch import nn
from torch.nn import functional as F
from .layers import InstanceEncoder, InstanceAttention, SemanticAttention


class MAGNN(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metapaths,
                 heads=8, dropout=0.5, instance_encoder='RotatE0', attention_channels=128, head_channels=None):
        super().__init__()
        self.paths = tuple(map(tuple, metapaths))
        self.projections = nn.ModuleDict({t: nn.Linear(c, hidden_channels) for t, c in in_channels_dict.items()})
        head_channels = head_channels or hidden_channels
        self.instance_projection = nn.Linear(hidden_channels, head_channels)
        self.encoders = nn.ModuleList([InstanceEncoder(head_channels, heads, instance_encoder) for _ in self.paths])
        self.attentions = nn.ModuleList([InstanceAttention(head_channels, heads, dropout) for _ in self.paths])
        self.semantic = SemanticAttention(head_channels * heads, attention_channels)
        self.output = nn.Linear(head_channels * heads, out_channels)
        self.relation_pairs = tuple(sorted({tuple(sorted((a, b))) for p in self.paths for a, b in zip(p, p[1:])}))
        self.rotations = nn.Parameter(torch.randn(len(self.relation_pairs), head_channels // 2, 2))
        self.dropout = dropout

    def forward(self, x_dict, instances):
        features = {t: self.instance_projection(F.dropout(layer(x_dict[t]), self.dropout, self.training))
                    for t, layer in self.projections.items()}
        views = []
        for path, encoder, attention in zip(self.paths, self.encoders, self.attentions):
            ids = instances[path].to(features[path[-1]].device)
            sequence = torch.stack([features[t][ids[:, i]] for i, t in enumerate(path)], dim=1)
            rotations = []
            for a, b in zip(path, path[1:]):
                pair = tuple(sorted((a, b)))
                rotation = torch.view_as_complex(F.normalize(self.rotations[self.relation_pairs.index(pair)], dim=-1).contiguous())
                rotations.append(rotation if (a, b) == pair else rotation.conj())
            views.append(F.elu(attention(encoder(sequence, rotations), ids[:, -1], len(features[path[-1]]))))
        return self.output(self.semantic(views))
