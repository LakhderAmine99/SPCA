"""Paper-based HetTree for the shared single-label classification experiments.

Reference: Guan et al., Heterogeneous Graph Neural Network on Semantic Tree,
AAAI 2025, Sections 4.1--4.3, equations (1)--(6).
https://doi.org/10.1609/aaai.v39i16.33860

The paper's microsoft/HetTree repository was unavailable when this module was
written. This is an independent implementation, not a port of the authors' code.
It uses relation-wise mean propagation and PyTorch only. Training labels are
matched to their feature paths after removing all return-walk contributions of
each destination node's own label. No held-out labels or pseudo-labels are used.
"""


import torch


from torch import nn


from torch.nn import functional as F


from ..sehgnn import precompute_metapath_features


from .layers import SubtreeAttention, _mlp
from .preprocessing import *

class HetTree(nn.Module):
    """Metapath feature/label matching, subtree attention, and two residuals."""

    def __init__(self, metapath_features, num_classes, label_features=None, hidden_channels=128, dropout=0.5):
        super().__init__()
        if not metapath_features or num_classes < 1 or hidden_channels < 1 or not 0 <= dropout < 1:
            raise ValueError('Features, positive channel/class counts, and dropout in [0, 1) are required')
        self.paths = tuple(sorted(metapath_features, key=lambda path: (len(path), path)))
        roots = [path for path in self.paths if len(path) == 1]
        if len(roots) != 1:
            raise ValueError('A semantic tree requires exactly one raw target feature root')
        self.target_type = roots[0][0]
        if any(not path or path[-1] != self.target_type for path in self.paths):
            raise ValueError('All feature paths must end at the same target type')
        path_indices = {path: i for i, path in enumerate(self.paths)}
        children = [[] for _ in self.paths]
        for i, path in enumerate(self.paths):
            if len(path) > 1:
                if path[1:] not in path_indices:
                    raise ValueError(f'Missing semantic-tree parent: {path[1:]}')
                children[path_indices[path[1:]]].append(i)
        self.tree_children = tuple(tuple(indices) for indices in children)
        closed_paths = {path for path in self.paths if len(path) > 1 and path[0] == self.target_type}
        if label_features is not None and not set(label_features) <= closed_paths:
            raise ValueError('Labels must match non-root target-to-target feature paths')
        self.label_paths = tuple(path for path in self.paths if label_features is not None and path in label_features)
        self.uses_labels = bool(self.label_paths)
        self.projections = nn.ModuleList()
        num_nodes = metapath_features[roots[0]].shape[0]
        for i, path in enumerate(self.paths):
            features = metapath_features[path]
            if features.ndim != 2 or features.shape[0] != num_nodes or not features.is_floating_point():
                raise ValueError('All features must be floating matrices with the same target node count')
            channels = features.shape[1]
            self.register_buffer(f'features_{i}', features.detach(), persistent=False)
            if path in self.label_paths:
                values = label_features[path]
                if values.shape != (num_nodes, num_classes) or values.device != features.device:
                    raise ValueError('Label features must match the target count, classes, and feature device')
                self.register_buffer(f'labels_{i}', values.detach().to(features.dtype), persistent=False)
                channels += num_classes
            self.projections.append(_mlp(channels, hidden_channels, hidden_channels, dropout))
        self.subtree_attention = nn.ModuleDict({
            str(i): SubtreeAttention(hidden_channels, dropout) for i, indices in enumerate(children) if indices
        })
        self.classifier = _mlp(hidden_channels, hidden_channels, num_classes, dropout)
        self.feature_residual = _mlp(metapath_features[roots[0]].shape[1], hidden_channels, num_classes, dropout)
        self.label_residual = _mlp(num_classes, hidden_channels, num_classes, dropout) if self.uses_labels else None

    def forward(self, x_dict=None, edge_index_dict=None):
        # All graph and training-label propagation was completed before fitting.
        transformed = []
        for i, path in enumerate(self.paths):
            features = getattr(self, f'features_{i}')
            if path in self.label_paths:
                features = torch.cat([features, getattr(self, f'labels_{i}')], dim=-1)
            transformed.append(self.projections[i](features))
        encoded = list(transformed)
        for i in range(len(self.paths) - 1, -1, -1):
            if self.tree_children[i]:
                encoded[i], _ = self.subtree_attention[str(i)](
                    transformed[i],
                    torch.stack([transformed[j] for j in self.tree_children[i]], dim=1),
                    torch.stack([encoded[j] for j in self.tree_children[i]], dim=1),
                )
        logits = self.classifier(torch.stack(encoded, dim=1).sum(dim=1))
        logits = logits + self.feature_residual(getattr(self, 'features_0'))
        if self.label_residual is not None:
            values = torch.stack([getattr(self, f'labels_{i}') for i, path in enumerate(self.paths)
                                  if path in self.label_paths], dim=1).sum(dim=1)
            logits = logits + self.label_residual(values)
        return logits


