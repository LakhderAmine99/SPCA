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


from .layers import MetapathLinear, SemanticFusion
from .preprocessing import *

class SeHGNN(nn.Module):
    """Supervised SeHGNN with cached metapath features and movie class logits."""

    def __init__(
        self, metapath_features, num_classes, hidden_channels=128,
        dropout=0.5, num_projection_layers=2, num_task_layers=4, num_heads=1,
    ):
        super().__init__()
        if not metapath_features or num_projection_layers < 1 or num_task_layers < 1:
            raise ValueError('Features and at least one projection/task layer are required')
        self.paths = tuple(sorted(metapath_features))
        self.embeddings = nn.ModuleList()
        num_nodes = metapath_features[self.paths[0]].shape[0]
        for i, path in enumerate(self.paths):
            features = metapath_features[path]
            if features.ndim != 2 or features.shape[0] != num_nodes:
                raise ValueError('All metapath features must have the same target node count')
            # Keep fixed features on the model's device, outside its checkpoint.
            self.register_buffer(f'features_{i}', features.detach(), persistent=False)
            self.embeddings.append(nn.Linear(features.shape[1], hidden_channels, bias=False))

        layers = []
        for _ in range(num_projection_layers):
            layers.extend([
                MetapathLinear(len(self.paths), hidden_channels),
                nn.LayerNorm((len(self.paths), hidden_channels)),
                nn.PReLU(), nn.Dropout(dropout),
            ])
        self.feature_projection = nn.Sequential(*layers)
        self.semantic_fusion = SemanticFusion(hidden_channels, num_heads=num_heads)
        self.fuse_projection = nn.Linear(len(self.paths) * hidden_channels, hidden_channels)
        self.task_input = nn.Sequential(nn.PReLU(), nn.Dropout(dropout))
        self.task_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels, affine=False),
                nn.PReLU(), nn.Dropout(dropout),
            ) for _ in range(num_task_layers - 1)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, num_classes),
            nn.LayerNorm(num_classes, elementwise_affine=False),
        )

    def forward(self, x_dict=None, edge_index_dict=None):
        # The graph is accepted for compatibility with the shared experiment
        # runner. All neighbor aggregation was completed before training.
        channels = [
            projection(getattr(self, f'features_{i}'))
            for i, projection in enumerate(self.embeddings)
        ]
        features = self.feature_projection(torch.stack(channels, dim=1))
        features = self.semantic_fusion(features).transpose(1, 2).flatten(start_dim=1)
        hidden = self.task_input(self.fuse_projection(features))
        for block in self.task_blocks:
            hidden = hidden + block(hidden)
        return self.classifier(hidden)


