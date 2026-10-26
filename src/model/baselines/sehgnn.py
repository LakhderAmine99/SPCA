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


@torch.no_grad()
@torch.sparse.check_sparse_tensor_invariants()
def precompute_metapath_features(
    x_dict, edge_index_dict, target_type='movie', num_hops=2, metapaths=None,
):
    """Aggregate selected paths, or all target-ending paths; retain raw features."""
    if num_hops < 0:
        raise ValueError('num_hops must be non-negative')
    if target_type not in x_dict:
        raise ValueError(f'Missing target features: {target_type}')

    selected = None if metapaths is None else {tuple(path) for path in metapaths}
    prefixes = None
    if selected is not None:
        if not selected:
            raise ValueError('metapaths must contain at least one path')
        if any(len(path) < 2 or path[-1] != target_type for path in selected):
            raise ValueError('Every metapath must have an edge and end at the target type')
        if any(len(path) - 1 > num_hops for path in selected):
            raise ValueError('A selected metapath exceeds num_hops')
        prefixes = {path[:length] for path in selected for length in range(1, len(path) + 1)}

    relations = []
    endpoints = set()
    for (src, _, dst), edge_index in edge_index_dict.items():
        if (src, dst) in endpoints:
            raise ValueError('Multiple relations with the same endpoints need distinct path keys')
        endpoints.add((src, dst))
        features = x_dict[src]
        edges = edge_index.to(features.device)
        values = torch.ones(edges.shape[1], dtype=features.dtype, device=features.device)
        adjacency = torch.sparse_coo_tensor(
            edges.flip(0), values, (x_dict[dst].shape[0], features.shape[0]), check_invariants=True,
        ).coalesce()
        degree = torch.zeros(x_dict[dst].shape[0], dtype=features.dtype, device=features.device)
        degree.index_add_(0, adjacency.indices()[0], adjacency.values())
        normalized = torch.sparse_coo_tensor(
            adjacency.indices(),
            adjacency.values() / degree[adjacency.indices()[0]].clamp_min(1),
            adjacency.shape, check_invariants=True,
        ).coalesce()
        relations.append((src, dst, normalized))

    current = {(node_type,): x.detach() for node_type, x in x_dict.items()}
    result = {(target_type,): current[(target_type,)]}
    if prefixes is not None:
        current = {path: features for path, features in current.items() if path in prefixes}
    for hop in range(1, num_hops + 1):
        next_features = {}
        for src, dst, adjacency in relations:
            if hop == num_hops and dst != target_type:
                continue
            for path, features in current.items():
                next_path = path + (dst,)
                if path[-1] == src and (prefixes is None or next_path in prefixes):
                    next_features[next_path] = torch.sparse.mm(adjacency, features)
        result.update({path: x for path, x in next_features.items()
                       if path[-1] == target_type and (selected is None or path in selected)})
        current = next_features
    if selected is not None and selected - result.keys():
        missing = sorted(selected - result.keys())
        raise ValueError(f'Cannot compute selected metapaths from the graph: {missing}')
    return result


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
