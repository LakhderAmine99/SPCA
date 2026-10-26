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

from .sehgnn import precompute_metapath_features


def _normalized_relations(x_dict, edge_index_dict):
    relations = {}
    for (source, _, destination), edge_index in edge_index_dict.items():
        key = (source, destination)
        if key in relations:
            raise ValueError('Relations with the same endpoints must be combined first')
        features = x_dict[source]
        edges = edge_index.to(features.device)
        adjacency = torch.sparse_coo_tensor(
            edges.flip(0), torch.ones(edges.shape[1], dtype=features.dtype, device=features.device),
            (x_dict[destination].shape[0], features.shape[0]),
        ).coalesce()
        degree = torch.zeros(adjacency.shape[0], dtype=features.dtype, device=features.device)
        degree.index_add_(0, adjacency.indices()[0], adjacency.values())
        relations[key] = torch.sparse_coo_tensor(
            adjacency.indices(),
            adjacency.values() / degree[adjacency.indices()[0]].clamp_min(1),
            adjacency.shape,
        ).coalesce()
    return relations


def _propagate_from_target(seed, paths, relations):
    """Share intermediate propagation across paths starting at one target type."""
    prefixes = {path[:length] for path in paths for length in range(1, len(path) + 1)}
    values = {}
    for path in sorted(prefixes, key=lambda value: (len(value), value)):
        values[path] = seed if len(path) == 1 else torch.sparse.mm(
            relations[path[-2:]], values[path[:-1]],
        )
    return {path: values[path] for path in paths}


@torch.no_grad()
def precompute_hettree_inputs(
    x_dict, edge_index_dict, target_type='movie', num_hops=2, *,
    labels=None, train_mask=None, num_classes=None, use_labels=True,
    label_batch_size=128, metapaths=None, include_ancestor_features=True,
):
    """Return semantic-tree features and optional self-excluded training labels.

    Keys use source-to-target order. A node's tree parent is therefore its suffix
    path[1:]. Selected paths are completed with these ancestors. By default all
    target-ending metapaths up to num_hops are retained, as in the paper.
    With include_ancestor_features=False, only the requested paths and raw root
    carry input features; missing ancestors are zero-input structural nodes.

    For a closed path's propagation matrix P, label inputs are
        (P @ Y - diag(P) * Y) / (P @ 1 - diag(P)).
    Y is one-hot only on train_mask and zero elsewhere. The diagonal is computed
    exactly in bounded-width identity blocks; no dense N-by-N matrix is stored.
    A row with no remaining neighbor mass receives zeros.
    """
    if target_type not in x_dict:
        raise ValueError(f'Missing target features: {target_type}')
    if use_labels:
        if labels is None or train_mask is None or num_classes is None:
            raise ValueError('Label propagation requires labels, train_mask, and num_classes')
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        if not isinstance(label_batch_size, int) or label_batch_size < 1:
            raise ValueError('label_batch_size must be a positive integer')
        if labels.ndim != 1 or train_mask.shape != labels.shape or train_mask.dtype != torch.bool:
            raise ValueError('labels and boolean train_mask must be one-dimensional and equally sized')
        if labels.shape[0] != x_dict[target_type].shape[0]:
            raise ValueError('Labels must match the target node count')
        labels = labels.to(x_dict[target_type].device)
        train_mask = train_mask.to(labels.device)
        training_labels = labels[train_mask]
        if training_labels.dtype != torch.long or torch.any((training_labels < 0) | (training_labels >= num_classes)):
            raise ValueError('Training labels must be integer class IDs in [0, num_classes)')

    selected = None
    ancestors = set()
    if metapaths is not None:
        selected = {tuple(path) for path in metapaths}
        if not selected or any(len(path) < 2 or path[-1] != target_type for path in selected):
            raise ValueError('Selected metapaths must have an edge and end at the target type')
        ancestors = {path[start:] for path in selected for start in range(1, len(path) - 1)} - selected
        if include_ancestor_features:
            selected = selected | ancestors
    features = precompute_metapath_features(
        x_dict, edge_index_dict, target_type=target_type,
        num_hops=num_hops, metapaths=selected,
    )
    if not use_labels:
        label_features = None
    else:
        label_features = _precompute_label_features(
            x_dict, edge_index_dict, target_type, features, labels, train_mask,
            num_classes, label_batch_size,
        )
    if not include_ancestor_features:
        for path in ancestors:
            features[path] = x_dict[path[0]].new_zeros(
                (x_dict[target_type].shape[0], x_dict[path[0]].shape[1]),
            )
    return features, label_features


def _precompute_label_features(
    x_dict, edge_index_dict, target_type, features, labels, train_mask,
    num_classes, label_batch_size,
):

    paths = tuple(sorted(path for path in features if len(path) > 1 and path[0] == target_type))
    if not paths:
        return {}
    target_features = x_dict[target_type]
    num_nodes = target_features.shape[0]
    seed = target_features.new_zeros((num_nodes, num_classes))
    seed[train_mask] = F.one_hot(labels[train_mask], num_classes=num_classes).to(seed.dtype)
    relations = _normalized_relations(x_dict, edge_index_dict)
    propagated = _propagate_from_target(torch.cat([seed, seed.new_ones((num_nodes, 1))], dim=1), paths, relations)
    diagonals = {path: seed.new_zeros(num_nodes) for path in paths}
    for start in range(0, num_nodes, label_batch_size):
        stop = min(start + label_batch_size, num_nodes)
        rows = torch.arange(start, stop, device=seed.device)
        columns = torch.arange(stop - start, device=seed.device)
        basis = seed.new_zeros((num_nodes, stop - start))
        basis[rows, columns] = 1
        for path, values in _propagate_from_target(basis, paths, relations).items():
            diagonals[path][rows] = values[rows, columns]
    label_features = {}
    epsilon = torch.finfo(seed.dtype).eps
    for path in paths:
        diagonal = diagonals[path].unsqueeze(1)
        values = (propagated[path][:, :num_classes] - diagonal * seed).clamp_min(0)
        mass = (propagated[path][:, num_classes:] - diagonal).clamp_min(0)
        label_features[path] = torch.where(mass > epsilon, values / mass.clamp_min(epsilon), 0)
    return label_features


def _mlp(in_channels, hidden_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels), nn.PReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_channels, out_channels),
    )


class SubtreeAttention(nn.Module):
    """Equations (4)--(5): parent/children reference and bottom-up attention."""

    def __init__(self, channels, dropout=0.5):
        super().__init__()
        self.reference = _mlp(2 * channels, channels, channels, dropout)
        self.score = nn.Linear(2 * channels, 1, bias=False)

    def forward(self, parent, child_features, child_encodings):
        # The reference uses transformed features M; attention values use the
        # already encoded child subtrees Z, as specified in the paper.
        reference = self.reference(torch.cat([parent, child_features.sum(dim=1)], dim=-1))
        reference = reference.unsqueeze(1).expand(-1, child_encodings.shape[1], -1)
        scores = F.leaky_relu(self.score(torch.cat([reference, child_encodings], dim=-1)), 0.2)
        attention = scores.softmax(dim=1)
        encoded = parent + F.elu((attention * child_encodings).sum(dim=1))
        return encoded, attention


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
