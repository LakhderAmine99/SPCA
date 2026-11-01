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


