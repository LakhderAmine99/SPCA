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


