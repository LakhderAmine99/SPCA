"""Label-free metapath connectivity shared by all paper models.

Edges follow PyG's source-to-destination convention. Boolean composition counts
an endpoint neighbor once, regardless of the number of connecting instances.
"""
import numpy as np
import scipy.sparse as sp
import torch

def relation_for(edge_index_dict, source, destination):
    matches = [edges for (a, _, b), edges in edge_index_dict.items()
               if a == source and b == destination]
    if len(matches) != 1:
        raise ValueError(f'Expected one relation {source} -> {destination}, found {len(matches)}')
    return matches[0]

def compose_metapaths(x_dict, edge_index_dict, metapaths):
    """Construct exact binary endpoint graphs, without adding artificial loops."""
    relations, prefixes, result = {}, {}, {}
    for path in map(tuple, metapaths):
        if len(path) < 2:
            raise ValueError('A metapath must contain at least one edge')
        for length in range(2, len(path) + 1):
            prefix = path[:length]
            if prefix in prefixes:
                continue
            pair = prefix[-2:]
            if pair not in relations:
                edge = relation_for(edge_index_dict, *pair).detach().cpu().numpy()
                relations[pair] = sp.csr_matrix((np.ones(edge.shape[1], dtype=bool), edge),
                    shape=(x_dict[pair[0]].shape[0], x_dict[pair[1]].shape[0]))
            prefixes[prefix] = (relations[pair] if length == 2 else
                                prefixes[prefix[:-1]] @ relations[pair])
        row, col = prefixes[path].nonzero()
        result[path] = torch.from_numpy(np.stack([row, col]).astype(np.int64)).to(x_dict[path[-1]].device)
    return result

def enumerate_instances(x_dict, edge_index_dict, metapaths, max_instances=None, seed=0):
    """MAGNN ordered instances, optionally sampled per receiving node.

    None preserves every instance. An explicit cap is an adaptation and is
    recorded by the runner. Prefix counts support uniform sampling with
    replacement without enumerating the exponentially large complete path set.
    """
    rng = np.random.default_rng(seed)
    if max_instances is not None and (not isinstance(max_instances, int) or max_instances < 1):
        raise ValueError('max_instances must be None or a positive integer')
    result = {}
    for path in map(tuple, metapaths):
        neighbors = []
        for source, destination in zip(path, path[1:]):
            edge = relation_for(edge_index_dict, source, destination).cpu().numpy()
            adjacency = sp.csr_matrix((np.ones(edge.shape[1], dtype=bool), edge),
                shape=(x_dict[source].shape[0], x_dict[destination].shape[0])).tocsc()
            neighbors.append(adjacency)
        counts = [np.ones(x_dict[path[0]].shape[0], dtype=np.float64)]
        for adjacency in neighbors:
            counts.append(np.asarray(adjacency.T.astype(np.float64) @ counts[-1]).ravel())
        rows = []
        for destination in range(x_dict[path[-1]].shape[0]):
            if max_instances is not None and counts[-1][destination] > max_instances:
                for _ in range(max_instances):
                    suffix = (destination,)
                    for position in range(len(neighbors) - 1, -1, -1):
                        adjacency = neighbors[position]
                        node = suffix[0]
                        candidates = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
                        weights = counts[position][candidates]
                        previous = int(rng.choice(candidates, p=weights / weights.sum()))
                        suffix = (previous,) + suffix
                    rows.append(suffix)
                continue
            reservoir, seen = [], 0
            def walk(position, suffix):
                nonlocal seen
                if position < 0:
                    seen += 1
                    if max_instances is None or len(reservoir) < max_instances:
                        reservoir.append(suffix)
                    else:
                        index = int(rng.integers(seen))
                        if index < max_instances:
                            reservoir[index] = suffix
                    return
                adjacency = neighbors[position]
                node = suffix[0]
                for previous in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]:
                    walk(position - 1, (int(previous),) + suffix)
            walk(len(path) - 2, (destination,))
            rows.extend(reservoir)
        result[path] = torch.tensor(rows, dtype=torch.long).reshape(-1, len(path))
    return result