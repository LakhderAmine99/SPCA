"""Task metadata and graph preparation shared by the benchmark runner."""

import torch
from torch_geometric.utils import coalesce


DATASET_INFO = {
    'imdb': ('movie', 'PyG/MAGNN IMDB (three-class)'),
    'dblp': ('author', 'PyG/MAGNN DBLP (four-class)'),
    'acm': ('paper', 'DGL HAN ACM raw (three-class)'),
}


def metapath_configuration(dataset_name):
    """Return the single metapath configuration shared by every compared model."""
    if dataset_name == 'imdb':
        paths = [
            ('movie', 'actor', 'movie'),
            ('movie', 'director', 'movie'),
            ('movie', 'actor', 'movie', 'director', 'movie'),
            ('movie', 'director', 'movie', 'actor', 'movie'),
        ]
    elif dataset_name == 'dblp':
        # APA, APTPA, APVPA: the loader names venue (V) nodes "conference".
        paths = [
            ('author', 'paper', 'author'),
            ('author', 'paper', 'term', 'paper', 'author'),
            ('author', 'paper', 'conference', 'paper', 'author'),
        ]
    elif dataset_name == 'acm':
        paths = [('paper', 'author', 'paper'), ('paper', 'subject', 'paper')]
    else:
        raise ValueError('dataset_name must be imdb, dblp, or acm')
    return paths, {path: [list(path)] for path in paths}


def prepare_benchmark(data, dataset_name):
    """Provide all node features and the relation names expected by the encoders."""
    target_type, _ = DATASET_INFO[dataset_name]
    target = data[target_type]
    if target.y.ndim != 1 or torch.any(target.y < 0):
        raise ValueError(f'{dataset_name} requires a known class label for every {target_type} node')
    for node_type in data.node_types:
        if data[node_type].get('x') is None:
            # DBLP conferences are featureless; retain their distinct identities.
            data[node_type].x = torch.eye(data[node_type].num_nodes)
        else:
            data[node_type].x = data[node_type].x.float()

    # The subpath encoder addresses relations by their endpoint node types.
    # Combine relations sharing endpoints into a binary adjacency matrix.
    relations = {}
    for source, relation, destination in list(data.edge_types):
        edges = data[source, relation, destination].edge_index
        relations.setdefault((source, 'to', destination), []).append(edges)
        del data[source, relation, destination]
    for edge_type, parts in relations.items():
        data[edge_type].edge_index = coalesce(torch.cat(parts, dim=1))
    for (source, _, destination), edges in list(data.edge_index_dict.items()):
        reverse = (destination, 'to', source)
        if reverse not in data.edge_types:
            data[reverse].edge_index = edges.flip(0)

    paths, _ = metapath_configuration(dataset_name)
    for path in paths:
        for source, destination in zip(path, path[1:]):
            if (source, 'to', destination) not in data.edge_types:
                raise ValueError(f'{dataset_name} is missing the {source} -> {destination} relation')
    data.target_type = target_type
    if dataset_name == 'acm':
        data.feature_source = 'DGL HAN ACM raw: paper words; author/subject neighbor means'
        data.preprocessing = 'han-raw-v1'
    return data
