"""Dataset loading only on explicit execution; no downloads on import or dry run."""
from pathlib import Path
import io
from urllib.request import urlopen
from zipfile import ZipFile
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import coalesce


def normalize_relations(data):
    grouped = {}
    for source, relation, target in list(data.edge_types):
        edge = data[source, relation, target].get('edge_index')
        if edge is not None:
            grouped.setdefault((source, 'to', target), []).append(edge)
        del data[source, relation, target]
    for key, parts in grouped.items():
        data[key].edge_index = coalesce(torch.cat(parts, 1))
    for (source, _, target), edges in list(data.edge_index_dict.items()):
        if (target, 'to', source) not in data.edge_types:
            data[target, 'to', source].edge_index = edges.flip(0)
    return data


def load_movielens(root):
    """All metadata movies, including unrated movies, as in manuscript Table 2."""
    raw = Path(root) / 'raw'
    names = ('users.dat', 'movies.dat', 'ratings.dat')
    if not all((raw / name).exists() for name in names):
        raw.mkdir(parents=True, exist_ok=True)
        with urlopen('https://files.grouplens.org/datasets/movielens/ml-1m.zip', timeout=60) as response:
            archive = ZipFile(io.BytesIO(response.read()))
        for name in names:
            (raw / name).write_bytes(archive.read('ml-1m/' + name))
    users = [row.split('::') for row in (raw / 'users.dat').read_text(encoding='latin-1').splitlines()]
    movies = [row.split('::') for row in (raw / 'movies.dat').read_text(encoding='latin-1').splitlines()]
    user_ids = {int(row[0]): i for i, row in enumerate(users)}
    movie_ids = {int(row[0]): i for i, row in enumerate(movies)}
    genres = sorted({genre for row in movies for genre in row[2].split('|')})
    genre_ids = {name: i for i, name in enumerate(genres)}
    data = HeteroData()
    # Metadata features have no rating-derived statistics or held-out interactions.
    user_features = []
    ages = sorted({row[2] for row in users})
    occupations = sorted({row[3] for row in users})
    for row in users:
        user_features.append([float(row[1] == 'M')] + [float(row[2] == a) for a in ages] +
                             [float(row[3] == o) for o in occupations])
    data['user'].x = torch.tensor(user_features)
    data['movie'].x = torch.zeros(len(movies), len(genres))
    genre_edges = []
    for i, row in enumerate(movies):
        for genre in row[2].split('|'):
            data['movie'].x[i, genre_ids[genre]] = 1
            genre_edges.append((i, genre_ids[genre]))
    data['genre'].x = torch.eye(len(genres))
    interactions = []
    for row in (raw / 'ratings.dat').read_text().splitlines():
        user, movie, _, _ = row.split('::')
        interactions.append((user_ids[int(user)], movie_ids[int(movie)]))
    data['user', 'to', 'movie'].edge_index = torch.tensor(interactions).T.contiguous()
    data['movie', 'to', 'genre'].edge_index = torch.tensor(genre_edges).T.contiguous()
    data.feature_source = 'MovieLens-1M metadata: user demographics; movie genres; genre identity'
    return data


def validate_paper_data(data, dataset, allow_mismatch=False):
    """Refuse silent substitutions of benchmark variants."""
    expected = {'dblp': {'author': 4057, 'paper': 14328, 'term': 7723, 'conference': 20},
                'imdb': {'movie': 4278, 'actor': 5257, 'director': 2081},
                'acm': {'paper': 3025, 'author': 5835, 'subject': 56},
                'lastfm': {'user': 1892, 'artist': 17632, 'tag': 1088},
                'movielens': {'user': 6040, 'movie': 3883, 'genre': 18}}[dataset]
    mismatches = {node: {'expected': count, 'actual': data[node].num_nodes if node in data.node_types else None}
                  for node, count in expected.items() if node not in data.node_types or data[node].num_nodes != count}
    feature_width = {'dblp': ('author', 334), 'imdb': ('movie', 3066), 'acm': ('paper', 1902)}
    if dataset in feature_width:
        node, width = feature_width[dataset]
        actual = data[node].x.size(1)
        if actual != width:
            mismatches['target_feature_width'] = {'expected': width, 'actual': actual}
    expected_edges = {
        'dblp': {('paper', 'to', 'author'): 19645, ('paper', 'to', 'conference'): 14328,
                 ('paper', 'to', 'term'): 85810},
        'imdb': {('movie', 'to', 'actor'): 12828, ('movie', 'to', 'director'): 4278},
        'acm': {('paper', 'to', 'author'): 9949, ('paper', 'to', 'subject'): 3025},
    }.get(dataset, {})
    for relation, count in expected_edges.items():
        actual = data[relation].edge_index.size(1) if relation in data.edge_types else None
        if actual != count:
            mismatches[str(relation)] = {'expected': count, 'actual': actual}
    if mismatches and not allow_mismatch:
        raise ValueError(f'{dataset} does not match manuscript Table 1/2: {mismatches}. '
            'Provide the matching local PyG graph with --graph, or explicitly use '
            '--allow-data-mismatch for a separately identified variant. '
            'The raw ACM.mat loader is not ACM3025.')
    return mismatches


def load_dataset(dataset, root, *, graph_path=None, feature_seed=0, feature_channels=128,
                 allow_data_mismatch=False):
    if graph_path:
        # This is an explicitly supplied, trusted local PyG graph, not a downloaded pickle.
        data = torch.load(graph_path, weights_only=False, map_location='cpu')
        if not isinstance(data, HeteroData):
            raise TypeError('--graph must contain a PyG HeteroData object')
    elif dataset in {'dblp', 'imdb', 'acm'}:
        from src.dblp import DBLP
        from src.imdb import IMDB
        from src.acm import ACM
        data = {'dblp': DBLP, 'imdb': IMDB, 'acm': ACM}[dataset](str(root))[0]
    elif dataset == 'lastfm':
        from torch_geometric.datasets import LastFM
        data = LastFM(str(root))[0]
        # Recover the entire observed interaction universe before making our splits.
        store = data['user', 'artist']
        store.edge_index = coalesce(torch.cat([store.edge_index] +
            [store[key] for key in ('val_pos_edge_index', 'test_pos_edge_index') if key in store], 1))
        if ('artist', 'to', 'user') in data.edge_types:
            data['artist', 'to', 'user'].edge_index = store.edge_index.flip(0)
        data.feature_source = f'Fixed Gaussian features, seed={feature_seed}, width={feature_channels}'
    else:
        data = load_movielens(root)
    data = normalize_relations(data)
    generator = torch.Generator().manual_seed(feature_seed)
    for node in data.node_types:
        if data[node].get('x') is None:
            data[node].x = (torch.randn(data[node].num_nodes, feature_channels, generator=generator)
                            if dataset in {'lastfm', 'movielens'} else torch.eye(data[node].num_nodes))
        data[node].x = data[node].x.float()
    discrepancies = validate_paper_data(data, dataset, allow_data_mismatch)
    data.paper_data_discrepancies = discrepancies
    return data
