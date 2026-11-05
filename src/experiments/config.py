"""Tables 3 and 11 and Sections 5.2/7.2 of the November 2026 manuscript."""
from itertools import product

MODELS = ('gcn', 'gat', 'rgcn', 'han', 'magnn', 'hgt', 'heco', 'gtc',
          'simplehgn', 'sehgnn', 'bghgnn', 'hettree', 'spca')
GRID = {'learning_rate': [0.0005, 0.001, 0.003, 0.005],
        'weight_decay': [0.0001, 0.0005], 'dropout': [0.1, 0.3, 0.5, 0.6],
        'hidden_channels': [64, 128, 256, 512], 'head_channels': [8, 16, 32, 64, 128]}

HEAD_WIDTH_MODELS = {'gat', 'han', 'magnn', 'simplehgn', 'bghgnn'}
PATH_CODES = {
    'imdb': {'movie': ['MDM', 'MAM', 'MDMDM', 'MDMAMDM', 'MAMAM', 'MAMDMAM', 'MAMDM', 'MDMAM']},
    'dblp': {'author': ['APA', 'APTPA', 'APVPA']},
    'acm': {'paper': ['PAP', 'PSP', 'PAPSP', 'PSPAP']},
    'lastfm': {'user': ['UAU', 'UATAU', 'UUAU', 'UUATAU', 'UAUUAU'], 'artist': ['AUA', 'ATA', 'AUUA']},
    'movielens': {'user': ['UMU', 'UMGMU', 'UMUMU'], 'movie': ['MUM', 'MGM', 'MUMGM', 'MUMGMUM']},
}
TYPE_CODES = {
    'imdb': dict(M='movie', A='actor', D='director'),
    'dblp': dict(A='author', P='paper', T='term', V='conference'),
    'acm': dict(P='paper', A='author', S='subject'),
    'lastfm': dict(U='user', A='artist', T='tag'),
    'movielens': dict(U='user', M='movie', G='genre'),
}
LP_RATIOS = {'lastfm': (0.7, 0.1, 0.2), 'movielens': (0.8, 0.1, 0.1)}


def metapaths(dataset, target=None):
    target = target or next(iter(PATH_CODES[dataset]))
    return [tuple(TYPE_CODES[dataset][letter] for letter in code) for code in PATH_CODES[dataset][target]]


def model_defaults(model, dataset):
    settings = dict(learning_rate=0.005, weight_decay=0.0005, dropout=0.5,
                    hidden_channels=128, embedding_channels=128, head_channels=16,
                    heads=8, num_layers=2)
    if model == 'heco':
        settings.update(temperature=0.9 if dataset == 'dblp' else 0.8, balance=0.5,
                        contrast_batch_size=512, positive_k=1000 if dataset == 'dblp' else 5)
    elif model == 'gtc':
        settings.update(temperature=0.9 if dataset == 'dblp' else 0.7,
                        balance=0.1 if dataset == 'dblp' else 0.4,
                        transformer_layers=5 if dataset == 'dblp' else 6,
                        num_layers=3 if dataset == 'dblp' else 2,
                        hops=1 if dataset == 'dblp' else 2, contrast_batch_size=512, positive_k=5)
    elif model == 'magnn':
        settings.update(instance_encoder='RotatE0', attention_channels=128, max_instances=100)
    elif model == 'bghgnn':
        settings.update(rank=4, type_channels=512, combine_channels=64)
    elif model == 'spca':
        settings.update(aggregator='concat-mlp', context_mode='subpath',
                        context_positions='intermediate', normalization=True, node_chunk_size=256,
                        negative_slope=0.01)
    elif model == 'rgcn':
        settings.update(num_bases=None)
    elif model == 'simplehgn':
        settings.update(edge_channels=64, negative_slope=0.05, attention_residual=0.05)
    elif model == 'sehgnn':
        settings.update(num_projection_layers=2, num_task_layers=4)
    elif model == 'hettree':
        settings.update(use_labels=True, label_batch_size=128)
    return settings


def configurations(model, overrides=None, grid=None):
    ranges = dict(GRID if grid is None else grid)
    if model not in HEAD_WIDTH_MODELS:
        ranges.pop('head_channels', None)
    overrides = overrides or {}
    for key in overrides:
        ranges.pop(key, None)
    if any(not isinstance(v, list) or not v for v in ranges.values()):
        raise ValueError('Every grid range must be a nonempty list')
    keys = tuple(ranges)
    for values in product(*(ranges[key] for key in keys)):
        yield {**dict(zip(keys, values)), **overrides}


def ratio(value):
    if value == 'full':
        return value
    number = float(value[:-1]) / 100 if isinstance(value, str) and value.endswith('%') else float(value)
    if not 0 < number <= 1:
        raise ValueError('Ratios must be in (0, 1]')
    return number
