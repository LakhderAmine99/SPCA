"""Lazy baseline discovery without importing optional upstream dependencies."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class BaselineSpec:
    symbol: str
    family: str
    integration: str
    pipeline: str


BASELINES = {
    'gcn': BaselineSpec('GCN', 'homogeneous', 'pyg', 'E2E'),
    'gat': BaselineSpec('GAT', 'homogeneous', 'pyg', 'E2E'),
    'han': BaselineSpec('HAN', 'metapath', 'local-pyg', 'Pre'),
    'magnn': BaselineSpec('MAGNN', 'metapath', 'upstream', 'Pre'),
    'rgcn': BaselineSpec('RGCN', 'relation-aware', 'local-pyg', 'E2E'),
    'hgt': BaselineSpec('HGT', 'relation-aware', 'local-pyg', 'E2E'),
    'gtc': BaselineSpec('GTC', 'relation-aware', 'upstream', 'Pre'),
    'heco': BaselineSpec('HeCo', 'advanced', 'upstream', 'Pre'),
    'simplehgn': BaselineSpec('SimpleHGN', 'advanced', 'local-adaptation', 'input-dependent'),
    'sehgnn': BaselineSpec('SeHGNN', 'advanced', 'local-adaptation', 'Pre'),
    'bghgnn': BaselineSpec('BGHGNN', 'advanced', 'external-required', 'Pre'),
    'hettree': BaselineSpec('HetTree', 'advanced', 'local-adaptation', 'Pre'),
}
ALIASES = {'hgb': 'simplehgn', 'simple-hgn': 'simplehgn', 'bg-hgnn': 'bghgnn'}


def get_baseline(name):
    """Return a model class or an UpstreamBaseline, according to integration."""
    key = name.lower()
    key = ALIASES.get(key, key)
    if key not in BASELINES:
        raise ValueError(f'Unknown baseline {name!r}; choose from {tuple(BASELINES)}')
    return getattr(import_module(f'{__package__}.{key}'), BASELINES[key].symbol)
