import sys
import subprocess
from pathlib import Path
import pytest
import torch
from torch_geometric.data import HeteroData
from src.experiments.config import configurations, metapaths
from src.experiments.splits import node_masks, link_split
from src.experiments.training import ValidationCheckpoint, summarize
from src.experiments.runner import parse_args, plan


def test_paper_grid_cartesian_and_fixed_overrides():
    assert len(list(configurations('spca'))) == 128
    assert len(list(configurations('gat'))) == 640
    values = list(configurations('gat', {'learning_rate': 0.001, 'head_channels': 16}))
    assert len(values) == 32
    assert all(v['learning_rate'] == 0.001 and v['head_channels'] == 16 for v in values)
    assert len(metapaths('imdb')) == 8 and len(metapaths('acm')) == 4
    assert len(metapaths('lastfm', 'artist')) == 3


def test_main_pool_low_label_and_fixed_holdouts():
    y = torch.tensor([0] * 20 + [1] * 20)
    original = {key: torch.zeros(40, dtype=torch.bool) for key in ('train_mask', 'val_mask', 'test_mask')}
    original['train_mask'][[0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]] = True
    original['val_mask'][[6, 26]] = True
    original['test_mask'][[7, 27]] = True
    main = node_masks(y, original, 0.5, 9, 'main')
    low = node_masks(y, original, 0.1, 9, 'low-label')
    assert int(main['train_mask'].sum()) == 6
    assert int(low['train_mask'].sum()) == 4
    larger = node_masks(y, original, 0.8, 9, 'main')
    assert torch.all(main['train_mask'] <= larger['train_mask'])
    other_seed = node_masks(y, original, 0.5, 12, 'main')
    for key in ('val_mask', 'test_mask'):
        assert torch.equal(main[key], other_seed[key])
    for masks in (main, low):
        assert not (masks['train_mask'] & (masks['val_mask'] | masks['test_mask'])).any()


def test_link_split_removes_forward_reverse_and_excludes_all_positives():
    data = HeteroData()
    data['user'].x = torch.randn(8, 4)
    data['movie'].x = torch.randn(8, 4)
    positive = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 1, 0]])
    data['user', 'to', 'movie'].edge_index = positive
    data['movie', 'to', 'user'].edge_index = positive.flip(0)
    graph, splits = link_split(data, ('user', 'to', 'movie'), (0.7, 0.1, 0.2), 42)
    assert graph['user', 'to', 'movie'].edge_index.size(1) == 7
    assert torch.equal(graph['user', 'to', 'movie'].edge_index.flip(0), graph['movie', 'to', 'user'].edge_index)
    all_positive = set(map(tuple, positive.T.tolist()))
    training = set(map(tuple, graph['user', 'to', 'movie'].edge_index.T.tolist()))
    negatives = []
    for name, split in splits.items():
        pos = split['edge_label_index'][:, split['edge_label'] == 1]
        neg = split['edge_label_index'][:, split['edge_label'] == 0]
        assert not all_positive.intersection(map(tuple, neg.T.tolist()))
        negatives.extend(map(tuple, neg.T.tolist()))
        if name != 'train':
            assert not training.intersection(map(tuple, pos.T.tolist()))
    assert len(set(negatives)) == len(negatives)
    assert torch.equal(data['user', 'to', 'movie'].edge_index, positive)


def test_validation_checkpoint_tiebreak_and_sample_std():
    model = torch.nn.Linear(2, 2)
    tracker = ValidationCheckpoint()
    assert tracker.consider(model, {'macro_f1': 0.7, 'loss': 1.0}, 1)
    assert not tracker.consider(model, {'macro_f1': 0.6, 'loss': 0.1}, 2)
    assert tracker.consider(model, {'macro_f1': 0.7, 'loss': 0.9}, 3)
    assert not tracker.consider(model, {'macro_f1': 0.7, 'loss': 0.9}, 4)
    assert tracker.epoch == 3
    assert summarize([{'x': 1}, {'x': 3}])['x']['std'] == pytest.approx(2 ** 0.5)
    assert summarize([{'x': 1}])['x']['std'] is None


def test_default_plan_and_custom_options():
    args = parse_args(['--dry-run'], dataset='dblp')
    result = plan(args)
    assert len(result['models']) == 13 and len(result['reporting_seeds']) == 5
    assert len(result['tuning_seeds']) == 3 and result['epochs'] == 500 and result['patience'] == 100
    custom = parse_args(['--models', 'spca', '--train-ratio', '10%', '--set', 'spca.aggregator=mean',
                         '--runs', '2', '--seed', '100', '--dry-run'], dataset='imdb')
    assert custom.seeds == [100, 101] and custom.train_ratios == [0.1]
    assert plan(custom)['models']['spca']['fixed_overrides']['aggregator'] == 'mean'


@pytest.mark.parametrize('dataset', ['dblp', 'imdb', 'acm', 'lastfm', 'movielens'])
def test_dry_run_without_torch_import_or_data_access(dataset, tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from src.experiments.runner import main; main(['--dry-run'], dataset=" + repr(dataset) + "); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert 'test_policy' in completed.stdout
