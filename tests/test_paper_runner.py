"""Orchestration checks with mocked fitting/evaluation; no experiment is run."""
import json
import torch
from torch_geometric.data import HeteroData
from src.experiments.runner import parse_args, run


def fixture_graph():
    data = HeteroData()
    for node in ('author', 'paper', 'term', 'conference'):
        data[node].x = torch.arange(24, dtype=torch.float).reshape(6, 4)
    edge = torch.stack([torch.arange(6), torch.arange(6)])
    for node in ('author', 'term', 'conference'):
        data['paper', 'to', node].edge_index = edge
        data[node, 'to', 'paper'].edge_index = edge
    data['author'].y = torch.tensor([0, 1, 0, 1, 0, 1])
    for name, start in [('train', 0), ('val', 2), ('test', 4)]:
        data['author'][name + '_mask'] = torch.zeros(6, dtype=torch.bool)
        data['author'][name + '_mask'][start:start + 2] = True
    data.paper_data_discrepancies = {}
    return data


def test_three_seed_tuning_five_reports_and_no_test_selection(monkeypatch, tmp_path):
    from src.experiments import data, training
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: fixture_graph())
    calls, evaluations = [], []
    def fake_fit(model, train, validation, settings, **kwargs):
        assert not (train['mask'] & validation['mask']).any()
        calls.append(dict(settings))
        return {'validation': {'macro_f1': settings['learning_rate'], 'loss': 1.0}, 'best_epoch': 1}
    def fake_evaluate(model, test, **kwargs):
        evaluations.append(test['mask'].clone())
        return {'macro_f1': 0.5, 'micro_f1': 0.5, 'loss': 1.0}
    monkeypatch.setattr(training, 'fit', fake_fit)
    monkeypatch.setattr(training, 'evaluate_test', fake_evaluate)
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps({'learning_rate': [0.001, 0.005]}))
    output = tmp_path / 'run'
    args = parse_args(['--models', 'spca', '--train-ratio', '60%', '--grid', str(grid),
                      '--output', str(output)])
    summary = run(args)
    assert len(calls) == 2 * 3 + 5
    assert len(evaluations) == 5
    assert all(x['learning_rate'] == 0.005 for x in calls[-5:])
    assert summary['0.6']['spca']['macro_f1']['runs'] == 5
    for path in output.glob('*/spca/tuning/*.json'):
        assert all('test' not in trial for trial in json.loads(path.read_text())['runs'])
    selected = json.loads((output / 'selected.json').read_text())
    assert selected['selected']['0.6']['spca']['learning_rate'] == 0.005
    assert selected['protocol_fingerprint']


def test_tune_only_never_calls_test(monkeypatch, tmp_path):
    from src.experiments import data, training
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: fixture_graph())
    monkeypatch.setattr(training, 'fit', lambda *a, **k: {'validation': {'macro_f1': 0.5, 'loss': 1.0}, 'best_epoch': 1})
    def forbidden(*args, **kwargs):
        raise AssertionError('Test evaluation during tuning')
    monkeypatch.setattr(training, 'evaluate_test', forbidden)
    grid = tmp_path / 'grid.json'
    grid.write_text('{}')
    args = parse_args(['--models', 'spca', '--train-ratio', '60%', '--phase', 'tune',
                      '--grid', str(grid), '--output', str(tmp_path / 'tuning')])
    run(args)
