"""Small synthetic graphs only: no benchmark datasets, downloads or experiments."""
import pytest
import torch
from torch_geometric.data import HeteroData
from src.experiments.config import MODELS, TYPE_CODES, PATH_CODES, metapaths, model_defaults
from src.experiments.models import PreparedGraph, BoundEncoder, NodeTask, LinkTask
from src.experiments.training import fit, evaluate_test


def tiny_graph(dataset):
    torch.manual_seed(11)
    data = HeteroData()
    for node in TYPE_CODES[dataset].values():
        data[node].x = torch.randn(6, 5)
    relations = set()
    for target in PATH_CODES[dataset]:
        for path in metapaths(dataset, target):
            relations.update(zip(path, path[1:]))
    edges = torch.stack([torch.arange(6).repeat(2), torch.cat([torch.arange(6), torch.arange(6).roll(1)])])
    for a, b in relations:
        data[a, 'to', b].edge_index = edges.clone()
    target = next(iter(PATH_CODES[dataset]))
    data[target].y = torch.tensor([0, 1, 0, 1, 0, 1])
    for name, start in [('train', 0), ('val', 2), ('test', 4)]:
        mask = torch.zeros(6, dtype=torch.bool)
        mask[start:start + 2] = True
        data[target][name + '_mask'] = mask
    return data


def small_settings(name, dataset):
    settings = model_defaults(name, dataset)
    settings.update(hidden_channels=16, embedding_channels=8, head_channels=4, heads=2,
                    dropout=0.0)
    if name == 'gtc':
        settings['transformer_layers'] = 1
    return settings


def test_json_metapaths_are_normalized_and_invalid_endpoints_rejected():
    data = tiny_graph('dblp')
    prepared = PreparedGraph(data, 'dblp', 'author', [['author', 'paper', 'author']])
    assert prepared.paths == [('author', 'paper', 'author')]
    assert BoundEncoder('spca', prepared, small_settings('spca', 'dblp'))().shape == (6, 8)
    with pytest.raises(ValueError, match='receiving type'):
        PreparedGraph(data, 'dblp', 'author', [['paper', 'author', 'paper']])
    with pytest.raises(ValueError, match='distinct'):
        PreparedGraph(data, 'dblp', 'author', [['author', 'paper', 'author']] * 2)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('dataset', ['dblp', 'imdb', 'acm', 'lastfm', 'movielens'])
@pytest.mark.parametrize('name', MODELS)
def test_all_model_dataset_adapters_forward_backward(name, dataset):
    data = tiny_graph(dataset)
    target = next(iter(PATH_CODES[dataset]))
    prepared = PreparedGraph(data, dataset, target)
    settings = small_settings(name, dataset)
    model = BoundEncoder(name, prepared, settings, labels=data[target].y,
                         train_mask=data[target].train_mask, num_classes=2)
    output = model()
    assert output.shape == (6, 8) and torch.isfinite(output).all()
    loss = output.square().mean() + model.auxiliary_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    state = model.state_dict()
    assert all('features_' not in key and 'labels_' not in key for key in state)


@pytest.mark.parametrize('name', MODELS)
def test_one_synthetic_nc_optimizer_step_and_selected_checkpoint(name):
    data = tiny_graph('dblp')
    settings = small_settings(name, 'dblp')
    encoder = BoundEncoder(name, PreparedGraph(data, 'dblp', 'author'), settings,
        labels=data['author'].y, train_mask=data['author'].train_mask, num_classes=2)
    model = NodeTask(encoder, 8, 2)
    train = {'labels': data['author'].y, 'mask': data['author'].train_mask}
    validation = {'labels': data['author'].y, 'mask': data['author'].val_mask}
    result = fit(model, train, validation, settings, epochs=1, patience=1)
    assert result['best_epoch'] == 1 and 'test' not in result
    metrics = evaluate_test(model, {'labels': data['author'].y, 'mask': data['author'].test_mask})
    assert 0 <= metrics['macro_f1'] <= 1


@pytest.mark.parametrize('name', MODELS)
def test_bipartite_lp_dot_decoder_and_gradients(name):
    data = tiny_graph('movielens')
    settings = small_settings(name, 'movielens')
    encoders = [BoundEncoder(name, PreparedGraph(data, 'movielens', target), settings)
                for target in ('user', 'movie')]
    model = LinkTask(*encoders).eval()
    query = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    result = model(query)
    left, right = model.encode()
    torch.testing.assert_close(result, (left[query[0]] * right[query[1]]).sum(-1))
    torch.nn.functional.binary_cross_entropy_with_logits(result, torch.tensor([1., 0., 1., 0.])).backward()
    for encoder in encoders:
        assert any(p.grad is not None for p in encoder.parameters())
