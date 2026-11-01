import pytest
import torch
from torch import nn

from src.NC import NodeClassifier, build_model as build_nc
from src.LP import LinkPredictor, DotProductDecoder, build_model as build_lp


def inputs(name):
    x = {'a': torch.randn(4, 3), 'b': torch.randn(3, 3)}
    edge = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    graph = {('a', 'to', 'b'): edge, ('b', 'to', 'a'): edge.flip(0)}
    path = ('a', 'b', 'a')
    if name in {'gcn', 'gat', 'rgcn'}:
        kwargs = dict(in_channels=3, hidden_channels=8, num_layers=2)
        args = (x['a'], torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
        if name == 'rgcn':
            kwargs['num_relations'] = 2
            args += (torch.tensor([0, 1, 0, 1]),)
    elif name in {'hgt', 'spca'}:
        kwargs = dict(in_channels_dict={'a': 3, 'b': 3}, hidden_channels=8,
                      metadata=(list(x), list(graph)), heads=2)
        if name == 'hgt':
            kwargs['target_type'] = 'a'
        else:
            kwargs.update(metapaths=[path], subpaths_dict={path: [('a', 'b')]})
        args = (x, graph)
    elif name == 'han':
        kwargs = dict(in_channels=3, hidden_channels=8, metapath_names=['aba'], target_type='a', heads=2)
        args = (x['a'], {('a', 'aba', 'a'): torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])})
    elif name in {'sehgnn', 'hettree'}:
        kwargs = dict(metapath_features={('a',): x['a']}, hidden_channels=8)
        args = ()
    else:
        kwargs = dict(x_dict=x, edge_index_dict=graph, hidden_channels=8, heads=2, target_type='a')
        args = (x, graph)
    return kwargs, args


@pytest.mark.parametrize('name', ['spca', 'gcn', 'gat', 'rgcn', 'hgt', 'han', 'sehgnn', 'hettree', 'simplehgn', 'hgb'])
@pytest.mark.parametrize('task', ['nc', 'lp'])
def test_native_models_adapt_and_train(name, task):
    torch.manual_seed(42)
    kwargs, args = inputs(name)
    if task == 'nc':
        model = build_nc(name, num_classes=2, **kwargs)
        output = model(*args)
        assert output.shape == (4, 2)
        loss = nn.functional.cross_entropy(output, torch.tensor([0, 1, 0, 1]))
    else:
        model = build_lp(name, embedding_channels=8, **kwargs)
        output = model(*args, edge_label_index=torch.tensor([[0, 1, 2], [2, 3, 0]]))
        assert output.shape == (3,)
        loss = nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([1., 0., 1.]))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_bipartite_lp_gradients_and_checkpoint():
    model = LinkPredictor(nn.Linear(3, 4), 4, destination_encoder=nn.Linear(5, 4))
    source, destination = torch.randn(3, 3), torch.randn(2, 5)
    edges = torch.tensor([[0, 2], [1, 0]])
    output = model(source, edge_label_index=edges, destination_inputs=((destination,), {}))
    output.sum().backward()
    assert model.encoder.weight.grad is not None
    assert model.destination_encoder.weight.grad is not None
    restored = LinkPredictor(nn.Linear(3, 4), 4, destination_encoder=nn.Linear(5, 4))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(output, restored(source, edge_label_index=edges,
                                               destination_inputs=((destination,), {})))
    with pytest.raises(ValueError, match='together'):
        model(source, edge_label_index=edges)


def test_dot_decoder_exact_scores_empty_queries_and_invalid_indices():
    decoder = DotProductDecoder()
    x = torch.tensor([[1., 2.], [3., 4.]])
    torch.testing.assert_close(decoder(x, x, torch.tensor([[0, 1], [1, 0]])), torch.tensor([11., 11.]))
    assert decoder(x, x, torch.empty((2, 0), dtype=torch.long)).shape == (0,)
    for edges in (torch.tensor([[-1], [0]]), torch.tensor([[2], [0]]), torch.zeros(2, 1)):
        with pytest.raises(ValueError):
            decoder(x, x, edges)


@pytest.mark.parametrize('name', ['magnn', 'heco', 'gtc', 'bg-hgnn'])
def test_previously_external_models_now_have_native_task_classes(name):
    from src.model.baselines.registry import get_baseline
    assert issubclass(get_baseline(name), nn.Module)


def test_nc_embedding_adapter_and_lp_label_guard():
    model = NodeClassifier(nn.Linear(3, 4), 4, 2)
    assert model(torch.randn(5, 3)).shape == (5, 2)
    with pytest.raises(ValueError, match='without NC label'):
        build_lp('hettree', 8, label_features={})
