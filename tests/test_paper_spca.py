"""Independent numerical checks against manuscript Eqs. (1)--(7)."""
import torch
import pytest
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from src.graph import compose_metapaths
from src.model.spca import SPCALayer, DotProductConv, SubpathContext


def graph():
    x = {'a': torch.tensor([[1., 2., 0., 1.], [0., 1., 2., 3.], [2., 0., 1., 1.]]),
         'b': torch.randn(2, 4)}
    ab = torch.tensor([[0, 1, 2], [0, 0, 1]])
    edges = {('a', 'to', 'b'): ab, ('b', 'to', 'a'): ab.flip(0)}
    return x, edges, ('a', 'b', 'a')


def test_dot_attention_matches_manual_receiving_neighborhoods():
    x = torch.tensor([[1., 2.], [2., -1.], [-2., 3.], [5., 4.]], requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 0], [1, 1, 1, 2]])
    conv = DotProductConv(dropout=0)
    assert isinstance(conv, MessagePassing)
    output = conv(x, edges)
    expected = torch.zeros_like(x)
    for destination in range(len(x)):
        neighbors = edges[0, edges[1] == destination]
        if neighbors.numel():
            scores = F.leaky_relu((x[neighbors] * x[destination]).sum(-1), 0.01).softmax(0)
            expected[destination] = (scores[:, None] * x[neighbors]).sum(0)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_spca_equations_and_intermediate_features_do_not_propagate():
    x, edges, path = graph()
    model = SPCALayer({'a': 4, 'b': 4}, 4, (list(x), list(edges)), [path], dropout=0).eval()
    graphs = compose_metapaths(x, edges, [path])
    assert model.subpaths[path] == (('b',),)
    assert model.context_encoder.sequences == (('b',),)
    projected = model.projections['a'](x['a'])
    structure = model.structural_norm(model.neighbor_encoder(projected, graphs[path]))
    manual = model.fusion(model.fusion_norm(structure + model.context_encoder(('b',))))
    actual = model(x, edges)
    torch.testing.assert_close(actual, manual)
    torch.testing.assert_close(actual, model({**x, 'b': 1000 * torch.randn_like(x['b'])}, edges))
    actual.square().sum().backward()
    assert model.context_encoder.type_embeddings.weight.grad.abs().sum() > 0


def test_binary_composition_deduplicates_instances_and_supports_odd_hops():
    x, edges, path = graph()
    edges[('a', 'to', 'b')] = torch.cat([edges[('a', 'to', 'b')]] * 3, 1)
    composed = compose_metapaths(x, edges, [path])[path]
    assert composed.size(1) == 5
    edges['a', 'to', 'a'] = torch.tensor([[0, 1, 2], [1, 2, 0]])
    odd = ('a', 'a', 'b', 'a')
    assert compose_metapaths(x, edges, [odd])[odd].shape[0] == 2


def test_full_context_and_shared_encoders_and_scalar_cancellation():
    x, edges, path = graph()
    full = SPCALayer({'a': 4, 'b': 4}, 4, (list(x), list(edges)), [path], context_positions='full')
    assert full.subpaths[path] == (path,)
    context = SubpathContext(['a', 'b'], [('a', 'b'), ('a', 'b'), ('b', 'a')], 8)
    assert len(context.encoders) == 2
    scalar = SPCALayer({'a': 4, 'b': 4}, 4, (list(x), list(edges)), [path],
                       context_mode='metapath-scalar-bias', dropout=0).eval()
    before = scalar(x, edges)
    with torch.no_grad():
        scalar.context_bias.fill_(3)
    torch.testing.assert_close(before, scalar(x, edges), atol=1e-6, rtol=1e-5)


def test_empty_neighborhood_and_reset_state():
    x, edges, path = graph()
    model = SPCALayer({'a': 4, 'b': 4}, 4, (list(x), list(edges)), [path], dropout=0)
    result = model(x, metapath_edge_index_dict={path: torch.empty(2, 0, dtype=torch.long)})
    assert result.shape == (3, 4) and torch.isfinite(result).all()
    old = model.fusion.weight.detach().clone()
    model.reset_parameters()
    assert not torch.equal(old, model.fusion.weight)


def test_chunked_attention_matches_unblocked_outputs_and_gradients():
    torch.manual_seed(10)
    x = torch.randn(128, 128, requires_grad=True)
    edges = torch.cartesian_prod(torch.arange(128), torch.arange(128)).T
    conv = DotProductConv(dropout=0, node_chunk_size=17)
    actual = conv(x, edges)
    expected = conv.propagate(edges, x=(x, x), size=(128, 128))
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    second = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(first, second, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('aggregator', ['concat-mlp', 'mean', 'lstm', 'transformer'])
def test_current_context_encoder_variants_have_gradients(aggregator):
    context = SubpathContext(['a', 'b'], [('a', 'b', 'a'), ('b',)], 16, aggregator, 0)
    output = context(('a', 'b', 'a')) + context(('b',))
    output.square().sum().backward()
    assert context.type_embeddings.weight.grad is not None
    assert torch.isfinite(output).all()
