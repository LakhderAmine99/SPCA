import pytest
import torch

from src.model.baselines.registry import BASELINES, get_baseline
from src.model.spca import SPCA


def test_registry_covers_paper_and_aliases():
    assert len(BASELINES) == 12
    for name in BASELINES:
        assert get_baseline(name) is not None
    assert get_baseline('hgb') is get_baseline('simplehgn')
    assert get_baseline('BG-HGNN') is get_baseline('bghgnn')
    with pytest.raises(ValueError):
        get_baseline('typo')


@pytest.mark.parametrize('name', ['gcn', 'gat', 'rgcn', 'hgt', 'han'])
def test_new_native_baselines_forward_backward(name):
    torch.manual_seed(1)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    x = torch.randn(4, 3, requires_grad=True)
    cls = get_baseline(name)
    if name == 'rgcn':
        model = cls(3, 8, 2, num_relations=2)
        logits = model(x, edges, torch.tensor([0, 1, 0, 1]))
    elif name == 'hgt':
        metadata = (['a'], [('a', 'to', 'a')])
        model = cls({'a': 3}, 8, 2, metadata, target_type='a', heads=2)
        logits = model({'a': x}, {('a', 'to', 'a'): edges})
    elif name == 'han':
        model = cls(3, 8, 2, ['aba'], target_type='a', heads=2)
        logits = model(x, {('a', 'aba', 'a'): edges})
    else:
        model = cls(in_channels=3, hidden_channels=8, num_layers=2, out_channels=2)
        logits = model(x, edges)
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('variant', ['base', 'regularized', 'stacked'])
def test_spca_classifier_and_state_restore(variant):
    path = ('a', 'b', 'a')
    metadata = (['a', 'b'], [('a', 'to', 'b'), ('b', 'to', 'a')])
    args = ({'a': 3, 'b': 3}, 8, 2, metadata, [path], {path: [('a', 'b')]})
    model = SPCA(*args, variant=variant, heads=2).eval()
    x = {'a': torch.randn(3, 3), 'b': torch.randn(2, 3)}
    edges = torch.tensor([[0, 1, 2], [0, 1, 0]])
    graph = {('a', 'to', 'b'): edges, ('b', 'to', 'a'): edges.flip(0)}
    logits = model(x, graph)
    assert logits.shape == (3, 2)
    logits.square().sum().backward()
    restored = SPCA(*args, variant=variant, heads=2).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(logits, restored(x, graph))
