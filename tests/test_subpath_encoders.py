import unittest
import torch
import index

class SequenceEncoderTests(unittest.TestCase):
    def make_encoder(self, cls, aggregator, mode='subpath'):
        path = ('a', 'b', 'a')
        return cls({'a': 3, 'b': 3}, 8, (['a', 'b'], [('a', 'to', 'b'), ('b', 'to', 'a')]),
                   [path], {path: [('a', 'b'), ('b', 'a')]},
                   aggregator=aggregator, context_mode=mode, heads=2)

    def test_forward_gradients_order_and_restore(self):
        for cls in (index.SubpathHANConv2, index.RegularizedSubpathHANConv):
            for kind in ('lstm', 'transformer'):
                with self.subTest(cls=cls.__name__, kind=kind):
                    torch.manual_seed(7)
                    encoder = self.make_encoder(cls, kind).eval()
                    x = {'a': torch.randn(3, 3), 'b': torch.randn(2, 3)}
                    edges = torch.tensor([[0, 1, 2], [0, 1, 0]])
                    graph = {('a', 'to', 'b'): edges, ('b', 'to', 'a'): edges.flip(0)}
                    output = encoder(x, graph)
                    self.assertEqual(output.shape, (3, 8))
                    self.assertTrue(torch.isfinite(output).all())
                    output.square().sum().backward()
                    for parameter in encoder.subpath_sequence_encoders.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    for sequence_encoder in encoder.subpath_sequence_encoders.values():
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in sequence_encoder.parameters()), 0)
                    path = ('a', 'b', 'a')
                    encoder.subpaths_dict[path] = [('a', 'b')]
                    forward = encoder._context_embedding(path)
                    encoder.subpaths_dict[path] = [('b', 'a')]
                    reverse = encoder._context_embedding(path)
                    self.assertFalse(torch.allclose(forward, reverse))
                    restored = self.make_encoder(cls, kind).eval()
                    restored.load_state_dict(encoder.state_dict())
                    encoder.subpaths_dict[path] = [('a', 'b'), ('b', 'a')]
                    torch.testing.assert_close(output, restored(x, graph))

    def test_subpaths_have_independent_parameters_and_gradients(self):
        for kind in ('lstm', 'transformer'):
            with self.subTest(kind=kind):
                encoder = self.make_encoder(index.SubpathHANConv2, kind).eval()
                self.assertEqual(set(encoder.subpath_sequence_encoders), {'a_b', 'b_a'})
                first = encoder.subpath_sequence_encoders['a_b']
                second = encoder.subpath_sequence_encoders['b_a']
                self.assertFalse({id(p) for p in first.parameters()} & {id(p) for p in second.parameters()})
                path = ('a', 'b', 'a')
                encoder.subpaths_dict[path] = [('a', 'b')]
                encoder._context_embedding(path).square().sum().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in first.parameters()))
                self.assertTrue(all(p.grad is None for p in second.parameters()))
                if kind == 'transformer':
                    self.assertEqual(set(encoder.subpath_positions), {'a_b', 'b_a'})
                    self.assertIsNotNone(encoder.subpath_positions['a_b'].grad)
                    self.assertIsNone(encoder.subpath_positions['b_a'].grad)

    def test_same_encoder_distinguishes_type_order(self):
        for kind in ('lstm', 'transformer'):
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                encoder = self.make_encoder(index.SubpathHANConv2, kind).eval()
                module = encoder.subpath_sequence_encoders['a_b']
                sequence = torch.stack([encoder.type_emb['a'], encoder.type_emb['b']])
                def encode(values):
                    if kind == 'lstm':
                        return module(values.unsqueeze(0))[1][0][-1, 0]
                    return module((values + encoder.subpath_positions['a_b']).unsqueeze(0))[0].mean(dim=0)
                self.assertFalse(torch.allclose(encode(sequence), encode(sequence.flip(0))))

    def test_repeated_subpath_reuses_one_encoder_and_variable_lengths_work(self):
        for kind in ('lstm', 'transformer'):
            with self.subTest(kind=kind):
                paths = [('a', 'b', 'a'), ('a', 'b', 'a', 'b', 'a')]
                subpaths = {paths[0]: [('a', 'b'), paths[0]],
                            paths[1]: [('a', 'b'), paths[1]]}
                encoder = index.SubpathHANConv2({'a': 3, 'b': 3}, 8, (['a', 'b'], []),
                                               paths, subpaths, aggregator=kind)
                self.assertEqual(len(encoder.subpath_sequence_encoders), 3)
                for path in paths:
                    context = encoder._context_embedding(path)
                    self.assertEqual(context.shape, (8,))
                    self.assertTrue(torch.isfinite(context).all())
                if kind == 'transformer':
                    self.assertEqual(encoder.subpath_positions['a_b_a_b_a'].shape, (5, 8))

    def test_ablations_do_not_allocate_sequence_encoder(self):
        for kind in ('lstm', 'transformer'):
            for mode in ('no-context', 'metapath-bias', 'metapath-scalar-bias'):
                encoder = self.make_encoder(index.SubpathHANConv2, kind, mode)
                self.assertFalse(hasattr(encoder, 'subpath_sequence_encoders'))

if __name__ == '__main__':
    unittest.main()
