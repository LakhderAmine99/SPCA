import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from index import RegularizedSubpathHANConv, StackedSubpathHAN, SubpathClassifier, SubpathHANConv2


class MetapathBiasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = {
            'movie': torch.randn(4, 3),
            'actor': torch.randn(2, 4),
            'director': torch.randn(3, 5),
        }
        self.edges = {}
        for node_type, count in [('actor', 2), ('director', 3)]:
            edges = torch.stack([torch.arange(4), torch.arange(4) % count])
            self.edges['movie', 'to', node_type] = edges
            self.edges[node_type, 'to', 'movie'] = edges.flip(0)
        self.metapaths = [('movie', 'actor', 'movie'), ('movie', 'director', 'movie')]
        self.subpaths = {
            path: [[path[1]], [path[0], path[1]]] for path in self.metapaths
        }
        self.options = dict(
            in_channels_dict={key: value.size(-1) for key, value in self.features.items()},
            out_channels=8, metadata=(list(self.features), list(self.edges)),
            metapaths=self.metapaths, subpaths_dict=self.subpaths,
        )

    def test_no_context_has_no_context_parameters_and_matches_zero_bias_outputs_and_gradients(self):
        for encoder_class in [SubpathHANConv2, RegularizedSubpathHANConv]:
            with self.subTest(encoder=encoder_class.__name__):
                options = {**self.options, 'subpaths_dict': None, 'heads': 2}
                encoder = encoder_class(**options, context_mode='no-context').eval()
                reference = encoder_class(**options, context_mode='metapath-bias').eval()
                encoder.load_state_dict({key: value for key, value in reference.state_dict().items()
                                         if not key.startswith('metapath_bias.')})
                for attribute in ['type_emb', 'subpath_mlps', 'metapath_bias']:
                    self.assertFalse(hasattr(encoder, attribute))
                self.assertEqual(
                    sum(p.numel() for p in reference.parameters()) - sum(p.numel() for p in encoder.parameters()),
                    len(self.metapaths) * self.options['out_channels'],
                )
                with patch.object(encoder, '_context_embedding', side_effect=AssertionError('Context must be bypassed')):
                    actual = encoder(self.features, self.edges)
                expected = reference(self.features, self.edges)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().sum().backward()
                expected.square().sum().backward()
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in encoder.named_parameters():
                    torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)

    def test_bias_mode_needs_no_subpath_definitions_or_encoder_parameters(self):
        for mode, dimensions in [('metapath-bias', 8), ('metapath-scalar-bias', 1)]:
            with self.subTest(mode=mode):
                model = SubpathHANConv2(
                    **{**self.options, 'subpaths_dict': None}, context_mode=mode,
                )
                self.assertFalse(hasattr(model, 'type_emb'))
                self.assertFalse(hasattr(model, 'subpath_mlps'))
                self.assertEqual(sum(p.numel() for p in model.metapath_bias.parameters()),
                                 len(self.metapaths) * dimensions)
                for bias in model.metapath_bias.values():
                    self.assertTrue(bias.requires_grad)
                    self.assertEqual(bias.shape, (dimensions,))
                    self.assertEqual(int(torch.count_nonzero(bias)), 0)
                self.assertEqual(model(self.features, self.edges).shape, (4, 8))

    def test_each_bias_changes_only_its_metapath_and_is_shared_across_nodes(self):
        encoder = SubpathHANConv2(**self.options, context_mode='metapath-bias').eval()
        captured = []
        handle = encoder.final_lin.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        self.addCleanup(handle.remove)
        encoder(self.features, self.edges)
        offset = torch.arange(8, dtype=torch.float)
        with torch.no_grad():
            encoder.metapath_bias['movie_actor_movie'].copy_(offset)
        encoder(self.features, self.edges)
        expected = torch.cat([offset, torch.zeros(8)]).expand(4, -1)
        torch.testing.assert_close(captured[1] - captured[0], expected)

    def test_scalar_bias_broadcasts_across_channels_only_for_its_metapath(self):
        encoder = SubpathHANConv2(**self.options, context_mode='metapath-scalar-bias').eval()
        captured = []
        handle = encoder.final_lin.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        self.addCleanup(handle.remove)
        encoder(self.features, self.edges)
        with torch.no_grad():
            encoder.metapath_bias['movie_actor_movie'].fill_(2.5)
        encoder(self.features, self.edges)
        expected = torch.cat([torch.full((8,), 2.5), torch.zeros(8)]).expand(4, -1)
        torch.testing.assert_close(captured[1] - captured[0], expected)

    def test_matching_context_vectors_preserve_original_predictions(self):
        for aggregator in ['concat-mlp', 'mean']:
            with self.subTest(aggregator=aggregator):
                original = SubpathHANConv2(**self.options, aggregator=aggregator).eval()
                bias = SubpathHANConv2(**self.options, context_mode='metapath-bias').eval()
                for name in ['proj', 'final_lin', 'norm']:
                    getattr(bias, name).load_state_dict(getattr(original, name).state_dict())
                with torch.no_grad():
                    for path in self.metapaths:
                        bias.metapath_bias['_'.join(path)].copy_(original._context_embedding(path))
                    torch.testing.assert_close(
                        original(self.features, self.edges), bias(self.features, self.edges),
                    )

    def test_all_biases_learn_and_checkpoint_restores_predictions(self):
        for mode in ['metapath-bias', 'metapath-scalar-bias']:
            with self.subTest(mode=mode):
                self.check_training_and_checkpoint(mode)

    def check_training_and_checkpoint(self, mode):
        model = SubpathClassifier(SubpathHANConv2(**self.options, context_mode=mode), 8, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        logits = model(self.features, self.edges)
        self.assertEqual(logits.shape, (4, 3))
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0]))
        loss.backward()
        for bias in model.encoder.metapath_bias.values():
            self.assertTrue(torch.isfinite(bias.grad).all())
            self.assertGreater(float(bias.grad.abs().sum()), 0)
        optimizer.step()
        for bias in model.encoder.metapath_bias.values():
            self.assertGreater(float(bias.detach().abs().sum()), 0)

        restored = SubpathClassifier(SubpathHANConv2(**self.options, context_mode=mode), 8, 3)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        with torch.no_grad():
            torch.testing.assert_close(
                restored(self.features, self.edges), model(self.features, self.edges),
            )

    def test_stacked_encoder_uses_selected_context_mode_in_every_layer(self):
        for mode in ['no-context', 'metapath-bias', 'metapath-scalar-bias']:
            with self.subTest(mode=mode):
                model = StackedSubpathHAN(
                    **{**self.options, 'subpaths_dict': None}, num_layers=2, context_mode=mode,
                )
                self.assertTrue(all(layer.context_mode == mode for layer in model.layers))
                self.assertEqual(model(self.features, self.edges).shape, (4, 8))

    def test_unknown_context_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'context_mode'):
            SubpathHANConv2(**self.options, context_mode='missing')


if __name__ == '__main__':
    unittest.main()
