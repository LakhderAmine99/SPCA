import copy
import math
import unittest

import torch
from torch import nn

from src.model.hettree import HetTree, SubtreeAttention, precompute_hettree_inputs


class HetTreeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = {
            'movie': torch.tensor([[2., 0.], [0., 2.], [4., 4.], [8., 8.]], requires_grad=True),
            'actor': torch.tensor([[1., 3., 5.], [5., 7., 9.]]),
        }
        edges = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        self.edges = {('movie', 'to', 'actor'): edges, ('actor', 'to', 'movie'): edges.flip(0)}
        self.labels = torch.tensor([0, 1, 999, -1])
        self.train_mask = torch.tensor([True, True, False, False])
        self.transition = torch.tensor([
            [.5, .5, 0., 0.], [.25, .5, .25, 0.],
            [0., .5, .5, 0.], [0., 0., 0., 0.],
        ])

    def precompute(self, **kwargs):
        options = dict(labels=self.labels, train_mask=self.train_mask, num_classes=2)
        options.update(kwargs)
        return precompute_hettree_inputs(self.features, self.edges, **options)

    def test_features_and_self_excluded_labels_match_hand_computed_two_hop_graph(self):
        features, labels = self.precompute(label_batch_size=2)
        self.assertEqual(set(features), {('movie',), ('actor', 'movie'), ('movie', 'actor', 'movie')})
        path = ('movie', 'actor', 'movie')
        torch.testing.assert_close(features[path], self.transition @ self.features['movie'])
        torch.testing.assert_close(labels[path], torch.tensor([[0., 1.], [.5, 0.], [0., 1.], [0., 0.]]))
        self.assertTrue(all(not value.requires_grad for value in features.values()))
        self.assertEqual(set(labels), {path})

    def test_four_hop_self_exclusion_matches_dense_operator_for_multiple_block_sizes(self):
        expected_operator = self.transition @ self.transition
        expected_operator.fill_diagonal_(0)
        denominator = expected_operator.sum(dim=1, keepdim=True)
        seed = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
        expected = expected_operator @ seed / denominator.clamp_min(1e-7)
        for batch_size in [1, 3, 20]:
            with self.subTest(batch_size=batch_size):
                _, labels = self.precompute(num_hops=4, label_batch_size=batch_size)
                torch.testing.assert_close(labels[('movie', 'actor', 'movie', 'actor', 'movie')], expected)

    def test_held_out_labels_and_own_label_cannot_leak_into_label_inputs(self):
        _, original = self.precompute(num_hops=4)
        changed = self.labels.clone()
        changed[~self.train_mask] = torch.tensor([-999, 555])
        _, held_out_changed = self.precompute(num_hops=4, labels=changed)
        changed[0] = 1
        _, own_label_changed = self.precompute(num_hops=4, labels=changed)
        for path, value in original.items():
            torch.testing.assert_close(value, held_out_changed[path], rtol=0, atol=0)
            torch.testing.assert_close(value[0], own_label_changed[path][0])
        self.assertGreater(float((original[('movie', 'actor', 'movie')][1] -
                                  own_label_changed[('movie', 'actor', 'movie')][1]).abs().sum()), 0)

    def test_self_only_components_have_no_label_input(self):
        edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        features = {'movie': torch.ones(4, 2), 'actor': torch.ones(4, 2)}
        graph = {('movie', 'to', 'actor'): edges, ('actor', 'to', 'movie'): edges.flip(0)}
        _, labels = precompute_hettree_inputs(
            features, graph, num_hops=4, labels=torch.tensor([0, 1, 0, 1]),
            train_mask=torch.ones(4, dtype=torch.bool), num_classes=2,
        )
        for values in labels.values():
            self.assertEqual(int(torch.count_nonzero(values)), 0)
            self.assertTrue(torch.isfinite(values).all())

    def test_selected_metapaths_include_their_tree_ancestors(self):
        path = ('movie', 'actor', 'movie', 'actor', 'movie')
        features, _ = self.precompute(num_hops=4, metapaths=[path])
        self.assertEqual(set(features), {path[i:] for i in range(len(path))})
        model = HetTree(features, 2)
        for i, children in enumerate(model.tree_children):
            for child in children:
                self.assertEqual(model.paths[child][1:], model.paths[i])

    def test_selected_metapaths_do_not_enable_other_closed_walks_within_the_hop_limit(self):
        path = ('movie', 'actor', 'movie')
        features, labels = self.precompute(num_hops=4, metapaths=[path])
        self.assertEqual(set(features), {('movie',), ('actor', 'movie'), path})
        self.assertEqual(set(labels), {path})
        original_features, original_labels = self.precompute(num_hops=2)
        for key in features:
            torch.testing.assert_close(features[key], original_features[key], rtol=0, atol=0)
        torch.testing.assert_close(labels[path], original_labels[path], rtol=0, atol=0)

    def test_shared_path_mode_zeroes_ancestors_and_ignores_their_input_features(self):
        path = ('movie', 'actor', 'movie', 'actor', 'movie')
        options = dict(num_hops=4, metapaths=[path], include_ancestor_features=False)
        features, labels = self.precompute(**options)
        full, full_labels = self.precompute(num_hops=4)
        torch.testing.assert_close(features[path], full[path])
        torch.testing.assert_close(labels[path], full_labels[path])
        self.assertEqual(set(labels), {path})
        for key, value in features.items():
            if key not in {('movie',), path}:
                self.assertEqual(torch.count_nonzero(value).item(), 0)
        changed = {key: value if key == 'movie' else value + 1000
                   for key, value in self.features.items()}
        after, _ = precompute_hettree_inputs(changed, self.edges, use_labels=False, **options)
        for key in features:
            torch.testing.assert_close(features[key], after[key])
        model = HetTree(features, 2, labels, hidden_channels=8, dropout=0)
        self.assertEqual(model.label_paths, (path,))
        loss = nn.functional.cross_entropy(model()[self.train_mask], self.labels[self.train_mask])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_subtree_attention_uses_children_in_reference_and_normalizes_weights(self):
        layer = SubtreeAttention(1, dropout=0)
        layer.reference = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.reference.weight.fill_(1)
            layer.score.weight.copy_(torch.tensor([[-1., 1.]]))
        parent = torch.tensor([[0.]])
        children = torch.tensor([[[1.], [0.]]], requires_grad=True)
        encoded_children = torch.tensor([[[0.], [2.]]])
        result, weights = layer(parent, children, encoded_children)
        expected_weight = math.exp(1) / (math.exp(-.2) + math.exp(1))
        torch.testing.assert_close(weights[0, :, 0], torch.tensor([1 - expected_weight, expected_weight]))
        torch.testing.assert_close(result, torch.tensor([[2 * expected_weight]]))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1))
        result.sum().backward()
        self.assertGreater(float(children.grad.abs().sum()), 0)
        _, changed_weights = layer(parent, torch.zeros_like(children), encoded_children)
        self.assertFalse(torch.allclose(weights, changed_weights))

    def test_training_and_checkpoint_restore_without_serializing_cached_features_or_labels(self):
        features, labels = self.precompute()
        model = HetTree(features, 2, labels, hidden_channels=8, dropout=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        before = model().detach().clone()
        self.assertEqual(before.shape, (4, 2))
        nn.functional.cross_entropy(model()[self.train_mask], self.labels[self.train_mask]).backward()
        self.assertGreater(float(model.projections[0][0].weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.allclose(before, model()))
        state = copy.deepcopy(model.state_dict())
        self.assertFalse(any(key.startswith(('features_', 'labels_')) for key in state))
        restored = HetTree(dict(reversed(list(features.items()))), 2, labels, hidden_channels=8, dropout=0)
        restored.load_state_dict(state)
        torch.testing.assert_close(model(), restored(), rtol=0, atol=0)
        restored.double()
        self.assertEqual(restored().dtype, torch.float64)
        self.assertTrue(torch.isfinite(restored()).all())

    def test_feature_only_mode_needs_no_labels_or_label_modules(self):
        features, labels = precompute_hettree_inputs(self.features, self.edges, use_labels=False)
        self.assertIsNone(labels)
        model = HetTree(features, 2, labels, hidden_channels=8, dropout=0)
        self.assertFalse(model.uses_labels)
        self.assertIsNone(model.label_residual)
        self.assertEqual(model.label_paths, ())
        self.assertEqual(model().shape, (4, 2))

    def test_invalid_training_inputs_and_incomplete_trees_are_rejected(self):
        for options in [dict(label_batch_size=0), dict(num_classes=0), dict(train_mask=torch.ones(4)),
                        dict(labels=torch.tensor([2, 1, 0, 0]))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.precompute(**options)
        with self.assertRaisesRegex(ValueError, 'requires'):
            precompute_hettree_inputs(self.features, self.edges)
        features, labels = self.precompute()
        incomplete = {key: value for key, value in features.items() if key != ('actor', 'movie')}
        with self.assertRaisesRegex(ValueError, 'parent'):
            HetTree(incomplete, 2)
        with self.assertRaisesRegex(ValueError, 'Labels must match'):
            HetTree(features, 2, {('movie',): torch.zeros(4, 2), **labels})


if __name__ == '__main__':
    unittest.main()
