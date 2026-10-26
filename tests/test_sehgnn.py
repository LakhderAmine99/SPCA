import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch_geometric.data import HeteroData

import index
from src.model.sehgnn import SeHGNN, precompute_metapath_features


class SeHGNNTests(unittest.TestCase):
    def setUp(self):
        self.features = {
            'movie': torch.tensor([[2., 0.], [0., 2.], [4., 4.]], requires_grad=True),
            'actor': torch.tensor([[1., 3.], [5., 7.]]),
            'director': torch.tensor([[10., 20.], [30., 40.]]),
        }
        self.edges = {
            ('movie', 'to', 'actor'): torch.tensor([[0, 1], [0, 0]]),
            ('actor', 'to', 'movie'): torch.tensor([[0, 0], [0, 1]]),
            ('movie', 'to', 'director'): torch.tensor([[0, 1], [0, 1]]),
            ('director', 'to', 'movie'): torch.tensor([[0, 1], [0, 1]]),
        }

    def test_mean_propagation_matches_hand_computed_paths(self):
        features = precompute_metapath_features(self.features, self.edges)
        self.assertEqual(len(features), 5)
        torch.testing.assert_close(features[('movie',)], self.features['movie'])
        torch.testing.assert_close(
            features[('actor', 'movie')], torch.tensor([[1., 3.], [1., 3.], [0., 0.]]),
        )
        torch.testing.assert_close(
            features[('director', 'movie')], torch.tensor([[10., 20.], [30., 40.], [0., 0.]]),
        )
        torch.testing.assert_close(
            features[('movie', 'actor', 'movie')], torch.tensor([[1., 1.], [1., 1.], [0., 0.]]),
        )
        torch.testing.assert_close(
            features[('movie', 'director', 'movie')], torch.tensor([[2., 0.], [0., 2.], [0., 0.]]),
        )
        self.assertTrue(all(not features.requires_grad for features in features.values()))

    def test_zero_hops_uses_raw_target_features_only(self):
        features = precompute_metapath_features(self.features, self.edges, num_hops=0)
        self.assertEqual(list(features), [('movie',)])
        with self.assertRaises(ValueError):
            precompute_metapath_features(self.features, self.edges, num_hops=-1)

    def test_selected_paths_match_full_propagation_and_ignore_other_type_features(self):
        paths = [
            ('movie', 'actor', 'movie'),
            ('movie', 'director', 'movie'),
            ('movie', 'actor', 'movie', 'director', 'movie'),
        ]
        full = precompute_metapath_features(self.features, self.edges, num_hops=4)
        selected = precompute_metapath_features(self.features, self.edges, num_hops=4, metapaths=paths)
        self.assertEqual(set(selected), {('movie',), *paths})
        for path in selected:
            torch.testing.assert_close(selected[path], full[path])

        # All requested paths start at movies, so other node types' raw
        # features must not introduce additional channels or affect values.
        changed = {key: value if key == 'movie' else value + 1000
                   for key, value in self.features.items()}
        after = precompute_metapath_features(changed, self.edges, num_hops=4, metapaths=paths)
        for path in selected:
            torch.testing.assert_close(after[path], selected[path])

    def test_invalid_or_missing_selected_paths_are_rejected(self):
        for paths, hops in [([], 2), ([('movie',)], 2),
                            ([('movie', 'actor')], 2),
                            ([('movie', 'actor', 'movie')], 1),
                            ([('movie', 'missing', 'movie')], 2)]:
            with self.subTest(paths=paths, hops=hops), self.assertRaises(ValueError):
                precompute_metapath_features(self.features, self.edges, num_hops=hops, metapaths=paths)

    def test_train_step_and_cached_forward(self):
        features = precompute_metapath_features(self.features, self.edges)
        model = SeHGNN(features, num_classes=3, hidden_channels=8, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        before = model.classifier[0].weight.detach().clone()
        # Graph aggregation must not be repeated inside a training epoch.
        with patch('torch.sparse.mm', side_effect=AssertionError('Features should be cached')):
            logits = model(self.features, self.edges)
            self.assertEqual(logits.shape, (3, 3))
            loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(model(self.features, self.edges)).all())
        self.assertFalse(torch.equal(before, model.classifier[0].weight))

    def test_checkpoint_restores_weights_without_serializing_feature_cache(self):
        features = precompute_metapath_features(self.features, self.edges)
        model = SeHGNN(features, num_classes=3, hidden_channels=8).eval()
        state = model.state_dict()
        self.assertFalse(any(name.startswith('features_') for name in state))
        restored = SeHGNN(features, num_classes=3, hidden_channels=8).eval()
        restored.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(model(), restored())


class IMDBComparisonTests(unittest.TestCase):
    @staticmethod
    def small_dataset():
        data = HeteroData()
        data['movie'].x = torch.randn(30, 4)
        data['actor'].x = torch.randn(5, 4)
        data['director'].x = torch.randn(3, 4)
        data['movie'].y = torch.arange(30) % 3
        for split, indices in [('train', range(6)), ('val', range(6, 12)), ('test', range(12, 30))]:
            mask = torch.zeros(30, dtype=torch.bool)
            mask[list(indices)] = True
            data['movie'][f'{split}_mask'] = mask
        for node_type, count in [('actor', 5), ('director', 3)]:
            edges = torch.stack([torch.arange(30), torch.arange(30) % count])
            data['movie', 'to', node_type].edge_index = edges
            data[node_type, 'to', 'movie'].edge_index = edges.flip(0)
        return data

    def test_both_models_share_masks_and_save_separate_loadable_checkpoints(self):
        self.check_shared_comparison('both', {'subpath', 'sehgnn'})

    def test_all_models_share_masks_and_save_separate_loadable_checkpoints(self):
        self.check_shared_comparison('all', {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias', 'sehgnn', 'simplehgn', 'hettree'})

    def test_hettree_can_run_alone(self):
        self.check_shared_comparison('hettree', {'hettree'})

    def test_hettree_can_run_without_labels_and_a_larger_hop_limit_adds_no_paths(self):
        self.check_shared_comparison('hettree', {'hettree'}, hettree_use_labels=False, hettree_hops=6)

    def test_ablation_shares_masks_and_common_initial_weights(self):
        self.check_shared_comparison('ablation', {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'})

    def test_no_context_can_run_alone(self):
        self.check_shared_comparison('no-context', {'no-context'})

    def test_metapath_bias_can_run_alone(self):
        self.check_shared_comparison('metapath-bias', {'metapath-bias'})

    def test_metapath_scalar_bias_can_run_alone(self):
        self.check_shared_comparison('metapath-scalar-bias', {'metapath-scalar-bias'})

    def check_shared_comparison(self, mode, expected_models, **run_options):
        data = self.small_dataset()
        expected_masks, _ = index.make_imdb_masks(
            data['movie'].y, data['movie'].train_mask, data['movie'].val_mask,
            data['movie'].test_mask, 0.1,
        )
        masks = []
        initial_weights = {}
        original_fit = index.fit_imdb_model

        def record_masks(*args, **kwargs):
            selected, _ = index.imdb_metapath_configuration()
            if args[1] == 'sehgnn':
                self.assertEqual(set(args[0].paths), {('movie',), *selected})
            elif args[1] == 'simplehgn':
                self.assertEqual(args[0].metapaths, tuple(selected))
                self.assertEqual(args[0].node_types, ('movie',))
            if args[1] == 'hettree':
                selected, _ = index.imdb_metapath_configuration()
                model = args[0]
                expected_tree_paths = {path[start:] for path in selected for start in range(len(path))}
                self.assertEqual(set(model.paths), expected_tree_paths)
                self.assertEqual(len(model.paths), 9)
                for i, path in enumerate(model.paths):
                    if path != ('movie',) and path not in selected:
                        self.assertEqual(torch.count_nonzero(getattr(model, f'features_{i}')).item(), 0)
                self.assertNotIn(('movie', 'actor', 'movie', 'actor', 'movie'), model.paths)
                self.assertNotIn(('movie', 'director', 'movie', 'director', 'movie'), model.paths)
                self.assertEqual(set(model.label_paths), set(selected) if run_options.get('hettree_use_labels', True) else set())
            masks.append(tuple(args[2]['movie'][f'{split}_mask'].clone()
                               for split in ['train', 'val', 'test']))
            initial_weights[args[1]] = {
                key: value.detach().clone() for key, value in args[0].state_dict().items()
            }
            return original_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch('index.IMDB', return_value=[data]), \
                patch('index.fit_imdb_model', side_effect=record_masks):
            results = index.run_imdb_example(
                train_ratio=0.1, model_name=mode, epochs=2,
                checkpoint_dir=os.path.relpath(directory, Path.cwd()),
                **run_options,
            )
            self.assertEqual(set(results), expected_models)
            tree_tag = ''
            if mode in {'hettree', 'all'}:
                tree_tag = f"sharedpaths_tree4h_{'labels' if run_options.get('hettree_use_labels', True) else 'features'}_"
            if mode in {'simplehgn', 'all'}:
                tree_tag += 'metagraph_'
            report = json.loads((Path(directory) / f'imdb_{mode}_sharedpaths_v2_{tree_tag}fixed_test10pct_10pct_seed42_results.json').read_text())
            selected, _ = index.imdb_metapath_configuration()
            for configuration in ['subpath', 'sehgnn', 'simplehgn', 'hettree']:
                self.assertEqual(report[f'{configuration}_configuration']['metapaths'],
                                 [list(path) for path in selected])
            self.assertEqual(report['sehgnn_configuration']['feature_hops'], 4)
            self.assertEqual(len({result['checkpoint'] for result in results.values()}), len(expected_models))
            for result in results.values():
                self.assertTrue(Path(result['checkpoint']).is_file())
                self.assertTrue(Path(result['checkpoint']).is_absolute())
                state = torch.load(result['checkpoint'], weights_only=True)
                self.assertTrue(state)
                self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
                self.assertLessEqual(result['best_epoch'], 2)
                self.assertTrue(0 <= result['metrics']['test_mask']['acc'] <= 1)
                self.assertGreater(result['num_parameters'], 0)
            for bias_mode in {'metapath-bias', 'metapath-scalar-bias'} & results.keys():
                dimensions = 1 if bias_mode == 'metapath-scalar-bias' else 128
                self.assertEqual(results[bias_mode]['context_mode'], bias_mode)
                self.assertEqual(results[bias_mode]['bias_dimensions'], dimensions)
                self.assertEqual(results[bias_mode]['context_parameters'],
                                 len(report['subpath_configuration']['metapaths']) * dimensions)
                state = torch.load(results[bias_mode]['checkpoint'], weights_only=True)
                self.assertFalse(any('type_emb.' in key or 'subpath_mlps.' in key for key in state))
                for key, value in state.items():
                    if 'metapath_bias.' in key:
                        self.assertEqual(value.shape, (dimensions,))
                        self.assertGreater(float(value.abs().sum()), 0)
            if 'no-context' in results:
                self.assertEqual(results['no-context']['context_mode'], 'no-context')
                self.assertEqual(results['no-context']['context_parameters'], 0)
                self.assertEqual(results['no-context']['context_injection'], 'none')
                self.assertNotIn('bias_dimensions', results['no-context'])
                state = torch.load(results['no-context']['checkpoint'], weights_only=True)
                self.assertFalse(any(key.startswith(('encoder.type_emb.', 'encoder.subpath_mlps.', 'encoder.metapath_bias.'))
                                     for key in state))
                if mode == 'no-context':
                    self.assertEqual(report['subpath_configuration']['context_injection'], 'none')
            if 'hettree' in results:
                result = results['hettree']
                use_labels = run_options.get('hettree_use_labels', True)
                self.assertEqual(result['implementation'], 'paper-equations-v1')
                self.assertEqual(result['label_propagation'], use_labels)
                self.assertEqual(result['feature_hops'], 4)
                self.assertEqual(report['hettree_configuration']['max_hops'], run_options.get('hettree_hops', 4))
                selected, _ = index.imdb_metapath_configuration()
                self.assertEqual(result['selected_metapaths'], [list(path) for path in selected])
                self.assertEqual(result['path_scope'], 'shared-metapaths-only')
                self.assertEqual(result['ancestor_feature_inputs'], 'zeros')
                self.assertEqual(report['hettree_configuration']['label_propagation'], use_labels)
                state = torch.load(result['checkpoint'], weights_only=True)
                self.assertFalse(any(key.startswith(('features_', 'labels_')) for key in state))
            self.assertEqual(report['models'], results)
            self.assertEqual(report['train_count'], 3)
            self.assertEqual(report['split_strategy'], 'fixed-nested')
            self.assertEqual(report['split_protocol'], 'fixed')
            self.assertEqual(report['test_count'], 3)
            self.assertEqual(report['unused_count'], 18)
            for split in ['train', 'val', 'test']:
                self.assertEqual(report['split_indices'][split],
                                 expected_masks[f'{split}_mask'].nonzero(as_tuple=True)[0].tolist())
        self.assertEqual(len(masks), len(expected_models))
        for snapshot in masks[1:]:
            for first, second in zip(masks[0], snapshot):
                self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(masks[0][0].sum()), 3)
        self.assertTrue(torch.equal(masks[0][1], expected_masks['val_mask']))
        self.assertTrue(torch.equal(masks[0][2], expected_masks['test_mask']))
        if 'subpath' in initial_weights:
            for bias_mode in {'no-context', 'metapath-bias', 'metapath-scalar-bias'} & initial_weights.keys():
                for key, value in initial_weights['subpath'].items():
                    if key.startswith(('encoder.proj.', 'encoder.final_lin.', 'encoder.norm.', 'classifier.')):
                        torch.testing.assert_close(value, initial_weights[bias_mode][key], rtol=0, atol=0)
                self.assertGreater(results['subpath']['context_parameters'],
                                   results[bias_mode]['context_parameters'])

    def test_both_models_share_fixed_masks_for_larger_ratios(self):
        data = self.small_dataset()
        observed_masks = []
        original_fit = index.fit_imdb_model

        def record_masks(*args, **kwargs):
            observed_masks.append(tuple(args[2]['movie'][f'{split}_mask'].clone()
                                        for split in ['train', 'val', 'test']))
            return original_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch('index.IMDB', return_value=[data]), \
                patch('index.fit_imdb_model', side_effect=record_masks):
            index.run_imdb_example(
                train_ratio=0.4, model_name='both', epochs=1,
                checkpoint_dir=directory,
            )
            report = json.loads((Path(directory) / 'imdb_both_sharedpaths_v2_fixed_test10pct_40pct_seed42_results.json').read_text())
            self.assertEqual(report['split_strategy'], 'fixed-nested')
            self.assertEqual([report[f'{split}_count'] for split in ['train', 'val', 'test']],
                             [12, 6, 3])
        self.assertEqual(len(observed_masks), 2)
        for first, second in zip(*observed_masks):
            self.assertTrue(torch.equal(first, second))
        membership = torch.stack(observed_masks[0]).sum(dim=0)
        self.assertTrue(torch.all(membership <= 1))
        self.assertEqual(int((membership == 0).sum()), 9)

    def test_legacy_split_protocol_preserves_masks_with_shared_path_report_names(self):
        data = self.small_dataset()
        with tempfile.TemporaryDirectory() as directory, patch('index.IMDB', return_value=[data]):
            index.run_imdb_example(
                train_ratio=0.4, model_name='subpath', epochs=1,
                checkpoint_dir=directory, split_protocol='legacy',
            )
            report = json.loads((Path(directory) / 'imdb_subpath_sharedpaths_v2_40pct_seed42_results.json').read_text())
            self.assertEqual(report['split_protocol'], 'legacy')
            self.assertEqual(report['split_strategy'], 'stratified')
            self.assertEqual([report[f'{split}_count'] for split in ['train', 'val', 'test']],
                             [12, 6, 12])

    def test_full_training_pool_is_saved_under_a_distinct_budget_name(self):
        data = self.small_dataset()
        with tempfile.TemporaryDirectory() as directory, patch('index.IMDB', return_value=[data]):
            results = index.run_imdb_example(
                train_ratio='full', model_name='subpath', epochs=1,
                checkpoint_dir=directory,
            )
            report = json.loads((Path(directory) / 'imdb_subpath_sharedpaths_v2_fixed_test10pct_full_seed42_results.json').read_text())
            self.assertEqual(report['requested_train_ratio'], 'full')
            self.assertEqual(report['train_ratio'], 21 / 30)
            self.assertEqual([report[f'{split}_count'] for split in ['train', 'val', 'test']],
                             [21, 6, 3])
            self.assertEqual(report['unused_count'], 0)
            self.assertEqual(Path(results['subpath']['checkpoint']).name,
                             'imdb_subpath_sharedpaths_v2_fixed_test10pct_full_seed42.pth')
            membership = torch.stack([
                data['movie'][f'{split}_mask'] for split in ['train', 'val', 'test']
            ]).sum(dim=0)
            self.assertTrue(torch.all(membership == 1))

    def test_final_evaluation_restores_lowest_validation_loss_weights(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        observed_weights = []

        def training_step(*args):
            with torch.no_grad():
                model.weight.add_(1)
            return 1.0

        def evaluation(*args):
            observed_weights.append(float(model.weight.detach().item()))
            loss = 0.2 if model.weight.item() == 1 else 0.8
            return {split: {'loss': loss, 'acc': 0.5, 'f1_micro': 0.5, 'f1_macro': 0.5}
                    for split in ['train_mask', 'val_mask', 'test_mask']}

        with tempfile.TemporaryDirectory() as directory, \
                patch('index.train', side_effect=training_step), \
                patch('index.test', side_effect=evaluation):
            result = index.fit_imdb_model(
                model, 'toy', None, torch.device('cpu'),
                Path(directory) / 'best.pth', epochs=3, patience=2,
            )
        self.assertEqual(result['best_epoch'], 1)
        self.assertEqual(observed_weights, [1.0, 2.0, 3.0, 1.0])

    def test_invalid_run_configuration_fails_before_loading_data(self):
        for kwargs in [{'model_name': 'missing'}, {'epochs': 0}, {'patience': 0},
                       {'split_protocol': 'missing'}, {'test_ratio': 0}, {'test_ratio': 1},
                       {'test_count': 0}, {'test_count': -800}, {'test_count': 800.5},
                       {'test_count': True}, {'test_count': 800, 'split_protocol': 'legacy'},
                       {'model_name': 'hettree', 'hettree_hops': 2},
                       {'model_name': 'all', 'hettree_hops': 2}, {'hettree_hops': 0},
                       {'train_ratio': 'full', 'split_protocol': 'legacy'}]:
            with self.subTest(kwargs=kwargs), patch('index.IMDB') as load:
                with self.assertRaises(ValueError):
                    index.run_imdb_example(**kwargs)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
