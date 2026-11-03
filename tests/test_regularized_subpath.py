import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

import index

try:  # Historical local-only script (gitignored); absent from public clones.
    import tune_imdb
except ImportError:
    tune_imdb = None


def tiny_graph():
    data = HeteroData()
    data['movie'].x = torch.randn(30, 4)
    data['movie'].y = torch.arange(30) % 3
    for split, start, stop in [('train', 0, 6), ('val', 6, 12), ('test', 12, 30)]:
        mask = torch.zeros(30, dtype=torch.bool)
        mask[start:stop] = True
        data['movie'][f'{split}_mask'] = mask
    for node_type, count in [('actor', 5), ('director', 3)]:
        data[node_type].x = torch.randn(count, 4)
        edges = torch.stack([torch.arange(30), torch.arange(30) % count])
        data['movie', 'to', node_type].edge_index = edges
        data[node_type, 'to', 'movie'].edge_index = edges.flip(0)
    return data


class RegularizedSubpathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.data = tiny_graph()
        self.paths, self.subpaths = index.imdb_metapath_configuration()

    def encoder(self, channels=8, heads=2, **kwargs):
        return index.RegularizedSubpathHANConv(
            {key: value.size(-1) for key, value in self.data.x_dict.items()},
            channels, self.data.metadata(), self.paths, self.subpaths,
            heads=heads, **kwargs,
        )

    def test_attention_is_independently_normalized_per_head(self):
        encoder = self.encoder(channels=4, heads=2)
        sources = torch.tensor([[1., 0., 0., 1.], [0., 1., 1., 0.]])
        targets = torch.tensor([[1., 0., 1., 0.]])
        edges = torch.tensor([[0, 1], [0, 0]])
        output = encoder._message_pass(sources, targets, edges)
        weight = torch.softmax(torch.tensor([1 / math.sqrt(2), 0.]), dim=0)[0]
        expected = torch.stack([weight, 1 - weight, weight, 1 - weight]).unsqueeze(0)
        torch.testing.assert_close(output, expected)

    def test_residual_preserves_feature_gradients_for_isolated_movies(self):
        encoder = self.encoder()
        features = {key: value.detach().clone().requires_grad_()
                    for key, value in self.data.x_dict.items()}
        empty_edges = {key: torch.empty(2, 0, dtype=torch.long)
                       for key in self.data.edge_index_dict}
        output = encoder(features, empty_edges)
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        self.assertGreater(float(features['movie'].grad.abs().sum()), 0)

    def test_context_variants_learn_and_restore_with_regularized_encoder(self):
        for mode in ['subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias']:
            with self.subTest(mode=mode):
                model = index.SubpathClassifier(self.encoder(context_mode=mode, dropout=0.3), 8, 3)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                logits = model(self.data.x_dict, self.data.edge_index_dict)
                loss = torch.nn.functional.cross_entropy(logits, self.data['movie'].y)
                loss.backward()
                if mode in {'metapath-bias', 'metapath-scalar-bias'}:
                    for bias in model.encoder.metapath_bias.values():
                        self.assertGreater(float(bias.grad.abs().sum()), 0)
                optimizer.step()
                restored = index.SubpathClassifier(self.encoder(context_mode=mode, dropout=0.3), 8, 3)
                restored.load_state_dict(model.state_dict())
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(self.data.x_dict, self.data.edge_index_dict),
                        restored(self.data.x_dict, self.data.edge_index_dict),
                    )

    def test_head_count_must_divide_embedding_dimensions(self):
        for heads in [0, 3]:
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                self.encoder(heads=heads)

    def test_regularized_ablation_uses_identical_common_initial_weights(self):
        snapshots = {}
        original_fit = index.fit_imdb_model

        def record_initialization(model, name, *args, **kwargs):
            snapshots[name] = {key: value.detach().clone() for key, value in model.state_dict().items()
                               if not key.startswith(('encoder.type_emb.', 'encoder.subpath_mlps.', 'encoder.metapath_bias.'))}
            return original_fit(model, name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch('index.IMDB', return_value=[self.data]), \
                patch('index.fit_imdb_model', side_effect=record_initialization), contextlib.redirect_stdout(io.StringIO()):
            results = index.run_imdb_example(
                train_ratio=0.1, model_name='ablation', epochs=1, subpath_variant='regularized',
                checkpoint_dir=directory,
            )
            self.assertEqual(set(results), {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'})
            self.assertEqual(results['no-context']['context_parameters'], 0)
            self.assertEqual(len(list(Path(directory).glob('*reg*_results.json'))), 1)
            for weights in snapshots.values():
                self.assertEqual(weights.keys(), snapshots['subpath'].keys())
                for key, weight in weights.items():
                    torch.testing.assert_close(weight, snapshots['subpath'][key], rtol=0, atol=0)

    def test_training_epochs_do_not_request_test_metrics(self):
        model = index.SubpathClassifier(self.encoder(), 8, 3)
        requested_splits = []
        original_test = index.test

        def record_splits(model, data, device, splits=None):
            requested_splits.append(splits)
            return original_test(model, data, device, splits)

        with tempfile.TemporaryDirectory() as directory, patch('index.test', side_effect=record_splits), \
                contextlib.redirect_stdout(io.StringIO()):
            index.fit_imdb_model(model, 'toy', self.data, torch.device('cpu'),
                                 Path(directory) / 'best.pth', epochs=2)
        self.assertEqual(requested_splits, [('train_mask', 'val_mask'), ('train_mask', 'val_mask'), None])

    def test_training_rejects_a_mask_that_contains_held_out_labels(self):
        model = index.SubpathClassifier(self.encoder(), 8, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for held_out in ['val_mask', 'test_mask']:
            data = self.data.clone()
            data['movie'].train_mask |= data['movie'][held_out]
            with self.subTest(held_out=held_out), self.assertRaisesRegex(ValueError, 'overlaps'):
                index.train(model, data, optimizer, torch.nn.CrossEntropyLoss(), torch.device('cpu'))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_macro_selection_restores_the_best_macro_checkpoint(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()

        def training_step(*args):
            with torch.no_grad():
                model.weight.add_(1)
            return 1.0

        def evaluation(*args):
            epoch = int(model.weight.item())
            loss, macro = {1: (0.2, 0.4), 2: (0.8, 0.7), 3: (0.7, 0.6)}[epoch]
            return {split: {'loss': loss, 'acc': macro, 'f1_micro': macro, 'f1_macro': macro}
                    for split in ['train_mask', 'val_mask', 'test_mask']}

        with tempfile.TemporaryDirectory() as directory, patch('index.train', side_effect=training_step), \
                patch('index.test', side_effect=evaluation), contextlib.redirect_stdout(io.StringIO()):
            result = index.fit_imdb_model(model, 'toy', None, torch.device('cpu'),
                                          Path(directory) / 'best.pth', epochs=3,
                                          selection_metric='f1_macro')
        self.assertEqual(result['best_epoch'], 2)
        self.assertEqual(result['best_val_loss'], 0.8)
        self.assertEqual(float(model.weight.item()), 2.0)

    @unittest.skipIf(tune_imdb is None, 'tune_imdb.py is a local-only historical script')
    def test_tuner_requests_test_only_once_after_configuration_selection(self):
        requested_splits = []
        original_test = tune_imdb.test

        def record_splits(model, data, device, splits=None):
            requested_splits.append(splits)
            return original_test(model, data, device, splits)

        with tempfile.TemporaryDirectory() as directory, patch('tune_imdb.IMDB', return_value=[self.data]), \
                patch('tune_imdb.CANDIDATES', tune_imdb.CANDIDATES[:2]), \
                patch('tune_imdb.test', side_effect=record_splits), contextlib.redirect_stdout(io.StringIO()):
            report = tune_imdb.run_validation_search(train_ratio=0.1, epochs=1, output_dir=directory)
            self.assertEqual(report['test_evaluations'], 1)
            self.assertEqual(requested_splits, [('train_mask', 'val_mask'), ('train_mask', 'val_mask'), None])
            for trial in report['trials']:
                self.assertNotIn('test_mask', trial['metrics'])


if __name__ == '__main__':
    unittest.main()
