import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import index
from test_regularized_subpath import tiny_graph


class AggregatorAblationTests(unittest.TestCase):
    def test_real_training_reports_and_shared_initialization(self):
        initial = []
        original_fit = index.fit_imdb_model
        def capture(model, *args, **kwargs):
            initial.append({key: value.clone() for key, value in model.state_dict().items()
                            if not key.startswith(('encoder.subpath_mlps.',
                                'encoder.subpath_sequence_encoders.', 'encoder.subpath_positions'))})
            return original_fit(model, *args, **kwargs)
        for variant in ('original', 'regularized'):
            initial.clear()
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory, \
                    patch('index.IMDB', return_value=[tiny_graph()]), \
                    patch('index.fit_imdb_model', side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = index.run_aggregator_ablation(
                    seeds=[42], train_ratio=0.1, epochs=2, patience=2,
                    checkpoint_dir=directory, subpath_variant=variant)
                self.assertTrue(report['complete'])
                self.assertEqual(len(report['runs']), 4)
                self.assertEqual(len({r['checkpoint'] for r in report['runs']}), 4)
                for weights in initial[1:]:
                    self.assertEqual(initial[0].keys(), weights.keys())
                    for key, value in weights.items():
                        torch.testing.assert_close(value, initial[0][key], rtol=0, atol=0)
                for run in report['runs']:
                    self.assertEqual(run['epochs_run'], 2)
                    self.assertEqual(len(run['epoch_times_seconds']), 2)
                    self.assertGreater(run['epoch_time_mean_seconds'], 0)
                    self.assertGreaterEqual(run['epoch_time_mean_seconds'], run['training_time_mean_seconds'])
                    self.assertGreater(run['context_parameters'], 0)
                    self.assertEqual(run['subpath_encoder_sharing'], 'per-distinct-subpath')
                    self.assertEqual(run['subpath_encoder_count'], 0 if run['aggregator'] == 'mean' else 4)
                    self.assertIn('persubpath_v2', run['checkpoint'])
                counts = {run['aggregator']: run['context_parameters'] for run in report['runs']}
                self.assertEqual(counts, {'concat-mlp': 330112, 'mean': 384,
                                          'lstm': 528768, 'transformer': 532352})
                for entry in report['summary']:
                    self.assertIsNone(entry['test_accuracy_std'])
                saved = list(Path(directory).glob('*aggregator_ablation*.json'))
                self.assertEqual(len(saved), 1)
                self.assertTrue(json.loads(saved[0].read_text())['complete'])
                self.assertEqual(report['configuration']['encoder_protocol'], 'per-subpath-v2')
                self.assertEqual(len(list(Path(directory).glob('*aggregator_ablation*.csv'))), 2)

    def test_summary_uses_sample_std_across_seeds(self):
        def fake_run(**options):
            value = options['split_seed'] / 100
            return {'subpath': dict(
                metrics={'test_mask': dict(acc=value, f1_micro=value, f1_macro=value, loss=1.)},
                num_parameters=100, context_parameters=10, best_epoch=1, best_val_loss=1.,
                epochs_run=2, epoch_time_mean_seconds=value, epoch_time_std_seconds=0.,
                epoch_time_median_seconds=value, epoch_time_min_seconds=value,
                epoch_time_max_seconds=value, training_time_mean_seconds=value / 2,
                total_epoch_time_seconds=value * 2, device='cpu', checkpoint='mock.pth',
                timing_scope='mock')}
        with tempfile.TemporaryDirectory() as directory, patch('index.run_experiment', side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            report = index.run_aggregator_ablation(seeds=[42, 44], checkpoint_dir=directory)
        self.assertEqual(len(report['runs']), 8)
        for entry in report['summary']:
            self.assertAlmostEqual(entry['test_accuracy_mean'], 0.43)
            self.assertAlmostEqual(entry['test_accuracy_std'], 0.02 / 2 ** 0.5)

    def test_repeated_seeds_rejected(self):
        with self.assertRaises(ValueError):
            index.run_aggregator_ablation(seeds=[42, 42])


if __name__ == '__main__':
    unittest.main()
