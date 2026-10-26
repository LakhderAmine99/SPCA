import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import savemat
from torch_geometric.data import HeteroData

import index
from src.acm import ACM
from src.datasets import DATASET_INFO, metapath_configuration, prepare_benchmark
from src.dblp import DBLP


class DatasetSelectionTests(unittest.TestCase):
    @staticmethod
    def small_graph(name):
        target_type = DATASET_INFO[name][0]
        paths, _ = metapath_configuration(name)
        types = sorted({node for path in paths for node in path})
        data = HeteroData()
        for node_type in types:
            count = 60 if node_type == target_type else 6
            data[node_type].num_nodes = count
            if node_type != 'conference':
                data[node_type].x = torch.randn(count, 4)
        data[target_type].y = torch.arange(60) % (4 if name == 'dblp' else 3)
        for split, indices in [('train', range(12)), ('val', range(12, 24)), ('test', range(24, 60))]:
            mask = torch.zeros(60, dtype=torch.bool)
            mask[list(indices)] = True
            data[target_type][f'{split}_mask'] = mask
        for path in paths:
            for source, destination in zip(path, path[1:]):
                if (source, 'relation', destination) in data.edge_types:
                    continue
                count = data[source].num_nodes
                data[source, 'relation', destination].edge_index = torch.stack([
                    torch.arange(count), torch.arange(count) % data[destination].num_nodes,
                ])
        return data

    def test_dblp_and_acm_run_all_models_with_shared_target_masks(self):
        for name in ['dblp', 'acm']:
            with self.subTest(dataset=name):
                self.check_experiment(name, 'all', 'original')

    def test_regularized_ablation_uses_author_targets(self):
        self.check_experiment('dblp', 'ablation', 'regularized')

    def test_all_models_share_an_exact_test_count_and_report_it(self):
        for name in ['dblp', 'acm']:
            with self.subTest(dataset=name):
                self.check_experiment(name, 'all', 'original', test_count=9)

    def test_hettree_runs_alone_on_dblp_and_acm(self):
        for name in ['dblp', 'acm']:
            with self.subTest(dataset=name):
                self.check_experiment(name, 'hettree', 'original')

    def check_experiment(self, name, model_name, variant, test_count=None):
        data = self.small_graph(name)
        target_type = DATASET_INFO[name][0]
        snapshots = []
        original_fit = index.fit_imdb_model
        selected, _ = metapath_configuration(name)

        def record_masks(model, model_name, graph, *args, **kwargs):
            self.assertEqual(graph.target_type, target_type)
            if model_name == 'sehgnn':
                self.assertEqual(set(model.paths), {(target_type,), *selected})
            elif model_name == 'simplehgn':
                self.assertEqual(model.metapaths, tuple(selected))
                self.assertEqual(model.graph_mode, 'metapath-projected')
                self.assertEqual(model.node_types, (target_type,))
            elif model_name == 'hettree':
                self.assertEqual(set(model.label_paths), set(selected))
                for i, path in enumerate(model.paths):
                    if path != (target_type,) and path not in selected:
                        self.assertEqual(torch.count_nonzero(getattr(model, f'features_{i}')).item(), 0)
            elif hasattr(model, 'encoder'):
                self.assertEqual(model.encoder.metapaths, selected)
            self.assertEqual(model(graph.x_dict, graph.edge_index_dict).shape,
                             (60, 4 if name == 'dblp' else 3))
            snapshots.append(tuple(graph[target_type][f'{split}_mask'].clone()
                                   for split in ['train', 'val', 'test']))
            return original_fit(model, model_name, graph, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch(f'index.{name.upper()}', return_value=[data]) as load, \
                patch('index.IMDB', side_effect=AssertionError('Wrong dataset loader')), \
                patch('index.fit_imdb_model', side_effect=record_masks), \
                contextlib.redirect_stdout(io.StringIO()):
            results = index.run_experiment(
                dataset_name=name.upper(), model_name=model_name, train_ratio=0.2,
                epochs=1, checkpoint_dir=directory, subpath_variant=variant,
                data_root=Path(directory) / 'cache', test_count=test_count,
            )
            load.assert_called_once_with(root=str((Path(directory) / 'cache').resolve()))
            report_path = next(Path(directory).glob(f'{name}_*_results.json'))
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(report['dataset_name'], name)
            self.assertEqual(report['target_type'], target_type)
            self.assertEqual(report['num_classes'], 4 if name == 'dblp' else 3)
            self.assertEqual(report['train_count'], 12)
            self.assertEqual(report['val_count'], 12)
            self.assertEqual(report['test_count'], 6 if test_count is None else test_count)
            self.assertEqual(report['requested_test_count'], test_count)
            if test_count is not None:
                self.assertEqual(report['test_ratio'], test_count / 60)
                self.assertIsNone(report['requested_test_ratio'])
                self.assertIn(f'fixed_test{test_count}nodes_', report_path.name)
                self.assertEqual(len(report['split_indices']['test']), test_count)
                self.assertTrue(all(f'fixed_test{test_count}nodes_' in result['checkpoint']
                                    for result in results.values()))
            self.assertEqual(report['sehgnn_configuration']['feature_hops'], max(len(path) - 1 for path in selected))
            self.assertTrue(report['sehgnn_configuration']['include_raw_features'])
            for configuration in ['subpath', 'sehgnn', 'simplehgn', 'hettree']:
                self.assertEqual(report[f'{configuration}_configuration']['metapaths'],
                                 [list(path) for path in selected])
            self.assertEqual(report['simplehgn_configuration']['graph_mode'], 'metapath-projected')
            self.assertIn('sharedpaths_v2', report_path.name)
            if model_name == 'all':
                self.assertIn('metagraph', report_path.name)
            self.assertEqual(len(results), 7 if model_name == 'all' else 4 if model_name == 'ablation' else 1)
            if 'no-context' in results:
                self.assertEqual(results['no-context']['context_parameters'], 0)
                self.assertEqual(results['no-context']['context_injection'], 'none')
            if 'hettree' in results:
                self.assertTrue(results['hettree']['label_propagation'])
                self.assertEqual(results['hettree']['feature_hops'], 4 if name == 'dblp' else 2)
                self.assertEqual(report['hettree_configuration']['label_source'], 'training-mask-only')
                self.assertEqual(report['hettree_configuration']['implementation'], 'paper-equations-v1')
                selected, _ = index.metapath_configuration(name)
                expected_tree_paths = {path[start:] for path in selected for start in range(len(path))}
                self.assertEqual(results['hettree']['selected_metapaths'], [list(path) for path in selected])
                self.assertEqual({tuple(path) for path in results['hettree']['feature_paths']},
                                 {(target_type,), *selected})
                self.assertEqual({tuple(path) for path in results['hettree']['tree_paths']}, expected_tree_paths)
                self.assertEqual({tuple(path) for path in results['hettree']['label_paths']}, set(selected))
                self.assertEqual(report['hettree_configuration']['metapaths'], [list(path) for path in selected])
                self.assertEqual(report['hettree_configuration']['path_scope'], 'shared-metapaths-only')
                self.assertEqual(results['hettree']['ancestor_feature_inputs'], 'zeros')
                self.assertIn('sharedpaths_tree', report_path.name)
            for result in results.values():
                self.assertEqual(result['selected_metapaths'], [list(path) for path in selected])
                checkpoint = Path(result['checkpoint'])
                self.assertTrue(checkpoint.name.startswith(name + '_'))
                self.assertTrue(torch.load(checkpoint, weights_only=True))
                self.assertTrue(np.isfinite(result['metrics']['test_mask']['loss']))
        for snapshot in snapshots:
            self.assertEqual(int(snapshot[2].sum()), 6 if test_count is None else test_count)
            for current, first in zip(snapshot, snapshots[0]):
                self.assertTrue(torch.equal(current, first))
            self.assertTrue(torch.all(torch.stack(snapshot).sum(dim=0) <= 1))

    def test_featureless_conferences_and_relation_names_are_prepared(self):
        data = prepare_benchmark(self.small_graph('dblp'), 'dblp')
        torch.testing.assert_close(data['conference'].x, torch.eye(6))
        self.assertEqual(data.target_type, 'author')
        self.assertTrue(all(relation == 'to' for _, relation, _ in data.edge_types))

    def test_unknown_dataset_fails_before_loading(self):
        with patch('index.IMDB') as load:
            with self.assertRaisesRegex(ValueError, 'dataset_name'):
                index.run_experiment(dataset_name='missing')
        load.assert_not_called()

    def test_unknown_target_labels_are_not_used_as_a_class(self):
        data = self.small_graph('dblp')
        data['author'].y[-1] = -1
        with self.assertRaisesRegex(ValueError, 'known class label'):
            prepare_benchmark(data, 'dblp')


class ACMProcessingTests(unittest.TestCase):
    def test_raw_graph_processing_has_real_labels_and_label_free_features(self):
        rows = np.arange(60)
        conferences = np.repeat([0, 1, 9, 10, 13], 12)
        raw = {
            'PvsC': sp.csr_matrix((np.ones(60), (rows, conferences)), shape=(60, 14)),
            'PvsA': sp.csr_matrix((np.ones(60), (rows, rows % 9)), shape=(60, 11)),
            'PvsL': sp.csr_matrix((np.ones(60), (rows, rows % 4)), shape=(60, 6)),
            'PvsT': sp.csr_matrix((np.ones(60), (rows, rows % 7)), shape=(60, 7)),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw').mkdir()
            savemat(root / 'raw' / 'ACM.mat', raw)
            with patch('src.acm.urlopen', side_effect=AssertionError('Cached data is offline')):
                data = ACM(root=str(root))[0]
                cached = ACM(root=str(root))[0]
            torch.testing.assert_close(data['paper'].y, torch.tensor(np.repeat([0, 1, 2, 2, 1], 12)))
            torch.testing.assert_close(data['paper'].x, torch.tensor(raw['PvsT'].toarray()).float())
            torch.testing.assert_close(cached['author'].x, data['author'].x)
            self.assertEqual(data['author'].num_nodes, 9)
            self.assertEqual(data['subject'].num_nodes, 4)
            self.assertNotIn('conference', data.node_types)
            membership = torch.stack([data['paper'][f'{split}_mask']
                                      for split in ['train', 'val', 'test']]).sum(dim=0)
            self.assertTrue(torch.all(membership == 1))
            edges = data['author', 'to', 'paper'].edge_index
            for author in range(9):
                papers = edges[1, edges[0] == author]
                torch.testing.assert_close(data['author'].x[author], data['paper'].x[papers].mean(0))


class DBLPDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = DBLP.__new__(DBLP)
        self.dataset.root = self.temporary.name
        self.raw_dir = Path(self.dataset.raw_dir)
        self.raw_dir.mkdir()

    def test_nested_cached_original_archive_loads_offline(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as archive:
            for name in self.dataset.raw_file_names:
                archive.writestr('DBLP_processed/' + name, name.encode())
        (self.raw_dir / 'DBLP_processed.zip').write_bytes(buffer.getvalue())
        with patch('src.dblp.urlopen', side_effect=AssertionError('Use cached archive')):
            self.dataset.download()
        for name in self.dataset.raw_file_names:
            self.assertEqual((self.raw_dir / name).read_bytes(), name.encode())

    def test_upstream_error_page_uses_raw_fallback_without_poisoning_cache(self):
        with patch('src.dblp.urlopen', return_value=io.BytesIO(b'<html>Temporarily disabled</html>')), \
                patch.object(self.dataset, '_build_from_source') as fallback, \
                contextlib.redirect_stdout(io.StringIO()):
            self.dataset.download()
        fallback.assert_called_once_with()
        self.assertEqual(list(self.raw_dir.iterdir()), [])

    def test_unavailable_sources_explain_manual_recovery(self):
        with patch('src.dblp.urlopen', side_effect=OSError('offline')), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, 'Both DBLP sources') as error:
            self.dataset.download()
        self.assertIn(str(self.raw_dir), str(error.exception))


if __name__ == '__main__':
    unittest.main()
