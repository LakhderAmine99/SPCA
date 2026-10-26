import math
import unittest

import torch

from src.model.simplehgn import SimpleHGN, SimpleHGNConv, pack_typed_graph, project_metapath_graph


class SimpleHGNTests(unittest.TestCase):
    @staticmethod
    def scalar_attention_layer(**kwargs):
        layer = SimpleHGNConv(
            1, 1, num_relations=2, heads=1, edge_channels=1,
            dropout=0.0, residual=False, **kwargs,
        )
        with torch.no_grad():
            layer.node_projection.weight.fill_(1)
            layer.edge_projection.weight.fill_(1)
            layer.edge_embedding.weight.zero_()
            layer.attn_src.zero_()
            layer.attn_dst.zero_()
            layer.attn_edge.fill_(1)
        return layer

    @staticmethod
    def graph_features():
        features = {'movie': torch.randn(3, 4), 'actor': torch.randn(2, 3),
                    'director': torch.randn(2, 5)}
        edges = {}
        for node_type in ['actor', 'director']:
            edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
            edges['movie', 'to', node_type] = edge_index
            edges[node_type, 'to', 'movie'] = edge_index.flip(0)
        return features, edges

    def test_attention_is_normalized_jointly_across_edge_types(self):
        layer = self.scalar_attention_layer()
        output, attention = layer(
            torch.tensor([[1.], [3.], [0.]]), torch.tensor([[0, 1], [2, 2]]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(attention[:, 0], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(output[:, 0, 0], torch.tensor([0., 0., 2.]))

    def test_edge_embeddings_change_attention_and_receive_gradients(self):
        layer = self.scalar_attention_layer()
        with torch.no_grad():
            layer.edge_embedding.weight[1, 0] = math.log(3)
        output, attention = layer(
            torch.tensor([[1.], [3.], [0.]]), torch.tensor([[0, 1], [2, 2]]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(attention[:, 0], torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(output[2, 0, 0], torch.tensor(2.5))
        output[2].sum().backward()
        self.assertGreater(float(layer.edge_embedding.weight.grad.abs().sum()), 0)

    def test_attention_residual_matches_weighted_previous_attention(self):
        layer = self.scalar_attention_layer(attention_residual=0.25)
        previous = torch.tensor([[1.], [0.]], requires_grad=True)
        output, attention = layer(
            torch.tensor([[1.], [3.], [0.]]), torch.tensor([[0, 1], [2, 2]]),
            torch.tensor([0, 1]), previous,
        )
        torch.testing.assert_close(attention[:, 0], torch.tensor([0.625, 0.375]))
        torch.testing.assert_close(output[2, 0, 0], torch.tensor(1.75))
        self.assertFalse(attention.requires_grad)
        output.sum().backward()
        self.assertIsNone(previous.grad)

    def test_node_residual_preserves_features_without_incoming_edges(self):
        layer = SimpleHGNConv(1, 1, 1, heads=1, dropout=0.0, residual=True)
        features = torch.tensor([[1.], [3.]])
        output, _ = layer(features, torch.empty(2, 0, dtype=torch.long),
                          torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(output[:, 0], features)

    def test_chunked_messages_preserve_outputs_and_all_gradients(self):
        normal = SimpleHGNConv(3, 2, 2, heads=2, edge_channels=2, dropout=0.0)
        chunked = SimpleHGNConv(3, 2, 2, heads=2, edge_channels=2, dropout=0.0)
        chunked.load_state_dict(normal.state_dict())
        chunked.message_chunk_size = 2
        features = torch.randn(4, 3, requires_grad=True)
        other_features = features.detach().clone().requires_grad_(True)
        edges = torch.tensor([[0, 1, 0, 2, 2, 3, 3], [1, 0, 1, 1, 3, 2, 1]])
        types = torch.tensor([0, 0, 1, 1, 0, 0, 1])
        expected, expected_attention = normal(features, edges, types)
        actual, actual_attention = chunked(other_features, edges, types)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_attention, expected_attention)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(other_features.grad, features.grad)
        for (name, parameter), (_, reference) in zip(chunked.named_parameters(), normal.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, reference.grad)

    def test_dense_messages_preserve_parallel_relation_outputs_and_gradients(self):
        normal = SimpleHGNConv(3, 2, 2, heads=2, edge_channels=2, dropout=0.0)
        dense = SimpleHGNConv(3, 2, 2, heads=2, edge_channels=2, dropout=0.0)
        dense.load_state_dict(normal.state_dict())
        normal.max_dense_elements = 0
        features = torch.randn(3, 3, requires_grad=True)
        other_features = features.detach().clone().requires_grad_(True)
        edges = torch.tensor([[0, 1, 0, 2, 2, 0, 1], [1, 0, 1, 1, 0, 2, 2]])
        types = torch.tensor([0, 0, 1, 1, 0, 1, 1])
        expected, expected_attention = normal(features, edges, types)
        actual, actual_attention = dense(other_features, edges, types)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_attention, expected_attention)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(other_features.grad, features.grad)
        for (name, parameter), (_, reference) in zip(dense.named_parameters(), normal.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, reference.grad)

    def test_dblp_metapath_graphs_have_only_the_requested_author_connections(self):
        features = {'author': torch.randn(4, 3), 'paper': torch.randn(3, 2),
                    'term': torch.randn(2, 2), 'conference': torch.randn(2, 2)}
        edges = {}
        for node_type, paper_edges in [
            ('author', torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])),
            ('term', torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])),
            ('conference', torch.tensor([[0, 1, 2], [0, 1, 0]])),
        ]:
            edges['paper', 'to', node_type] = paper_edges
            edges[node_type, 'to', 'paper'] = paper_edges.flip(0)
        paths = [
            ('author', 'paper', 'author'),
            ('author', 'paper', 'term', 'paper', 'author'),
            ('author', 'paper', 'conference', 'paper', 'author'),
        ]
        projected = project_metapath_graph(features, edges, paths, 'author')
        expected = [
            {(0, 1), (1, 0)},
            {(0, 1), (1, 0), (1, 2), (2, 1)},
            {(a, b) for a in range(3) for b in range(3) if a != b},
        ]
        self.assertEqual(len(projected), 3)
        for path, pairs in zip(paths, expected):
            actual = projected['author', '_'.join(path), 'author']
            self.assertEqual(set(map(tuple, actual.t().tolist())), pairs)

        model = SimpleHGN(features, edges, 3, target_type='author', metapaths=paths,
                          hidden_channels=4, heads=2, edge_channels=2, dropout=0.0).eval()
        self.assertEqual(model.node_types, ('author',))
        self.assertEqual(model.graph_mode, 'metapath-projected')
        self.assertEqual(model.metapaths, tuple(paths))
        self.assertEqual(model.edge_index.shape[1], sum(map(len, expected)) + 4)
        logits = model(features)
        self.assertEqual(logits.shape, (4, 3))
        changed = {key: value if key == 'author' else value + 1000 for key, value in features.items()}
        torch.testing.assert_close(model(changed), logits)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
        self.assertGreater(float(model.projections['author'].weight.grad.abs().sum()), 0)

    def test_graph_packing_preserves_relations_and_adds_one_loop_per_node(self):
        features, edges = self.graph_features()
        edges['movie', 'self', 'movie'] = torch.tensor([[0], [0]])
        node_types, offsets, packed, types = pack_typed_graph(features, edges)
        self.assertEqual(node_types, tuple(sorted(features)))
        is_loop = packed[0] == packed[1]
        self.assertEqual(int(is_loop.sum()), 7)
        torch.testing.assert_close(packed[0, is_loop].sort().values, torch.arange(7))
        self.assertTrue(torch.all(types[is_loop] == len(edges)))
        for relation_id, relation in enumerate(sorted(edges)):
            if relation[1] == 'self':
                continue
            expected = edges[relation] + torch.tensor([[offsets[relation[0]]], [offsets[relation[2]]]])
            torch.testing.assert_close(packed[:, types == relation_id], expected)

    def test_model_train_step_output_norm_and_checkpoint_round_trip(self):
        features, edges = self.graph_features()
        model = SimpleHGN(features, edges, 3, hidden_channels=4, heads=2,
                          edge_channels=4, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        before = model.projections['movie'].weight.detach().clone()
        logits = model(features, edges)
        self.assertEqual(logits.shape, (3, 3))
        torch.testing.assert_close(logits.norm(dim=-1), torch.ones(3))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.projections['movie'].weight, before))
        self.assertFalse(any(key in model.state_dict() for key in ['edge_index', 'edge_types']))
        restored = SimpleHGN(features, edges, 3, hidden_channels=4, heads=2,
                             edge_channels=4, dropout=0.0).eval()
        restored.load_state_dict(model.state_dict())
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features))
            reordered = dict(reversed(list(features.items())))
            torch.testing.assert_close(model(reordered), model(features))


if __name__ == '__main__':
    unittest.main()
