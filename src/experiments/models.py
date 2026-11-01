"""Bind all models to the same exact selected metapath graphs and task heads."""
import inspect
import torch
from torch import nn
from torch_geometric.utils import coalesce
from src.graph import compose_metapaths, enumerate_instances
from src.model.baselines.registry import get_baseline
from src.model.spca import SPCALayer
from .config import metapaths


def selected_kwargs(cls, settings):
    keys = inspect.signature(cls).parameters
    return {k: v for k, v in settings.items() if k in keys}


class PreparedGraph:
    """One preparation per graph/split; contains no held-out labels."""
    def __init__(self, data, dataset, target, paths=None):
        self.data, self.target = data, target
        self.paths = [tuple(path) for path in (metapaths(dataset, target) if paths is None else paths)]
        if not self.paths or any(len(path) < 3 or path[0] != target or path[-1] != target for path in self.paths):
            raise ValueError('Shared baseline metapaths must have internal types and start/end at the receiving type')
        if len(set(self.paths)) != len(self.paths):
            raise ValueError('Metapaths must be distinct')
        self.graphs = compose_metapaths(data.x_dict, data.edge_index_dict, self.paths)
        self.typed_graphs = {(target, f'metapath_{i}', target): self.graphs[path]
                             for i, path in enumerate(self.paths)}
        self.edges = torch.cat(list(self.graphs.values()), 1)
        self.edge_types = torch.cat([torch.full((edges.size(1),), i, dtype=torch.long, device=edges.device)
                                    for i, edges in enumerate(self.graphs.values())])
        self.union = coalesce(self.edges)
        self.cache = {}

    def positive_edges(self, top_k):
        # Upstream positive mining counts the selected normalized semantic views.
        import numpy as np
        import scipy.sparse as sp
        n = self.data[self.target].num_nodes
        scores = sp.csr_matrix((n, n), dtype=np.float32)
        for edges in self.graphs.values():
            rows, cols = edges.detach().cpu().numpy()
            matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
            matrix = sp.diags(1 / np.maximum(np.asarray(matrix.sum(1)).ravel(), 1)) @ matrix
            scores = scores + matrix
        rows, cols = [], []
        for i in range(n):
            a, b = scores.indptr[i:i + 2]
            order = np.argsort(-scores.data[a:b], kind='stable')[:top_k]
            ids = scores.indices[a:b][order]
            rows.extend([i] * len(ids))
            cols.extend(ids.tolist())
        return torch.tensor([rows, cols], dtype=torch.long, device=self.edges.device)


class BoundEncoder(nn.Module):
    def __init__(self, name, prepared, settings, *, labels=None, train_mask=None, num_classes=None, seed=0):
        super().__init__()
        self.name, self.prepared = name, prepared
        data, target, paths = prepared.data, prepared.target, prepared.paths
        x = data[target].x
        width, output = settings['hidden_channels'], settings['embedding_channels']
        options = dict(settings)
        options.update(out_channels=output, num_classes=output)
        graphs = list(prepared.graphs.values())
        self.args, self.kwargs = (), {}
        self.positive_edges = None
        if name == 'spca':
            self.model = SPCALayer({k: v.size(1) for k, v in data.x_dict.items()}, width,
                data.metadata(), paths, output_channels=output, **selected_kwargs(SPCALayer, {k: v for k, v in settings.items()
                    if k not in {'hidden_channels', 'out_channels', 'in_channels_dict', 'metadata', 'metapaths'}}))
            self.output = nn.Identity()
            self.args = (data.x_dict,)
            self.kwargs = {'metapath_edge_index_dict': prepared.graphs}
            return
        cls = get_baseline(name)
        if name in {'gcn', 'gat', 'rgcn', 'bghgnn'}:
            options.update(in_channels=x.size(1), num_relations=len(paths))
            self.args = ((x, prepared.edges, prepared.edge_types) if name in {'rgcn', 'bghgnn'} else (x, prepared.union))
        elif name == 'han':
            options.update(in_channels=x.size(1), metapath_names=[key[1] for key in prepared.typed_graphs], target_type=target)
            self.args = (x, prepared.typed_graphs)
        elif name == 'hgt':
            options.update(in_channels_dict={target: x.size(1)}, metadata=([target], list(prepared.typed_graphs)), target_type=target)
            self.args = ({target: x}, prepared.typed_graphs)
        elif name == 'simplehgn':
            options.update(x_dict={target: x}, edge_index_dict=prepared.typed_graphs, target_type=target)
            self.args = ({target: x},)
        elif name == 'magnn':
            options.update(in_channels_dict={k: v.size(1) for k, v in data.x_dict.items()}, metapaths=paths)
            key = ('instances', settings.get('max_instances'), seed)
            if key not in prepared.cache:
                prepared.cache[key] = enumerate_instances(data.x_dict, data.edge_index_dict, paths,
                                                          settings.get('max_instances'), seed)
            self.args = (data.x_dict, prepared.cache[key])
        elif name in {'sehgnn', 'hettree'}:
            from src.model.baselines.sehgnn import precompute_metapath_features
            from src.model.baselines.hettree import precompute_hettree_inputs
            hops = max(len(p) - 1 for p in paths)
            if name == 'sehgnn':
                if 'features' not in prepared.cache:
                    prepared.cache['features'] = precompute_metapath_features(data.x_dict, data.edge_index_dict,
                        target, hops, metapaths=paths)
                options.update(metapath_features=prepared.cache['features'], num_heads=settings['heads'])
            else:
                features, label_features = precompute_hettree_inputs(data.x_dict, data.edge_index_dict,
                    target, hops, labels=labels, train_mask=train_mask, num_classes=num_classes,
                    use_labels=settings.get('use_labels', True) and labels is not None,
                    metapaths=paths, include_ancestor_features=False,
                    label_batch_size=settings.get('label_batch_size', 128))
                # HetTree's label channels have class width, independent of output embedding width.
                if label_features:
                    options['num_classes'] = num_classes
                options.update(metapath_features=features, label_features=label_features)
        elif name in {'heco', 'gtc'}:
            options.update(in_channels=x.size(1), num_metapaths=len(paths))
            self.args = (x, graphs)
            key = ('positives', settings['positive_k'])
            if key not in prepared.cache:
                prepared.cache[key] = prepared.positive_edges(settings['positive_k'])
            self.positive_edges = prepared.cache[key]
            if name == 'gtc':
                from src.model.baselines.gtc.preprocessing import hop_tokens
                key = ('tokens', settings['hops'])
                if key not in prepared.cache:
                    prepared.cache[key] = hop_tokens(x, graphs, settings['hops'])
                self.args += (prepared.cache[key],)
        else:
            raise ValueError(name)
        self.model = cls(**selected_kwargs(cls, options))
        if name == 'hettree' and options['num_classes'] != output:
            # Keep the existing class and label preprocessing unchanged; adapt
            # only terminal task projections to a common embedding width.
            for branch in ('classifier', 'feature_residual', 'label_residual'):
                head = getattr(self.model, branch)
                if head is not None:
                    head[-1] = nn.Linear(head[-1].in_features, output)
        self.output = nn.Identity()

    def forward(self, detach_backbone=False):
        kwargs = dict(self.kwargs)
        if detach_backbone:
            kwargs['detach_encoder'] = True
        return self.output(self.model(*self.args, **kwargs))

    def auxiliary_loss(self):
        if self.positive_edges is None:
            return next(self.parameters()).new_zeros(())
        return self.model.contrastive_loss(*self.args, self.positive_edges)


class NodeTask(nn.Module):
    def __init__(self, encoder, channels, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self):
        return self.classifier(self.encoder())

    def auxiliary_loss(self):
        return self.encoder.auxiliary_loss()


class LinkTask(nn.Module):
    def __init__(self, source_encoder, destination_encoder):
        super().__init__()
        self.source, self.destination = source_encoder, destination_encoder

    def encode(self):
        return self.source(), self.destination()

    def forward(self, edge_label_index):
        source, destination = self.encode()
        return (source[edge_label_index[0]] * destination[edge_label_index[1]]).sum(-1)

    def auxiliary_loss(self):
        return self.source.auxiliary_loss() + self.destination.auxiliary_loss()
