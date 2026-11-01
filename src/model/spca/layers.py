"""SPCA equations (1)--(7), implemented as independently reusable PyG layers."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import softmax
from src.graph import compose_metapaths


class DotProductConv(MessagePassing):
    """Parameter-free endpoint attention, Eqs. (1)--(2).

    No intermediate feature updates, score scaling, learned attention vectors,
    or implicit self-loops. Destination blocks bound message memory exactly.
    """
    def __init__(self, dropout=0.5, negative_slope=0.01, node_chunk_size=256):
        super().__init__(aggr='add', node_dim=0)
        if not 0 <= dropout < 1 or node_chunk_size < 1:
            raise ValueError('Require dropout in [0, 1) and positive chunk size')
        self.dropout, self.negative_slope = dropout, negative_slope
        self.node_chunk_size = node_chunk_size

    def forward(self, x, edge_index, size=None):
        source, destination = (x, x) if isinstance(x, torch.Tensor) else x
        size = size or (source.size(0), destination.size(0))
        if edge_index.size(1) * source.size(1) < 2_000_000:
            return self.propagate(edge_index, x=(source, destination), size=size)
        order = edge_index[1].argsort(stable=True)
        edges = edge_index[:, order]
        boundaries = torch.searchsorted(edges[1], torch.arange(
            0, size[1] + self.node_chunk_size, self.node_chunk_size, device=edges.device))
        output = []
        for block, start in enumerate(range(0, size[1], self.node_chunk_size)):
            stop = min(start + self.node_chunk_size, size[1])
            selected = edges[:, boundaries[block]:boundaries[block + 1]].clone()
            selected[1] -= start
            def aggregate(a, b, e):
                return self.propagate(e, x=(a, b), size=(a.size(0), b.size(0)))
            if self.training and torch.is_grad_enabled():
                output.append(checkpoint(aggregate, source, destination[start:stop], selected, use_reentrant=False))
            else:
                output.append(aggregate(source, destination[start:stop], selected))
        return torch.cat(output)

    def message(self, x_j, x_i, index, ptr, size_i):
        scores = F.leaky_relu((x_j * x_i).sum(-1), negative_slope=self.negative_slope)
        attention = softmax(scores, index, ptr, size_i)
        return x_j * F.dropout(attention, self.dropout, self.training).unsqueeze(-1)


class SubpathContext(nn.Module):
    """Globally shared type vectors and one encoder per distinct ordered sequence."""
    def __init__(self, node_types, sequences, channels, aggregator='concat-mlp', dropout=0.5):
        super().__init__()
        if aggregator not in {'concat-mlp', 'mean', 'lstm', 'transformer'}:
            raise ValueError(f'Unknown context encoder: {aggregator}')
        self.aggregator = aggregator
        self.type_indices = {name: i for i, name in enumerate(node_types)}
        self.type_embeddings = nn.Embedding(len(node_types), channels)
        self.sequences = tuple(dict.fromkeys(tuple(s) for s in sequences))
        if any(not s for s in self.sequences):
            raise ValueError('Context requires nonempty subpaths')
        self.sequence_indices = {s: i for i, s in enumerate(self.sequences)}
        self.encoders = nn.ModuleList()
        self.positions = nn.ParameterList()
        for sequence in self.sequences:
            if aggregator == 'concat-mlp':
                encoder = nn.Sequential(nn.Linear(len(sequence) * channels, channels), nn.ReLU(),
                                        nn.Linear(channels, channels), nn.LayerNorm(channels))
            elif aggregator == 'mean':
                encoder = nn.Identity()
            elif aggregator == 'lstm':
                encoder = nn.LSTM(channels, channels, batch_first=True)
            else:
                heads = next(h for h in (8, 4, 2, 1) if channels % h == 0)
                encoder = nn.TransformerEncoderLayer(channels, heads, 2 * channels,
                    dropout=dropout, activation='gelu', batch_first=True)
                self.positions.append(nn.Parameter(torch.empty(len(sequence), channels)))
            self.encoders.append(encoder)
        self.reset_parameters()

    def reset_parameters(self):
        self.type_embeddings.reset_parameters()
        for encoder in self.encoders:
            if isinstance(encoder, nn.LSTM):
                encoder.reset_parameters()
            else:
                for module in encoder.modules():
                    if isinstance(module, (nn.Linear, nn.LayerNorm)):
                        module.reset_parameters()
                    elif isinstance(module, nn.MultiheadAttention):
                        module._reset_parameters()
        for position in self.positions:
            nn.init.normal_(position, std=0.02)

    def forward(self, sequence):
        sequence = tuple(sequence)
        index = self.sequence_indices[sequence]
        ids = torch.tensor([self.type_indices[t] for t in sequence], device=self.type_embeddings.weight.device)
        values = self.type_embeddings(ids)
        if self.aggregator == 'concat-mlp':
            return self.encoders[index](values.flatten())
        if self.aggregator == 'mean':
            return values.mean(0)
        if self.aggregator == 'lstm':
            return self.encoders[index](values.unsqueeze(0))[1][0][-1, 0]
        return self.encoders[index]((values + self.positions[index]).unsqueeze(0))[0].mean(0)


class SPCALayer(nn.Module):
    """One SPCA layer; returns embeddings of the selected receiving node type.

    Supply precomputed metapath_edge_index_dict keyed by type-sequence tuples to
    avoid recomputing connectivity. Raw PyG edge_index_dict is also accepted.
    Explicit subpaths_dict supports selected positions and multiple contexts.
    """
    def __init__(self, in_channels_dict, out_channels, metadata, metapaths,
                 subpaths_dict=None, *, dropout=0.5, aggregator='concat-mlp',
                 context_mode='subpath', context_positions='intermediate',
                 normalization=True, negative_slope=0.01, node_chunk_size=256, output_channels=None):
        super().__init__()
        self.metapaths = tuple(map(tuple, metapaths))
        if not self.metapaths or any(len(p) < 3 for p in self.metapaths):
            raise ValueError('SPCA requires metapaths with nonempty internal type sequences')
        if len({p[-1] for p in self.metapaths}) != 1:
            raise ValueError('One SPCALayer encodes one receiving node type')
        if context_positions not in {'intermediate', 'full'}:
            raise ValueError('context_positions must be intermediate or full')
        if context_mode not in {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'}:
            raise ValueError('Unknown context mode')
        self.context_mode = context_mode
        self.target_type = self.metapaths[0][-1]
        self.subpaths = {p: tuple(tuple(s) for s in subpaths_dict[p]) if subpaths_dict is not None else
                        (p[1:-1] if context_positions == 'intermediate' else p,)
                        for p in self.metapaths}
        if any(not seqs or any(not s for s in seqs) for seqs in self.subpaths.values()):
            raise ValueError('Each metapath needs a nonempty context sequence')
        self.projections = nn.ModuleDict({t: Linear(c, out_channels) for t, c in in_channels_dict.items()})
        self.neighbor_encoder = DotProductConv(dropout, negative_slope, node_chunk_size)
        self.structural_norm = nn.LayerNorm(out_channels) if normalization else nn.Identity()
        self.fusion_norm = nn.LayerNorm(out_channels) if normalization else nn.Identity()
        self.context_encoder = (SubpathContext(metadata[0],
            [s for seqs in self.subpaths.values() for s in seqs], out_channels, aggregator, dropout)
            if context_mode == 'subpath' else None)
        if context_mode in {'metapath-bias', 'metapath-scalar-bias'}:
            self.context_bias = nn.Parameter(torch.zeros(len(self.metapaths),
                1 if context_mode == 'metapath-scalar-bias' else out_channels))
        self.fusion = Linear(len(self.metapaths) * out_channels, output_channels or out_channels)

    def reset_parameters(self):
        for projection in self.projections.values():
            projection.reset_parameters()
        for norm in (self.structural_norm, self.fusion_norm):
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if self.context_encoder is not None:
            self.context_encoder.reset_parameters()
        if hasattr(self, 'context_bias'):
            nn.init.zeros_(self.context_bias)
        self.fusion.reset_parameters()

    def forward(self, x_dict, edge_index_dict=None, *, metapath_edge_index_dict=None):
        graphs = metapath_edge_index_dict
        if graphs is None:
            graphs = compose_metapaths(x_dict, edge_index_dict, self.metapaths)
        projected = {t: layer(x_dict[t]) for t, layer in self.projections.items()}
        outputs = []
        contexts = ({s: self.context_encoder(s) for s in self.context_encoder.sequences}
                    if self.context_encoder is not None else {})
        for i, path in enumerate(self.metapaths):
            structural = self.structural_norm(self.neighbor_encoder(
                (projected[path[0]], projected[path[-1]]), graphs[path]))
            if self.context_mode == 'subpath':
                structural = structural + torch.stack([contexts[s] for s in self.subpaths[path]]).mean(0)
            elif self.context_mode != 'no-context':
                structural = structural + self.context_bias[i]
            outputs.append(self.fusion_norm(structural))
        return self.fusion(torch.cat(outputs, dim=-1))
