"""SPCA encoders and ablations, independent of experiment/data loading code."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class SimpleMetapathConv(MessagePassing):
    def __init__(self, out_channels, edge_types):
        super().__init__(aggr='add', node_dim=0)
        self.edge_types = edge_types
        self.out_channels = out_channels

        # Attention parameters per edge type (optional, here simple scalar weights)
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        for et in edge_types:
            et_key = '__'.join(et)
            self.lin_src[et_key] = nn.Parameter(torch.randn(1, out_channels))
            self.lin_dst[et_key] = nn.Parameter(torch.randn(1, out_channels))

    def forward(self, x_dict, edge_index_dict):
        # Initialize output dict with zeros same shape as input features
        out_dict = {k: torch.zeros_like(v) for k, v in x_dict.items()}

        for et in self.edge_types:
            if et not in edge_index_dict:
                continue
            edge_index = edge_index_dict[et]
            et_key = '__'.join(et)

            x_src = x_dict[et[0]]  # Already projected features
            x_dst = x_dict[et[2]]

            # propagate messages on this edge type
            out = self.propagate(edge_index, x=(x_src, x_dst))

            # aggregate to destination node type
            out_dict[et[2]] += out

        return out_dict

    def message(self, x_j, x_i):
        # Simple message: pass source node features as is
        return x_j


class SubpathHANConv(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        metapaths,  # List of metapaths, e.g. [('movie', 'actor', 'movie'), ...]
        subpaths_dict,  # Dict: metapath -> list of subpaths (each subpath is a list of node types)
        aggregator='concat-mlp',  # or 'mean'
        dropout=0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.metadata = metadata
        self.metapaths = metapaths
        self.subpaths_dict = subpaths_dict
        self.aggregator = aggregator
        self.dropout = dropout

        # 1. Global type embeddings (for subpaths)
        self.type_emb = nn.ParameterDict()
        for node_type in metadata[0]:
            self.type_emb[node_type] = nn.Parameter(torch.randn(out_channels))

        # 2. Node feature projection - project all node types once here
        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = nn.Linear(in_channels, out_channels)

        # 3. MLPs for subpath aggregation (if concat-mlp)
        if aggregator == 'concat-mlp':
            self.subpath_mlps = nn.ModuleDict()
            for metapath in metapaths:
                for subpath in subpaths_dict[metapath]:
                    key = '_'.join(subpath)
                    self.subpath_mlps[key] = nn.Sequential(
                        nn.Linear(len(subpath) * out_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels)
                    )

        # 4. One SimpleMetapathConv per metapath (expects projected features)
        self.metapath_convs = nn.ModuleDict()
        for metapath in metapaths:
            conv_name = '_'.join(metapath)
            edge_types = self._edge_types_for_metapath(metapath)
            self.metapath_convs[conv_name] = SimpleMetapathConv(
                out_channels=out_channels,
                edge_types=edge_types
            )

        # 5. Final linear to aggregate metapath embeddings
        self.final_lin = nn.Linear(len(metapaths) * out_channels, out_channels)

    def _edge_types_for_metapath(self, metapath):
        # Extract edge types (triplets) from metapath node types:
        edge_types = []
        for i in range(len(metapath) - 1):
            edge_types.append((metapath[i], 'to', metapath[i + 1]))
        return edge_types

    def forward(self, x_dict, edge_index_dict):
        # Project input node features once
        x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}

        metapath_embs = []
        for metapath in self.metapaths:
            conv_name = '_'.join(metapath)
            edge_types = self._edge_types_for_metapath(metapath)

            # Collect only edges for this metapath
            metapath_edge_index_dict = {
                et: edge_index_dict[et] for et in edge_types if et in edge_index_dict
            }

            # Run message passing conv on projected features
            han_out = self.metapath_convs[conv_name](x_proj, metapath_edge_index_dict)

            # Target node type is last in metapath
            target_type = metapath[-1]
            within_emb = han_out[target_type]

            # Compute subpath embeddings
            sub_embs = []
            for subpath in self.subpaths_dict[metapath]:
                type_embs = [self.type_emb[t] for t in subpath]
                if self.aggregator == 'concat-mlp':
                    key = '_'.join(subpath)
                    emb = torch.cat(type_embs, dim=-1)
                    emb = self.subpath_mlps[key](emb)
                elif self.aggregator == 'mean':
                    emb = torch.stack(type_embs).mean(dim=0)
                else:
                    raise NotImplementedError

                emb = emb.unsqueeze(0).expand(within_emb.size(0), -1)  # batch expand
                sub_embs.append(emb)

            subpath_emb = torch.stack(sub_embs, dim=0).mean(dim=0)  # mean over subpaths

            combined = within_emb + subpath_emb  # combine embeddings
            metapath_embs.append(combined)

        # Concatenate all metapath embeddings
        out = torch.cat(metapath_embs, dim=-1)
        out = self.final_lin(out)
        return out


class SubpathHANConv2(nn.Module):
    """Metapath encoder with optional subpath context or vector/scalar biases."""

    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        metapaths,
        subpaths_dict,
        heads=1,
        dropout=0.0,
        aggregator='concat-mlp',
        context_mode='subpath',
    ):
        super().__init__()
        if context_mode not in {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'}:
            raise ValueError('context_mode must be subpath, no-context, metapath-bias, or metapath-scalar-bias')
        self.out_channels = out_channels
        self.heads = heads
        self.metadata = metadata
        self.metapaths = metapaths
        self.subpaths_dict = subpaths_dict
        self.dropout = dropout
        self.aggregator = aggregator
        self.context_mode = context_mode
        if aggregator not in {'concat-mlp', 'mean', 'lstm', 'transformer'}:
            raise ValueError('aggregator must be concat-mlp, mean, lstm, or transformer')

        if context_mode == 'subpath':
            self.type_emb = nn.ParameterDict()
            for node_type in metadata[0]:
                self.type_emb[node_type] = nn.Parameter(torch.randn(out_channels))
        elif context_mode in {'metapath-bias', 'metapath-scalar-bias'}:
            # Each metapath learns its own context, shared across its nodes.
            # A scalar is also broadcast across all embedding dimensions.
            # Start with no context offset and learn it through the same loss.
            bias_channels = 1 if context_mode == 'metapath-scalar-bias' else out_channels
            self.metapath_bias = nn.ParameterDict({
                '_'.join(metapath): nn.Parameter(torch.zeros(bias_channels))
                for metapath in metapaths
            })

        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        if context_mode == 'subpath':
            self.subpath_mlps = nn.ModuleDict()
            if aggregator == 'concat-mlp':
                for metapath in metapaths:
                    for subpath in subpaths_dict[metapath]:
                        key = '_'.join(subpath)
                        self.subpath_mlps[key] = nn.Sequential(
                            nn.Linear(len(subpath) * out_channels, out_channels),
                            nn.ReLU(),
                            nn.Linear(out_channels, out_channels),
                            nn.LayerNorm(out_channels)
                        )

        if context_mode == 'subpath' and aggregator in {'lstm', 'transformer'}:
            # Match the MLP policy: one encoder per distinct type sequence.
            self.subpath_sequence_encoders = nn.ModuleDict()
            if aggregator == 'transformer':
                self.subpath_positions = nn.ParameterDict()
                sequence_heads = next(h for h in (4, 2, 1) if out_channels % h == 0)
            for metapath in metapaths:
                for subpath in subpaths_dict[metapath]:
                    key = '_'.join(subpath)
                    if key in self.subpath_sequence_encoders:
                        continue
                    if aggregator == 'lstm':
                        self.subpath_sequence_encoders[key] = nn.LSTM(
                            out_channels, out_channels, num_layers=1, batch_first=True,
                        )
                    else:
                        self.subpath_positions[key] = nn.Parameter(
                            torch.empty(len(subpath), out_channels))
                        nn.init.normal_(self.subpath_positions[key], std=0.02)
                        self.subpath_sequence_encoders[key] = nn.TransformerEncoderLayer(
                            d_model=out_channels, nhead=sequence_heads,
                            dim_feedforward=2 * out_channels, dropout=dropout,
                            activation='gelu', batch_first=True,
                        )

        self.final_lin = Linear(len(metapaths) * out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def _context_embedding(self, metapath):
        if self.context_mode in {'metapath-bias', 'metapath-scalar-bias'}:
            return self.metapath_bias['_'.join(metapath)]

        sub_embs = []
        for subpath in self.subpaths_dict[metapath]:
            type_embs = [self.type_emb[t] for t in subpath]
            if self.aggregator == 'concat-mlp':
                emb = self.subpath_mlps['_'.join(subpath)](torch.cat(type_embs, dim=-1))
            elif self.aggregator == 'mean':
                emb = torch.stack(type_embs).mean(dim=0)
            elif self.aggregator == 'lstm':
                encoder = self.subpath_sequence_encoders['_'.join(subpath)]
                _, (hidden, _) = encoder(torch.stack(type_embs).unsqueeze(0))
                emb = hidden[-1, 0]
            elif self.aggregator == 'transformer':
                key = '_'.join(subpath)
                sequence = torch.stack(type_embs) + self.subpath_positions[key]
                emb = self.subpath_sequence_encoders[key](sequence.unsqueeze(0))[0].mean(dim=0)
            else:
                raise NotImplementedError(f'Unknown subpath aggregator: {self.aggregator}')
            sub_embs.append(emb)
        return torch.stack(sub_embs, dim=0).mean(dim=0)

    def _message_pass(self, x_src, x_dst, edge_index):
        # Simple attention-like weighted aggregation
        alpha = (x_src[edge_index[0]] * x_dst[edge_index[1]]).sum(dim=-1, keepdim=True)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[1], num_nodes=x_dst.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h = x_src[edge_index[0]] * alpha
        out = torch.zeros_like(x_dst)
        out.index_add_(0, edge_index[1], h)
        return out

    def forward(self, x_dict, edge_index_dict):
        x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}

        metapath_embs = []
        for metapath in self.metapaths:
            assert len(metapath) % 2 == 1, "Metapath must be odd-length"
            num_hops = len(metapath) // 2
            src_type = metapath[0]
            dst_type = metapath[-1]

            h = x_proj[src_type]
            for i in range(num_hops):
                edge1 = (metapath[i * 2], 'to', metapath[i * 2 + 1])
                edge2 = (metapath[i * 2 + 1], 'to', metapath[i * 2 + 2])
                h = self._message_pass(h, x_proj[metapath[i * 2 + 1]], edge_index_dict[edge1])
                h = self._message_pass(h, x_proj[metapath[i * 2 + 2]], edge_index_dict[edge2])
            h = self.norm(h)
            within_emb = h

            if self.context_mode != 'no-context':
                # Broadcast the chosen context to every node on this metapath.
                within_emb = within_emb + self._context_embedding(metapath)
            metapath_embs.append(within_emb)

        out = torch.cat(metapath_embs, dim=-1)
        out = self.final_lin(out)
        return out


class RegularizedSubpathHANConv(SubpathHANConv2):
    """Opt-in encoder with head-wise attention, feature residuals and dropout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.heads < 1 or self.out_channels % self.heads:
            raise ValueError('out_channels must be divisible by a positive number of heads')
        self.head_channels = self.out_channels // self.heads
        self.hop_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(self.out_channels) for node_type in self.metadata[0]
        })
        self.output_norm = nn.LayerNorm(self.out_channels)

    def _message_pass(self, x_src, x_dst, edge_index):
        source = x_src[edge_index[0]].reshape(-1, self.heads, self.head_channels)
        target = x_dst[edge_index[1]].reshape(-1, self.heads, self.head_channels)
        scores = (source * target).sum(dim=-1) / math.sqrt(self.head_channels)
        attention = softmax(F.leaky_relu(scores), edge_index[1], num_nodes=x_dst.size(0))
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        messages = (source * attention.unsqueeze(-1)).reshape(-1, self.out_channels)
        output = torch.zeros_like(x_dst)
        output.index_add_(0, edge_index[1], messages)
        return output

    def forward(self, x_dict, edge_index_dict):
        projected = {
            key: F.dropout(self.proj[key](F.normalize(value, p=2, dim=-1)),
                           p=self.dropout, training=self.training)
            for key, value in x_dict.items()
        }
        embeddings = []
        for path in self.metapaths:
            h = projected[path[0]]
            for source_type, target_type in zip(path, path[1:]):
                messages = self._message_pass(
                    h, projected[target_type], edge_index_dict[source_type, 'to', target_type],
                )
                h = F.elu(self.hop_norm[target_type](messages + projected[target_type]))
            within_emb = self.norm(h)
            if self.context_mode != 'no-context':
                within_emb = within_emb + self._context_embedding(path)
            embeddings.append(within_emb)
        target_type = self.metapaths[0][-1]
        hidden = self.final_lin(torch.cat(embeddings, dim=-1)) + projected[target_type]
        return F.dropout(F.gelu(self.output_norm(hidden)), p=self.dropout, training=self.training)


class SubpathHANConv3(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        metapaths,
        subpaths_dict,
        heads=1,
        dropout=0.0,
        aggregator='concat-mlp',
    ):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.metadata = metadata
        self.metapaths = metapaths
        self.subpaths_dict = subpaths_dict
        self.dropout = dropout
        self.aggregator = aggregator

        self.type_emb = nn.ParameterDict()
        for node_type in metadata[0]:
            self.type_emb[node_type] = nn.Parameter(torch.randn(out_channels))

        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.subpath_mlps = nn.ModuleDict()
        if aggregator == 'concat-mlp':
            for metapath in metapaths:
                for subpath in subpaths_dict[metapath]:
                    key = '_'.join(subpath)
                    self.subpath_mlps[key] = nn.Sequential(
                        nn.Linear(len(subpath) * out_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels),
                        nn.LayerNorm(out_channels)
                    )

        self.norm = nn.LayerNorm(out_channels)

        # HAN-style semantic attention
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.randn(out_channels))

    def _message_pass(self, x_src, x_dst, edge_index):
        # Basic attention-like aggregation
        alpha = (x_src[edge_index[0]] * x_dst[edge_index[1]]).sum(dim=-1, keepdim=True)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[1], num_nodes=x_dst.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h = x_src[edge_index[0]] * alpha
        out = torch.zeros_like(x_dst)
        out.index_add_(0, edge_index[1], h)
        return out

    def forward(self, x_dict, edge_index_dict):
        x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}

        metapath_embs = []
        for metapath in self.metapaths:
            assert len(metapath) % 2 == 1, "Metapath must be odd-length"
            num_hops = len(metapath) // 2
            src_type = metapath[0]
            dst_type = metapath[-1]

            h = x_proj[src_type]
            for i in range(num_hops):
                edge1 = (metapath[i * 2], 'to', metapath[i * 2 + 1])
                edge2 = (metapath[i * 2 + 1], 'to', metapath[i * 2 + 2])
                h = self._message_pass(h, x_proj[metapath[i * 2 + 1]], edge_index_dict[edge1])
                h = self._message_pass(h, x_proj[metapath[i * 2 + 2]], edge_index_dict[edge2])
            h = self.norm(h)
            within_emb = h  # node embeddings from this metapath

            # Subpath Embeddings
            sub_embs = []
            for subpath in self.subpaths_dict[metapath]:
                type_embs = [self.type_emb[t] for t in subpath]
                if self.aggregator == 'concat-mlp':
                    key = '_'.join(subpath)
                    emb = torch.cat(type_embs, dim=-1)
                    emb = self.subpath_mlps[key](emb)
                elif self.aggregator == 'mean':
                    emb = torch.stack(type_embs).mean(dim=0)
                emb = emb.unsqueeze(0).expand(within_emb.size(0), -1)
                sub_embs.append(emb)

            subpath_emb = torch.stack(sub_embs, dim=0).mean(dim=0)
            final_emb = within_emb + subpath_emb  # [num_nodes, out_channels]
            metapath_embs.append(final_emb)

        # === HAN-style semantic attention over metapath embeddings ===
        H = torch.stack(metapath_embs, dim=0).transpose(0, 1)  # [num_nodes, num_metapaths, out_channels]
        Wh = torch.tanh(self.k_lin(H))                          # [num_nodes, num_metapaths, out_channels]
        attn_logits = (Wh * self.q).sum(dim=-1)                 # [num_nodes, num_metapaths]
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [num_nodes, num_metapaths, 1]
        out = (H * attn_weights).sum(dim=1)                     # [num_nodes, out_channels]
        return out


class StackedSubpathHAN(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        metapaths,
        subpaths_dict,
        num_layers=2,
        heads=1,
        dropout=0.0,
        aggregator='concat-mlp',
        context_mode='subpath',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.metadata = metadata
        self.metapaths = metapaths
        self.subpaths_dict = subpaths_dict

        # Assume all metapaths have the same target node type
        self.target_type = metapaths[0][-1]

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = (
                in_channels_dict if i == 0
                else {k: (out_channels if k == self.target_type else in_channels_dict[k]) for k in in_channels_dict}
            )
            self.layers.append(
                SubpathHANConv2(
                    in_channels_dict=layer_in_channels,
                    out_channels=out_channels,
                    metadata=metadata,
                    metapaths=metapaths,
                    subpaths_dict=subpaths_dict,
                    heads=heads,
                    dropout=dropout,
                    aggregator=aggregator,
                    context_mode=context_mode,
                )
            )

    def forward(self, x_dict, edge_index_dict):
        h_dict = x_dict.copy()
        for layer in self.layers:
            # Only pass the current target node features to the next layer
            out = layer(h_dict, edge_index_dict)
            # out: [num_target_nodes, out_channels]
            # Update only the target node type in h_dict
            h_dict = h_dict.copy()
            h_dict[self.target_type] = out
        return out  # Only the target node embeddings


class SubpathClassifier(nn.Module):
    """Use the subpath encoder with a head for the target node classes."""

    def __init__(self, encoder, hidden_channels, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        return self.classifier(self.encoder(x_dict, edge_index_dict))

