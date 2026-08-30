import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import IMDB
from torch_geometric.nn import Linear, HANConv
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing

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

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax


class SubpathHANConv2(nn.Module):
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

        self.final_lin = Linear(len(metapaths) * out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

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

            metapath_embs.append(within_emb + subpath_emb)

        out = torch.cat(metapath_embs, dim=-1)
        out = self.final_lin(out)
        return out

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax


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

import torch
import torch.nn as nn
from torch_geometric.nn import Linear

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
        aggregator='concat-mlp'
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
                    aggregator=aggregator
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

def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['movie'].train_mask
    loss = criterion(out[mask], data['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, device):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=-1)

    results = {}
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['movie'][split]
        y_true = data['movie'].y[mask].cpu()
        y_pred = preds[mask].cpu()

        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        results[split] = {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return results


def run_imdb_example():
    global dataset  # So classifier knows num_classes
    dataset = IMDB(root='/data/hgb/imdb/')
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # data['movie'].train_mask[: 492] = True # 20%
    # data['movie'].train_mask[: 970] = True # 30%
    data['movie'].train_mask[: 2380] = True # 60%

    print('data: ', data['movie'].train_mask.sum())
    
    metapaths = [
        ('movie', 'actor', 'movie'),
        ('movie', 'director', 'movie'),
        # ('movie', 'actor', 'movie', 'director', 'movie'),
        # ('movie', 'director', 'movie', 'actor', 'movie')
    ]

    subpaths_dict = {
        ('movie', 'actor', 'movie'): [
            ['actor'],
            #  ['movie', 'actor'],
            #  ['movie', 'actor', 'movie'],
        ],
        ('movie', 'director', 'movie'): [
            ['director'],
            #  ['movie', 'director'],
            #  ['movie', 'director', 'movie'],
        ],
        # ('movie', 'actor', 'movie', 'director', 'movie'): [
        #     # ['actor', 'director'],
        #     ['actor', 'movie', 'director'],
        # ],
        # ('movie', 'director', 'movie', 'actor', 'movie'): [
        #     # ['director', 'actor'],
        #     ['director', 'movie', 'actor'],
        # ],
    }

    # metapaths = [
    #     ('movie', 'actor', 'movie'),
    #     ('movie', 'director', 'movie'),
    # ]

    # subpaths_dict = {
    #     ('movie', 'actor', 'movie'): [
    #         ['actor'],
    #         # ['movie', 'actor', 'movie'],
    #     ],
    #     ('movie', 'director', 'movie'): [
    #         ['director'],
    #         # ['movie', 'director', 'movie'],
    #     ],
    # }

    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}

    # model = StackedSubpathHAN(
    #     in_channels_dict=in_channels_dict,
    #     out_channels=128,
    #     metadata=data.metadata(),
    #     metapaths=metapaths,
    #     num_layers=2,
    #     subpaths_dict=subpaths_dict,
    #     heads=8,
    #     aggregator='concat-mlp',
    # ).to(device)
    model = SubpathHANConv2(
        in_channels_dict=in_channels_dict,
        out_channels=128,
        metadata=data.metadata(),
        metapaths=metapaths,
        subpaths_dict=subpaths_dict,
        heads=8,
        aggregator='concat-mlp',
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0

    for epoch in range(1, 250):
        loss = train(model, data, optimizer, criterion, device)
        results = test(model, data, device)
        val_loss = criterion(model(data.x_dict, data.edge_index_dict)[data['movie'].val_mask], data['movie'].y[data['movie'].val_mask]).item()

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {results['train_mask']['acc']:.4f} | Val Acc: {results['val_mask']['acc']:.4f} | Test Acc: {results['test_mask']['acc']:.4f} | "
              f"Test F1-micro: {results['test_mask']['f1_micro']:.4f} | Test F1-macro: {results['test_mask']['f1_macro']:.4f}")


        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test
    # model.load_state_dict(torch.load('best_model.pth'))
    results = test(model, data, device)
    print("Final Test Results:", results['test_mask'])

if __name__ == '__main__':
    run_imdb_example()
