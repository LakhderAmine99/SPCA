import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import AMiner
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ==================== Model ====================
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

# ==================== Training ====================
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['venue'].train_mask
    loss = criterion(out[mask], data['venue'].y[mask])
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
        mask = data['venue'][split]
        y_true = data['venue'].y[mask].cpu()
        y_pred = preds[mask].cpu()

        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        results[split] = {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return results

# ==================== Main ====================
def run_aminer_example():
    dataset = AMiner(root='./data/AMiner')
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add features if missing
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            data[node_type].x = torch.eye(data[node_type].num_nodes)

    # Setup masks for 'venue' classification
    y = data['venue'].y
    idx = torch.arange(y.size(0))
    train_idx, rest_idx = train_test_split(idx, train_size=0.6, stratify=y)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx])

    data['venue'].train_mask = torch.zeros_like(y, dtype=torch.bool)
    data['venue'].val_mask = torch.zeros_like(y, dtype=torch.bool)
    data['venue'].test_mask = torch.zeros_like(y, dtype=torch.bool)

    data['venue'].train_mask[train_idx] = True
    data['venue'].val_mask[val_idx] = True
    data['venue'].test_mask[test_idx] = True

    data = data.to(device)

    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}

    metapaths = [
        ('venue', 'published_in_rev', 'paper', 'written_by', 'author'),
        ('venue', 'published_in_rev', 'paper', 'cites', 'paper', 'published_in', 'venue'),
    ]

    subpaths_dict = {
        ('venue', 'published_in_rev', 'paper', 'written_by', 'author'): [
            ['published_in_rev', 'written_by'],
            ['paper'],
        ],
        ('venue', 'published_in_rev', 'paper', 'cites', 'paper', 'published_in', 'venue'): [
            ['cites'],
            ['published_in'],
        ],
    }

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

    for epoch in range(1, 251):
        loss = train(model, data, optimizer, criterion, device)
        results = test(model, data, device)
        val_loss = criterion(model(data.x_dict, data.edge_index_dict)[data['venue'].val_mask], data['venue'].y[data['venue'].val_mask]).item()

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {results['train_mask']['acc']:.4f} | Val Acc: {results['val_mask']['acc']:.4f} | "
              f"Test Acc: {results['test_mask']['acc']:.4f} | F1-Micro: {results['test_mask']['f1_micro']:.4f} | "
              f"F1-Macro: {results['test_mask']['f1_macro']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_aminer.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_model_aminer.pth'))
    results = test(model, data, device)
    print("Final Test Results:", results['test_mask'])


if __name__ == '__main__':
    run_aminer_example()
