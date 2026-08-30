import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear, HANConv
from torch_geometric.utils import softmax
from sklearn.metrics import f1_score

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

class SubpathHANConv(nn.Module):
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

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for mlp in self.subpath_mlps.values():
            for p in mlp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        self.final_lin.reset_parameters()
        self.norm.reset_parameters()
        for proj in self.proj.values():
            proj.reset_parameters()

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

            # sum = within_emb + subpath_emb
            # sum = self.norm(sum)

            # metapath_embs.append(sum)
            metapath_embs.append(within_emb + subpath_emb)

        out = torch.cat(metapath_embs, dim=-1)
        out = self.final_lin(out)
        return out

# impl of a han model
class HAN(nn.Module):
    def __init__(self, in_channels_dict, out_channels, metadata, heads=1, dropout=0.0):
        super().__init__()
        self.conv = HANConv(
            in_channels=in_channels_dict,
            out_channels=out_channels,
            metadata=metadata,
            heads=heads,
            dropout=dropout
        )
        self.lin = Linear(out_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv(x_dict, edge_index_dict)
        return self.lin(x)

def count_parameters(model):
    """Calculate the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = criterion(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=-1)

    results = {}
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        y_true = data['author'].y[mask].cpu()
        y_pred = preds[mask].cpu()

        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        results[split] = {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return results


def run_dblp():
    dataset = DBLP(root='./data/dblp')
    data = dataset[0]
    
    # data['author'].train_mask[:970] = True
    
    print(data['author'].train_mask.sum())
    # return

    if 'conference' not in data.x_dict or data['conference'].x is None:
        num_nodes = data['conference'].num_nodes
        data['conference'].x = torch.ones((num_nodes, 64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    metapaths = [
        ('author', 'paper', 'author'),
        ('author', 'paper', 'conference', 'paper', 'author'),
    ]

    subpaths_dict = {
        ('author', 'paper', 'author'): [
            ['paper'],
            ['author', 'paper', 'author'],
        ],
        ('author', 'paper', 'conference', 'paper', 'author'): [
            ['conference'],
            ['paper', 'conference', 'paper'],
        ],
    }

    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}

    model = SubpathHANConv(
        in_channels_dict=in_channels_dict,
        out_channels=64,
        metadata=data.metadata(),
        metapaths=metapaths,
        subpaths_dict=subpaths_dict,
        heads=4,
        aggregator='concat-mlp',
    ).to(device)
    
    # model = HAN(
    #     in_channels_dict=in_channels_dict,
    #     out_channels=64,
    #     metadata=data.metadata(),
    #     heads=4,
    # ).to(device)

    # Print total number of parameters
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model: {model.__class__.__name__}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion)
        results = test(model, data)
        val_loss = criterion(model(data.x_dict, data.edge_index_dict)[data['author'].val_mask], data['author'].y[data['author'].val_mask]).item()

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {results['train_mask']['acc']:.4f} | Val Acc: {results['val_mask']['acc']:.4f} | Test Acc: {results['test_mask']['acc']:.4f} | "
              f"Test F1-micro: {results['test_mask']['f1_micro']:.4f} | Test F1-macro: {results['test_mask']['f1_macro']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_dblp_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_dblp_model.pth'))
    results = test(model, data)
    print("Final Test Results:", results['test_mask'])

if __name__ == '__main__':
    run_dblp()