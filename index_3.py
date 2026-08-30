import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv
from torch_geometric.transforms import AddMetaPaths
from sklearn.metrics import f1_score

# 1. Load and transform dataset
metapaths = [
    [('author', 'to', 'paper'), ('paper', 'to', 'author')],
    [('author', 'to', 'paper'), ('paper', 'to', 'conference'), ('conference', 'to', 'paper'), ('paper', 'to', 'author')],
]

dataset = DBLP(root='./data/dblp', transform=AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True))
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 2. Create model
class HAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super().__init__()
        self.han_conv = HANConv(in_channels, out_channels, metadata=metadata, heads=8, dropout=0.6)
        self.lin = Linear(out_channels, 4)

    def forward(self, x_dict, edge_index_dict):
        x = self.han_conv(x_dict, edge_index_dict)
        return self.lin(x['author'])

# 3. Train/Test logic
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['author'].train_mask], data['author'].y[data['author'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)

    results = {}
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        y_true = data['author'].y[mask].cpu()
        y_pred = pred[mask].cpu()
        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        results[split] = {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return results

# 4. Run training
model = HAN(in_channels=-1, out_channels=64, metadata=data.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

best_val_loss = float('inf')
patience = 30
patience_counter = 0

for epoch in range(1, 201):
    loss = train(model, data, optimizer, criterion)
    val_loss = criterion(model(data.x_dict, data.edge_index_dict)[data['author'].val_mask], data['author'].y[data['author'].val_mask]).item()
    results = test(model, data)

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {results['train_mask']['acc']:.4f} | Val Acc: {results['val_mask']['acc']:.4f} | Test Acc: {results['test_mask']['acc']:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'han_dblp_best.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# Final evaluation
model.load_state_dict(torch.load('han_dblp_best.pt'))
final_results = test(model, data)
print("Final Test Results:", final_results['test_mask'])
