import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import HGBDataset
import torch_geometric.transforms as T
from torch_geometric.nn import HANConv
from sklearn.metrics import f1_score
from torch_geometric.transforms import ToUndirected, AddMetaPaths
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define metapaths for Freebase dataset based on the actual structure
# Target: book classification, so we need metapaths that start and end with 'book'
metapaths = [
    # book -> people -> book (books connected through people)
    [('book', 'rev_to', 'people'), ('people', 'to', 'book')],
    
    # book -> film -> book (books connected through films)
    # [('book', 'to', 'film'), ('film', 'rev_to', 'book')],
    
    # # book -> music -> book (books connected through music)
    # [('book', 'rev_in', 'music'), ('music', 'in', 'book')],
    
    # # book -> location -> book (books connected through locations)
    # [('book', 'on', 'location'), ('location', 'rev_on', 'book')],
    
    # # book -> organization -> book (books connected through organizations)
    # [('book', 'about', 'organization'), ('organization', 'rev_about', 'book')],
    
    # # book -> business -> book (books connected through business)
    # [('book', 'rev_about', 'business'), ('business', 'about', 'book')],
    
    # # Longer paths for richer semantics
    # # book -> people -> film -> book
    # [('book', 'rev_to', 'people'), ('people', 'to', 'film'), ('film', 'rev_to', 'book')],
    
    # # book -> location -> people -> book
    # [('book', 'on', 'location'), ('location', 'rev_on', 'people'), ('people', 'to', 'book')],
]

# Create transform with metapaths
transform = T.Compose([
    T.ToUndirected(),  # Make edges undirected and add reverse edges
    T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False, drop_unconnected_node_types=False)
])

# Load dataset with metapaths
print("Loading Freebase dataset...")
dataset = HGBDataset(root='../data/hgb/freebase', name='Freebase', transform=transform)
data = dataset[0]

print(f"Dataset info:")
print(f"Node types: {list(data.node_types)}")
print(f"Edge types: {list(data.edge_types)}")
print(f"Number of edge types: {len(data.edge_types)}")

# Check book node properties
print(f"\nBook node info:")
print(f"Number of books: {data['book'].num_nodes}")
if hasattr(data['book'], 'x') and data['book'].x is not None:
    print(f"Book features shape: {data['book'].x.shape}")
else:
    print("No pre-computed features found, using node degree as features")
    # Create simple features based on node degrees
    from torch_geometric.transforms import ToUndirected
    from torch_geometric.utils import degree
    
    # Calculate degree for each node type
    features = {}
    for node_type in data.node_types:
        # Get all edges involving this node type
        degrees = torch.zeros(data[node_type].num_nodes)
        for edge_type in data.edge_types:
            if edge_type[0] == node_type:
                edge_index = data[edge_type].edge_index
                deg = degree(edge_index[0], num_nodes=data[node_type].num_nodes)
                degrees += deg
            elif edge_type[2] == node_type:
                edge_index = data[edge_type].edge_index
                deg = degree(edge_index[1], num_nodes=data[node_type].num_nodes)
                degrees += deg
        
        # Use degree as feature (add small constant to avoid zero features)
        features[node_type] = (degrees + 1).float().unsqueeze(1)
    
    data.x_dict = features
    print(f"Created degree-based features for all node types")

print(f"Number of classes: {data['book'].y.max().item() + 1}")

# Create validation mask from training data (20% of train for validation)
train_idx = data['book'].train_mask.nonzero(as_tuple=True)[0]
val_split = int(0.8 * len(train_idx))
val_idx = train_idx[val_split:]

# Create validation mask
data['book'].val_mask = torch.zeros(data['book'].num_nodes, dtype=torch.bool)
data['book'].train_mask[val_idx] = False  # Remove from train
data['book'].val_mask[val_idx] = True     # Add to validation

print(f"\nData splits:")
print(f"Train: {data['book'].train_mask.sum().item()}")
print(f"Val: {data['book'].val_mask.sum().item()}")
print(f"Test: {data['book'].test_mask.sum().item()}")

# Move data to device
data = data.to(device)

# Get number of classes
num_classes = int(data['book'].y.max().item()) + 1
print(f"Number of classes: {num_classes}")

# HAN Model
class HAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=8, dropout=0.6):
        super().__init__()
        self.han_conv = HANConv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            metadata=metadata, 
            heads=heads, 
            dropout=dropout
        )
        self.lin = Linear(out_channels, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict):
        # Apply HAN convolution
        x = self.han_conv(x_dict, edge_index_dict)
        
        # Apply dropout and linear layer to book embeddings
        x_book = self.dropout(x['book'])
        return self.lin(x_book)

# Initialize model
model = HAN(
    in_channels=-1,  # Lazy initialization 
    out_channels=64, 
    metadata=data.metadata(),
    heads=8,
    dropout=0.6
).to(device)

# Print model info
print(f"\nModel initialized:")
print(f"Model: {model}")

# Training function
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['book'].train_mask], data['book'].y[data['book'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def evaluate(model, data, mask_name):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)
    mask = data['book'][mask_name]
    
    y_true = data['book'].y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    acc = (y_pred == y_true).mean()
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    loss = F.cross_entropy(out[mask], data['book'].y[mask]).item()
    
    return {
        'acc': acc, 
        'f1_micro': f1_micro, 
        'f1_macro': f1_macro
    }, loss

# Initialize optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training parameters
best_val_loss = float('inf')
patience = 50  # Reduced patience for faster convergence
patience_counter = 0
best_model_path = 'han_freebase_best.pth'

# print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")
# print("\nStarting training...")
# print("=" * 100)

# Training loop
for epoch in range(1, 201):  # Reduced epochs for faster execution
    # Train
    loss = train(model, data, optimizer, criterion)
    
    # Evaluate
    val_results, val_loss = evaluate(model, data, 'val_mask')
    test_results, _ = evaluate(model, data, 'test_mask')
    train_results, _ = evaluate(model, data, 'train_mask')
    
    # Print results every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_results['acc']:.4f} | "
              f"Val Acc: {val_results['acc']:.4f} | "
              f"Test Acc: {test_results['acc']:.4f} | "
              f"Val F1: {val_results['f1_micro']:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print("=" * 100)
print("Training completed!")

# Load best model and final evaluation
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(best_model_path))

final_train_results, _ = evaluate(model, data, 'train_mask')
final_val_results, _ = evaluate(model, data, 'val_mask')
final_test_results, _ = evaluate(model, data, 'test_mask')

print("\nFinal Results:")
print("=" * 60)
print(f"Train - Acc: {final_train_results['acc']:.4f}, "
      f"F1-Micro: {final_train_results['f1_micro']:.4f}, "
      f"F1-Macro: {final_train_results['f1_macro']:.4f}")
print(f"Val   - Acc: {final_val_results['acc']:.4f}, "
      f"F1-Micro: {final_val_results['f1_micro']:.4f}, "
      f"F1-Macro: {final_val_results['f1_macro']:.4f}")
print(f"Test  - Acc: {final_test_results['acc']:.4f}, "
      f"F1-Micro: {final_test_results['f1_micro']:.4f}, "
      f"F1-Macro: {final_test_results['f1_macro']:.4f}")

# Analyze metapaths that were created
print(f"\nMetapath Analysis:")
print(f"Original edge types: {len([et for et in data.edge_types if not et[1].startswith('metapath')])}")
print(f"Metapath edge types: {len([et for et in data.edge_types if et[1].startswith('metapath')])}")
print(f"Total edge types after metapaths: {len(data.edge_types)}")

metapath_edges = [et for et in data.edge_types if et[1].startswith('metapath')]
print(f"Created metapath edges: {len(metapath_edges)}")
for i, edge_type in enumerate(metapath_edges[:5]):  # Show first 5
    print(f"  {i+1}. {edge_type}")
if len(metapath_edges) > 5:
    print(f"  ... and {len(metapath_edges) - 5} more")

# Clean up
if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"\nCleaned up temporary model file: {best_model_path}")

print("\nHAN model training on Freebase dataset completed successfully!")
print("The model leverages heterogeneous metapaths to capture rich semantic relationships")
print("between books and other entities in the knowledge graph.")