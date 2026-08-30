import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import LastFM
from torch_geometric.nn import HANConv, Linear
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class HANModel(nn.Module):
    """
    Standard HAN (Heterogeneous Attention Network) model for link prediction
    """
    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        heads=8,
        dropout=0.6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # HAN layers
        self.han_conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=64,
            metadata=metadata,
            heads=heads,
            dropout=dropout
        )
        
        self.han_conv2 = HANConv(
            in_channels=64,
            out_channels=out_channels,
            metadata=metadata,
            heads=1,
            dropout=dropout
        )
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # First HAN layer
        x_dict = self.han_conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
        
        # Second HAN layer
        x_dict = self.han_conv2(x_dict, edge_index_dict)
        
        return x_dict


class LinkPredictor(nn.Module):
    """
    Link prediction module that uses node embeddings to predict edges
    """
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embeddings, edge_index):
        """
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Edge scores [num_edges]
        """
        src_emb = embeddings[edge_index[0]]
        dst_emb = embeddings[edge_index[1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        return self.predictor(edge_emb).squeeze(-1)


def train_link_prediction(model, link_predictor, data, optimizer_emb, optimizer_lp, device, target_edge_type=('user', 'to', 'artist')):
    """Train both embedding model and link predictor"""
    model.train()
    link_predictor.train()
    
    # Train embedding model
    optimizer_emb.zero_grad()
    embeddings_dict = model(data.x_dict, data.edge_index_dict)
    
    # Get target edges for link prediction
    if target_edge_type in data.edge_index_dict:
        target_edges = data.edge_index_dict[target_edge_type]
        src_type, _, dst_type = target_edge_type
        
        # Get embeddings for source and destination node types
        src_embeddings = embeddings_dict[src_type]
        dst_embeddings = embeddings_dict[dst_type]
        
        # Create combined embedding space for link prediction
        # For user-artist prediction, we'll use a hybrid approach
        max_nodes = max(src_embeddings.size(0), dst_embeddings.size(0))
        
        # Pad embeddings to same size
        if src_embeddings.size(0) < max_nodes:
            padding = torch.zeros(max_nodes - src_embeddings.size(0), src_embeddings.size(1)).to(device)
            src_embeddings_padded = torch.cat([src_embeddings, padding], dim=0)
        else:
            src_embeddings_padded = src_embeddings
            
        if dst_embeddings.size(0) < max_nodes:
            padding = torch.zeros(max_nodes - dst_embeddings.size(0), dst_embeddings.size(1)).to(device)
            dst_embeddings_padded = torch.cat([dst_embeddings, padding], dim=0)
        else:
            dst_embeddings_padded = dst_embeddings
        
        # Use user embeddings as the primary embedding space (users are typically fewer)
        if src_type == 'user':
            embeddings_for_lp = src_embeddings[:data[src_type].num_nodes]
            num_nodes_lp = data[src_type].num_nodes
        else:
            embeddings_for_lp = dst_embeddings[:data[dst_type].num_nodes]
            num_nodes_lp = data[dst_type].num_nodes
        
        # Limit the number of edges for training efficiency
        max_edges = min(target_edges.size(1), 2000)
        sampled_edges = target_edges[:, :max_edges]
        
        # Generate negative samples
        neg_edges = negative_sampling(
            sampled_edges,
            num_nodes=num_nodes_lp,
            num_neg_samples=max_edges
        )
        
        # Combine positive and negative edges
        all_edges = torch.cat([sampled_edges, neg_edges], dim=1)
        labels = torch.cat([
            torch.ones(max_edges),
            torch.zeros(max_edges)
        ]).to(device)
        
        # Predict links
        optimizer_lp.zero_grad()
        link_scores = link_predictor(embeddings_for_lp, all_edges)
        link_loss = F.binary_cross_entropy_with_logits(link_scores, labels)
        
        total_loss = link_loss
        total_loss.backward()
        optimizer_emb.step()
        optimizer_lp.step()
        
        return total_loss.item()
    return 0.0


@torch.no_grad()
def evaluate_link_prediction(model, link_predictor, data, device, target_edge_type=('user', 'to', 'artist')):
    """Evaluate link prediction performance"""
    model.eval()
    link_predictor.eval()
    
    embeddings_dict = model(data.x_dict, data.edge_index_dict)
    
    if target_edge_type not in data.edge_index_dict:
        return {'auc': 0.0, 'ap': 0.0}
    
    target_edges = data.edge_index_dict[target_edge_type]
    src_type, _, dst_type = target_edge_type
    
    # Get embeddings for appropriate node type
    if src_type == 'user':
        embeddings_for_lp = embeddings_dict[src_type]
        num_nodes = data[src_type].num_nodes
    else:
        embeddings_for_lp = embeddings_dict[dst_type] 
        num_nodes = data[dst_type].num_nodes
    
    # Limit edges for evaluation efficiency
    max_eval_edges = min(target_edges.size(1), 1500)
    test_edges = target_edges[:, :max_eval_edges]
    
    # Generate test negative samples
    neg_edges = negative_sampling(
        test_edges,
        num_nodes=num_nodes,
        num_neg_samples=max_eval_edges
    )
    
    # Combine positive and negative edges
    all_edges = torch.cat([test_edges, neg_edges], dim=1)
    labels = torch.cat([
        torch.ones(max_eval_edges),
        torch.zeros(max_eval_edges)
    ]).cpu().numpy()
    
    # Predict
    link_scores = link_predictor(embeddings_for_lp, all_edges)
    scores = torch.sigmoid(link_scores).cpu().numpy()
    
    # Compute metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    return {'auc': auc, 'ap': ap}


def run_lastfm_han_link_prediction():
    """Main function to run LastFM link prediction with HAN model"""
    print("Loading LastFM dataset...")
    dataset = LastFM(root='./data/lastfm')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    print(f"Dataset: {data}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {list(data.edge_index_dict.keys())}")
    
    # Initialize node features if they don't exist
    feature_dim = 64  # Initial feature dimension
    data.x_dict = {}
    print("Initializing node features...")
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        # Initialize with random features (in practice, you might use node2vec, etc.)
        features = torch.randn(num_nodes, feature_dim)
        data[node_type].x = features
        data.x_dict[node_type] = features
    print("Node features initialized!")
    
    # Get input channels
    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}
    print(f"Input channels: {in_channels_dict}")
    
    # Initialize HAN model
    embedding_dim = 128
    model = HANModel(
        in_channels_dict=in_channels_dict,
        out_channels=embedding_dim,
        metadata=data.metadata(),
        heads=8,
        dropout=0.6
    ).to(device)
    
    # Initialize link predictor
    link_predictor = LinkPredictor(embedding_dim, hidden_dim=64).to(device)
    
    # Optimizers
    optimizer_emb = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer_lp = torch.optim.Adam(link_predictor.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("\nStarting HAN training...")
    
    # Target edge type for link prediction (user-artist relationships)
    target_edge_type = ('user', 'to', 'artist')
    if target_edge_type not in data.edge_index_dict:
        # Try alternative edge type naming
        possible_targets = [k for k in data.edge_index_dict.keys() if 'user' in k[0] and 'artist' in k[2]]
        if possible_targets:
            target_edge_type = possible_targets[0]
            print(f"Using edge type: {target_edge_type}")
        else:
            print("No suitable user-artist edge type found!")
            return
    
    best_auc = 0.0
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, 201):
        # Train
        loss = train_link_prediction(model, link_predictor, data, optimizer_emb, optimizer_lp, device, target_edge_type)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            results = evaluate_link_prediction(model, link_predictor, data, device, target_edge_type)
            auc, ap = results['auc'], results['ap']
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
            
            # Early stopping
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model': model.state_dict(),
                    'link_predictor': link_predictor.state_dict(),
                    'epoch': epoch,
                    'auc': auc,
                    'ap': ap
                }, 'best_lastfm_han_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # Load best model and final evaluation
    checkpoint = torch.load('best_lastfm_han_model.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    final_results = evaluate_link_prediction(model, link_predictor, data, device, target_edge_type)
    print(f"\nFinal HAN Results:")
    print(f"AUC: {final_results['auc']:.4f}")
    print(f"AP: {final_results['ap']:.4f}")
    
    return model, link_predictor, final_results


if __name__ == '__main__':
    run_lastfm_han_link_prediction()