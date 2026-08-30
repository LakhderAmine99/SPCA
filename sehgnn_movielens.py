"""
Link Prediction on MovieLens using SeHGNN
MovieLens schema: user, movie
Task: Predict user-movie ratings/links
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import zipfile
import urllib.request


class SeHGNN(nn.Module):
    """
    Self-supervised Heterogeneous Graph Neural Network (SeHGNN)
    Based on: "Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning"
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=64,
        out_channels=128,
        num_layers=2,
        dropout=0.5,
        edge_types=None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = list(in_channels_dict.keys())
        
        # Node type transformation layers
        self.node_transformers = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.node_transformers[node_type] = nn.Sequential(
                Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, hidden_channels)
            )
        
        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_channels)
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)
        
        # Output projection
        self.output_proj = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.output_proj[node_type] = nn.Sequential(
                Linear(hidden_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(out_channels, out_channels)
            )
        
        # Self-supervised contrastive projection
        self.contrast_proj = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.contrast_proj[node_type] = nn.Sequential(
                Linear(out_channels, hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
        
        # Semantic-level projection
        self.semantic_proj = Linear(out_channels, hidden_channels)
        
        # Temperature parameters
        self.tau_node = 0.5
        self.tau_semantic = 0.8
        
    def forward(self, x_dict, edge_index_dict, return_embeddings=True):
        # Initial node transformation
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_transformers:
                h_dict[node_type] = self.node_transformers[node_type](x)
        
        # Multi-layer heterogeneous convolution
        for conv in self.convs:
            h_dict_new = conv(h_dict, edge_index_dict)
            # Add residual connection and activation
            for key in h_dict_new:
                if key in h_dict:
                    h_dict_new[key] = F.relu(h_dict_new[key] + h_dict[key])
                else:
                    h_dict_new[key] = F.relu(h_dict_new[key])
            h_dict = {key: F.dropout(h, p=self.dropout, training=self.training) 
                     for key, h in h_dict_new.items()}
        
        # Output projection
        out_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.output_proj:
                out_dict[node_type] = self.output_proj[node_type](h)
        
        if return_embeddings:
            return out_dict
        
        # For self-supervised training, return contrastive projections too
        contrast_dict = {}
        for node_type, emb in out_dict.items():
            if node_type in self.contrast_proj:
                contrast_dict[node_type] = self.contrast_proj[node_type](emb)
        
        return out_dict, contrast_dict
    
    def node_level_contrast(self, z_dict, edge_index_dict):
        """Node-level contrastive loss"""
        total_loss = 0
        count = 0
        
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(1) == 0:
                continue
                
            src_type, _, dst_type = edge_type
            if src_type not in z_dict or dst_type not in z_dict:
                continue
            
            # Positive pairs from edges
            num_pos = min(edge_index.size(1), 1000)
            perm = torch.randperm(edge_index.size(1))[:num_pos]
            
            src_emb = z_dict[src_type][edge_index[0, perm]]
            dst_emb = z_dict[dst_type][edge_index[1, perm]]
            
            # Cosine similarity for positive pairs
            pos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1) / self.tau_node
            
            # Negative sampling
            neg_src_idx = torch.randint(0, z_dict[src_type].size(0), (num_pos,)).to(edge_index.device)
            neg_dst_idx = torch.randint(0, z_dict[dst_type].size(0), (num_pos,)).to(edge_index.device)
            
            neg_src_emb = z_dict[src_type][neg_src_idx]
            neg_dst_emb = z_dict[dst_type][neg_dst_idx]
            neg_sim = F.cosine_similarity(neg_src_emb, neg_dst_emb, dim=1) / self.tau_node
            
            # InfoNCE loss
            logits = torch.cat([pos_sim, neg_sim], dim=0)
            labels = torch.cat([torch.ones(num_pos), torch.zeros(num_pos)]).to(edge_index.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1)
    
    def semantic_level_contrast(self, emb_dict):
        """Semantic-level contrastive loss between node types"""
        node_types = list(emb_dict.keys())
        if len(node_types) < 2:
            return torch.tensor(0.0).to(next(iter(emb_dict.values())).device)
        
        # Aggregate embeddings for each node type
        type_embs = []
        for node_type in node_types:
            type_emb = emb_dict[node_type].mean(dim=0)
            type_embs.append(self.semantic_proj(type_emb))
        
        type_embs = torch.stack(type_embs)
        
        # Contrastive loss between node types
        sim_matrix = F.cosine_similarity(type_embs.unsqueeze(1), type_embs.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.tau_semantic
        
        labels = torch.arange(len(node_types)).to(sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class LinkPredictor(nn.Module):
    """Link prediction using MLP"""
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
    
    def forward(self, src_emb, dst_emb, edge_index):
        src = src_emb[edge_index[0]]
        dst = dst_emb[edge_index[1]]
        edge_emb = torch.cat([src, dst], dim=-1)
        return self.predictor(edge_emb).squeeze(-1)


def load_movielens_manual(root='./data/movielens_manual'):
    """Load MovieLens 100K dataset manually"""
    os.makedirs(root, exist_ok=True)
    data_file = os.path.join(root, 'ml-100k', 'u.data')
    
    # Download if needed
    if not os.path.exists(data_file):
        print("Downloading MovieLens 100K...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_path = os.path.join(root, 'ml-100k.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.remove(zip_path)
    
    # Load ratings
    print("Processing ratings...")
    ratings = []
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = int(parts[0]) - 1
            movie_id = int(parts[1]) - 1
            ratings.append((user_id, movie_id))
    
    ratings = np.array(ratings)
    user_ids = ratings[:, 0].astype(int)
    movie_ids = ratings[:, 1].astype(int)
    
    num_users = user_ids.max() + 1
    num_movies = movie_ids.max() + 1
    
    print(f"  Users: {num_users}, Movies: {num_movies}, Ratings: {len(ratings)}")
    
    # Build HeteroData
    data = HeteroData()
    data['user'].x = torch.randn(num_users, 64)
    data['movie'].x = torch.randn(num_movies, 64)
    data['user'].num_nodes = num_users
    data['movie'].num_nodes = num_movies
    
    edge_index = torch.tensor([user_ids, movie_ids], dtype=torch.long)
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['movie', 'rated_by', 'user'].edge_index = edge_index.flip(0)
    
    return data, num_users, num_movies


def create_link_splits(edge_index, num_nodes_src, num_nodes_dst, 
                       train_ratio=0.8, val_ratio=0.1):
    """Split edges into train/val/test"""
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    train_size = int(train_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    train_pos = edge_index[:, perm[:train_size]]
    val_pos = edge_index[:, perm[train_size:train_size + val_size]]
    test_pos = edge_index[:, perm[train_size + val_size:]]
    
    # Generate negatives
    def sample_negatives(num_neg):
        neg_src = torch.randint(0, num_nodes_src, (num_neg,))
        neg_dst = torch.randint(0, num_nodes_dst, (num_neg,))
        return torch.stack([neg_src, neg_dst])
    
    train_neg = sample_negatives(train_pos.size(1))
    val_neg = sample_negatives(val_pos.size(1))
    test_neg = sample_negatives(test_pos.size(1))
    
    return {
        'train_pos': train_pos, 'train_neg': train_neg,
        'val_pos': val_pos, 'val_neg': val_neg,
        'test_pos': test_pos, 'test_neg': test_neg
    }


def train_epoch(model, link_predictor, x_dict, edge_index_dict,
                train_pos, train_neg, optimizer, device, alpha=0.1, beta=0.1):
    """Train one epoch with self-supervised loss"""
    model.train()
    link_predictor.train()
    optimizer.zero_grad()
    
    # Forward pass with contrastive projections
    out_dict, contrast_dict = model(x_dict, edge_index_dict, return_embeddings=False)
    
    # Link prediction loss
    user_emb = out_dict['user']
    movie_emb = out_dict['movie']
    
    pos_scores = link_predictor(user_emb, movie_emb, train_pos)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    
    neg_scores = link_predictor(user_emb, movie_emb, train_neg)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    
    link_loss = pos_loss + neg_loss
    
    # Self-supervised losses
    node_contrast_loss = model.node_level_contrast(contrast_dict, edge_index_dict)
    semantic_contrast_loss = model.semantic_level_contrast(out_dict)
    
    # Total loss
    total_loss = link_loss + alpha * node_contrast_loss + beta * semantic_contrast_loss
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total': total_loss.item(),
        'link': link_loss.item(),
        'node_contrast': node_contrast_loss.item(),
        'semantic_contrast': semantic_contrast_loss.item()
    }


@torch.no_grad()
def evaluate(model, link_predictor, x_dict, edge_index_dict,
             pos_edges, neg_edges, device):
    """Evaluate link prediction"""
    model.eval()
    link_predictor.eval()
    
    out_dict = model(x_dict, edge_index_dict, return_embeddings=True)
    user_emb = out_dict['user']
    movie_emb = out_dict['movie']
    
    pos_scores = torch.sigmoid(link_predictor(user_emb, movie_emb, pos_edges)).cpu().numpy()
    neg_scores = torch.sigmoid(link_predictor(user_emb, movie_emb, neg_edges)).cpu().numpy()
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    return {'auc': auc, 'ap': ap}


def run_sehgnn_movielens():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MovieLens dataset...")
    data, num_users, num_movies = load_movielens_manual('./data/movielens_manual')
    
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"  user: {num_users} nodes")
    print(f"  movie: {num_movies} nodes")
    
    # Prepare features
    x_dict = {
        'user': data['user'].x.to(device),
        'movie': data['movie'].x.to(device)
    }
    
    # Edge index dict (use original edge type names for HeteroConv)
    edge_index_dict = {}
    for edge_type in data.edge_types:
        edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)
    
    # Target edges for link prediction
    target_edges = data['user', 'rates', 'movie'].edge_index.to(device)
    print(f"Total edges: {target_edges.size(1)}")
    
    # Create splits
    splits = create_link_splits(target_edges, num_users, num_movies)
    splits = {k: v.to(device) for k, v in splits.items()}
    
    # Model config
    in_channels_dict = {k: v.size(-1) for k, v in x_dict.items()}
    hidden_channels = 64
    out_channels = 64
    
    model = SeHGNN(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.3,
        edge_types=list(edge_index_dict.keys()),
    ).to(device)
    
    link_predictor = LinkPredictor(out_channels, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    print("\nTraining SeHGNN...")
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(1, 151):
        losses = train_epoch(
            model, link_predictor, x_dict, edge_index_dict,
            splits['train_pos'], splits['train_neg'],
            optimizer, device, alpha=0.1, beta=0.1
        )
        
        if epoch % 10 == 0:
            val_metrics = evaluate(
                model, link_predictor, x_dict, edge_index_dict,
                splits['val_pos'], splits['val_neg'], device
            )
            
            print(f"Epoch {epoch:03d} | Loss: {losses['total']:.4f} "
                  f"(link: {losses['link']:.4f}, node: {losses['node_contrast']:.4f}, "
                  f"sem: {losses['semantic_contrast']:.4f}) | "
                  f"Val AUC: {val_metrics['auc']:.4f} | Val AP: {val_metrics['ap']:.4f}")
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                torch.save({
                    'model': model.state_dict(),
                    'link_predictor': link_predictor.state_dict(),
                }, 'best_sehgnn_movielens.pth')
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Final test
    checkpoint = torch.load('best_sehgnn_movielens.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    test_metrics = evaluate(
        model, link_predictor, x_dict, edge_index_dict,
        splits['test_pos'], splits['test_neg'], device
    )
    
    print(f"\n{'='*50}")
    print(f"Final SeHGNN Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"{'='*50}")
    
    return test_metrics


if __name__ == '__main__':
    run_sehgnn_movielens()
