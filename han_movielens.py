"""
Link Prediction on MovieLens using HAN (Heterogeneous Attention Network)
Uses PyG's built-in HANConv
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HANConv, Linear
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import zipfile
import urllib.request


class HANModel(nn.Module):
    """
    Standard HAN (Heterogeneous Attention Network) model
    Uses metapath-based attention + semantic-level attention
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels,
        out_channels,
        metadata,
        heads=8,
        dropout=0.6,
    ):
        super().__init__()
        self.out_channels = out_channels
        
        # Two-layer HAN
        self.han_conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
            dropout=dropout
        )
        
        self.han_conv2 = HANConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            heads=1,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # First HAN layer
        x_dict = self.han_conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Second HAN layer
        x_dict = self.han_conv2(x_dict, edge_index_dict)
        
        return x_dict


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
    
    if not os.path.exists(data_file):
        print("Downloading MovieLens 100K...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_path = os.path.join(root, 'ml-100k.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.remove(zip_path)
    
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
                train_pos, train_neg, optimizer, device):
    """Train one epoch"""
    model.train()
    link_predictor.train()
    optimizer.zero_grad()
    
    out_dict = model(x_dict, edge_index_dict)
    user_emb = out_dict['user']
    movie_emb = out_dict['movie']
    
    pos_scores = link_predictor(user_emb, movie_emb, train_pos)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    
    neg_scores = link_predictor(user_emb, movie_emb, train_neg)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, link_predictor, x_dict, edge_index_dict,
             pos_edges, neg_edges, device):
    """Evaluate link prediction"""
    model.eval()
    link_predictor.eval()
    
    out_dict = model(x_dict, edge_index_dict)
    user_emb = out_dict['user']
    movie_emb = out_dict['movie']
    
    pos_scores = torch.sigmoid(link_predictor(user_emb, movie_emb, pos_edges)).cpu().numpy()
    neg_scores = torch.sigmoid(link_predictor(user_emb, movie_emb, neg_edges)).cpu().numpy()
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    return {'auc': auc, 'ap': ap}


def run_han_movielens():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading MovieLens dataset...")
    data, num_users, num_movies = load_movielens_manual('./data/movielens_manual')
    
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"  user: {num_users} nodes")
    print(f"  movie: {num_movies} nodes")
    
    x_dict = {
        'user': data['user'].x.to(device),
        'movie': data['movie'].x.to(device)
    }
    
    edge_index_dict = {}
    for edge_type in data.edge_types:
        edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)
    
    target_edges = data['user', 'rates', 'movie'].edge_index.to(device)
    print(f"Total edges: {target_edges.size(1)}")
    
    splits = create_link_splits(target_edges, num_users, num_movies)
    splits = {k: v.to(device) for k, v in splits.items()}
    
    # HAN needs channel dict
    in_channels_dict = {k: v.size(-1) for k, v in x_dict.items()}
    hidden_channels = 64
    out_channels = 64
    
    model = HANModel(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=data.metadata(),
        heads=4,
        dropout=0.3,
    ).to(device)
    
    link_predictor = LinkPredictor(out_channels, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    print("\nTraining HAN...")
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(1, 151):
        loss = train_epoch(
            model, link_predictor, x_dict, edge_index_dict,
            splits['train_pos'], splits['train_neg'],
            optimizer, device
        )
        
        if epoch % 10 == 0:
            val_metrics = evaluate(
                model, link_predictor, x_dict, edge_index_dict,
                splits['val_pos'], splits['val_neg'], device
            )
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | Val AP: {val_metrics['ap']:.4f}")
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                torch.save({
                    'model': model.state_dict(),
                    'link_predictor': link_predictor.state_dict(),
                }, 'best_han_movielens.pth')
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    checkpoint = torch.load('best_han_movielens.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    test_metrics = evaluate(
        model, link_predictor, x_dict, edge_index_dict,
        splits['test_pos'], splits['test_neg'], device
    )
    
    print(f"\n{'='*50}")
    print(f"Final HAN Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"{'='*50}")
    
    return test_metrics


if __name__ == '__main__':
    run_han_movielens()
