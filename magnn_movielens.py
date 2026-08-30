"""
Link Prediction on MovieLens using MAGNN (Metapath Aggregated Graph Neural Network)
Based on: "MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding"
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import zipfile
import urllib.request


class IntraMetapathAggregation(nn.Module):
    """
    Intra-metapath aggregation with attention mechanism.
    Aggregates neighbors along a single metapath.
    """
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear = Linear(in_channels, out_channels)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_src, x_dst, edge_index):
        """
        x_src: source node features
        x_dst: destination node features  
        edge_index: edges from src to dst
        """
        if edge_index.size(1) == 0:
            return self.linear(x_dst)
        
        x_src = self.linear(x_src)
        x_dst = self.linear(x_dst)
        
        # Compute attention scores
        h_src = x_src[edge_index[0]]
        h_dst = x_dst[edge_index[1]]
        
        att_input = torch.cat([h_src, h_dst], dim=-1)
        alpha = (att_input * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index[1], num_nodes=x_dst.size(0))
        alpha = self.dropout(alpha)
        
        # Aggregate
        out = torch.zeros_like(x_dst)
        out.index_add_(0, edge_index[1], h_src * alpha.unsqueeze(-1))
        
        return out


class InterMetapathAggregation(nn.Module):
    """
    Inter-metapath aggregation with semantic attention.
    Combines embeddings from multiple metapaths.
    """
    def __init__(self, hidden_channels, num_metapaths):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.semantic_att = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            Linear(hidden_channels, 1, bias=False)
        )
    
    def forward(self, metapath_embeddings):
        """
        metapath_embeddings: list of [num_nodes, hidden_channels] tensors
        """
        if len(metapath_embeddings) == 1:
            return metapath_embeddings[0]
        
        # Stack: [num_metapaths, num_nodes, hidden_channels]
        stacked = torch.stack(metapath_embeddings, dim=0)
        
        # Compute attention weights
        # [num_metapaths, num_nodes, 1]
        att_scores = self.semantic_att(stacked)
        
        # Average over nodes, then softmax over metapaths
        # [num_metapaths, 1]
        att_weights = att_scores.mean(dim=1)
        att_weights = F.softmax(att_weights, dim=0)
        
        # Weighted combination: [num_nodes, hidden_channels]
        out = (stacked * att_weights.unsqueeze(-1)).sum(dim=0)
        
        return out


class MAGNN(nn.Module):
    """
    Metapath Aggregated Graph Neural Network
    
    Architecture:
    1. Intra-metapath aggregation: Aggregate along each metapath
    2. Inter-metapath aggregation: Semantic attention across metapaths
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels,
        out_channels,
        metadata,
        metapaths,  # List of metapaths, e.g., [('user', 'movie', 'user')]
        num_layers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.metapaths = metapaths
        self.num_layers = num_layers
        self.node_types = metadata[0]
        
        # Input projection
        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = Linear(in_channels, hidden_channels)
        
        # Intra-metapath aggregation for each metapath
        self.intra_agg = nn.ModuleDict()
        for metapath in metapaths:
            mp_key = '_'.join(metapath)
            self.intra_agg[mp_key] = IntraMetapathAggregation(
                hidden_channels, hidden_channels, dropout
            )
        
        # Inter-metapath aggregation for each node type
        self.inter_agg = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            # Count metapaths that end with this node type
            num_mp = sum(1 for mp in metapaths if mp[-1] == node_type)
            if num_mp > 0:
                self.inter_agg[node_type] = InterMetapathAggregation(
                    hidden_channels, num_mp
                )
        
        # Output projection
        self.output_lin = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.output_lin[node_type] = Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x_dict, edge_index_dict):
        # Project input features
        h_dict = {k: self.proj[k](v) for k, v in x_dict.items()}
        
        # Collect embeddings per node type from each metapath
        node_embeddings = {nt: [] for nt in self.node_types}
        
        for metapath in self.metapaths:
            mp_key = '_'.join(metapath)
            target_type = metapath[-1]
            
            # Simple two-hop aggregation for length-3 metapaths
            # metapath: (A, B, A) means A -> B -> A
            if len(metapath) == 3:
                src_type = metapath[0]
                mid_type = metapath[1]
                dst_type = metapath[2]
                
                # Edge: src -> mid
                edge1 = None
                for et, ei in edge_index_dict.items():
                    if et[0] == src_type and et[2] == mid_type:
                        edge1 = (et, ei)
                        break
                
                # Edge: mid -> dst
                edge2 = None
                for et, ei in edge_index_dict.items():
                    if et[0] == mid_type and et[2] == dst_type:
                        edge2 = (et, ei)
                        break
                
                if edge1 is not None and edge2 is not None:
                    # First hop: src -> mid (aggregate src features to mid)
                    h_mid = self.intra_agg[mp_key](h_dict[src_type], h_dict[mid_type], edge1[1])
                    h_mid = F.elu(h_mid)
                    h_mid = self.dropout(h_mid)
                    
                    # Second hop: mid -> dst (aggregate mid features to dst)
                    h_out = self.intra_agg[mp_key](h_mid, h_dict[dst_type], edge2[1])
                    h_out = self.norm(h_out)
                    
                    node_embeddings[target_type].append(h_out)
        
        # Inter-metapath aggregation
        out_dict = {}
        for node_type in self.node_types:
            if node_embeddings[node_type]:
                if node_type in self.inter_agg:
                    aggregated = self.inter_agg[node_type](node_embeddings[node_type])
                else:
                    aggregated = node_embeddings[node_type][0]
                out_dict[node_type] = self.output_lin[node_type](aggregated)
            else:
                # Fallback: use projected features
                out_dict[node_type] = self.output_lin[node_type](h_dict[node_type])
        
        return out_dict


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


def run_magnn_movielens():
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
    
    # MAGNN config with metapaths
    in_channels_dict = {k: v.size(-1) for k, v in x_dict.items()}
    hidden_channels = 64
    out_channels = 64
    
    # Metapaths for MovieLens
    metapaths = [
        ('user', 'movie', 'user'),   # Users connected through movies
        ('movie', 'user', 'movie'),  # Movies connected through users
    ]
    
    model = MAGNN(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=(list(x_dict.keys()), list(edge_index_dict.keys())),
        metapaths=metapaths,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    
    link_predictor = LinkPredictor(out_channels, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    print("\nTraining MAGNN...")
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
                }, 'best_magnn_movielens.pth')
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    checkpoint = torch.load('best_magnn_movielens.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    test_metrics = evaluate(
        model, link_predictor, x_dict, edge_index_dict,
        splits['test_pos'], splits['test_neg'], device
    )
    
    print(f"\n{'='*50}")
    print(f"Final MAGNN Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"{'='*50}")
    
    return test_metrics


if __name__ == '__main__':
    run_magnn_movielens()
