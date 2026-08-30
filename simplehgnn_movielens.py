"""
Link Prediction on MovieLens using SimpleHGN
MovieLens schema: user, movie
Task: Predict user-movie ratings/links
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


class SimpleHGNConv(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        out_channels,
        metadata,
        heads=1,
        dropout=0.0,
        use_norm=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.metadata = metadata
        self.dropout = dropout
        self.use_norm = use_norm

        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = Linear(in_channels, out_channels * heads)

        self.edge_type_emb = nn.ParameterDict()
        for edge_type in metadata[1]:
            key = '__'.join(edge_type)
            self.edge_type_emb[key] = nn.Parameter(torch.randn(heads, out_channels))

        self.att = nn.Parameter(torch.Tensor(heads, 3 * out_channels))
        nn.init.xavier_uniform_(self.att)

        self.residual = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for node_type in metadata[0]:
            self.residual[node_type] = Linear(out_channels * heads, out_channels * heads)
            if use_norm:
                self.norm[node_type] = nn.LayerNorm(out_channels * heads)

    def forward(self, x_dict, edge_index_dict):
        x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}
        out_dict = {k: torch.zeros_like(x_proj[k]) for k in x_proj}

        for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
            key = '__'.join((src_type, rel, dst_type))
            x_src = x_proj[src_type]
            x_dst = x_proj[dst_type]
            edge_emb = self.edge_type_emb[key]

            h_src = x_src[edge_index[0]].view(-1, self.heads, self.out_channels)
            h_dst = x_dst[edge_index[1]].view(-1, self.heads, self.out_channels)
            e = edge_emb.unsqueeze(0).expand(h_src.size(0), -1, -1)

            att_input = torch.cat([h_src, h_dst, e], dim=-1)
            score = (att_input * self.att.unsqueeze(0)).sum(dim=-1)
            score = F.leaky_relu(score, negative_slope=0.2)
            score = softmax(score, edge_index[1], num_nodes=x_dst.size(0))
            score = F.dropout(score, p=self.dropout, training=self.training)

            msg = h_src * score.unsqueeze(-1)
            out = torch.zeros_like(x_proj[dst_type].view(-1, self.heads, self.out_channels))
            out.index_add_(0, edge_index[1], msg)
            out_dict[dst_type] += out.view(-1, self.heads * self.out_channels)

        for node_type in out_dict:
            out = out_dict[node_type] + self.residual[node_type](x_proj[node_type])
            if self.use_norm:
                out = self.norm[node_type](out)
            out_dict[node_type] = out

        return out_dict


class SimpleHGN(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        hidden_channels,
        out_channels,
        metadata,
        heads=8,
        dropout=0.0,
        use_norm=True,
    ):
        super().__init__()
        self.encoder = SimpleHGNConv(
            in_channels_dict=in_channels_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
            use_norm=use_norm,
        )
        # Output linear for each node type
        self.lin = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin[node_type] = Linear(hidden_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.encoder(x_dict, edge_index_dict)
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.lin[node_type](h)
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
                train_pos, train_neg, optimizer, device):
    """Train one epoch"""
    model.train()
    link_predictor.train()
    optimizer.zero_grad()
    
    # Forward pass
    out_dict = model(x_dict, edge_index_dict)
    user_emb = out_dict['user']
    movie_emb = out_dict['movie']
    
    # Positive edges
    pos_scores = link_predictor(user_emb, movie_emb, train_pos)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    
    # Negative edges
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


def run_simplehgn_movielens():
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
    
    # Edge index dict
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
    hidden_channels = 32
    out_channels = 64
    
    # Metadata: (node_types, edge_types)
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    
    model = SimpleHGN(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=metadata,
        heads=4,
        dropout=0.2,
        use_norm=True,
    ).to(device)
    
    link_predictor = LinkPredictor(out_channels, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    print("\nTraining SimpleHGN...")
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
                }, 'best_simplehgn_movielens.pth')
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Final test
    checkpoint = torch.load('best_simplehgn_movielens.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    test_metrics = evaluate(
        model, link_predictor, x_dict, edge_index_dict,
        splits['test_pos'], splits['test_neg'], device
    )
    
    print(f"\n{'='*50}")
    print(f"Final SimpleHGN Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"{'='*50}")
    
    return test_metrics


if __name__ == '__main__':
    run_simplehgn_movielens()
