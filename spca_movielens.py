"""
Link Prediction on MovieLens using SubpathHANConv2
MovieLens schema: user, movie, (optional: genre)
Task: Predict user-movie ratings/links
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax, negative_sampling
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import zipfile
import urllib.request


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
                    if key not in self.subpath_mlps:
                        self.subpath_mlps[key] = nn.Sequential(
                            nn.Linear(len(subpath) * out_channels, out_channels),
                            nn.ReLU(),
                            nn.Linear(out_channels, out_channels),
                            nn.LayerNorm(out_channels)
                        )

        self.final_lin = Linear(len(metapaths) * out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def _message_pass(self, x_src, x_dst, edge_index):
        if edge_index.size(1) == 0:
            return torch.zeros_like(x_dst)
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

            h = x_proj[src_type]
            for i in range(num_hops):
                edge1 = (metapath[i * 2], 'to', metapath[i * 2 + 1])
                edge2 = (metapath[i * 2 + 1], 'to', metapath[i * 2 + 2])
                if edge1 in edge_index_dict:
                    h = self._message_pass(h, x_proj[metapath[i * 2 + 1]], edge_index_dict[edge1])
                if edge2 in edge_index_dict:
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


class LinkPredictor(nn.Module):
    """Link prediction using dot product + MLP"""
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
    """
    Load MovieLens 100K dataset manually without sentence-transformers.
    Downloads from GroupLens if not present.
    """
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
    
    # Load ratings: user_id, movie_id, rating, timestamp
    print("Processing ratings...")
    ratings = []
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = int(parts[0]) - 1  # 0-indexed
            movie_id = int(parts[1]) - 1
            rating = float(parts[2])
            ratings.append((user_id, movie_id, rating))
    
    ratings = np.array(ratings)
    
    # Get unique users and movies
    user_ids = ratings[:, 0].astype(int)
    movie_ids = ratings[:, 1].astype(int)
    
    num_users = user_ids.max() + 1
    num_movies = movie_ids.max() + 1
    
    print(f"  Users: {num_users}, Movies: {num_movies}, Ratings: {len(ratings)}")
    
    # Build HeteroData
    data = HeteroData()
    
    # Random node features (64-dim)
    data['user'].x = torch.randn(num_users, 64)
    data['movie'].x = torch.randn(num_movies, 64)
    data['user'].num_nodes = num_users
    data['movie'].num_nodes = num_movies
    
    # Edge index for user-movie ratings
    edge_index = torch.tensor([user_ids, movie_ids], dtype=torch.long)
    data['user', 'rates', 'movie'].edge_index = edge_index
    
    # Reverse edges for message passing
    data['movie', 'rated_by', 'user'].edge_index = edge_index.flip(0)
    
    return data, num_users, num_movies


def prepare_data(data, device):
    """Prepare MovieLens data for link prediction"""
    # Initialize features if missing
    x_dict = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            x_dict[node_type] = data[node_type].x.to(device)
        else:
            # Random initialization
            num_nodes = data[node_type].num_nodes
            x_dict[node_type] = torch.randn(num_nodes, 64).to(device)
    
    # Build edge_index_dict with bidirectional edges
    edge_index_dict = {}
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index.to(device)
        # Forward
        edge_index_dict[(src, 'to', dst)] = edge_index
        # Backward
        edge_index_dict[(dst, 'to', src)] = edge_index.flip(0)
    
    return x_dict, edge_index_dict


def create_link_splits(edge_index, num_nodes_src, num_nodes_dst, 
                       train_ratio=0.8, val_ratio=0.1):
    """Split edges into train/val/test"""
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    train_size = int(train_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    
    train_pos = edge_index[:, train_idx]
    val_pos = edge_index[:, val_idx]
    test_pos = edge_index[:, test_idx]
    
    # Generate negative samples
    def sample_negatives(pos_edges, num_neg):
        neg_src = torch.randint(0, num_nodes_src, (num_neg,))
        neg_dst = torch.randint(0, num_nodes_dst, (num_neg,))
        return torch.stack([neg_src, neg_dst])
    
    train_neg = sample_negatives(train_pos, train_pos.size(1))
    val_neg = sample_negatives(val_pos, val_pos.size(1))
    test_neg = sample_negatives(test_pos, test_pos.size(1))
    
    return {
        'train_pos': train_pos, 'train_neg': train_neg,
        'val_pos': val_pos, 'val_neg': val_neg,
        'test_pos': test_pos, 'test_neg': test_neg
    }


def train_epoch(model, link_predictor, x_dict, edge_index_dict,
                user_emb_cache, movie_emb_cache,
                train_pos, train_neg, optimizer, device):
    """Train one epoch"""
    model.train()
    link_predictor.train()
    optimizer.zero_grad()
    
    # Get user embeddings (for user metapaths)
    user_emb = model(x_dict, edge_index_dict)
    
    # Positive edges
    pos_scores = link_predictor(user_emb, x_dict['movie'], train_pos)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    
    # Negative edges
    neg_scores = link_predictor(user_emb, x_dict['movie'], train_neg)
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
    
    user_emb = model(x_dict, edge_index_dict)
    
    pos_scores = torch.sigmoid(link_predictor(user_emb, x_dict['movie'], pos_edges)).cpu().numpy()
    neg_scores = torch.sigmoid(link_predictor(user_emb, x_dict['movie'], neg_edges)).cpu().numpy()
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    return {'auc': auc, 'ap': ap}


def run_movielens_link_prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MovieLens dataset manually (no sentence-transformers needed)
    print("Loading MovieLens dataset...")
    data, num_users, num_movies = load_movielens_manual('./data/movielens_manual')
    
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"  user: {num_users} nodes")
    print(f"  movie: {num_movies} nodes")
    
    # Prepare data
    x_dict, edge_index_dict = prepare_data(data, device)
    
    # Get target edges (user-movie)
    target_edge_type = ('user', 'rates', 'movie')
    
    print(f"Target edge type: {target_edge_type}")
    target_edges = data[target_edge_type].edge_index.to(device)
    print(f"Total edges: {target_edges.size(1)}")
    
    # Create train/val/test splits
    splits = create_link_splits(
        target_edges,
        num_users,
        num_movies,
        train_ratio=0.8,
        val_ratio=0.1
    )
    splits = {k: v.to(device) for k, v in splits.items()}
    
    # Define metapaths
    # user -> movie -> user (users who like same movies)
    metapaths = [
        ('user', 'movie', 'user'),
    ]
    
    subpaths_dict = {
        ('user', 'movie', 'user'): [
            ['movie'],
        ],
    }
    
    in_channels_dict = {k: v.size(-1) for k, v in x_dict.items()}
    out_channels = 64
    
    model = SubpathHANConv2(
        in_channels_dict=in_channels_dict,
        out_channels=out_channels,
        metadata=(list(x_dict.keys()), list(edge_index_dict.keys())),
        metapaths=metapaths,
        subpaths_dict=subpaths_dict,
        heads=4,
        dropout=0.2,
        aggregator='concat-mlp',
    ).to(device)
    
    link_predictor = LinkPredictor(out_channels, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    print("\nTraining...")
    best_val_auc = 0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, 151):
        loss = train_epoch(
            model, link_predictor, x_dict, edge_index_dict,
            None, None,  # caches not used
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
                }, 'best_movielens_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 3:  # 30 epochs
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Final test evaluation
    checkpoint = torch.load('best_movielens_model.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    test_metrics = evaluate(
        model, link_predictor, x_dict, edge_index_dict,
        splits['test_pos'], splits['test_neg'], device
    )
    
    print(f"\n{'='*40}")
    print(f"Final Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"{'='*40}")
    
    return test_metrics


if __name__ == '__main__':
    run_movielens_link_prediction()
