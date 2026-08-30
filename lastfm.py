import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import LastFM
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class SubpathHANConv2(nn.Module):
    """
    Enhanced subpath-aware HAN model with semantic attention for link prediction
    """
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

        # Type embeddings for subpaths
        self.type_emb = nn.ParameterDict()
        for node_type in metadata[0]:
            self.type_emb[node_type] = nn.Parameter(torch.randn(out_channels))

        # Node feature projection
        self.proj = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        # MLPs for subpath aggregation
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

        # HAN-style semantic attention
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.randn(out_channels))

    def _message_pass(self, x_src, x_dst, edge_index):
        """Simple attention-like weighted aggregation"""
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
        # Project input features
        x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}

        # Store embeddings for all node types
        all_embeddings = {}
        
        for node_type in x_dict.keys():
            metapath_embs = []
            
            # Process metapaths that end with this node type
            for metapath in self.metapaths:
                if metapath[-1] != node_type:
                    continue
                    
                assert len(metapath) % 2 == 1, "Metapath must be odd-length"
                num_hops = len(metapath) // 2
                src_type = metapath[0]
                dst_type = metapath[-1]

                h = x_proj[src_type]
                # Multi-hop message passing
                for i in range(num_hops):
                    edge1 = (metapath[i * 2], 'to', metapath[i * 2 + 1])
                    edge2 = (metapath[i * 2 + 1], 'to', metapath[i * 2 + 2])
                    
                    if edge1 in edge_index_dict:
                        h = self._message_pass(h, x_proj[metapath[i * 2 + 1]], edge_index_dict[edge1])
                    if edge2 in edge_index_dict:
                        h = self._message_pass(h, x_proj[metapath[i * 2 + 2]], edge_index_dict[edge2])
                
                h = self.norm(h)
                within_emb = h

                # Subpath embeddings
                sub_embs = []
                if metapath in self.subpaths_dict:
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

                if sub_embs:
                    subpath_emb = torch.stack(sub_embs, dim=0).mean(dim=0)
                    final_emb = within_emb + subpath_emb
                else:
                    final_emb = within_emb
                
                metapath_embs.append(final_emb)
            
            # If no metapaths end with this node type, use projected features
            if not metapath_embs:
                all_embeddings[node_type] = x_proj[node_type]
            else:
                # Semantic attention over metapaths
                if len(metapath_embs) == 1:
                    all_embeddings[node_type] = metapath_embs[0]
                else:
                    H = torch.stack(metapath_embs, dim=0).transpose(0, 1)  # [num_nodes, num_metapaths, out_channels]
                    Wh = torch.tanh(self.k_lin(H))
                    attn_logits = (Wh * self.q).sum(dim=-1)
                    attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)
                    all_embeddings[node_type] = (H * attn_weights).sum(dim=1)
        
        return all_embeddings


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


def train_embeddings(model, data, optimizer, device):
    """Train the embedding model (without link prediction loss)"""
    model.train()
    optimizer.zero_grad()
    
    # Get embeddings
    embeddings_dict = {}
    for node_type in data.node_types:
        if node_type in data.x_dict:
            # Forward pass for each node type
            out = model(data.x_dict, data.edge_index_dict)
            embeddings_dict[node_type] = out
            break  # For now, assume we get embeddings for all node types together
    
    # Simple reconstruction loss on existing edges
    total_loss = 0
    num_edge_types = 0
    
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.size(1) > 0:
            src_type, _, dst_type = edge_type
            if src_type in embeddings_dict and dst_type in embeddings_dict:
                src_emb = embeddings_dict[src_type][edge_index[0]]
                dst_emb = embeddings_dict[dst_type][edge_index[1]]
                # Simple dot product similarity
                pos_scores = (src_emb * dst_emb).sum(dim=-1)
                
                # Negative sampling
                neg_edge_index = negative_sampling(
                    edge_index, 
                    num_nodes=(embeddings_dict[src_type].size(0), embeddings_dict[dst_type].size(0)),
                    num_neg_samples=edge_index.size(1)
                )
                neg_src_emb = embeddings_dict[src_type][neg_edge_index[0]]
                neg_dst_emb = embeddings_dict[dst_type][neg_edge_index[1]]
                neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=-1)
                
                # Binary cross entropy loss
                pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                total_loss += pos_loss + neg_loss
                num_edge_types += 1
    
    if num_edge_types > 0:
        total_loss = total_loss / num_edge_types
        total_loss.backward()
        optimizer.step()
        return total_loss.item()
    return 0.0


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
        max_nodes = max(src_embeddings.size(0), dst_embeddings.size(0))
        if src_embeddings.size(0) < max_nodes:
            padding = torch.zeros(max_nodes - src_embeddings.size(0), src_embeddings.size(1)).to(device)
            src_embeddings = torch.cat([src_embeddings, padding], dim=0)
        if dst_embeddings.size(0) < max_nodes:
            padding = torch.zeros(max_nodes - dst_embeddings.size(0), dst_embeddings.size(1)).to(device)
            dst_embeddings = torch.cat([dst_embeddings, padding], dim=0)
        
        # Use user embeddings for link prediction
        if src_type == 'user':
            embeddings_for_lp = src_embeddings[:data[src_type].num_nodes]
        else:
            embeddings_for_lp = dst_embeddings[:data[dst_type].num_nodes]
        
        # Generate negative samples
        neg_edges = negative_sampling(
            target_edges,
            num_nodes=data[src_type].num_nodes if src_type == 'user' else data[dst_type].num_nodes,
            num_neg_samples=min(target_edges.size(1), 1000)
        )
        
        # Combine positive and negative edges
        all_edges = torch.cat([target_edges[:, :min(target_edges.size(1), 1000)], neg_edges], dim=1)
        labels = torch.cat([
            torch.ones(min(target_edges.size(1), 1000)),
            torch.zeros(neg_edges.size(1))
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


def evaluate_link_prediction(model, link_predictor, data, device, target_edge_type=('user', 'to', 'artist')):
    """Evaluate link prediction performance"""
    model.eval()
    link_predictor.eval()
    
    embeddings_dict = model(data.x_dict, data.edge_index_dict)
    
    if target_edge_type not in data.edge_index_dict:
        return {'auc': 0.0, 'ap': 0.0, 'mrr': 0.0}
    
    target_edges = data.edge_index_dict[target_edge_type]
    src_type, _, dst_type = target_edge_type
    
    # Get embeddings for appropriate node type
    if src_type == 'user':
        embeddings_for_lp = embeddings_dict[src_type]
        num_nodes = data[src_type].num_nodes
    else:
        embeddings_for_lp = embeddings_dict[dst_type] 
        num_nodes = data[dst_type].num_nodes
    
    # Generate test negative samples
    test_edges = target_edges[:, :min(target_edges.size(1), 1000)]
    neg_edges = negative_sampling(
        test_edges,
        num_nodes=num_nodes,
        num_neg_samples=test_edges.size(1)
    )
    
    # Combine positive and negative edges
    all_edges = torch.cat([test_edges, neg_edges], dim=1)
    labels = torch.cat([
        torch.ones(test_edges.size(1)),
        torch.zeros(neg_edges.size(1))
    ]).cpu().numpy()
    
    # Predict
    link_scores = link_predictor(embeddings_for_lp, all_edges)
    scores = torch.sigmoid(link_scores).detach().cpu().numpy()
    
    # Compute metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    # Compute MRR (per-query: each positive vs sampled negatives)
    num_pos = test_edges.size(1)
    pos_scores = scores[:num_pos]
    neg_scores = scores[num_pos:]
    
    mrr_sum = 0.0
    num_neg_per_pos = 50  # Sample 50 negatives per positive
    for i, pos_score in enumerate(pos_scores):
        # Sample random negatives for this query
        neg_indices = np.random.choice(len(neg_scores), min(num_neg_per_pos, len(neg_scores)), replace=False)
        sampled_neg_scores = neg_scores[neg_indices]
        # Rank: count how many negatives score >= positive, then +1
        rank = (sampled_neg_scores >= pos_score).sum() + 1
        mrr_sum += 1.0 / rank
    mrr = mrr_sum / len(pos_scores)
    
    return {'auc': auc, 'ap': ap, 'mrr': mrr}


def run_lastfm_link_prediction():
    """Main function to run LastFM link prediction with subpath2 model"""
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
        # Initialize with random features
        features = torch.randn(num_nodes, feature_dim)
        data[node_type].x = features
        data.x_dict[node_type] = features
    print("Node features initialized!")
    
    # Define metapaths for LastFM (user-artist-user patterns)
    metapaths = [
        ('user', 'artist', 'user'),  # Users connected through artists
        ('artist', 'user', 'artist'),  # Artists connected through users
    ]
    
    # Define subpaths for semantic encoding
    subpaths_dict = {
        ('user', 'artist', 'user'): [
            ['artist'],  # Artist as intermediate
            ['user', 'artist'],  # User-Artist pattern
        ],
        ('artist', 'user', 'artist'): [
            ['user'],  # User as intermediate
            ['artist', 'user'],  # Artist-User pattern
        ],
    }
    
    # Get input channels
    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}
    print(f"Input channels: {in_channels_dict}")
    
    # Initialize model
    embedding_dim = 128
    model = SubpathHANConv2(
        in_channels_dict=in_channels_dict,
        out_channels=embedding_dim,
        metadata=data.metadata(),
        metapaths=metapaths,
        subpaths_dict=subpaths_dict,
        heads=4,
        dropout=0.1,
        aggregator='concat-mlp',
    ).to(device)
    
    # Initialize link predictor
    link_predictor = LinkPredictor(embedding_dim, hidden_dim=64).to(device)
    
    # Optimizers
    optimizer_emb = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_lp = torch.optim.Adam(link_predictor.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("\nStarting training...")
    
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
            auc, ap, mrr = results['auc'], results['ap'], results['mrr']
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f} | MRR: {mrr:.4f}")
            
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
                    'ap': ap,
                    'mrr': mrr
                }, 'best_lastfm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # Load best model and final evaluation
    checkpoint = torch.load('best_lastfm_model.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    final_results = evaluate_link_prediction(model, link_predictor, data, device, target_edge_type)
    print(f"\nFinal Results:")
    print(f"AUC: {final_results['auc']:.4f}")
    print(f"AP: {final_results['ap']:.4f}")
    print(f"MRR: {final_results['mrr']:.4f}")
    
    return model, link_predictor, final_results


if __name__ == '__main__':
    run_lastfm_link_prediction()
