import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import LastFM
from torch_geometric.nn import Linear, HeteroConv, SAGEConv
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class SeHGNN(nn.Module):
    """
    Self-supervised Heterogeneous Graph Neural Network (SeHGNN)
    Based on the paper: "Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning"
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=64,
        out_channels=128,
        num_layers=2,
        dropout=0.5,
        heads=4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
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
            for edge_type in [('user', 'to', 'artist'), ('artist', 'to', 'user'), 
                             ('user', 'to', 'user'), ('artist', 'to', 'tag'), ('tag', 'to', 'artist')]:
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
        
        # Self-supervised components
        self.contrast_proj = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.contrast_proj[node_type] = nn.Sequential(
                Linear(out_channels, hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
        
        # Semantic-level contrast projection
        self.semantic_proj = Linear(out_channels, hidden_channels)
        
        # Node-level contrast temperature
        self.tau_node = 0.5
        # Semantic-level contrast temperature  
        self.tau_semantic = 0.8
        
    def forward(self, x_dict, edge_index_dict, return_embeddings=True):
        # Initial node transformation
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.node_transformers[node_type](x)
        
        # Multi-layer heterogeneous convolution
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}
            h_dict = {key: F.dropout(h, p=self.dropout, training=self.training) 
                     for key, h in h_dict.items()}
        
        # Output projection
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.output_proj[node_type](h)
        
        if return_embeddings:
            return out_dict
        
        # For self-supervised training, return projections too
        contrast_dict = {}
        for node_type, emb in out_dict.items():
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
            src_emb = z_dict[src_type][edge_index[0]]
            dst_emb = z_dict[dst_type][edge_index[1]]
            
            # Cosine similarity for positive pairs
            pos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1) / self.tau_node
            
            # Negative sampling for contrastive learning
            num_neg = min(edge_index.size(1), 1000)  # Limit negatives
            neg_src_idx = torch.randint(0, z_dict[src_type].size(0), (num_neg,)).to(edge_index.device)
            neg_dst_idx = torch.randint(0, z_dict[dst_type].size(0), (num_neg,)).to(edge_index.device)
            
            neg_src_emb = z_dict[src_type][neg_src_idx]
            neg_dst_emb = z_dict[dst_type][neg_dst_idx]
            neg_sim = F.cosine_similarity(neg_src_emb, neg_dst_emb, dim=1) / self.tau_node
            
            # InfoNCE loss
            logits = torch.cat([pos_sim[:num_neg], neg_sim], dim=0)
            labels = torch.zeros(num_neg).long().to(edge_index.device)
            
            loss = F.cross_entropy(logits.unsqueeze(0).repeat(num_neg, 1), labels)
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
            # Mean pooling of node embeddings for each type
            type_emb = emb_dict[node_type].mean(dim=0)
            type_embs.append(self.semantic_proj(type_emb))
        
        type_embs = torch.stack(type_embs)  # [num_types, hidden_dim]
        
        # Contrastive loss between node types
        sim_matrix = F.cosine_similarity(type_embs.unsqueeze(1), type_embs.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.tau_semantic
        
        # Positive pairs: same type with itself (diagonal)
        labels = torch.arange(len(node_types)).to(sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class LinkPredictor(nn.Module):
    """Link prediction module"""
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
        src_emb = embeddings[edge_index[0]]
        dst_emb = embeddings[edge_index[1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        return self.predictor(edge_emb).squeeze(-1)


def train_sehgnn(model, link_predictor, data, optimizer_emb, optimizer_lp, device, 
                target_edge_type=('user', 'to', 'artist'), alpha=0.1, beta=0.1):
    """Training function with self-supervised loss"""
    model.train()
    link_predictor.train()
    
    optimizer_emb.zero_grad()
    
    # Forward pass for both embeddings and contrastive projections
    embeddings_dict, contrast_dict = model(data.x_dict, data.edge_index_dict, return_embeddings=False)
    
    # Self-supervised losses
    node_contrast_loss = model.node_level_contrast(contrast_dict, data.edge_index_dict)
    semantic_contrast_loss = model.semantic_level_contrast(embeddings_dict)
    
    # Link prediction loss
    link_loss = 0.0
    if target_edge_type in data.edge_index_dict:
        target_edges = data.edge_index_dict[target_edge_type]
        src_type, _, dst_type = target_edge_type
        
        # Get embeddings
        if src_type == 'user':
            embeddings_for_lp = embeddings_dict[src_type]
            num_nodes_lp = data[src_type].num_nodes
        else:
            embeddings_for_lp = embeddings_dict[dst_type]
            num_nodes_lp = data[dst_type].num_nodes
        
        # Sample edges for training
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
        
        # Link prediction
        optimizer_lp.zero_grad()
        link_scores = link_predictor(embeddings_for_lp, all_edges)
        link_loss = F.binary_cross_entropy_with_logits(link_scores, labels)
    
    # Combined loss
    total_loss = link_loss + alpha * node_contrast_loss + beta * semantic_contrast_loss
    total_loss.backward()
    optimizer_emb.step()
    if link_loss > 0:
        optimizer_lp.step()
    
    return {
        'total_loss': total_loss.item(),
        'link_loss': link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss,
        'node_contrast': node_contrast_loss.item(),
        'semantic_contrast': semantic_contrast_loss.item()
    }


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


def preprocess_hetero_graph(data):
    """
    Enhanced preprocessing for heterogeneous graphs in SeHGNN
    """
    print("Enhanced preprocessing...")
    
    # 1. Compute node degrees for each edge type
    node_degrees = {}
    for node_type in data.node_types:
        degrees = torch.zeros(data[node_type].num_nodes)
        
        # Count incoming and outgoing edges for each edge type
        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type == node_type:
                # Outgoing edges
                degrees.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
            if dst_type == node_type:
                # Incoming edges  
                degrees.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
        
        node_degrees[node_type] = degrees
    
    # 2. Create enhanced node features
    feature_dim = 64
    enhanced_features = {}
    
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        
        # Base random features
        base_features = torch.randn(num_nodes, feature_dim - 4)
        
        # Degree-based features
        degrees = node_degrees[node_type].unsqueeze(1)
        log_degrees = torch.log(degrees + 1).unsqueeze(1)
        
        # Node type one-hot (simple encoding)
        type_id = data.node_types.index(node_type)
        type_features = torch.zeros(num_nodes, 2)
        type_features[:, type_id % 2] = 1.0
        
        # Combine all features
        enhanced_features[node_type] = torch.cat([
            base_features, degrees, log_degrees, type_features
        ], dim=1)
    
    # 3. Ensure bidirectional edges exist
    edge_types_to_add = []
    for edge_type in list(data.edge_index_dict.keys()):
        src_type, rel, dst_type = edge_type
        reverse_edge_type = (dst_type, rel, src_type)
        
        if reverse_edge_type not in data.edge_index_dict and src_type != dst_type:
            # Add reverse edges
            reverse_edge_index = data.edge_index_dict[edge_type].flip(0)
            edge_types_to_add.append((reverse_edge_type, reverse_edge_index))
    
    # Add the reverse edges
    for edge_type, edge_index in edge_types_to_add:
        data.edge_index_dict[edge_type] = edge_index
    
    # 4. Update data with enhanced features
    for node_type, features in enhanced_features.items():
        data[node_type].x = features
        data.x_dict[node_type] = features
    
    print(f"Enhanced features shape: {[(k, v.shape) for k, v in enhanced_features.items()]}")
    print(f"Total edge types after preprocessing: {len(data.edge_index_dict)}")
    
    return data


def create_augmented_views(data, drop_edge_rate=0.1, mask_feature_rate=0.1):
    """
    Create augmented graph views for contrastive learning
    """
    import copy
    
    # Create two augmented views
    data_aug1 = copy.deepcopy(data)
    data_aug2 = copy.deepcopy(data)
    
    # Edge dropping augmentation
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.size(1) > 0:
            # Randomly drop edges
            mask1 = torch.rand(edge_index.size(1)) > drop_edge_rate
            mask2 = torch.rand(edge_index.size(1)) > drop_edge_rate
            
            data_aug1.edge_index_dict[edge_type] = edge_index[:, mask1]
            data_aug2.edge_index_dict[edge_type] = edge_index[:, mask2]
    
    # Feature masking augmentation
    for node_type in data.node_types:
        features1 = data.x_dict[node_type].clone()
        features2 = data.x_dict[node_type].clone()
        
        # Random feature masking
        mask1 = torch.rand_like(features1) > mask_feature_rate
        mask2 = torch.rand_like(features2) > mask_feature_rate
        
        data_aug1.x_dict[node_type] = features1 * mask1
        data_aug2.x_dict[node_type] = features2 * mask2
    
    return data_aug1, data_aug2


def run_lastfm_sehgnn_link_prediction():
    """Main function to run LastFM link prediction with SeHGNN"""
    print("Loading LastFM dataset for SeHGNN...")
    dataset = LastFM(root='./data/lastfm')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    print(f"Dataset: {data}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {list(data.edge_index_dict.keys())}")
    
    # Initialize node features
    feature_dim = 64
    data.x_dict = {}
    print("Initializing node features...")
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        features = torch.randn(num_nodes, feature_dim)
        data[node_type].x = features
        data.x_dict[node_type] = features
    print("Node features initialized!")
    
    # Get input channels
    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}
    print(f"Input channels: {in_channels_dict}")
    
    # Initialize SeHGNN model
    embedding_dim = 128
    model = SeHGNN(
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        out_channels=embedding_dim,
        num_layers=2,
        dropout=0.5,
        heads=4
    ).to(device)
    
    # Initialize link predictor
    link_predictor = LinkPredictor(embedding_dim, hidden_dim=64).to(device)
    
    # Optimizers
    optimizer_emb = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_lp = torch.optim.Adam(link_predictor.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("\nStarting SeHGNN training...")
    
    # Target edge type for link prediction
    target_edge_type = ('user', 'to', 'artist')
    if target_edge_type not in data.edge_index_dict:
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
        # Train with self-supervised learning
        losses = train_sehgnn(model, link_predictor, data, optimizer_emb, optimizer_lp, device, 
                             target_edge_type, alpha=0.1, beta=0.05)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            results = evaluate_link_prediction(model, link_predictor, data, device, target_edge_type)
            auc, ap = results['auc'], results['ap']
            
            print(f"Epoch {epoch:03d} | Total Loss: {losses['total_loss']:.4f} | "
                  f"Link: {losses['link_loss']:.4f} | Node: {losses['node_contrast']:.4f} | "
                  f"Semantic: {losses['semantic_contrast']:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
            
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
                }, 'best_lastfm_sehgnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:03d} | Total Loss: {losses['total_loss']:.4f} | "
                  f"Link: {losses['link_loss']:.4f} | Node: {losses['node_contrast']:.4f} | "
                  f"Semantic: {losses['semantic_contrast']:.4f}")
    
    # Load best model and final evaluation
    checkpoint = torch.load('best_lastfm_sehgnn_model.pth')
    model.load_state_dict(checkpoint['model'])
    link_predictor.load_state_dict(checkpoint['link_predictor'])
    
    final_results = evaluate_link_prediction(model, link_predictor, data, device, target_edge_type)
    print(f"\nFinal SeHGNN Results:")
    print(f"AUC: {final_results['auc']:.4f}")
    print(f"AP: {final_results['ap']:.4f}")
    
    return model, link_predictor, final_results


if __name__ == '__main__':
    run_lastfm_sehgnn_link_prediction()