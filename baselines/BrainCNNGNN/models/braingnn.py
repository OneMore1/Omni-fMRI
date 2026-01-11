"""
BrainGNN: Graph Neural Network for Brain Network Analysis.
Implementation based on graph neural network architectures for brain connectivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class BrainGNN(nn.Module):
    """
    Brain Graph Neural Network for brain network analysis.
    
    Args:
        num_node_features (int): Number of node features
        num_classes (int): Number of output classes
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of GNN layers
        dropout (float): Dropout probability
        pooling (str): Global pooling method ('mean' or 'max')
        use_attention (bool): Whether to use GAT layers instead of GCN
    """
    
    def __init__(self, num_node_features=1, num_classes=2, hidden_dim=64, 
                 num_layers=3, dropout=0.5, pooling='mean', use_attention=False):
        super(BrainGNN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_attention = use_attention
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if use_attention:
            self.convs.append(GATConv(num_node_features, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # MLP for classification
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, num_node_features)
            edge_index (torch.Tensor): Graph connectivity in COO format of shape (2, num_edges)
            edge_weight (torch.Tensor, optional): Edge weights of shape (num_edges,)
            batch (torch.Tensor, optional): Batch vector of shape (num_nodes,)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            if self.use_attention:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is None:
            # Single graph case
            if self.pooling == 'mean':
                x = x.mean(dim=0, keepdim=True)
            else:
                x = x.max(dim=0, keepdim=True)[0]
        else:
            # Batch case
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            else:
                x = global_max_pool(x, batch)
        
        # MLP classification head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc_out(x)
        
        return x
    
    def get_num_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
