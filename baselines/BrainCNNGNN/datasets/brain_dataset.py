"""
Brain network dataset for connectivity matrices.
"""

import os
import numpy as np
import torch
from .base_dataset import BaseDataset


class BrainDataset(BaseDataset):
    """
    Dataset for brain connectivity matrices.
    
    Args:
        data_dir (str): Directory containing the data
        split (str): Dataset split ('train', 'val', or 'test')
        num_nodes (int): Number of ROIs/nodes in the brain network
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, data_dir, split='train', num_nodes=90, transform=None):
        super(BrainDataset, self).__init__(data_dir, split, transform)
        self.num_nodes = num_nodes
        self.load_data()
    
    def load_data(self):
        """
        Load brain connectivity data from disk.
        Expected format: .npy files containing connectivity matrices.
        """
        data_file = os.path.join(self.data_dir, f'{self.split}_data.npy')
        label_file = os.path.join(self.data_dir, f'{self.split}_labels.npy')
        
        if os.path.exists(data_file) and os.path.exists(label_file):
            self.data = np.load(data_file)
            self.labels = np.load(label_file)
        else:
            print(f"Warning: Data files not found at {self.data_dir}")
            print(f"Expected files: {data_file} and {label_file}")
            print("Creating dummy data for demonstration purposes.")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration purposes."""
        num_samples = 100 if self.split == 'train' else 20
        self.data = np.random.randn(num_samples, self.num_nodes, self.num_nodes).astype(np.float32)
        # Make matrices symmetric
        for i in range(num_samples):
            self.data[i] = (self.data[i] + self.data[i].T) / 2
        self.labels = np.random.randint(0, 2, size=num_samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Dictionary containing 'data' (connectivity matrix) and 'label'
        """
        # Get connectivity matrix and label
        connectivity = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        connectivity = torch.FloatTensor(connectivity).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([label]).squeeze()
        
        sample = {
            'data': connectivity,
            'label': label,
            'num_nodes': self.num_nodes
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_connectivity_matrix(self, idx):
        """Get the connectivity matrix at the given index."""
        return self.data[idx]
    
    def get_edge_index(self, idx, threshold=0.0):
        """
        Convert connectivity matrix to edge index format for graph neural networks.
        
        Args:
            idx (int): Index
            threshold (float): Threshold for edge creation
            
        Returns:
            tuple: (edge_index, edge_weight)
        """
        connectivity = self.data[idx]
        edge_index = []
        edge_weight = []
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and abs(connectivity[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_weight.append(connectivity[i, j])
        
        edge_index = torch.LongTensor(edge_index).t()
        edge_weight = torch.FloatTensor(edge_weight)
        
        return edge_index, edge_weight
