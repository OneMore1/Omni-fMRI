"""
Base dataset class for brain network data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset class for brain network data.
    
    Args:
        data_dir (str): Directory containing the data
        split (str): Dataset split ('train', 'val', or 'test')
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.data = []
        self.labels = []
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Dictionary containing 'data' and 'label'
        """
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def load_data(self):
        """Load data from disk. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset."""
        return len(np.unique(self.labels))
    
    def get_statistics(self):
        """Get dataset statistics."""
        return {
            'num_samples': len(self.data),
            'num_classes': self.get_num_classes(),
            'split': self.split
        }
