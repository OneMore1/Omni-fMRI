"""
Example script demonstrating how to use the framework.
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from models import BrainNetCNN, BrainGNN
from datasets import BrainDataset
from tasks import ClassificationTask


def test_brainnetcnn():
    """Test BrainNetCNN model."""
    print("Testing BrainNetCNN...")
    
    # Create model
    model = BrainNetCNN(num_nodes=90, num_features=1, num_classes=2, dropout=0.5)
    print(f"Model created with {model.get_num_params()} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 90, 90)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("BrainNetCNN test passed!\n")


def test_braingnn():
    """Test BrainGNN model."""
    print("Testing BrainGNN...")
    
    # Create model
    model = BrainGNN(num_node_features=1, num_classes=2, hidden_dim=64, num_layers=3)
    model.eval()  # Set to eval mode to avoid batch norm issues
    print(f"Model created with {model.get_num_params()} parameters")
    
    # Test forward pass
    num_nodes = 90
    x = torch.randn(num_nodes, 1)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, 200))  # Random edges
    edge_weight = torch.randn(200)
    
    output = model(x, edge_index, edge_weight)
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {output.shape}")
    print("BrainGNN test passed!\n")


def test_dataset():
    """Test dataset loading."""
    print("Testing BrainDataset...")
    
    # Create dataset (will create dummy data)
    dataset = BrainDataset(data_dir='./data', split='train', num_nodes=90)
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Test data loading
    sample = dataset[0]
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample label: {sample['label']}")
    print("Dataset test passed!\n")


def test_training():
    """Test training pipeline."""
    print("Testing training pipeline...")
    
    # Create datasets
    train_dataset = BrainDataset(data_dir='./data', split='train', num_nodes=90)
    val_dataset = BrainDataset(data_dir='./data', split='val', num_nodes=90)
    
    # Create model
    model = BrainNetCNN(num_nodes=90, num_features=1, num_classes=2, dropout=0.5)
    
    # Create task with minimal epochs
    config = {
        'batch_size': 16,
        'num_epochs': 2,  # Just 2 epochs for testing
        'learning_rate': 0.001,
        'weight_decay': 5e-4
    }
    
    task = ClassificationTask(model, train_dataset, val_dataset, config=config)
    print("Training for 2 epochs...")
    history = task.train()
    
    print(f"Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train acc: {history['train_acc'][-1]:.2f}%")
    print("Training pipeline test passed!\n")


if __name__ == '__main__':
    print("="*60)
    print("Running ProjectBrainBaseline Tests")
    print("="*60 + "\n")
    
    # Test individual components
    test_brainnetcnn()
    test_braingnn()
    test_dataset()
    test_training()
    
    print("="*60)
    print("All tests passed!")
    print("="*60)
