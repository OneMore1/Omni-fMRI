# Adding a New Model to ProjectBrainBaseline

This guide shows how to add a new model to the framework.

## Steps

### 1. Create the Model File

Create a new file in the `models/` directory (e.g., `models/my_new_model.py`):

```python
"""
My New Model for brain network analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNewModel(nn.Module):
    """
    Description of your model.
    
    Args:
        num_nodes (int): Number of nodes/ROIs in the brain network
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, num_nodes=90, num_features=1, num_classes=2, 
                 hidden_dim=128, dropout=0.5):
        super(MyNewModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Define your layers here
        self.conv1 = nn.Conv2d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, num_nodes, num_nodes)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Implement your forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x
    
    def get_num_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 2. Register the Model

After creating your model file, update `models/__init__.py` to include your new model:

```python
"""
Models package for brain network analysis.
Supports BrainNetCNN, BrainGNN, MyNewModel, and other models.
"""

from .brainnetcnn import BrainNetCNN
from .braingnn import BrainGNN
from .my_new_model import MyNewModel  # Add this line

__all__ = ['BrainNetCNN', 'BrainGNN', 'MyNewModel']  # Add your model here
```

### 3. Update Training Script

Update the `get_model()` function in `train.py`:

```python
def get_model(config):
    """Initialize model based on configuration."""
    model_name = config['model']['name']
    model_params = config['model']['params']
    
    if model_name == 'BrainNetCNN':
        model = BrainNetCNN(**model_params)
    elif model_name == 'BrainGNN':
        model = BrainGNN(**model_params)
    elif model_name == 'MyNewModel':  # Add this
        model = MyNewModel(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model
```

### 4. Create Configuration File

Create a configuration file in `configs/` (e.g., `configs/my_new_model.yaml`):

```yaml
# Configuration for MyNewModel Classification

seed: 42

data:
  data_dir: './data'
  num_nodes: 90

model:
  name: 'MyNewModel'
  params:
    num_nodes: 90
    num_features: 1
    num_classes: 2
    hidden_dim: 128
    dropout: 0.5

task:
  name: 'classification'
  params:
    batch_size: 32
    num_epochs: 100
    learning_rate: 0.001
    weight_decay: 0.0005

log_dir: './logs'
checkpoint_dir: './checkpoints'
```

### 5. Run Training

Train your model:

```bash
python train.py --config configs/my_new_model.yaml
```

## Tips

- Make sure your model's `forward()` method is compatible with the data format used by your dataset
- Add proper documentation and comments
- Test your model independently before integrating
- Consider adding unit tests in a test file
- If your model requires special preprocessing, you can add transforms to the dataset

## Example: Adding a Graph-based Model

If your model works with graph data (like BrainGNN), you may need to modify the dataset to return graph data format. See `datasets/brain_dataset.py` for the `get_edge_index()` method that converts connectivity matrices to edge format.
