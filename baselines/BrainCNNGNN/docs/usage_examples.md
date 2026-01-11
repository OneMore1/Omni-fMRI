# Usage Examples

This document provides common usage examples for ProjectBrainBaseline.

## Basic Usage

### Training BrainNetCNN on Classification Task

```bash
python train.py --config configs/brainnetcnn_classification.yaml
```

### Training BrainGNN on Classification Task

```bash
python train.py --config configs/braingnn_classification.yaml
```

### Training on Regression Task

```bash
python train.py --config configs/brainnetcnn_regression.yaml
```

## Using Custom Data

### Prepare Your Data

1. Create numpy arrays for your connectivity matrices and labels:

```python
import numpy as np

# Training data
train_data = np.random.randn(100, 90, 90)  # 100 samples, 90x90 connectivity matrices
train_labels = np.random.randint(0, 2, 100)  # Binary classification

# Validation data
val_data = np.random.randn(20, 90, 90)
val_labels = np.random.randint(0, 2, 20)

# Test data
test_data = np.random.randn(20, 90, 90)
test_labels = np.random.randint(0, 2, 20)

# Save to disk
np.save('data/train_data.npy', train_data)
np.save('data/train_labels.npy', train_labels)
np.save('data/val_data.npy', val_data)
np.save('data/val_labels.npy', val_labels)
np.save('data/test_data.npy', test_data)
np.save('data/test_labels.npy', test_labels)
```

2. Run training:

```bash
python train.py --config configs/brainnetcnn_classification.yaml
```

## Creating a Custom Configuration

Create a new YAML file in `configs/`:

```yaml
# my_experiment.yaml
seed: 42

data:
  data_dir: './my_data'
  num_nodes: 116  # Different number of ROIs

model:
  name: 'BrainNetCNN'
  params:
    num_nodes: 116
    num_features: 1
    num_classes: 3  # Multi-class classification
    dropout: 0.6

task:
  name: 'classification'
  params:
    batch_size: 16  # Smaller batch size
    num_epochs: 50
    learning_rate: 0.0005  # Different learning rate
    weight_decay: 0.001

log_dir: './my_logs'
checkpoint_dir: './my_checkpoints'
```

Then run:

```bash
python train.py --config configs/my_experiment.yaml
```

## Using Models Programmatically

### BrainNetCNN Example

```python
import torch
from models import BrainNetCNN

# Create model
model = BrainNetCNN(
    num_nodes=90,
    num_features=1,
    num_classes=2,
    dropout=0.5
)

# Sample input: batch of connectivity matrices
batch_size = 8
connectivity_matrices = torch.randn(batch_size, 1, 90, 90)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(connectivity_matrices)
    predictions = outputs.argmax(dim=1)

print(f"Predictions: {predictions}")
```

### BrainGNN Example

```python
import torch
from models import BrainGNN

# Create model
model = BrainGNN(
    num_node_features=1,
    num_classes=2,
    hidden_dim=64,
    num_layers=3,
    dropout=0.5,
    pooling='mean',
    use_attention=False
)

# Sample graph input
num_nodes = 90
node_features = torch.randn(num_nodes, 1)
edge_index = torch.randint(0, num_nodes, (2, 200))
edge_weight = torch.randn(200)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(node_features, edge_index, edge_weight)
    predictions = outputs.argmax(dim=1)

print(f"Predictions: {predictions}")
```

## Using Dataset Directly

```python
from datasets import BrainDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = BrainDataset(
    data_dir='./data',
    split='train',
    num_nodes=90
)

# Create data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through data
for batch in loader:
    connectivity = batch['data']  # Shape: [batch_size, 1, 90, 90]
    labels = batch['label']       # Shape: [batch_size]
    
    print(f"Batch connectivity shape: {connectivity.shape}")
    print(f"Batch labels shape: {labels.shape}")
    break
```

## Evaluating a Trained Model

```python
import torch
from models import BrainNetCNN
from datasets import BrainDataset
from utils import compute_metrics

# Load model
model = BrainNetCNN(num_nodes=90, num_features=1, num_classes=2, dropout=0.5)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test dataset
test_dataset = BrainDataset('./data', split='test', num_nodes=90)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
all_preds = []
all_labels = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    for batch in test_loader:
        data = batch['data'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(data)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
metrics = compute_metrics(all_labels, all_preds, task='classification')
print(metrics)
```

## Visualizing Results

```python
from utils import visualize_results, visualize_connectivity_matrix
import numpy as np

# Load training history from checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')
history = checkpoint['history']

# Visualize training curves
visualize_results(history, save_dir='./my_visualizations')

# Visualize a connectivity matrix
dataset = BrainDataset('./data', split='train', num_nodes=90)
connectivity = dataset.get_connectivity_matrix(0)
visualize_connectivity_matrix(connectivity, save_path='connectivity_example.png')
```

## Running Tests

```bash
# Run the framework test script
python test_framework.py
```

This will test:
- BrainNetCNN model
- BrainGNN model
- Dataset loading
- Training pipeline

## Common Issues

### Out of Memory

If you encounter out-of-memory errors, try:
- Reducing `batch_size` in the config
- Using a smaller model or fewer layers
- Reducing the number of nodes if possible

### Slow Training

To speed up training:
- Increase `batch_size` if memory allows
- Use GPU by ensuring CUDA is available
- Reduce `num_epochs` for quick experiments
- Use fewer `num_layers` in BrainGNN

### Custom Number of ROIs

If your brain network has a different number of ROIs (e.g., 116 instead of 90):
- Update `num_nodes` in both `data` and `model` sections of the config
- Ensure your data files have the correct dimensions
