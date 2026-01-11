# ProjectBrainBaseline

A PyTorch-based deep learning framework for brain network analysis with support for multiple datasets, models, and tasks.

## Features

- **Multiple Model Support**: BrainNetCNN, BrainGNN, and extensible architecture for additional models
- **Multiple Task Support**: Classification and regression tasks with extensible task framework
- **Multiple Dataset Support**: Flexible dataset loaders for brain connectivity data
- **Modular Design**: Clean separation of models, datasets, tasks, and utilities
- **Easy Configuration**: YAML-based configuration files for experiments
- **Comprehensive Utilities**: Logging, metrics computation, and visualization tools

## Project Structure

```
ProjectBrainBaseline/
├── models/                 # Neural network models
│   ├── __init__.py
│   ├── brainnetcnn.py     # BrainNetCNN implementation
│   └── braingnn.py        # BrainGNN implementation
├── datasets/              # Dataset loaders
│   ├── __init__.py
│   ├── base_dataset.py    # Base dataset class
│   └── brain_dataset.py   # Brain connectivity dataset
├── tasks/                 # Task definitions
│   ├── __init__.py
│   ├── classification.py  # Classification task
│   └── regression.py      # Regression task
├── configs/               # Configuration files
│   ├── brainnetcnn_classification.yaml
│   ├── braingnn_classification.yaml
│   └── brainnetcnn_regression.yaml
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── logger.py          # Logging utilities
│   ├── metrics.py         # Metrics computation
│   └── visualization.py   # Visualization tools
├── data/                  # Data directory (place your datasets here)
├── logs/                  # Training logs and visualizations
├── checkpoints/           # Model checkpoints
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/iViolinSolo/ProjectBrainBaseline.git
cd ProjectBrainBaseline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Place your brain connectivity data in the `data/` directory. The expected format is:
- `train_data.npy`: Training connectivity matrices (shape: [num_samples, num_nodes, num_nodes])
- `train_labels.npy`: Training labels (shape: [num_samples])
- `val_data.npy`: Validation connectivity matrices
- `val_labels.npy`: Validation labels
- `test_data.npy`: Test connectivity matrices
- `test_labels.npy`: Test labels

If no data is provided, the framework will generate dummy data for demonstration.

### 2. Train a Model

**BrainNetCNN for Classification:**
```bash
python train.py --config configs/brainnetcnn_classification.yaml
```

**BrainGNN for Classification:**
```bash
python train.py --config configs/braingnn_classification.yaml
```

**BrainNetCNN for Regression:**
```bash
python train.py --config configs/brainnetcnn_regression.yaml
```

### 3. View Results

Training logs and visualizations will be saved in the `logs/` directory:
- `loss_curve.png`: Training and validation loss curves
- `accuracy_curve.png`: Training and validation accuracy curves (classification)
- `*.log`: Detailed training logs

Model checkpoints will be saved in the `checkpoints/` directory.

## Supported Models

### BrainNetCNN
Convolutional Neural Network designed for brain connectivity matrices. Features:
- Edge-to-Edge (E2E) convolutional layers
- Edge-to-Node (E2N) convolutional layers
- Node-to-Graph (N2G) pooling
- Fully connected classification/regression head

**Reference:** Kawahara et al., "BrainNetCNN: Convolutional neural networks for brain networks"

### BrainGNN
Graph Neural Network for brain network analysis. Features:
- GCN or GAT layers for graph convolution
- Flexible number of layers
- Global mean/max pooling
- MLP classification/regression head

## Configuration

Configuration files are in YAML format and include:

```yaml
seed: 42                    # Random seed for reproducibility

data:
  data_dir: './data'        # Data directory
  num_nodes: 90             # Number of ROIs

model:
  name: 'BrainNetCNN'       # Model name
  params:                   # Model parameters
    num_nodes: 90
    num_features: 1
    num_classes: 2
    dropout: 0.5

task:
  name: 'classification'    # Task type
  params:                   # Task parameters
    batch_size: 32
    num_epochs: 100
    learning_rate: 0.001
    weight_decay: 0.0005

log_dir: './logs'           # Log directory
checkpoint_dir: './checkpoints'  # Checkpoint directory
```

## Adding New Models

To add a new model:

1. Create a new file in `models/` (e.g., `my_model.py`)
2. Implement your model as a `torch.nn.Module`
3. Add a `get_num_params()` method
4. Import and register in `models/__init__.py`
5. Update `train.py` to support your model
6. Create a configuration file in `configs/`

## Adding New Datasets

To add a new dataset:

1. Create a new file in `datasets/` (e.g., `my_dataset.py`)
2. Inherit from `BaseDataset` or `torch.utils.data.Dataset`
3. Implement `__len__()`, `__getitem__()`, and `load_data()`
4. Import and register in `datasets/__init__.py`

## Adding New Tasks

To add a new task:

1. Create a new file in `tasks/` (e.g., `my_task.py`)
2. Implement training, validation, and testing logic
3. Import and register in `tasks/__init__.py`
4. Update `train.py` to support your task

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- PyYAML >= 6.0

See `requirements.txt` for the complete list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{projectbrainbaseline,
  title = {ProjectBrainBaseline: A PyTorch Framework for Brain Network Analysis},
  author = {iViolinSolo},
  year = {2024},
  url = {https://github.com/iViolinSolo/ProjectBrainBaseline}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.
