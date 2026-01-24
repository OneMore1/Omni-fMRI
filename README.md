# 

This repository contains the official implementation of Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model, which introduces a dynamic patching mechanism that significantly reduces computational costs while preserving informative spatial structures.

<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

---

## Installation

Setting up the environment requires Python 3.13 and CUDA-compatible PyTorch for GPU acceleration:



## Project Structure

The codebase is organized into modular components for easy navigation and extension:

```

```


## Data Preparation

### Preprocessing Pipeline

We provide the end-to-end data preparation pipeline under ```data_preparation/```. Volumes were resampled with cubic spline interpolation to a $96\times96\times96$ grid at 2 mm isotropic resolution in MNI space. Time series with TR outside 0.7–0.8 s were voxel-wise resampled to 0.72 s with cubic splines, and signals were globally z-scored within the brain mask. We followed the preprocessing settings in [Swift](https://github.com/Transconnectome/SwiFT).

### Test Data Structure

The repository includes randomly generated placeholder data provided only for structural reference:

```
test_data/
├── 100001__REST1_LR_hp2000_clean/  # 10 synthetic subjects
├── train_list.txt                  # Training split
├── val_list.txt                    # Validation split  
├── test_list.txt                   # Test split
├── fake_labels.csv                 # Synthetic labels
└── fake_checkpoint.pth             # Placeholder checkpoint
```

## Training

### Pre-training

Self-supervised pre-training learns general representations from unlabeled fMRI data using masked prediction tasks:

```bash
# Basic training with default settings and test data
./run_training.sh

# Custom configuration with specific parameters
./run_training.sh --batch-size 8 --num-epochs 200 --save-dir ./checkpoints/my_experiment

# Resume training from a previous checkpoint
./run_training.sh --resume ./checkpoints/my_experiment/checkpoint.pth

# Training with specific GPU
CUDA_VISIBLE_DEVICES=1 ./run_training.sh --batch-size 4

# Debug mode with reduced epochs for testing
./run_training.sh --debug --num-epochs 5
```

### Downstream Tasks

Fine-tune or Linear probe pre-trained models on specific neuroimaging classification tasks or extract features for custom analysis:

#### Available Tasks

The framework supports several standard neuroimaging prediction tasks:

```bash
# Gender classification (binary classification)
./run_downstream.sh --task gender

# Age group classification (multi-class classification)
./run_downstream.sh --task age_group
```

#### Feature Extraction

Extract learned representations for custom downstream analysis:

```bash
# Extract features for all data splits
./run_downstream.sh --extract-features

# Extract features for specific split only
./run_downstream.sh --extract-features --task gender

# Alternative: Direct Python execution
python downstream.py --task gender --extract_feature
```

#### Model Checkpoints

Our pre-trained model weights can be found in the checkpoints directory:  `./checkpoints/best_model.pth`




