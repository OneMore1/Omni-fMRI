"""
Main training script for brain network analysis.
"""

import os
import yaml
from omegaconf import OmegaConf
import argparse
import torch
import numpy as np

from models import BrainNetCNN, BrainGNN, BrainGNNVanilla
from datasets import BrainDataset, fmri_collate_fn, FMRIAdjDataset, FMRIGraphDataset
from tasks import ClassificationTask, RegressionTask, GraphClassificationTask, GraphRegressionTask
from utils import setup_logger, compute_metrics, visualize_results


def load_config(config_path):
    """Load configuration from YAML file."""
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    config = OmegaConf.load(config_path)
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(config):
    """Initialize model based on configuration."""
    model_name = config['model']['name']
    model_params = config['model']['params']
    
    if model_name == 'BrainNetCNN':
        model = BrainNetCNN(**model_params)
    elif model_name == 'BrainGNN':
        model = BrainGNN(**model_params)
    elif model_name == 'BrainGNNVanilla':
        model = BrainGNNVanilla(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_datasets(config):
    """Load datasets based on configuration."""

    if config['data'].get('dataset_class', None) == 'FMRIAdjDataset':
        train_dataset = FMRIAdjDataset(
            paths_txt=config['data']['train_paths_txt'],
            labels_csv=config['data']['train_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            mmap_mode='r',
            transform=None,
            return_path=False,
            dtype=np.float32
        )
        val_dataset = FMRIAdjDataset(
            paths_txt=config['data']['val_paths_txt'],
            labels_csv=config['data']['val_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            mmap_mode='r',
            transform=None,
            return_path=False,
            dtype=np.float32
        )
        test_dataset = FMRIAdjDataset(
            paths_txt=config['data']['test_paths_txt'],
            labels_csv=config['data']['test_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            mmap_mode='r',
            transform=None,
            return_path=False,
            dtype=np.float32
        )
    elif config['data'].get('dataset_class', None) == 'FMRIGraphDataset':
        train_dataset = FMRIGraphDataset(
            root=config['data']['data_dir'],
            paths_txt=config['data']['train_paths_txt'],
            labels_csv=config['data']['train_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            transform=None,
            dtype=np.float32,
            pre_transform=None,
            use_adj_as_x=config['data'].get('use_adj_as_x', True)
        )
        val_dataset = FMRIGraphDataset(
            root=config['data']['data_dir'],
            paths_txt=config['data']['val_paths_txt'],
            labels_csv=config['data']['val_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            transform=None,
            dtype=np.float32,
            pre_transform=None,
            use_adj_as_x=config['data'].get('use_adj_as_x', True)
        )
        test_dataset = FMRIGraphDataset(
            root=config['data']['data_dir'],
            paths_txt=config['data']['test_paths_txt'],
            labels_csv=config['data']['test_labels_csv'],
            path_col=config['data'].get('path_col', 'path'),
            label_col=config['data'].get('label_col', 'label'),
            transform=None,
            dtype=np.float32,
            pre_transform=None,
            use_adj_as_x=config['data'].get('use_adj_as_x', True)
        )
    else:
        data_dir = config['data']['data_dir']
        num_nodes = config['data'].get('num_nodes', 90)
        train_dataset = BrainDataset(data_dir, split='train', num_nodes=num_nodes)
        val_dataset = BrainDataset(data_dir, split='val', num_nodes=num_nodes)
        test_dataset = BrainDataset(data_dir, split='test', num_nodes=num_nodes)
    
    return train_dataset, val_dataset, test_dataset


def get_task(model, train_dataset, val_dataset, test_dataset, config, logger=None):
    """Initialize task based on configuration."""
    task_name = config['task']['name']
    # task_config = config['task']['params']
    
    if task_name == 'classification':
        task = ClassificationTask(model, train_dataset, val_dataset, test_dataset, config, logger=logger)
    elif task_name == 'graph_classification':
        task = GraphClassificationTask(model, train_dataset, val_dataset, test_dataset, config, logger=logger)
    elif task_name == 'regression':
        task = RegressionTask(model, train_dataset, val_dataset, test_dataset, config, logger=logger)
    elif task_name == 'graph_regression':
        task = GraphRegressionTask(model, train_dataset, val_dataset, test_dataset, config, logger=logger)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return task


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Override config parameters with command line arguments
    if args.downstream_dataset is not None:
        config['downstream_dataset'] = args.downstream_dataset
    if args.upstream_model is not None:
        config['upstream_model'] = args.upstream_model
    if args.num_rois is not None:
        config['n_roi'] = args.num_rois
    if args.batch_size is not None:
        config['task']['params']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['task']['params']['learning_rate'] = args.lr
    if args.wd is not None:
        config['task']['params']['weight_decay'] = args.wd
    if args.seed is not None:
        config['seed'] = args.seed
    if args.label_col is not None:
        config['data']['label_col'] = args.label_col
    if args.n_classes is not None:
        if 'nclass' in config['model']['params']:  # braingnn config style
            config['model']['params']['nclass'] = args.n_classes
        if 'num_classes' in config['model']['params']: # brainnetcnn config style
            config['model']['params']['num_classes'] = args.n_classes

    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup logger
    logger = setup_logger('train', log_dir=config.get('log_dir', 'logs'))
    logger.info(f"Configuration: {config}")
    
    # Get model
    model = get_model(config)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Number of parameters: {model.get_num_params()}")
    
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    logger.info(f"Train samples: {len(train_dataset)} - type: {type(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)} - type: {type(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)} - type: {type(test_dataset)}")
    
    # Get task
    task = get_task(model, train_dataset, val_dataset, test_dataset, config, logger=logger)
    logger.info(f"Task: {config['task']['name']}")
    
    # Train
    logger.info("Starting training...")
    history = task.train()
    
    # Visualize results
    visualize_results(history, save_dir=config.get('log_dir', 'logs'))
    logger.info("Training completed!")
    
    # Test
    if test_dataset:
        logger.info("Evaluating on test set...")
        if config['task']['name'] == 'classification' or config['task']['name'] == 'graph_classification':
            test_acc, test_f1 = task.test()
            logger.info(f"Test Accuracy: {test_acc:.2f}%, Test F1: {test_f1:.2f}")
        else:
            test_mse, test_r2 = task.test()
            logger.info(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain network models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--downstream_dataset', type=str, default=None, help='Override downstream dataset from config')
    parser.add_argument('--upstream_model', type=str, default=None, help='Override upstream model from config')
    parser.add_argument('--num_rois', type=int, default=None, help='Override number of ROIs from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--wd', type=float, default=None, help='Override weight decay from config')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed from config')
    parser.add_argument('--label_col', type=str, default=None, help='Override label column name from config')
    parser.add_argument('--n_classes', type=int, default=None, help='Override number of classes from config')
    args = parser.parse_args()
    
    main(args)
