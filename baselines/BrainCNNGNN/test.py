'''
Author: ViolinSolo
Date: 2025-11-06 10:49:03
LastEditTime: 2025-11-06 11:49:50
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/test.py
'''

import os
import yaml
from omegaconf import OmegaConf
import argparse
import torch
import numpy as np

from models import BrainNetCNN, BrainGNN, BrainGNNVanilla
from datasets import BrainDataset, fmri_collate_fn, FMRIAdjDataset, FMRIGraphDataset
from tasks import ClassificationTask, RegressionTask, GraphClassificationTask
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
        test_dataset = BrainDataset(data_dir, split='test', num_nodes=num_nodes)
    
    return test_dataset


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
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return task


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
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
    test_dataset = get_datasets(config)
    logger.info(f"Test samples: {len(test_dataset)} - type: {type(test_dataset)}")
    
    # Get task
    task = get_task(model, test_dataset, test_dataset, test_dataset, config, logger=logger)
    logger.info(f"Task: {config['task']['name']}")
    
    # # Train
    # logger.info("Starting training...")
    # history = task.train()
    
    # # Visualize results
    # visualize_results(history, save_dir=config.get('log_dir', 'logs'))
    # logger.info("Training completed!")
    
    # Test
    if test_dataset:
        logger.info("Evaluating on test set...")
        if config['task']['name'] == 'classification' or config['task']['name'] == 'graph_classification':
            test_acc, test_f1 = task.test(load_best=not args.use_last)
            logger.info(f"Test Accuracy: {test_acc:.2f}%, Test F1: {test_f1:.2f}")
        else:
            test_mse = task.test()
            logger.info(f"Test MSE: {test_mse:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain network models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--use_last', action='store_true', help='Use last checkpoint instead of best for testing')
    args = parser.parse_args()
    
    main(args)
