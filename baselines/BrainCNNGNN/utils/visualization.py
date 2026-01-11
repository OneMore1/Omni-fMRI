"""
Visualization utilities for training and results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_results(history, save_dir='logs'):
    """
    Visualize training history.
    
    Args:
        history (dict): Training history dictionary
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.close()
    
    # Plot training and validation accuracy (for classification)
    if 'train_acc' in history and 'val_acc' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
        plt.close()
    
    # Plot training and validation MSE (for regression)
    if 'train_mse' in history and 'val_mse' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_mse'], label='Train MSE')
        plt.plot(history['val_mse'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'mse_curve.png'))
        plt.close()


def visualize_confusion_matrix(cm, class_names=None, save_path='confusion_matrix.png'):
    """
    Visualize confusion matrix.
    
    Args:
        cm (array-like): Confusion matrix
        class_names (list, optional): List of class names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_connectivity_matrix(matrix, save_path='connectivity_matrix.png'):
    """
    Visualize brain connectivity matrix.
    
    Args:
        matrix (array-like): Connectivity matrix
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Brain Connectivity Matrix')
    plt.xlabel('ROI')
    plt.ylabel('ROI')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_learning_curves(train_scores, val_scores, metric_name='Accuracy', save_path='learning_curves.png'):
    """
    Plot learning curves.
    
    Args:
        train_scores (list): Training scores
        val_scores (list): Validation scores
        metric_name (str): Name of the metric
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
