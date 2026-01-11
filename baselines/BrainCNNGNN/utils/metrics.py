"""
Metrics computation utilities.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, task='classification'):
    """
    Compute metrics for evaluation.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        task (str): Task type ('classification' or 'regression')
        
    Returns:
        dict: Dictionary of metrics
    """
    if task == 'classification':
        return compute_classification_metrics(y_true, y_pred)
    elif task == 'regression':
        return compute_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task}")


def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary of classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics


def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary of regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': np.mean(np.abs(y_true - y_pred))
    }
    
    return metrics


def print_metrics(metrics):
    """
    Pretty print metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print("="*50)
    
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key:15s}: {value:.4f}")
    
    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
    
    print("="*50 + "\n")
