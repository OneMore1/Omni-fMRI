"""
Tasks package for different brain analysis tasks.
Supports classification, regression, and other tasks.
"""

from .classification import ClassificationTask
from .graph_classification import GraphClassificationTask
from .regression import RegressionTask
from .graph_regression import GraphRegressionTask

__all__ = ['ClassificationTask', 'GraphClassificationTask', 'RegressionTask', 'GraphRegressionTask']
