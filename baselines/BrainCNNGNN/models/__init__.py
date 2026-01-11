"""
Models package for brain network analysis.
Supports BrainNetCNN, BrainGNN, and other models.
"""

from .brainnetcnn import BrainNetCNN
from .braingnn import BrainGNN
from .braingnn_vanilla import Network as BrainGNNVanilla

__all__ = ['BrainNetCNN', 'BrainGNN', 'BrainGNNVanilla']
