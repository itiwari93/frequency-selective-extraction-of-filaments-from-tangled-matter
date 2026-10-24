"""
Validation module for cross-validation and model assessment.
"""
from .cross_validation import UniformSplitter
from .bootstrap import generate_bootstrap_samples, compute_bootstrap_statistics
from .metrics import mean_squared_relative_error

__all__ = ['UniformSplitter', 
           'generate_bootstrap_samples', 
           'compute_bootstrap_statistics',
           'mean_squared_relative_error']