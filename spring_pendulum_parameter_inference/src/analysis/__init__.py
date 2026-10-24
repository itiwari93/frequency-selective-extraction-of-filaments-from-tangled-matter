"""
Analysis module for mathematical and numerical methods.
"""
from .numerical import differentiate_finitediff, finite_difference_derivative, fourier_derivative, generate_time_intervals
from .trajectories import decompose_radial_tangential, compute_trajectory_observables

# Define what gets imported with "from src.analysis import *"
__all__ = [
    'finite_difference_derivative', 'fourier_derivative', 'generate_time_intervals', 
    'decompose_radial_tangential', 'compute_trajectory_observables'
]