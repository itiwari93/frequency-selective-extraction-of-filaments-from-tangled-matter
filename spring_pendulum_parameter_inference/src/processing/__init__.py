"""
Processing module for data cleaning, denoising, 
transformations and feature extraction.
"""
from .preprocessing import (tangle_data_load, tangle_data_standardize_save, tangle_data_standardize_save_moments)
from .features import transform_spring_pendulum_coordinates, generate_spring_pendulum_equations, generate_spring_pendulum_candidates, FourierFeatures
from .denoise import (
    MovingAverageSmoother,
    GaussianKernelSmoother, 
    LocalPolynomialSmoother,
    SplineSmoother,
    FrequencyDomainSmoother,
    add_noise
)

__all__ = [
    'tangle_data_load',
    'tangle_data_standardize_save',
    'tangle_data_standardize_save_moments',
    'transform_spring_pendulum_coordinates',
    'generate_spring_pendulum_equations',
    'generate_spring_pendulum_candidates',
    'FourierFeatures',
    'MovingAverageSmoother',
    'GaussianKernelSmoother', 
    'LocalPolynomialSmoother',
    'SplineSmoother',
    'FrequencyDomainSmoother',
    'add_noise'
]