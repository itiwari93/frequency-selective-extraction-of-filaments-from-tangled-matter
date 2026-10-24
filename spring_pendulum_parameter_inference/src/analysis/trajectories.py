"""
Functions for analyzing trajectory data, including decomposition into radial and tangential components
and computation of various trajectory observables.
"""

import numpy as np

def decompose_radial_tangential(x_vec_series, u_vec_series, x0=None, eps=1e-12, return_unit_vectors=False):
    """
    Decompose a vector time series into radial and tangential vector components relative to a reference point.

    For each sample i, define the radial direction using the displacement
        r_vec[i] = x_vec_series[i] - x0
    with unit vector
        r_hat[i] = r_vec[i] / ||r_vec[i]||,
    then decompose
        ur_vec_series[i] = (u_vec_series[i] · r_hat[i]) * r_hat[i]   (radial vector component)
        ut_vec_series[i] = u_vec_series[i] - ur_vec_series[i]        (tangential vector component)

    Parameters:
    -----------
    x_vec_series : array
        Position vectors used to define the radial directions. Shape (N, 3).
    u_vec_series : array
        Vector series to decompose (e.g., velocity or acceleration). Shape (N, 3).
    x0 : array, optional
        Reference point defining the origin for the radial direction. Default is [0, 0, 0].
        Note: this is distinct from any equilibrium point used for displacement metrics.
    eps : float, optional
        Small threshold for handling samples with ||x - x0|| close to zero. Default: 1e-12.
        For samples with radius < eps, outputs are set to NaN.
    return_unit_vectors : bool, optional
        If True, also return r_hat. Default: False.

    Returns:
    --------
    dict : Dictionary containing:
        - 'ur_vec_series'   : radial vector component (N, 3)
        - 'ut_vec_series'   : tangential vector component (N, 3)
        If return_unit_vectors is True, also includes:
        - 'rhat_vec_series' : radial unit vectors (N, 3)
    """
    # Set origin to zero if not provided
    if x0 is None:
        x0 = np.array([0., 0., 0.])
    
    # Compute radial vectors from origin
    r_vec_series = x_vec_series - x0
    r_mag_series = np.linalg.norm(r_vec_series, axis=1, keepdims=True)
    
    # Initialize outputs with NaN
    N = len(x_vec_series)
    rhat_vec_series = np.full((N, 3), np.nan)
    ur_vec_series = np.full((N, 3), np.nan)
    ut_vec_series = np.full((N, 3), np.nan)
    
    # Mask for valid samples (radius > eps)
    valid_mask = (r_mag_series[:, 0] > eps)
    
    # Compute radial unit vectors for valid samples
    rhat_vec_series[valid_mask] = r_vec_series[valid_mask] / r_mag_series[valid_mask]
    
    # Radial component (vector projection)
    u_dot_rhat = np.sum(u_vec_series[valid_mask] * rhat_vec_series[valid_mask], axis=1, keepdims=True)
    ur_vec_series[valid_mask] = u_dot_rhat * rhat_vec_series[valid_mask]
    
    # Tangential component (vector rejection)
    ut_vec_series[valid_mask] = u_vec_series[valid_mask] - ur_vec_series[valid_mask]
    
    # Prepare return dictionary
    results = {
        'ur_vec_series': ur_vec_series,
        'ut_vec_series': ut_vec_series
    }
    
    if return_unit_vectors:
        results['rhat_vec_series'] = rhat_vec_series
    
    return results

def compute_trajectory_observables(x_vec_series=None, v_vec_series=None, a_vec_series=None, x_eq=None, observables=['ms_vel'], key_suffix=''):
    """
    Compute observables for trajectory data.
    
    Parameters:
    -----------
    x_vec_series : array
        Position vectors
    v_vec_series : array
        Velocity vectors
    a_vec_series : array, optional
        Acceleration vectors. Required for acceleration-based observables.
    x_eq : array, optional
        Equilibrium position for displacement-based metrics. Default is origin [0, 0, 0].
    observables : list, optional
        List of observable names to compute. Options:
        - 'ma_pos', 'ma_vel', 'ma_acc' (mean absolute)
        - 'ms_pos', 'ms_vel', 'ms_acc' (mean squared)
        - 'rms_pos', 'rms_vel', 'rms_acc' (root mean squared)
        Default: ['ms_vel']
    key_suffix : str, optional
        Suffix to append to all returned keys (e.g., '_rad', '_tan', '_rel'). Default: ''
        
    Returns:
    --------
    dict : Dictionary of computed observables (with optional keys appended with suffix)
    """
    results = {}
    
    # Set equilibrium position to origin if not provided
    if x_eq is None:
        x_eq = np.array([0., 0., 0.])
    
    # Position-based
    if any(observable in observables for observable in ['ma_pos', 'ms_pos', 'rms_pos']):
        if x_vec_series is None:
            raise ValueError("x_vec_series must be provided for position-based observables")
        if 'ma_pos' in observables:
            results['ma_pos'] = np.mean(np.linalg.norm(x_vec_series - x_eq, axis=1))
        if 'ms_pos' in observables:
            results['ms_pos'] = np.mean(np.linalg.norm(x_vec_series - x_eq, axis=1)**2)
        if 'rms_pos' in observables:
            results['rms_pos'] = np.sqrt(np.mean(np.linalg.norm(x_vec_series - x_eq, axis=1)**2))
    
    # Velocity-based
    if any(observable in observables for observable in ['ma_vel', 'ms_vel', 'rms_vel']):
        if v_vec_series is None:
            raise ValueError("v_vec_series must be provided for velocity-based observables")
        if 'ma_vel' in observables:
            results['ma_vel'] = np.mean(np.linalg.norm(v_vec_series, axis=1))
        if 'ms_vel' in observables:
            results['ms_vel'] = np.mean(np.linalg.norm(v_vec_series, axis=1)**2)
        if 'rms_vel' in observables:
            results['rms_vel'] = np.sqrt(np.mean(np.linalg.norm(v_vec_series, axis=1)**2))
    
    # Acceleration-based
    if any(observable in observables for observable in ['ma_acc', 'ms_acc', 'rms_acc']):
        if a_vec_series is None:
            raise ValueError("a_vec_series must be provided for acceleration-based observables")
        if 'ma_acc' in observables:
            results['ma_acc'] = np.mean(np.linalg.norm(a_vec_series, axis=1))
        if 'ms_acc' in observables:
            results['ms_acc'] = np.mean(np.linalg.norm(a_vec_series, axis=1)**2)
        if 'rms_acc' in observables:
            results['rms_acc'] = np.sqrt(np.mean(np.linalg.norm(a_vec_series, axis=1)**2))
    
    # Apply key suffix if provided
    if key_suffix:
        results = {key + key_suffix: value for key, value in results.items()}
    
    return results
