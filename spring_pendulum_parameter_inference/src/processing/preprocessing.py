"""
Preprocessing functions for experimental tangle data.

This module provides functions to load and standardize experimental data
from MATLAB files for the tangled fibers project.
"""

import numpy as np
import h5py
from pathlib import Path
from src.utilities import save_datasets_to_label


def tangle_data_load(
    data_file: Path,
    exp_name_mapping: dict,
    exp_params: dict,
    cos_sig: np.ndarray,
    piston_left: np.ndarray,
    piston_ref: np.ndarray,
    video_results_key: str = 'videoResults',
    verbose: bool = True
) -> dict:
    """
    Load and process experimental data from MATLAB file.
    
    This function reads dual-camera tracking data from a MATLAB HDF5 file,
    extracts time series and coordinate data, removes NaNs, and organizes
    into a standardized dictionary structure.
    
    Parameters
    ----------
    data_file : Path
        Path to the MATLAB data.mat file
    exp_name_mapping : dict
        Mapping from original experiment names to clean names
    exp_params : dict
        Dictionary with experiment parameters for each clean name
    cos_sig : np.ndarray
        Cosine signal array for piston motion
    piston_left : np.ndarray
        Piston left data array
    piston_ref : np.ndarray
        Piston reference coordinates array
    video_results_key : str, optional
        Key name for video results in MATLAB file.
        Use 'videoResults' for tangle_00, 'videoResults1' for tangle_01.
        Default: 'videoResults'
    verbose : bool, optional
        Print progress information. Default: True
        
    Returns
    -------
    processed_datasets : dict
        Dictionary mapping clean experiment names to processed data.
        Each entry contains:
        - 'time_series': 1D array of time points
        - 'data_series_left': 2D array (10, n_points) from left camera
        - 'data_series_right': 2D array (10, n_points) from right camera
        - 'pist_data_series': 2D array (2, n_points) of piston data
        - 'length_scale_left': float, calibration scale
        - 'length_scale_right': float, calibration scale
        - 'piston_ref_left': array, left reference coordinates
        - 'piston_ref_right': array, right reference coordinates
        - 'exp_param': dict with full experiment metadata
        
    Notes
    -----
    Camera data indices:
    - Left camera: [Avg_Z, Variance_Z, Skewness_Z, Kurtosis_Z, 
                    Avg_Y, Variance_Y, Skewness_Y, Kurtosis_Y, Cov_ZY, Signal]
    - Right camera: [Avg_X, Variance_X, Skewness_X, Kurtosis_X,
                     Avg_Y, Variance_Y, Skewness_Y, Kurtosis_Y, Cov_XY, Signal]
    """
    processed_datasets = {}
    
    with h5py.File(data_file, 'r') as mat_data:
        if verbose:
            print(f"Keys in data.mat: {list(mat_data.keys())}")
        
        video_results = mat_data[video_results_key]
        N_exp = len(video_results['Filename'])
        
        if verbose:
            print(f"Found {N_exp} experiments in data.mat")
            print("original_name".ljust(20), "\tclean_name".ljust(12), 
                  "\ttotal_time".ljust(10), "\tstatus")
        
        for n_exp in range(N_exp):
            # Extract original experiment name from matlab file
            original_exp_name = mat_data[video_results['Filename'][n_exp][0]][()]
            original_exp_name = ''.join(chr(x[0]) for x in original_exp_name).replace('.mp4','').replace('.','-')
            
            # Check if we have parameters for this experiment
            if original_exp_name not in exp_name_mapping:
                if verbose:
                    print(f"{original_exp_name.ljust(20)}\t{'N/A'.ljust(12)}\t{'N/A'.ljust(10)}\tSKIPPED (no parameters)")
                continue
            
            # Get clean name and parameters
            clean_exp_name = exp_name_mapping[original_exp_name]
            exp_param = exp_params[clean_exp_name]
            
            # Extract time series
            time_series = mat_data[video_results['Time'][n_exp][0]][()].flatten()
            
            # Extract left camera data
            data_keys_left = [
                'Avg_Z', 'Variance_Z', 'Skewness_Z', 'Kurtosis_Z',
                'Avg_Y', 'Variance_Y', 'Skewness_Y', 'Kurtosis_Y',
                'Cov_ZY', 'Signal']
            data_series_left = [mat_data[video_results['Left'][n_exp][0]][key][()].flatten() 
                               for key in data_keys_left]
            data_series_left = np.vstack(data_series_left)
            
            # Extract right camera data
            data_keys_right = [
                'Avg_X', 'Variance_X', 'Skewness_X', 'Kurtosis_X',
                'Avg_Y', 'Variance_Y', 'Skewness_Y', 'Kurtosis_Y',
                'Cov_XY', 'Signal']
            data_series_right = [mat_data[video_results['Right'][n_exp][0]][key][()].flatten() 
                                for key in data_keys_right]
            data_series_right = np.vstack(data_series_right)
            
            # Extract piston data
            pist_data_series = np.vstack([cos_sig[0][n_exp][0], piston_left[0][n_exp][0]])
            piston_ref_coord = np.array([piston_ref[0][n_exp][i].item() 
                                        for i in range(len(piston_ref[0][n_exp]))])
            piston_ref_left, piston_ref_right = piston_ref_coord[:2], piston_ref_coord[2:]
            
            # Remove NaNs
            valid_indices = ~np.any(np.isnan(data_series_left) | np.isnan(data_series_right), axis=0)
            time_series = time_series[valid_indices]
            data_series_left = data_series_left[:, valid_indices]
            data_series_right = data_series_right[:, valid_indices]
            pist_data_series = pist_data_series[:, valid_indices]
            
            if verbose:
                print(f"  {original_exp_name.ljust(20)}\t{clean_exp_name.ljust(12)}\t{time_series[-1]:.5f}".ljust(48) + "\tprocessed")
            
            # Store processed data using clean name
            processed_datasets[clean_exp_name] = {
                'time_series': time_series,
                'data_series_left': data_series_left,
                'data_series_right': data_series_right,
                'pist_data_series': pist_data_series,
                'length_scale_left': exp_param['length_scale_left'],
                'length_scale_right': exp_param['length_scale_right'],
                'piston_ref_left': piston_ref_left,
                'piston_ref_right': piston_ref_right,
                'exp_param': exp_param
            }
    
    if verbose:
        print(f"\nSuccessfully processed {len(processed_datasets)} experiments.")
    
    return processed_datasets


def tangle_data_standardize_save(
    processed_datasets: dict,
    exp_annotations: dict,
    frequency_mapping: dict,
    piston_amplitude_scaling: float,
    output_path: Path,
    hdf5_options: dict = None,
    verbose: bool = True
) -> None:
    """
    Convert experimental data to standardized 3D coordinate format and save to HDF5.
    
    This function:
    1. Extracts 3D coordinates from dual-camera data
    2. Centers coordinates at piston origin
    3. Converts units from mm to meters
    4. Creates standardized parameter dictionaries
    5. Saves to HDF5 with compression
    
    Parameters
    ----------
    processed_datasets : dict
        Dictionary from tangle_data_load() mapping experiment names to data
    exp_annotations : dict
        Dictionary mapping experiment names to annotations with:
        - 'data_quality': str ('good', 'bad', etc.)
        - 'extracted': bool
        - 'comments': str
        - 'time_range': tuple (optional)
    frequency_mapping : dict
        Mapping from frequency_type (e.g., 'low', 'opt', 'high') to Hz
    piston_amplitude_scaling : float
        Piston amplitude in mm (will be converted to meters)
    output_path : Path
        Path to output HDF5 file
    hdf5_options : dict, optional
        HDF5 compression options. Default: {'compression':'gzip', 'compression_opts':4}
    verbose : bool, optional
        Print progress information. Default: True
        
    Notes
    -----
    Coordinate system transformation:
    - x: from left camera Avg_Z (index 0) / length_scale_left
    - y: from right camera Avg_X (index 0) / length_scale_right
    - z: -1 * right camera Avg_Y (index 4) / length_scale_right (negated, vertical)
    - Piston x: constant from piston_ref_left[0]
    - Piston y: constant from piston_ref_right[0]
    - Piston z: amplitude * (cos_signal - 1) + piston_ref_right[1]
    - Origin: centered at piston center (midpoint of piston trajectory)
    - Units: converted from mm to meters (divide by 1000)
    """
    if hdf5_options is None:
        hdf5_options = {'compression': 'gzip', 'compression_opts': 4}
    
    if verbose:
        print("Convert to standardized coordinate format and save to HDF5")
        print("exp_name".ljust(12), "\tpoints".ljust(8), "\ttotal_time".ljust(10), "\tstatus")
    
    for exp_name, exp_data in processed_datasets.items():
        # Extract time series and camera data
        time_series = exp_data['time_series']
        data_series_left = exp_data['data_series_left']
        data_series_right = exp_data['data_series_right']
        pist_data_series = exp_data['pist_data_series']
        
        # Extract coordinate data (indices match data_keys_left/right)
        avgZ_left_series = data_series_left[0]    # Avg_Z from left camera -> x coordinate
        avgX_right_series = data_series_right[0]  # Avg_X from right camera -> y coordinate
        avgY_right_series = data_series_right[4]  # Avg_Y from right camera -> z coordinate
        piston_cos_series = pist_data_series[0]   # Cosine signal for piston motion
        
        # Get length scales and piston references
        length_scale_left = exp_data['length_scale_left']
        length_scale_right = exp_data['length_scale_right']
        piston_ref_left = exp_data['piston_ref_left']
        piston_ref_right = exp_data['piston_ref_right']
        
        # Create 3D coordinates for mass (in mm)
        X_m = np.column_stack([
            avgZ_left_series / length_scale_left,      # x coordinate
            avgX_right_series / length_scale_right,    # y coordinate
            -avgY_right_series / length_scale_right    # z coordinate (negated)
        ])
        
        # Create 3D coordinates for piston (in mm)
        n_points = X_m.shape[0]
        X_p = np.column_stack([
            np.ones(n_points) * piston_ref_left[0] / length_scale_left,
            np.ones(n_points) * piston_ref_right[0] / length_scale_right,
            -(piston_amplitude_scaling * (piston_cos_series - 1) + 
              piston_ref_right[1] / length_scale_right)
        ])
        
        # Center piston at origin
        piston_center = (np.max(X_p, axis=0) + np.min(X_p, axis=0)) / 2
        X_m_centered = X_m - piston_center
        X_p_centered = X_p - piston_center
        
        # Convert to standardized format (convert mm to meters)
        datasets = {
            't': time_series,
            'x': X_m_centered[:, 0] / 1000,  # Convert mm to m
            'y': X_m_centered[:, 1] / 1000,
            'z': X_m_centered[:, 2] / 1000,
            'xp': X_p_centered[:, 0] / 1000,
            'yp': X_p_centered[:, 1] / 1000,
            'zp': X_p_centered[:, 2] / 1000
        }
        
        # Create experiment parameters dict
        exp_param = exp_data['exp_param']
        annotation = exp_annotations.get(exp_name, {})
        params = {
            # Clean identifiers
            'exp_name': exp_name,
            'original_name': exp_param['original_name'],
            
            # Experimental conditions
            'tumbling_weights': exp_param['tumbling_weights'],
            'frequency_type': exp_param['frequency_type'],
            'piston_frequency': frequency_mapping[exp_param['frequency_type']],
            'piston_amplitude': piston_amplitude_scaling / 1000,
            
            # Calibration data
            'length_scale_left': length_scale_left,
            'length_scale_right': length_scale_right,
            'piston_ref_left': piston_ref_left.tolist(),
            'piston_ref_right': piston_ref_right.tolist(),
            
            # Data characteristics
            'time_range': annotation.get('time_range', (0, 100)),
            'total_time': float(time_series[-1]),
            'n_points': len(time_series),
            'dt_mean': float(np.mean(np.diff(time_series))),
            
            # Manual annotations
            'extracted': annotation.get('extracted', None),
            'data_quality': annotation.get('data_quality', 'unknown'),
            'comments': annotation.get('comments', '')
        }
        
        # Save to HDF5
        save_datasets_to_label(
            data_path=output_path,
            data_label=exp_name,
            datasets=datasets,
            params=params,
            hdf5_options=hdf5_options
        )
        
        if verbose:
            print(f"{exp_name.ljust(12)}\t{len(time_series):<8}\t{time_series[-1]:<10.5f}\tsaved")
    
    if verbose:
        print(f"\nSaved {len(processed_datasets)} experiments to {output_path}.")


def tangle_data_standardize_save_moments(
    processed_datasets: dict,
    exp_annotations: dict,
    frequency_mapping: dict,
    piston_amplitude_scaling: float,
    output_path: Path,
    exp_key: str = None,
    moment_orders: list = None,
    hdf5_options: dict = None,
    verbose: bool = True
) -> None:
    """
    Convert experimental data to standardized 3D coordinate format with higher moments and save to HDF5.
    
    This function:
    1. Extracts 3D coordinates from dual-camera data
    2. Extracts higher statistical moments (variance, skewness, kurtosis, covariances)
    3. Centers coordinates at piston origin
    4. Converts units from mm to meters (and mm^2 to m^2 for second moments)
    5. Creates standardized parameter dictionaries
    6. Saves to HDF5 with compression
    
    Parameters
    ----------
    processed_datasets : dict
        Dictionary from tangle_data_load() mapping experiment names to data
    exp_annotations : dict
        Dictionary mapping experiment names to annotations with:
        - 'data_quality': str ('good', 'bad', etc.)
        - 'extracted': bool
        - 'comments': str
        - 'time_range': tuple (optional)
    frequency_mapping : dict
        Mapping from frequency_type (e.g., 'low', 'opt', 'high') to Hz
    piston_amplitude_scaling : float
        Piston amplitude in mm (will be converted to meters)
    output_path : Path
        Path to output HDF5 file
    exp_key : str, optional
        Experiment key to use as group prefix (e.g., 'tangle_00', 'tangle_01').
        If provided, data will be saved as 'exp_key/exp_name'. Default: None
    moment_orders : list, optional
        List of moment orders to process (e.g., [2, 3, 4] for variance, skewness, kurtosis).
        First moment (mean) is always processed. Default: [2]
    hdf5_options : dict, optional
        HDF5 compression options. Default: {'compression':'gzip', 'compression_opts':4}
    verbose : bool, optional
        Print progress information. Default: True
        
    Notes
    -----
    Coordinate system transformation:
    - x: from left camera Avg_Z (index 0) / length_scale_left
    - y: from right camera Avg_X (index 0) / length_scale_right
    - z: -1 * right camera Avg_Y (index 4) / length_scale_right (negated, vertical)
    - Piston x: constant from piston_ref_left[0]
    - Piston y: constant from piston_ref_right[0]
    - Piston z: amplitude * (cos_signal - 1) + piston_ref_right[1]
    - Origin: centered at piston center (midpoint of piston trajectory)
    - Units: converted from mm to meters (divide by 1000), mm^2 to m^2 (divide by 1000^2)
    
    Higher moments processing:
    - Second moments (order=2): var_x, var_y, var_z, std_x, std_y, std_z, cov_yz, cov_xz
    - Third moments (order=3): skew_x, skew_y, skew_z (standardized)
    - Fourth moments (order=4): kurt_x, kurt_y, kurt_z (standardized)
    - Sign flips applied for z-coordinate (negated from camera Y)
    """
    if hdf5_options is None:
        hdf5_options = {'compression': 'gzip', 'compression_opts': 4}
    
    if moment_orders is None:
        moment_orders = [2]
    
    if verbose:
        print("Convert to standardized coordinate format and save to HDF5")
        print("exp_name".ljust(12), "\tpoints".ljust(8), "\ttotal_time".ljust(10), "\tstatus")
    
    for exp_name, exp_data in processed_datasets.items():
        ####################################
        # 1. Extract data and length scales
        # Extract time series and camera data
        time_series = exp_data['time_series']
        data_series_left = exp_data['data_series_left']
        data_series_right = exp_data['data_series_right']
        pist_data_series = exp_data['pist_data_series']
        
        # Extract coordinate data (indices match data_keys_left/right)
        avgZ_left_series = data_series_left[0]    # Avg_Z from left camera -> x coordinate
        avgX_right_series = data_series_right[0]  # Avg_X from right camera -> y coordinate
        avgY_right_series = data_series_right[4]  # Avg_Y from right camera -> z coordinate
        piston_cos_series = pist_data_series[0]   # Cosine signal for piston motion

        # Extract higher moments
        if 2 in moment_orders or 3 in moment_orders or 4 in moment_orders:
            # Extract higher moment data from left camera (for x coordinate)
            varZ_left_series = data_series_left[1]   # Variance_Z -> Var(x)
            skewZ_left_series = data_series_left[2]  # Skewness_Z -> Skew(x)
            kurtZ_left_series = data_series_left[3]  # Kurtosis_Z -> Kurt(x)
            covZY_left_series = data_series_left[8]  # Cov_ZY -> Cov(x,z) (sign flip later)
            
            # Extract higher moment data from right camera (for y and z coordinates)
            varX_right_series = data_series_right[1]  # Variance_X -> Var(y)
            skewX_right_series = data_series_right[2] # Skewness_X -> Skew(y)
            kurtX_right_series = data_series_right[3] # Kurtosis_X -> Kurt(y)
            varY_right_series = data_series_right[5]  # Variance_Y -> Var(z)
            skewY_right_series = data_series_right[6] # Skewness_Y -> Skew(z) (sign flip later)
            kurtY_right_series = data_series_right[7] # Kurtosis_Y -> Kurt(z)
            covXY_right_series = data_series_right[8] # Cov_XY -> Cov(y,z) (sign flip later)
        
        # Get length scales and piston references
        length_scale_left = exp_data['length_scale_left']
        length_scale_right = exp_data['length_scale_right']
        piston_ref_left = exp_data['piston_ref_left']
        piston_ref_right = exp_data['piston_ref_right']
        
        ####################################
        # 2. Process average coordinates and piston motion
        # Create 3D coordinates for mass (in mm)
        X_m = np.column_stack([
            avgZ_left_series / length_scale_left,      # x coordinate
            avgX_right_series / length_scale_right,    # y coordinate
            -avgY_right_series / length_scale_right    # z coordinate (negated)
        ])
        
        # Create 3D coordinates for piston (in mm)
        n_points = X_m.shape[0]
        X_p = np.column_stack([
            np.ones(n_points) * piston_ref_left[0] / length_scale_left,
            np.ones(n_points) * piston_ref_right[0] / length_scale_right,
            -(piston_amplitude_scaling * (piston_cos_series - 1) + 
              piston_ref_right[1] / length_scale_right)
        ])
        
        # Center piston at origin
        piston_center = (np.max(X_p, axis=0) + np.min(X_p, axis=0)) / 2
        X_m_centered = X_m - piston_center
        X_p_centered = X_p - piston_center

        # Convert mm to m
        X_m_centered /= 1000
        X_p_centered /= 1000

        ####################################
        # 3. Process higher moments

        # Second moments: variances and covariances
        if 2 in moment_orders:
            var_x = varZ_left_series / (length_scale_left ** 2)
            var_y = varX_right_series / (length_scale_right ** 2)
            var_z = varY_right_series / (length_scale_right ** 2)
            cov_yz = - covXY_right_series / (length_scale_right ** 2)  # Sign flip due to z = -Y
            cov_xz = - covZY_left_series / (length_scale_left ** 2)    # Sign flip due to z = -Y (proxy: left camera ZY approximates global xz)

            # Convert mm^2 to m^2
            var_x /= (1000 ** 2)
            var_y /= (1000 ** 2)
            var_z /= (1000 ** 2)
            cov_yz /= (1000 ** 2)
            cov_xz /= (1000 ** 2)

        # Third moments: skewness (standardized)
        if 3 in moment_orders:
            skew_x = skewZ_left_series
            skew_y = skewX_right_series
            skew_z = - skewY_right_series  # Sign flip due to z = -Y

        # Fourth moments: kurtosis (standardized)
        if 4 in moment_orders:
            kurt_x = kurtZ_left_series
            kurt_y = kurtX_right_series
            kurt_z = kurtY_right_series


        ####################################
        # 4. Prepare datasets and parameters and save to HDF5
        datasets = {
            't': time_series,
            'x': X_m_centered[:, 0],
            'y': X_m_centered[:, 1],
            'z': X_m_centered[:, 2],
            'xp': X_p_centered[:, 0],
            'yp': X_p_centered[:, 1], 
            'zp': X_p_centered[:, 2]
        }
        if 2 in moment_orders:
            datasets.update({
                'std_x': np.sqrt(var_x),
                'std_y': np.sqrt(var_y),
                'std_z': np.sqrt(var_z),
                'var_x': var_x,
                'var_y': var_y,
                'var_z': var_z,
                'cov_yz': cov_yz,
                'cov_xz': cov_xz
            })
        if 3 in moment_orders:
            datasets.update({
                'skew_x': skew_x,
                'skew_y': skew_y,
                'skew_z': skew_z
            })
        if 4 in moment_orders:
            datasets.update({
                'kurt_x': kurt_x,
                'kurt_y': kurt_y,
                'kurt_z': kurt_z
            })
        
        # Create experiment parameters dict
        exp_param = exp_data['exp_param']
        annotation = exp_annotations.get(exp_name, {})
        params = {
            # Clean identifiers
            'exp_name': exp_name,
            'original_name': exp_param['original_name'],
            
            # Experimental conditions
            'tumbling_weights': exp_param['tumbling_weights'],
            'frequency_type': exp_param['frequency_type'],
            'piston_frequency': frequency_mapping[exp_param['frequency_type']],
            'piston_amplitude': piston_amplitude_scaling / 1000,
            
            # Calibration data
            'length_scale_left': length_scale_left,
            'length_scale_right': length_scale_right,
            'piston_ref_left': piston_ref_left.tolist(),
            'piston_ref_right': piston_ref_right.tolist(),
            
            # Data characteristics
            'total_time': float(time_series[-1]),
            'n_points': len(time_series),
            'dt_mean': float(np.mean(np.diff(time_series))),
            
            # Manual annotations
            'extracted': annotation.get('extracted', None),
            'data_quality': annotation.get('data_quality', 'unknown'),
            'comments': annotation.get('comments', '')
        }
        
        # Save to HDF5
        save_datasets_to_label(
            data_path=output_path,
            data_label=exp_key + '/' + exp_name if exp_key else exp_name,
            datasets=datasets,
            params=params,
            hdf5_options=hdf5_options
        )
        
        if verbose:
            print(f"{exp_name.ljust(12)}\t{len(time_series):<8}\t{time_series[-1]:<10.5f}\tsaved")
    
    if verbose:
        print(f"\nSaved {len(processed_datasets)} experiments to {output_path}.")