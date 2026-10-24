import numpy as np
import pandas as pd
import h5py
import os
import shutil
import json
from natsort import natsorted

# %% Saving and reading datasets

def save_datasets_to_label(
    data_path: str,
    data_label: str,
    datasets: dict[str, any],
    params: dict[str, any] = None,
    overwrite: bool = True,
    hdf5_options: dict | None = None
):
    '''Save datasets and parameters to an HDF5 file under the specified label.'''
    with h5py.File(data_path, 'a') as f:
        
        # Check if the group already exists
        if data_label not in f: group = f.create_group(data_label)
        elif overwrite:
            del f[data_label]
            group = f.create_group(data_label)
        else: group = f[data_label]
        
        # Set default HDF5 options if not provided
        if hdf5_options is None: 
            hdf5_options = {}
        if 'compression' in hdf5_options and 'chunks' not in hdf5_options:
            hdf5_options['chunks'] = True

        # Save each dataset in the group
        for key, value in datasets.items(): 
            if key in group and not overwrite: 
                continue
            group.create_dataset(key, data=np.asarray(value), **hdf5_options)  # Convert to numpy array
        
        # Save parameters as JSON string in group attributes 
        if params is not None:
            safe_params = to_serializable(params)  # Convert to a serializable format
            if overwrite or 'params' not in group.attrs:
                group.attrs['params'] = json.dumps(safe_params)

def read_datasets_from_label(
    data_path: str,
    data_label: str
) -> tuple[dict[str, np.ndarray], dict[str, any]]:
    '''Read datasets and parameters from an HDF5 file under the specified label.'''
    with h5py.File(data_path, 'r') as f:
        if data_label not in f:
            return {}, {}
        group = f[data_label]

        # Read each dataset (skip subgroups)
        datasets = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):  # Only read datasets, skip groups
                datasets[key] = group[key][...]

        # Read parameters from JSON‐serialized attrs
        raw = group.attrs.get('params', '{}')
        params = json.loads(raw)

    return datasets, params


# %% Saving and reading denoised datasets

def create_denoise_config(coordinate_keys, passthrough_keys, smoother_pipeline, smoother_cv):
    """Create denoise configuration dictionary from pipeline objects."""
    return {
        'coordinate_keys': coordinate_keys,
        'passthrough_keys': passthrough_keys,
        'smoother_pipeline': {
            'name': smoother_pipeline.named_steps['smoother'].__class__.__name__,
            'params': smoother_pipeline.named_steps['smoother'].get_params()
        },
        'smoother_cv': {
            'param_grid': smoother_cv.param_grid,
            'cv': {
                'type': smoother_cv.cv.__class__.__name__,
                **vars(smoother_cv.cv)
            },
            'scoring': smoother_cv.scoring,
            'return_train_score': smoother_cv.return_train_score,
            'n_jobs': smoother_cv.n_jobs
        }
    }

def save_denoise_results(data_path, data_label, denoised_datasets, data_params, denoise_config, coordinate_cv_results, **kwargs):
    """Save denoised data with denoise subgroup containing CV results and metadata."""
    save_datasets_to_label(data_path, data_label, denoised_datasets, data_params, **kwargs)  # Save datasets under the specified label
    
    # Add denoise subgroup
    with h5py.File(data_path, 'a') as f:
        if 'denoise' in f[data_label]:
            del f[data_label]['denoise']
        denoise_group = f[data_label].create_group('denoise')

        # Save denoising config as separate attributes
        denoise_group.attrs['denoise_params'] = json.dumps(to_serializable(denoise_config))

        # Save CV results datasets
        for coord_key, cv_results in coordinate_cv_results.items():
            denoise_group.attrs[f'cv_{coord_key}_best_params'] = json.dumps(to_serializable(cv_results['best_params']))
            for metric, values in cv_results.items():
                if metric == 'best_params':
                    continue
                denoise_group.create_dataset(f'cv_{coord_key}_{metric}', data=np.asarray(values)) # e.g., cv_x_mean_test_score, cv_y_std_train_score
        
def read_denoise_results(data_path, data_label):
    """Read denoised data with CV results and metadata from denoise subgroup."""
    # Read main datasets (now safely skips subgroups)
    denoised_datasets, params = read_datasets_from_label(data_path, data_label)
    
    # Read denoise subgroup
    with h5py.File(data_path, 'r') as f:
        if 'denoise' not in f[data_label]:
            return denoised_datasets, params, None, {}
            
        denoise_group = f[data_label]['denoise']
        
        # Read denoise config
        denoise_config = json.loads(denoise_group.attrs['denoise_params'])
        
        # Reconstruct CV results for each coordinate
        coordinate_cv_results = {}
        for coord_key in denoise_config['coordinate_keys']:
            cv_results = {
                'best_params': json.loads(denoise_group.attrs[f'cv_{coord_key}_best_params'])
            }
            
            # Read CV datasets
            for key in denoise_group.keys():
                if key.startswith(f'cv_{coord_key}_') and 'best_params' not in key:
                    metric = key.replace(f'cv_{coord_key}_', '')
                    cv_results[metric] = denoise_group[key][...]
                    
            coordinate_cv_results[coord_key] = cv_results
    
    return denoised_datasets, params, denoise_config, coordinate_cv_results


# %% Saving and reading coordinate transforms

def save_coord_transform_results(data_path, data_label, coord_datasets, data_params, 
                               coord_transform_config, equations_data, 
                               candidates_data=None, **kwargs):
    """
    Save coordinate transformation data with coord_transform subgroup containing 
    features and targets for regression. Optionally includes candidate features and targets.
    """
    
    save_datasets_to_label(data_path, data_label, coord_datasets, data_params, **kwargs)
    
    with h5py.File(data_path, 'a') as f:
        

        # Save coordinate transformation config as attributes
        f[data_label].attrs['coord_transform_config'] = json.dumps(to_serializable(coord_transform_config))
        
        # Save equations_data
        if 'equations_data' in f[data_label]:
            del f[data_label]['equations_data']
        equations_group = f[data_label].create_group('equations_data')
        
        for eq_name, eq_data in equations_data.items():
            eq_prefix = eq_name.replace('_equation', '')
            
            # Save metadata as attributes
            equations_group.attrs[f'{eq_prefix}_feature_names'] = json.dumps(to_serializable(eq_data['feature_names']))
            equations_group.attrs[f'{eq_prefix}_feature_labels'] = json.dumps(to_serializable(eq_data['feature_labels']))
            equations_group.attrs[f'{eq_prefix}_target_name'] = eq_data['target_name'] 
            equations_group.attrs[f'{eq_prefix}_target_label'] = eq_data['target_label']
            equations_group.attrs[f'{eq_prefix}_equation_string'] = eq_data['equation_string']
            equations_group.attrs[f'{eq_prefix}_equation_label'] = eq_data['equation_label']
            
            # Save arrays as datasets
            equations_group.create_dataset(f'{eq_prefix}_feature_data', data=np.asarray(eq_data['feature_data']))
            equations_group.create_dataset(f'{eq_prefix}_target_data', data=np.asarray(eq_data['target_data']))
        
        # Save candidates_data (optional)
        include_candidates = coord_transform_config.get('include_candidates', False)
        if include_candidates and candidates_data is not None:
            if 'candidates_data' in f[data_label]:
                del f[data_label]['candidates_data']
            candidates_group = f[data_label].create_group('candidates_data')
            
            # Save arrays
            candidates_group.create_dataset('feature_candidate_data', data=candidates_data['feature_candidate_data'])
            candidates_group.create_dataset('target_candidate_data', data=candidates_data['target_candidate_data'])
            
            # Save metadata
            candidates_group.attrs['feature_candidate_names'] = json.dumps(candidates_data['feature_candidate_names'])
            candidates_group.attrs['feature_candidate_labels'] = json.dumps(candidates_data['feature_candidate_labels'])
            candidates_group.attrs['target_candidate_names'] = json.dumps(candidates_data['target_candidate_names'])
            candidates_group.attrs['target_candidate_labels'] = json.dumps(candidates_data['target_candidate_labels'])


def read_coord_transform_results(data_path, data_label):
    """Read coordinate transformation results, with optional candidates data."""
    datasets, params = read_datasets_from_label(data_path, data_label)
    
    with h5py.File(data_path, 'r') as f:
        if data_label not in f:
            return datasets, params, {}, {}, None
        
        # Read coordinate transformation config
        coord_transform_config = json.loads(f[data_label].attrs['coord_transform_config'])
        
        # Read equations_data
        equations_data = {}
        if 'equations_data' in f[data_label]:
            equations_group = f[data_label]['equations_data']
            equations = coord_transform_config.get('equations', [])
            
            for eq_name in equations:
                eq_prefix = eq_name.replace('_equation', '')
                
                # Read metadata and arrays
                eq_data = {
                    'feature_data': equations_group[f'{eq_prefix}_feature_data'][...],
                    'feature_names': json.loads(equations_group.attrs[f'{eq_prefix}_feature_names']),
                    'feature_labels': json.loads(equations_group.attrs[f'{eq_prefix}_feature_labels']),
                    'target_data': equations_group[f'{eq_prefix}_target_data'][...],
                    'target_name': equations_group.attrs[f'{eq_prefix}_target_name'],
                    'target_label': equations_group.attrs[f'{eq_prefix}_target_label'],
                    'equation_string': equations_group.attrs[f'{eq_prefix}_equation_string'],
                    'equation_label': equations_group.attrs[f'{eq_prefix}_equation_label']
                }
                
                # Reconstruct DataFrames
                eq_data['feature_df'] = pd.DataFrame(eq_data['feature_data'], columns=eq_data['feature_names'])
                eq_data['target_df'] = pd.DataFrame(eq_data['target_data'], columns=[eq_data['target_name']])
                
                # Store in equations_data
                equations_data[eq_name] = eq_data
        
        # Read candidates_data
        candidates_data = None
        include_candidates = coord_transform_config.get('include_candidates', False)
        
        if include_candidates and 'candidates_data' in f[data_label]:
            candidates_group = f[data_label]['candidates_data']
            
            candidates_data = {
                # Read metadata
                'feature_candidate_names': json.loads(candidates_group.attrs['feature_candidate_names']),
                'feature_candidate_labels': json.loads(candidates_group.attrs['feature_candidate_labels']),
                'target_candidate_names': json.loads(candidates_group.attrs['target_candidate_names']),
                'target_candidate_labels': json.loads(candidates_group.attrs['target_candidate_labels']),
                
                # Read arrays
                'feature_candidate_data': candidates_group['feature_candidate_data'][...],
                'target_candidate_data': candidates_group['target_candidate_data'][...],
            }
            
            # Reconstruct DataFrames
            candidates_data['feature_candidate_df'] = pd.DataFrame(
                candidates_data['feature_candidate_data'], 
                columns=candidates_data['feature_candidate_names']
            )
            candidates_data['target_candidate_df'] = pd.DataFrame(
                candidates_data['target_candidate_data'], 
                columns=candidates_data['target_candidate_names']
            )
    
    return datasets, params, coord_transform_config, equations_data, candidates_data


# %% Saving and reading inference results
def save_inference_results(data_path, data_label, data_params, 
                          regressor_config, cv_config, bootstrap_config, equation, inference_results, **kwargs):
    """
    Save inference results with time intervals, spring pendulum parameters, and validation metrics.
    """
    
    # Save params (no main datasets in this case)
    save_datasets_to_label(data_path, data_label, {}, data_params, **kwargs)
    
    with h5py.File(data_path, 'a') as f:
        
        # Save inference configs as attributes
        f[data_label].attrs['regressor_config'] = json.dumps(to_serializable(regressor_config))
        f[data_label].attrs['cv_config'] = json.dumps(to_serializable(cv_config))
        f[data_label].attrs['bootstrap_config'] = json.dumps(to_serializable(bootstrap_config))
        f[data_label].attrs['equation'] = equation

        # Save time intervals
        f[data_label].create_dataset('time_intervals', data=np.asarray(inference_results['time_intervals']))
        
        # Save spring_pendulum_params
        if 'spring_pendulum_params' in f[data_label]:
            del f[data_label]['spring_pendulum_params']
        params_group = f[data_label].create_group('spring_pendulum_params')
        
        params_group.attrs['param_names'] = json.dumps(inference_results['spring_pendulum_params']['param_names'])
        params_group.attrs['param_labels'] = json.dumps(to_serializable(inference_results['spring_pendulum_params']['param_labels']))
        
        for key in ['params_fitted', 'bootstrap_means', 'bootstrap_stds', 'bootstrap_ci_lowers', 'bootstrap_ci_uppers']:
            # Transpose to get shape: (n_params, n_intervals)
            params_group.create_dataset(key, data=np.asarray(inference_results['spring_pendulum_params'][key]).T)

        # Save validation
        if 'validation' in f[data_label]:
            del f[data_label]['validation']
        validation_group = f[data_label].create_group('validation')
        
        validation_group.attrs['cv_best_params'] = json.dumps(to_serializable(inference_results['validation']['cv_best_params']))
        for key in ['cv_best_scores', 'cv_param_grid', 'mse_scores', 'r2_scores']:
            validation_group.create_dataset(key, data=np.asarray(inference_results['validation'][key]))
        if 'msre_scores' in inference_results['validation'].keys():  # Save msre_scores if it exists
            validation_group.create_dataset('msre_scores', data=np.asarray(inference_results['validation']['msre_scores']))

        # Optional CV curves if they exist
        for key in ['cv_mean_test_score', 'cv_std_test_score', 'cv_mean_train_score', 'cv_std_train_score']:
            if key in inference_results['validation'] and inference_results['validation'][key]:
                # Convert list of arrays to 2D array for storage
                cv_data = np.array(inference_results['validation'][key])
                validation_group.create_dataset(key, data=cv_data)

def read_inference_results(data_path, data_label):
    """Read inference results with time intervals, spring pendulum parameters, and validation metrics."""
    datasets, params = read_datasets_from_label(data_path, data_label)
    time_intervals = datasets.get('time_intervals', None)
    
    with h5py.File(data_path, 'r') as f:
        if data_label not in f:
            return datasets, params, {}, {}, {}
        
        # Read inference configs
        regressor_config = json.loads(f[data_label].attrs['regressor_config'])
        cv_config = json.loads(f[data_label].attrs['cv_config']) 
        bootstrap_config = json.loads(f[data_label].attrs['bootstrap_config'])
        equation = f[data_label].attrs['equation']
        
        # Read time intervals
        time_intervals = f[data_label]['time_intervals'][...]
        
        # Read spring_pendulum_params
        inference_results = {}
        if 'spring_pendulum_params' in f[data_label]:
            params_group = f[data_label]['spring_pendulum_params']
            
            inference_results['spring_pendulum_params'] = {
                'param_names': json.loads(params_group.attrs['param_names']),
                'param_labels': json.loads(params_group.attrs['param_labels']),
                'params_fitted': params_group['params_fitted'][...],
                'bootstrap_means': params_group['bootstrap_means'][...],
                'bootstrap_stds': params_group['bootstrap_stds'][...],
                'bootstrap_ci_lowers': params_group['bootstrap_ci_lowers'][...],
                'bootstrap_ci_uppers': params_group['bootstrap_ci_uppers'][...]
            }
        
        # Read validation
        if 'validation' in f[data_label]:
            validation_group = f[data_label]['validation']
            
            inference_results['validation'] = {
                'cv_best_params': json.loads(validation_group.attrs['cv_best_params']),
                'cv_param_grid': validation_group['cv_param_grid'][...],
                'cv_best_scores': validation_group['cv_best_scores'][...],
                'mse_scores': validation_group['mse_scores'][...],
                'r2_scores': validation_group['r2_scores'][...]
            }
            if 'msre_scores' in validation_group:  # Read msre_scores if it exists
                inference_results['validation']['msre_scores'] = validation_group['msre_scores'][...]
            
            # Read optional CV curves if they exist
            for key in ['cv_mean_test_score', 'cv_std_test_score', 'cv_mean_train_score', 'cv_std_train_score']:
                if key in validation_group:
                    inference_results['validation'][key] = validation_group[key][...]
        
        # Add time_intervals to inference_results
        inference_results['time_intervals'] = time_intervals
    
    return params, regressor_config, cv_config, bootstrap_config, equation, time_intervals, inference_results


# %% Helper functions

def get_keys_h5(h5_path, group_path=None, sort=True):
    """
    Get keys from an HDF5 file, optionally from a specific group.
    
    Parameters:
    -----------
    h5_path : str or Path
        Path to the HDF5 file
    group_path : str, optional
        Path to a specific group (e.g., 'group' or 'group/subgroup')
        If None, returns top-level keys
    sort : bool, optional
        Whether to sort the keys naturally (default: True)
    
    Returns:
    --------
    list
        List of keys in the file or group
    """
    with h5py.File(h5_path, 'r') as f:
        if group_path is not None:
            keys = list(f[group_path].keys())
        else:
            keys = list(f.keys())
        
        if sort:
            return natsorted(keys)
        else:
            return keys

def get_attrs_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return dict(f.attrs)

def inspect_h5(file_path):
    '''Print the structure and attributes of an HDF5 file.'''
    def summarize_attr(value):
        if isinstance(value, (str, bytes)):
            return f'(type={type(value).__name__}, len={len(value)})'
        elif isinstance(value, np.ndarray):
            return f'(type=ndarray, shape={value.shape}, dtype={value.dtype})'
        elif isinstance(value, (int, float, bool)):
            return f'(type={type(value).__name__})'
        elif hasattr(value, 'shape'):
            return f'(type={type(value).__name__}, shape={value.shape})'
        else:
            return f'(type={type(value).__name__})'

    def print_attrs(obj, prefix):
        for key, val in obj.attrs.items():
            summary = summarize_attr(val)
            print(f"{prefix}@{key} {summary}")

    def print_h5_structure(name, obj, prefix=""):
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{name}/")
            print_attrs(obj, prefix + "    ")
            for key in obj:
                print_h5_structure(key, obj[key], prefix + "    ")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}{name} (shape={obj.shape}, dtype={obj.dtype})")
            print_attrs(obj, prefix + "    ")
    
    with h5py.File(file_path, 'r') as f:
        print(f"{file_path}/")
        print_attrs(f, prefix="    ")
        for key in f:
            print_h5_structure(key, f[key], prefix="    ")

def repack_h5(path: str, in_place: bool = True):
    tmp = f"{path}.tmp"
    try:
        with h5py.File(path,'r') as src, h5py.File(tmp,'w') as dst:
            dst.attrs.update(src.attrs)
            for name in src: src.copy(name, dst)
            dst.flush()
        shutil.copystat(path, tmp)
        if in_place: os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        raise
   
def to_serializable(obj):
    '''Convert an object to a serializable format for JSON or HDF5 storage.'''
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)): return round(obj.item(), 12)
    elif 'sklearn' in str(type(obj)): return str(obj)  # Convert sklearn objects to string representation
    else: return obj

# %% Saving and reading FFT results

def save_fft_results(
    data_path: str,
    data_group_path: str,
    params: dict,
    time_interval_config: dict,
    time_intervals: list,
    fft_config: dict,
    fft_data: dict,
    hdf5_options: dict | None = None,
    overwrite: bool = True
) -> None:
    """
    Save FFT spectra for all moment keys and time intervals to an HDF5 file.

    Parameters
    ----------
    data_path : str or Path
        Path to the output HDF5 file.
    data_group_path : str
        Group path within the file, e.g. 'tangle_00/tw10_low'.
    params : dict
        Experiment parameters (copied from input file).
    time_interval_config : dict
        Time interval configuration (copied from input file).
    time_intervals : list
        List of [t_start, t_end] pairs that were computed.
    fft_config : dict
        FFT computation settings, e.g. {mean_subtracted, normalization, include_full}.
    fft_data : dict
        Nested dict: fft_data[moment_key][interval_key] = {'freq': array, 'magnitude': array}.
        interval_key is e.g. 'interval_000', 'interval_001', ..., 'full'.
    hdf5_options : dict, optional
        HDF5 compression options. Default: no compression.
    overwrite : bool, optional
        If True, delete and recreate the group. Default: True.
    """
    if hdf5_options is None:
        hdf5_options = {}
    if 'compression' in hdf5_options and 'chunks' not in hdf5_options:
        hdf5_options['chunks'] = True

    with h5py.File(data_path, 'a') as f:
        # Create or overwrite the data group
        if data_group_path in f:
            if overwrite:
                del f[data_group_path]
            else:
                raise ValueError(f"Group '{data_group_path}' already exists and overwrite=False.")
        group = f.require_group(data_group_path)

        # Store metadata as attributes
        group.attrs['params'] = json.dumps(to_serializable(params))
        group.attrs['time_interval_config'] = json.dumps(to_serializable(time_interval_config))
        group.attrs['time_intervals'] = json.dumps(to_serializable(time_intervals))
        group.attrs['fft_config'] = json.dumps(to_serializable(fft_config))

        # Store FFT data: {moment_key}/{interval_key}/freq + magnitude
        for moment_key, intervals in fft_data.items():
            mk_group = group.create_group(moment_key)
            for interval_key, arrays in intervals.items():
                iv_group = mk_group.create_group(interval_key)
                iv_group.create_dataset('freq', data=np.asarray(arrays['freq']), **hdf5_options)
                iv_group.create_dataset('magnitude', data=np.asarray(arrays['magnitude']), **hdf5_options)


def read_fft_results(
    data_path: str,
    data_group_path: str
) -> tuple:
    """
    Read FFT spectra from an HDF5 file.

    Parameters
    ----------
    data_path : str or Path
        Path to the HDF5 file.
    data_group_path : str
        Group path, e.g. 'tangle_00/tw10_low'.

    Returns
    -------
    params : dict
    time_interval_config : dict
    time_intervals : list of [t_start, t_end]
    fft_config : dict
    fft_data : dict
        fft_data[moment_key][interval_key] = {'freq': np.ndarray, 'magnitude': np.ndarray}
    """
    with h5py.File(data_path, 'r') as f:
        if data_group_path not in f:
            raise KeyError(f"Group '{data_group_path}' not found in {data_path}.")
        group = f[data_group_path]

        params = json.loads(group.attrs.get('params', '{}'))
        time_interval_config = json.loads(group.attrs.get('time_interval_config', '{}'))
        time_intervals = json.loads(group.attrs.get('time_intervals', '[]'))
        fft_config = json.loads(group.attrs.get('fft_config', '{}'))

        fft_data = {}
        for moment_key in group.keys():
            if not isinstance(group[moment_key], h5py.Group):
                continue
            mk_group = group[moment_key]
            fft_data[moment_key] = {}
            for interval_key in mk_group.keys():
                iv_group = mk_group[interval_key]
                fft_data[moment_key][interval_key] = {
                    'freq': iv_group['freq'][...],
                    'magnitude': iv_group['magnitude'][...]
                }

    return params, time_interval_config, time_intervals, fft_config, fft_data


def get_param_vals(data_path, param_keys):
    """
    Extract parameter values for each data label in the given HDF5 file.
    """
    param_vals = {key: [] for key in param_keys}
    data_labels = get_keys_h5(data_path)
    for data_label in data_labels:
        params = read_datasets_from_label(data_path, data_label)[-1]
        for key in param_keys:
            if key in params: param_vals[key].append(params[key])
    param_vals = {key: list(vals) for key, vals in param_vals.items()}
    return data_labels, param_vals
