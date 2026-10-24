import numpy as np

FITTED_PARAM_NAMES = ['g', 'gamma_m', 'kL0_m', 'k_m', 'c_m', 'xi']


def read_inference_params(inference_path, data_label, params_plot=None,
                          derived_params=None, metrics_plot=None):
    """
    Read inference results from an HDF5 file, compute derived parameters, and
    filter down to the requested subset of parameters.

    Parameters
    ----------
    inference_path : path-like
        Path to the inference results HDF5 file.
    data_label : str
        Key identifying the experiment within the HDF5 file.
    params_plot : list of str, optional
        Ordered list of parameter names to return. If None, all parameters are returned.
    derived_params : list of str, optional
        Additional derived parameters to compute before filtering (e.g. ['L0']).
        Each name must be handled by compute_derived_params_bootstrap.
        If None, no derived parameters are computed.
    metrics_plot : list of str, optional
        Validation metric keys to extract from inference_results['validation']
        (e.g. ['r2_scores', 'mse_scores']). If None or empty, metric_values is
        an empty (0, n_intervals) array.

    Returns
    -------
    time_midpoints : ndarray, shape (n_intervals,)
    param_names    : list of str
    param_labels   : dict
    params_fitted  : ndarray, shape (n_params, n_intervals)
    bootstrap_means   : ndarray, shape (n_params, n_intervals)
    bootstrap_stds    : ndarray, shape (n_params, n_intervals)
    bootstrap_ci_lowers : ndarray, shape (n_params, n_intervals)
    bootstrap_ci_uppers : ndarray, shape (n_params, n_intervals)
    metric_values  : ndarray, shape (n_metrics, n_intervals)
    """
    # Local imports to avoid circular dependencies at module load time
    from src.utilities import read_inference_results
    from .inference import compute_derived_params_bootstrap

    _, _, _, _, _, time_intervals, inference_results = read_inference_results(inference_path, data_label)

    time_midpoints = time_intervals.mean(axis=1)
    param_names = list(inference_results['spring_pendulum_params']['param_names'])
    param_labels = dict(inference_results['spring_pendulum_params']['param_labels'])

    params_fitted       = np.array(inference_results['spring_pendulum_params']['params_fitted'])
    bootstrap_means     = np.array(inference_results['spring_pendulum_params']['bootstrap_means'])
    bootstrap_stds      = np.array(inference_results['spring_pendulum_params']['bootstrap_stds'])
    bootstrap_ci_lowers = np.array(inference_results['spring_pendulum_params']['bootstrap_ci_lowers'])
    bootstrap_ci_uppers = np.array(inference_results['spring_pendulum_params']['bootstrap_ci_uppers'])

    # Compute additional derived parameters if requested and not already present
    if derived_params is not None:
        if any(p not in param_names for p in derived_params):
            (param_names, param_labels,
             params_fitted, bootstrap_means, bootstrap_stds,
             bootstrap_ci_lowers, bootstrap_ci_uppers) = compute_derived_params_bootstrap(
                param_names, param_labels,
                params_fitted, bootstrap_means, bootstrap_stds,
                bootstrap_ci_lowers, bootstrap_ci_uppers,
                derived_params=derived_params,
            )

    # Filter to the requested parameter subset
    if params_plot is not None:
        param_indices = [param_names.index(p) for p in params_plot]
        param_names         = [param_names[i] for i in param_indices]
        params_fitted       = params_fitted[param_indices]
        bootstrap_means     = bootstrap_means[param_indices]
        bootstrap_stds      = bootstrap_stds[param_indices]
        bootstrap_ci_lowers = bootstrap_ci_lowers[param_indices]
        bootstrap_ci_uppers = bootstrap_ci_uppers[param_indices]

    # Extract validation metrics
    n_intervals = len(time_midpoints)
    metric_values = np.empty((0, n_intervals))
    if metrics_plot:
        metric_values = np.array([
            np.array(inference_results['validation'][m]) for m in metrics_plot
        ])

    return (time_midpoints, param_names, param_labels,
            params_fitted, bootstrap_means, bootstrap_stds,
            bootstrap_ci_lowers, bootstrap_ci_uppers,
            metric_values)


def initialize_inference_results(time_intervals):
    """
    Create an empty inference results accumulator for the given time intervals.

    Parameters
    ----------
    time_intervals : list of (float, float)
        List of (t_start, t_end) tuples defining the inference windows.

    Returns
    -------
    dict
        Initialized results dict ready to be populated by store_interval_inference_results.
    """
    return {
        'time_intervals': time_intervals,
        'spring_pendulum_params': {
            'param_names': None,      # set after first fit
            'param_labels': None,     # set after first fit
            'params_fitted': [],
            'bootstrap_means': [],
            'bootstrap_stds': [],
            'bootstrap_ci_lowers': [],
            'bootstrap_ci_uppers': [],
            'bootstrap_mins': [],
            'bootstrap_maxs': [],
        },
        'validation': {
            'cv_param_grid': None,    # set after first fit
            'cv_best_params': [],
            'cv_best_scores': [],
            'mse_scores': [],
            'r2_scores': [],
            # cv curves (one array per interval)
            'cv_mean_test_score': [],
            'cv_std_test_score': [],
            'cv_mean_train_score': [],
            'cv_std_train_score': []
        }
    }


def finalize_inference_results(inference_results):
    """
    Convert per-interval lists in inference_results to numpy arrays in place.
    Call once after the interval loop, before saving.

    Parameters
    ----------
    inference_results : dict
        Accumulator dict populated by store_interval_inference_results.
    """
    for key in ['params_fitted', 'bootstrap_means', 'bootstrap_stds',
                'bootstrap_ci_lowers', 'bootstrap_ci_uppers',
                'bootstrap_mins', 'bootstrap_maxs']:
        inference_results['spring_pendulum_params'][key] = np.array(
            inference_results['spring_pendulum_params'][key])
    for key in ['cv_best_scores', 'mse_scores', 'r2_scores']:
        inference_results['validation'][key] = np.array(
            inference_results['validation'][key])


def print_inference_results(params=None, equation_data=None, spring_pendulum_params=None,
                            validation=None, fmt='.5f', extra=None):
    """
    Pretty-print inference results for a single time interval.

    Parameters
    ----------
    params : dict, optional
        Experiment metadata (must contain 'exp_name' and 'comments').
    equation_data : dict, optional
        Equation data dict (must contain 'equation_string').
    spring_pendulum_params : dict, optional
        Dict mapping param name -> (value, bootstrap_stats). Fitted params
        (listed in FITTED_PARAM_NAMES) are printed first; derived params follow.
    validation : dict, optional
        Validation results dict (cv_best_params, cv_mean_test_score, r2_score, mse_score).
    fmt : str
        Python format spec for numeric values (default '.5f').
    extra : str, optional
        Extra line printed below the experiment header (e.g. interval info).
    """
    if params is not None:
        print('=' * 80)
        print(f"{params['exp_name'].ljust(10)} ({params['comments']})")
        if extra:
            print(extra)

    if equation_data is not None:
        print('-' * 80)
        print(f"\n    {equation_data['equation_string']}")
        print()

    if spring_pendulum_params is not None:
        fitted_items  = [(p, v) for p, v in spring_pendulum_params.items() if p in FITTED_PARAM_NAMES]
        derived_items = [(p, v) for p, v in spring_pendulum_params.items() if p not in FITTED_PARAM_NAMES]
        print('-' * 60)
        for param, (value, stats) in fitted_items:
            print(f'{param.ljust(10)}: {format(value, fmt).ljust(10)} \u00b1 {format(stats["std"], fmt)}')
        if derived_items:
            print()
            print('derived params:')
            for param, (value, stats) in derived_items:
                print(f'{param.ljust(10)}: {format(value, fmt).ljust(10)} \u00b1 {format(stats["std"], fmt)}')
    print()

    if validation is not None:
        print('-' * 60)
        cv_print = {key: format(value, fmt) for key, value in validation['cv_best_params'].items()}
        print(f"cv_results: {cv_print} (score = {format(validation['cv_mean_test_score'].min(), fmt)})")
        print(f'r2        : {format(validation["r2_score"], fmt)}')
        print(f'mse       : {format(validation["mse_score"], fmt)}')
        print()
    print()


def store_interval_inference_results(inference_results, spring_pendulum_params,
                                     spring_pendulum_param_labels, validation):
    """
    Append results from one time interval into the running inference_results structure.

    Parameters
    ----------
    inference_results : dict
        Accumulator dict (modified in place) built before the interval loop.
    spring_pendulum_params : dict
        Dict mapping param name -> (value, bootstrap_stats) for this interval.
    spring_pendulum_param_labels : dict
        Dict mapping param name -> LaTeX label string.
    validation : dict
        Validation metrics dict for this interval.
    """
    # Set param names/labels after first fit
    if inference_results['spring_pendulum_params']['param_names'] is None:
        inference_results['spring_pendulum_params']['param_names'] = list(spring_pendulum_params.keys())
        inference_results['spring_pendulum_params']['param_labels'] = spring_pendulum_param_labels

    # Store parameter results (maintain order using param_names)
    param_names = inference_results['spring_pendulum_params']['param_names']
    inference_results['spring_pendulum_params']['params_fitted'].append(
        [spring_pendulum_params[p][0] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_means'].append(
        [spring_pendulum_params[p][1]['mean'] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_stds'].append(
        [spring_pendulum_params[p][1]['std'] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_ci_lowers'].append(
        [spring_pendulum_params[p][1]['ci_lower'] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_ci_uppers'].append(
        [spring_pendulum_params[p][1]['ci_upper'] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_mins'].append(
        [spring_pendulum_params[p][1]['min'] for p in param_names])
    inference_results['spring_pendulum_params']['bootstrap_maxs'].append(
        [spring_pendulum_params[p][1]['max'] for p in param_names])

    # Store validation results
    inference_results['validation']['cv_best_params'].append(validation['cv_best_params'])
    if inference_results['validation']['cv_param_grid'] is None:
        inference_results['validation']['cv_param_grid'] = validation['cv_param_grid']
    inference_results['validation']['cv_best_scores'].append(validation['cv_mean_test_score'].min())
    inference_results['validation']['mse_scores'].append(validation['mse_score'])
    inference_results['validation']['r2_scores'].append(validation['r2_score'])

    # Optional cv curves
    for key in ['cv_mean_test_score', 'cv_std_test_score', 'cv_mean_train_score', 'cv_std_train_score']:
        if key in inference_results['validation'] and key in validation:
            inference_results['validation'][key].append(validation[key])


def print_inference_summary(params, equation_data, inference_results, fmt='.5f'):
    """
    Print a compact summary of inferred parameters across all time intervals.

    Parameters
    ----------
    params : dict
        Experiment metadata (must contain 'exp_name' and 'comments').
    equation_data : dict
        Equation data dict (must contain 'equation_string').
    inference_results : dict
        Finalized inference results accumulator.
    fmt : str
        Python format spec for numeric values (default '.5f').
    """
    print('=' * 80)
    print(f"{params['exp_name'].ljust(10)} ({params['comments']})")
    print('-' * 80)
    print(f"\n    {equation_data['equation_string']}")
    print()

    param_names = inference_results['spring_pendulum_params']['param_names']
    params_fitted = np.array(inference_results['spring_pendulum_params']['params_fitted'])

    fitted_names  = [p for p in param_names if p in FITTED_PARAM_NAMES]
    derived_names = [p for p in param_names if p not in FITTED_PARAM_NAMES]

    print('-' * 60)
    for param, values in zip(param_names, params_fitted.T):
        if param in fitted_names:
            print(f"{param.ljust(10)}: [{', '.join(format(v, fmt) for v in values)}]")

    if derived_names:
        print()
        print('derived params:')
        for param, values in zip(param_names, params_fitted.T):
            if param in derived_names:
                print(f"{param.ljust(10)}: [{', '.join(format(v, fmt) for v in values)}]")

    print()
    print('-' * 60)
    print(f"cv_results: [{', '.join(format(v, fmt) for v in inference_results['validation']['cv_best_scores'])}]")
    print(f"r2        : [{', '.join(format(v, fmt) for v in inference_results['validation']['r2_scores'])}]")
    print(f"mse       : [{', '.join(format(v, fmt) for v in inference_results['validation']['mse_scores'])}]")
    print("\n")
