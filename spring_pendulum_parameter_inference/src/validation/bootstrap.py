import numpy as np
import pandas as pd

def generate_bootstrap_samples(estimator, X, y, extract_params, n_bootstrap=100, random_state=None):
    """
    Generic bootstrap uncertainty estimation for any estimator.
    
    Parameters:
    -----------
    estimator : sklearn-compatible estimator
        Fitted or unfitted estimator with fit() method
    X : array-like or DataFrame
        Feature matrix
    y : array-like  
        Target values
    extract_params : callable
        Function that takes fitted estimator and returns dict of parameters
        Example: lambda est: {'coef': est.coef_, 'intercept': est.intercept_}
    n_bootstrap : int, default=500
        Number of bootstrap samples
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    bootstrap_samples : list of dict
        List of parameter dictionaries from each bootstrap sample
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert to DataFrame if needed for consistent indexing
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = np.array(y)
    
    n_samples = len(X)
    bootstrap_samples = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[indices].reset_index(drop=True)
        y_boot = y[indices]
        
        try:
            # Clone estimator to avoid modifying original
            estimator_boot = estimator.__class__(**estimator.get_params())
            estimator_boot.fit(X_boot, y_boot)
            
            # Extract parameters using provided function
            params = extract_params(estimator_boot)
            bootstrap_samples.append(params)
            
        except Exception:
            # Skip failed fits (can happen with difficult bootstrap samples)
            continue
    
    return bootstrap_samples

def compute_bootstrap_statistics(bootstrap_samples, confidence_level=0.95):
    """
    Compute summary statistics from bootstrap samples.
    
    Parameters:
    -----------
    bootstrap_samples : list of dict
        Output from bootstrap_resampling()
    confidence_level : float, default=0.95
        Confidence level for intervals
        
    Returns:
    --------
    stats : dict
        Statistics for each parameter: mean, std, ci_lower, ci_upper
    """
    if not bootstrap_samples:
        return {}
    
    param_names = bootstrap_samples[0].keys()
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    stats = {}
    for param in param_names:
        values = [sample[param] for sample in bootstrap_samples]
        
        if values:
            stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'ci_lower': np.percentile(values, lower_percentile),
                'ci_upper': np.percentile(values, upper_percentile),
                'min': np.min(values),
                'max': np.max(values),
                'n_samples': len(values)
            }
    
    return stats

