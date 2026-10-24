import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# Regressors
# ==============================================================================

class SpringPendulumRegressorConstrained(BaseEstimator, RegressorMixin):
    """
    Regressor for driven spring-pendulum systems with optional half-spring behavior
    and L0 monotonicity constraint.
    
    Fits the model: m*z_ddot = -mg - gamma*z_dot - 1{r>L0}*[k*(r-L0)*cos(theta) + c*r_dot*cos(theta)]
    
    Expected DataFrame columns: ['constant', 'z_dot', 'r', 'cos_theta', 'r_cos_theta', 'r_dot_cos_theta']
    
    Parameters:
    -----------
    breakpoint_value : float, default=0.05
        Threshold L0 for half-spring behavior  
    half_spring : bool, default=True
        Whether to apply half-spring step function
    use_features : list of str, default=None
        Which feature columns to include in regression. If None, uses all except 'r'.
    base_regressor : sklearn regressor, default=None
        Underlying regression method. If None, uses LinearRegression(fit_intercept=False)
    fix_gravity : bool, default=False
        Whether to fix gravity constant rather than fit it
    g_value : float, default=1.0
        Fixed gravity value when fix_gravity=True
    constrain_positive : bool, default=False
        Whether to constrain coefficients to be positive
    L0_min : float, default=None
        Minimum value for L0. If provided, enforces L0 >= L0_min by reformulating features.
    """
    
    def __init__(
        self, 
        breakpoint_value=1.0, 
        half_spring=True, 
        use_features=None, 
        base_regressor=None, 
        fix_gravity=False, 
        g_value=1.0, 
        constrain_positive=False,
        L0_min=None
    ):
        self.breakpoint_value = breakpoint_value
        self.half_spring = bool(half_spring)
        self.use_features = use_features
        self.base_regressor = base_regressor
        self.fix_gravity = bool(fix_gravity)
        self.g_value = g_value
        self.constrain_positive = bool(constrain_positive)
        self.L0_min = L0_min


    def fit(self, X, y):
        """
        Fit the driven spring regressor with optional L0 constraint.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix with columns including 'r' for breakpoint logic
        y : array-like
            Target values (z acceleration)
        """
        # Validate inputs
        if isinstance(X, dict):
            X_fit = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_fit = X.copy()
        else:
            raise ValueError("X must be pandas DataFrame or dict with column_name: array pairs")
        y_fit = np.array(y).flatten()
        
        # Extract feature names from DataFrame
        self.feature_names_in_ = X_fit.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        
        # 1. Feature selection based on use_features
        r_values = X_fit['r'].values                    # Always need r for breakpoint logic
        if self.use_features is not None:
            use_features = [col for col in self.use_features if col in self.feature_names_in_]
        else:
            use_features = [col for col in self.feature_names_in_ if col != 'r']     # Default: all except 'r'
        X_fit = X_fit[use_features]
        
        # Store original selected features (before any transformations) for predict
        self.original_selected_features_ = use_features.copy()
        
        # 2. Handle gravity fixing
        if self.fix_gravity:
            X_fit = X_fit.drop(columns=['constant'], errors='ignore')
            y_fit = y_fit - (-self.g_value)                             # Subtract -g from targets
            self.regression_feature_names_ = [col for col in use_features if col != 'constant']
        else:
            X_fit = X_fit.copy()
            y_fit = y_fit.copy()
            self.regression_feature_names_ = use_features.copy()

        # 3. Apply breakpoint logic
        X_fit = self._apply_breakpoint(X_fit, r_values)

        # 4. Transform features for positive parameters
        X_fit = self._apply_negative_features(X_fit)        # Negative 'constant', 'z_dot', 'r_cos_theta', 'r_dot_cos_theta'
        self.regression_feature_names_ = [          # Update regression feature names to reflect transformation
            f'neg_{col}' if col in ['constant', 'z_dot', 'r_cos_theta', 'r_dot_cos_theta'] 
            else col for col in self.regression_feature_names_
        ]
        
        # 5. Handle scaling (avoid constant columns)
        stds = X_fit.std()
        is_constant = stds < 1e-12

        # Store for prediction and parameter extraction
        self.is_constant_ = is_constant
        
        if (~is_constant).any():
            self.scaler_ = StandardScaler(with_mean=False)
            non_constant_cols = X_fit.columns[~is_constant]
            X_fit[non_constant_cols] = self.scaler_.fit_transform(X_fit[non_constant_cols])      # Scale only non-constant features
            
            # Create full scale array for coefficient extraction
            self.scale_full_ = np.ones(len(self.regression_feature_names_))
            self.scale_full_[~is_constant] = self.scaler_.scale_
        else: 
            # If only constant column present, no scaling needed
            self.scale_full_ = np.ones(len(self.regression_feature_names_))
            self.scaler_ = None

        # 6. Handle L0 constraint
        self.constraint_active_ = False
        if self.L0_min is not None:
            # Try unconstrained fit first to check if constraint is needed
            temp_regressor = LinearRegression(fit_intercept=False, positive=self.constrain_positive)
            temp_regressor.fit(X_fit, y_fit)
            
            # Temporarily store state to extract parameters
            temp_state = {
                'regressor_': self.regressor_ if hasattr(self, 'regressor_') else None,
                'scale_full_': self.scale_full_.copy(),
                'regression_feature_names_': self.regression_feature_names_.copy(),
                'is_constant_': self.is_constant_.copy(),
                'scaler_': self.scaler_
            }
            self.regressor_ = temp_regressor
            
            # Check constraint with robustness to division issues
            try:
                trial_params, _ = self.get_physical_parameters()
                k_m = trial_params['k_m']
                kL0_m = trial_params['kL0_m']
                
                # Check for valid values before division
                if np.isfinite(k_m) and np.isfinite(kL0_m) and abs(k_m) > 1e-10:
                    L0_trial = kL0_m / abs(k_m)
                else:
                    # If k_m is too small or invalid, assume constraint is violated
                    L0_trial = 0.0
                    
            except Exception:
                # If extraction fails, assume constraint is violated
                L0_trial = 0.0
            
            # Restore state
            if temp_state['regressor_'] is not None:
                self.regressor_ = temp_state['regressor_']
            
            if L0_trial < self.L0_min:
                # Constraint violated - refit with constrained features
                X_fit = self._apply_L0_constraint(X_fit, r_values)
                self.constraint_active_ = True
                self.L0_constrained_value_ = self.L0_min
                
                # Update regression_feature_names_ for constrained case
                self.regression_feature_names_ = self.regression_feature_names_constrained_
                
                # Recompute scaling with new features
                stds = X_fit.std()
                is_constant = stds < 1e-12
                self.is_constant_ = is_constant
                
                if (~is_constant).any():
                    self.scaler_ = StandardScaler(with_mean=False)
                    non_constant_cols = X_fit.columns[~is_constant]
                    X_fit[non_constant_cols] = self.scaler_.fit_transform(X_fit[non_constant_cols])
                    self.scale_full_ = np.ones(len(self.regression_feature_names_))
                    self.scale_full_[~is_constant] = self.scaler_.scale_
                else:
                    self.scale_full_ = np.ones(len(self.regression_feature_names_))
                    self.scaler_ = None
        
        # 7. Fit regressor (final fit with possibly constrained features)
        if self.base_regressor is None:
            self.regressor_ = LinearRegression(
                fit_intercept=False, 
                positive=self.constrain_positive
            )
        else:
            self.regressor_ = self.base_regressor
            
        self.regressor_.fit(X_fit, y_fit)
        
        return self
    

    def predict(self, X):
        """Make predictions using the fitted model"""
        # Validate inputs
        if isinstance(X, dict):
            X_predict = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_predict = X.copy()
        else:
            raise ValueError("X must be pandas DataFrame or dict with column_name: array pairs")
        
        # Apply same preprocessing pipeline as in fit
        # 1. Feature selection - use the original selected features stored during fit
        r_values = X_predict['r'].values
        X_predict = X_predict[self.original_selected_features_]
        
        # 2. Handle gravity fixing (already handled by feature selection above)
        if self.fix_gravity:
            X_predict = X_predict.drop(columns=['constant'], errors='ignore')

        # 3. Apply breakpoint
        X_predict = self._apply_breakpoint(X_predict, r_values)

        # 4. Transform features for positive parameters
        X_predict = self._apply_negative_features(X_predict)
        
        # 5. Apply L0 constraint if it was active during fit
        if hasattr(self, 'constraint_active_') and self.constraint_active_:
            X_predict = self._apply_L0_constraint(X_predict, r_values)

        # 6. Apply scaling
        if self.scaler_ is not None:
            X_predict = X_predict.copy()
            non_constant_cols = X_predict.columns[~self.is_constant_]
            X_predict[non_constant_cols] = self.scaler_.transform(X_predict[non_constant_cols])
        else:
            X_predict = X_predict.copy()
        
        # 7. Make predictions
        predictions = self.regressor_.predict(X_predict)
        
        # 8. Adjust for gravity
        if self.fix_gravity:
            predictions += (-self.g_value)  # Add back -g to get original z_ddot
            
        return predictions
    

    def _apply_breakpoint(self, X_fit, r_values):
        """Apply half-spring breakpoint logic to relevant features"""
        if self.half_spring:
            # Identify spring features that get step function applied
            spring_feature_names = ['cos_theta', 'r_cos_theta', 'r_dot_cos_theta']
            spring_indices = [i for i, name in enumerate(X_fit.columns) 
                            if name in spring_feature_names]
            
            # Apply step function: multiply spring features by 1{r > breakpoint}
            step_function = (r_values > self.breakpoint_value).astype(float)
            X_with_breakpoint = X_fit.copy()
            X_with_breakpoint.iloc[:, spring_indices] *= step_function[:, np.newaxis]
        else:
            X_with_breakpoint = X_fit.copy()
            
        return X_with_breakpoint
    

    def _apply_negative_features(self, X_fit):
        """Always create negative features for positive physical parameters"""
        features_to_negate = ['constant', 'z_dot', 'r_cos_theta', 'r_dot_cos_theta']
        
        # Build result with preserved column order
        result_data = {}
        for col in X_fit.columns:
            if col in features_to_negate:
                result_data[f'neg_{col}'] = -X_fit[col]
            else:
                result_data[col] = X_fit[col]
        
        return pd.DataFrame(result_data)


    def _apply_L0_constraint(self, X_fit, r_values):
        """
        Reformulate features to enforce L0 = L0_min.
        Replaces separate cos_theta and r_cos_theta features with (r - L0_min)*cos_theta.
        """
        X_constrained = X_fit.copy()
        
        # Check if both spring features are present
        has_cos = 'cos_theta' in X_fit.columns
        has_r_cos = 'neg_r_cos_theta' in X_fit.columns
        
        if has_cos and has_r_cos:
            # Get cos_theta values
            cos_theta_vals = X_fit['cos_theta'].values
            
            # Create constrained feature: -(r - L0_min)*cos_theta (note the negative!)
            # This is because we already negated r_cos_theta, so we need consistency
            X_constrained['neg_r_minus_L0_cos_theta'] = -(r_values - self.L0_min) * cos_theta_vals
            
            # Remove the unconstrained features
            X_constrained = X_constrained.drop(['cos_theta', 'neg_r_cos_theta'], axis=1)
            
            # Update regression feature names for constrained case
            self.regression_feature_names_constrained_ = [
                name for name in self.regression_feature_names_ 
                if name not in ['cos_theta', 'neg_r_cos_theta']
            ]
            self.regression_feature_names_constrained_.append('neg_r_minus_L0_cos_theta')
        
        return X_constrained


    def get_physical_parameters(self):
        """Extract physical parameters from fitted coefficients"""
        # Extract and unscale coefficients
        coeffs_scaled = self.regressor_.coef_
        coeffs_unscaled = coeffs_scaled / self.scale_full_
        
        # Correct for constant term based on means of non-constant features, if scaler shifted means
        if self.is_constant_.any() and self.scaler_ is not None and self.scaler_.with_mean==True:
            constant_indices = np.where(self.is_constant_)[0]
            for const_idx in constant_indices:
                non_constant_indices = np.where(~self.is_constant_)[0]
                if len(non_constant_indices) > 0:
                    correction = np.sum(coeffs_unscaled[non_constant_indices] * self.scaler_.mean_)
                    coeffs_unscaled[const_idx] = coeffs_scaled[const_idx] - correction
        
        # Pad coefficients to full feature length for consistent parameter extraction
        full_coeffs = np.full(len(self.feature_names_in_), np.nan)
        
        # Handle constrained case - need to reconstruct kL0_m and k_m from constrained feature
        if hasattr(self, 'constraint_active_') and self.constraint_active_:
            # Find the constrained feature coefficient
            for i, feature_name in enumerate(self.regression_feature_names_):
                if feature_name == 'neg_r_minus_L0_cos_theta':
                    # This coefficient is k/m (from -(r - L0_min)*cos_theta)
                    k_m = coeffs_unscaled[i]
                    kL0_m = self.L0_constrained_value_ * k_m
                    
                    # Place into appropriate positions
                    if 'cos_theta' in self.feature_names_in_:
                        full_coeffs[self.feature_names_in_.index('cos_theta')] = kL0_m
                    if 'r_cos_theta' in self.feature_names_in_:
                        full_coeffs[self.feature_names_in_.index('r_cos_theta')] = k_m
                elif feature_name.replace('neg_', '') in self.feature_names_in_:
                    # Handle other features normally
                    feature_name_clean = feature_name.replace('neg_', '')
                    full_idx = self.feature_names_in_.index(feature_name_clean)
                    full_coeffs[full_idx] = coeffs_unscaled[i]
        else:
            # Unconstrained case - normal mapping
            for i, feature_name in enumerate(self.regression_feature_names_):
                feature_name_clean = feature_name.replace('neg_', '')
                if feature_name_clean in self.feature_names_in_:
                    full_idx = self.feature_names_in_.index(feature_name_clean)
                    full_coeffs[full_idx] = coeffs_unscaled[i]
        
        # Handle fixed gravity
        if self.fix_gravity:
            const_idx = self.feature_names_in_.index('constant')
            full_coeffs[const_idx] = self.g_value
        
        # Map to physical parameters, return as dict of values and labels
        g = full_coeffs[self.feature_names_in_.index('constant')]
        gamma_m = full_coeffs[self.feature_names_in_.index('z_dot')]
        kL0_m = full_coeffs[self.feature_names_in_.index('cos_theta')]
        k_m = full_coeffs[self.feature_names_in_.index('r_cos_theta')]
        c_m = full_coeffs[self.feature_names_in_.index('r_dot_cos_theta')]
        xi = self.breakpoint_value if self.half_spring else np.nan

        return (
            {'g': g, 'gamma_m': gamma_m, 'kL0_m': kL0_m, 'k_m': k_m, 'c_m': c_m, 'xi': xi}, 
            {'g': r'$g$', 'gamma_m': r'$\gamma/m$', 'kL0_m': r'$kL_0/m$', 'k_m': r'$k/m$', 'c_m': r'$c/m$', 'xi': r'$\xi$'}
        )


# Backward-compatible alias: old code importing SpringPendulumRegressor keeps working
SpringPendulumRegressor = SpringPendulumRegressorConstrained


# ==============================================================================
# Equation factory
# ==============================================================================

def create_spring_pendulum_equation(spring_pendulum_params, equation='z_equation', breakpoint_key='xi'):
    """
    Create a function that computes x_vec_ddot from the spring-pendulum equation.
    
    Equation: z_ddot = -g - (gamma/m)*z_dot - (k/m)*(r - L0)*cos_theta - (c/m)*r_dot*cos_theta
    With half-spring: multiply spring terms by 1{r > breakpoint}
    
    Parameters:
    -----------
    spring_pendulum_params : dict
        Dictionary of physical parameters
    equation : str, default='z_equation'
        Which equation to return
    breakpoint_key : str, default='xi'
        Which parameter to use as the half-spring breakpoint threshold.
        Use 'xi' for the fitted cross-validation breakpoint, or 'L0' to split at the natural length.
    """
    def spring_pendulum_z_ddot(r, cos_theta, r_dot, z_dot):
        # Extract parameters
        g = spring_pendulum_params.get('g', 0)
        gamma_m = spring_pendulum_params.get('gamma_m', 0)
        kL0_m = spring_pendulum_params.get('kL0_m', 0)
        k_m = spring_pendulum_params.get('k_m', 0)
        c_m = spring_pendulum_params.get('c_m', 0)
        xi = spring_pendulum_params.get('xi', 0)
        
        # Select breakpoint value
        breakpoint = spring_pendulum_params.get(breakpoint_key, xi)
        
        # Compute terms
        gravity_term = -g
        drag_term = -gamma_m * z_dot
        spring_term = kL0_m * cos_theta - k_m * r * cos_theta  # This is -k/m * (r - L0) * cos_theta
        damping_term = -c_m * r_dot * cos_theta
        
        # Apply half-spring step function
        step_function = (r > breakpoint).astype(float)
        spring_damping_terms = step_function * (spring_term + damping_term)
        
        return gravity_term + drag_term + spring_damping_terms
    
    if equation == 'z_equation':
        # Return the function for z_ddot
        return spring_pendulum_z_ddot
    else:
        raise ValueError(f"Unknown equation: {equation}. Supported: 'z_equation'.")
    

# ==============================================================================
# Derived parameters
# ==============================================================================

def compute_derived_params_bootstrap(param_names, param_labels, params_fitted, bootstrap_means, bootstrap_stds, bootstrap_ci_lowers, bootstrap_ci_uppers, derived_params=None):
    """
    Compute additional derived parameters and their bootstrap statistics from fitted parameters.
    
    Parameters
    ----------
    param_names : list
        List of parameter names
    param_labels : dict
        Dictionary mapping parameter names to LaTeX labels    
    params_fitted : np.ndarray
        Fitted parameter values (shape: n_params x n_intervals)
    bootstrap_means : np.ndarray
        Bootstrap mean values
    bootstrap_stds : np.ndarray
        Bootstrap standard deviations
    bootstrap_ci_lowers : np.ndarray
        Bootstrap lower confidence intervals
    bootstrap_ci_uppers : np.ndarray
        Bootstrap upper confidence intervals
    derived_params : list, optional
        List of additional parameters to compute. Default: ['L0']
        
    Returns
    -------
    tuple
        Updated (param_names, param_labels, params_fitted, bootstrap_means, bootstrap_stds, 
                 bootstrap_ci_lowers, bootstrap_ci_uppers)
    """
    if derived_params is None:
        derived_params = ['L0']
    
    # Make copies to avoid modifying originals
    param_names = param_names.copy()
    param_labels = param_labels.copy()
    params_fitted = params_fitted.copy()
    bootstrap_means = bootstrap_means.copy()
    bootstrap_stds = bootstrap_stds.copy()
    bootstrap_ci_lowers = bootstrap_ci_lowers.copy()
    bootstrap_ci_uppers = bootstrap_ci_uppers.copy()
    
    for param in derived_params:
        if param == 'L0':
            # Compute L0 = (kL0/m) / (k/m)
            if 'kL0_m' in param_names and 'k_m' in param_names:
                kL0_m_idx = param_names.index('kL0_m')
                k_m_idx = param_names.index('k_m')
                
                # Compute L0 values
                L0_fitted = params_fitted[kL0_m_idx] / np.abs(params_fitted[k_m_idx])
                L0_bootstrap_means = bootstrap_means[kL0_m_idx] / np.abs(bootstrap_means[k_m_idx])
                L0_bootstrap_stds = bootstrap_stds[kL0_m_idx] / np.abs(bootstrap_means[k_m_idx])  # Approximate
                L0_bootstrap_ci_lowers = bootstrap_ci_lowers[kL0_m_idx] / np.abs(bootstrap_ci_uppers[k_m_idx])  # Conservative
                L0_bootstrap_ci_uppers = bootstrap_ci_uppers[kL0_m_idx] / np.abs(bootstrap_ci_lowers[k_m_idx])  # Conservative
                
                # Add to arrays
                param_names.append('L0')
                param_labels['L0'] = r'$L_0$'
                params_fitted = np.vstack([params_fitted, L0_fitted])
                bootstrap_means = np.vstack([bootstrap_means, L0_bootstrap_means])
                bootstrap_stds = np.vstack([bootstrap_stds, L0_bootstrap_stds])
                bootstrap_ci_lowers = np.vstack([bootstrap_ci_lowers, L0_bootstrap_ci_lowers])
                bootstrap_ci_uppers = np.vstack([bootstrap_ci_uppers, L0_bootstrap_ci_uppers])
        
        # Add more derived parameters here in the future
    
    return (param_names, param_labels, params_fitted, bootstrap_means, bootstrap_stds,
            bootstrap_ci_lowers, bootstrap_ci_uppers)


def compute_derived_params(spring_pendulum_params, derived_param_names=None, Omega=None):
    """
    Compute derived parameters from fitted spring pendulum parameters.

    Parameters
    ----------
    spring_pendulum_params : dict
        Dictionary of fitted parameters with float values (e.g., 'g', 'k_m', 'kL0_m', ...).
    derived_param_names : list, optional
        List of derived parameter names to compute.
        Default: ['L0', 'omega_0', 'zeta', 'omega'].
    Omega : float, optional
        Driving angular frequency (rad/s). Required to compute the normalized
        driving frequency omega = Omega / omega_0.

    Returns
    -------
    derived_params : dict
        Dictionary of derived parameter values.
    derived_param_labels : dict
        Dictionary of LaTeX labels for the derived parameters.
    """
    if derived_param_names is None:
        derived_param_names = ['L0', 'omega_0', 'zeta', 'omega']

    derived_params = {}
    derived_param_labels = {}

    for param in derived_param_names:

        if param == 'L0':
            # L0 = (kL0/m) / (k/m)
            if 'kL0_m' in spring_pendulum_params and 'k_m' in spring_pendulum_params:
                kL0_m = spring_pendulum_params['kL0_m']
                k_m = spring_pendulum_params['k_m']
                derived_params['L0'] = kL0_m / np.abs(k_m)
                derived_param_labels['L0'] = r'$L_0$'

        elif param == 'omega_0':
            # omega_0 = sqrt(k/m)  -- natural frequency
            if 'k_m' in spring_pendulum_params:
                k_m = spring_pendulum_params['k_m']
                derived_params['omega_0'] = np.sqrt(np.abs(k_m))
                derived_param_labels['omega_0'] = r'$\omega_0$'

        elif param == 'zeta':
            # zeta = (gamma/m + c/m) / (2 * sqrt(k/m))
            # NaN values for gamma_m or c_m are treated as 0
            if 'k_m' in spring_pendulum_params:
                k_m = spring_pendulum_params['k_m']
                gamma_m = spring_pendulum_params.get('gamma_m', 0)
                c_m = spring_pendulum_params.get('c_m', 0)
                gamma_m = 0 if (gamma_m is None or np.isnan(gamma_m)) else gamma_m
                c_m = 0 if (c_m is None or np.isnan(c_m)) else c_m
                omega_0 = np.sqrt(np.abs(k_m))
                derived_params['zeta'] = (gamma_m + c_m) / (2 * omega_0) if omega_0 > 0 else np.nan
                derived_param_labels['zeta'] = r'$\zeta$'

        elif param == 'omega':
            # Normalized driving frequency: omega = Omega / omega_0
            if Omega is not None and 'k_m' in spring_pendulum_params:
                k_m = spring_pendulum_params['k_m']
                omega_0 = np.sqrt(np.abs(k_m))
                derived_params['omega'] = Omega / omega_0 if omega_0 > 0 else np.nan
                derived_param_labels['omega'] = r'$\omega$'

        # Add more derived parameters here in the future

    return derived_params, derived_param_labels