"""
Coordinate transformations and feature engineering.
"""
import numpy as np
import pandas as pd
from src.analysis import finite_difference_derivative
from sklearn.base import BaseEstimator, TransformerMixin

# %% Coordinate transformations and feature engineering for driven spring pendulum

def transform_spring_pendulum_coordinates(datasets):
    """
    Transform to spring-pendulum coordinate system (relative to piston).
    
    Parameters:
    -----------
    datasets : dict
        Dictionary with keys: 't', 'x', 'y', 'z', 'xp', 'yp', 'zp'
    
    Returns:
    --------
    coord_data : dict
        Coordinate representations, including: original coordinates,
        relative coordinates, and spherical coordinates.
    """
    
    # Extract time and coordinates
    t = datasets['t']
    x, y, z = datasets['x'], datasets['y'], datasets['z']
    xp, yp, zp = datasets['xp'], datasets['yp'], datasets['zp']
    
    # 1. Compute relative coordinates (mass - piston)
    x_rel = x - xp
    y_rel = y - yp  
    z_rel = z - zp
    
    # 2. Transform to spherical coordinates
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    theta = np.arccos(z_rel / (r + 1e-12))              # Avoid division by zero
    phi = np.mod(np.arctan2(y_rel, x_rel), 2 * np.pi)   # Ensure phi is in [0, 2*pi)
    
    return {
        't': t,
        'x': x, 'y': y, 'z': z,          # Original coordinates
        'xp': xp, 'yp': yp, 'zp': zp,    # Piston coordinates  
        'x_rel': x_rel, 'y_rel': y_rel, 'z_rel': z_rel,     # Relative coordinates (mass - piston)
        'r': r, 'theta': theta, 'phi': phi,         # Spherical coordinates
    }


def generate_spring_pendulum_equations(coord_data, equations=['z_equation']):
    """
    Generate regression data for the vertical spring-pendulum equation from coordinate data.
    
    Creates feature matrices, target vectors, and metadata for regression 
    of spring-pendulum dynamics.
    
    Parameters:
    -----------
    coord_data : dict
        Output from transform_spring_pendulum_coordinates()
    equations : list
        Equations to create features for. Only 'z_equation' is supported.
    
    Returns:
    --------
    results : dict
        Features, targets, and metadata keyed by 'z_equation'.
    """
    
    # Extract coordinates
    t = coord_data['t']
    x = coord_data['x']
    y = coord_data['y']
    z = coord_data['z']
    r = coord_data['r']
    theta = coord_data['theta']
    phi = coord_data['phi']
    
    # Generate features for each equation
    equations_data = {}

    for eq in equations:
        if eq == 'z_equation':
            # Compute derivatives
            z_dot = finite_difference_derivative(t, z, order=1)
            z_ddot = finite_difference_derivative(t, z, order=2)
            r_dot = finite_difference_derivative(t, r, order=1)
            
            # Compute trigonometric terms
            cos_theta = np.cos(theta)
            
            # Create feature and target data matrix and df
            feature_data = np.column_stack([
                np.ones(len(t)),           # Constant term (coefficient: -g)
                z_dot,                     # Velocity term (coefficient: -gamma/m)
                r,                         # Radial distance term (breakpoint at r=L0)
                cos_theta,                 # Angular term (coefficient: k*L0/m)
                r * cos_theta,             # Spring term (coefficient: -k/m)
                r_dot * cos_theta          # Damping term (coefficient: -c/m)
            ])
            feature_names = ['constant', 'z_dot', 'r', 'cos_theta', 'r_cos_theta', 'r_dot_cos_theta']
            feature_labels = {
                'constant': '1',
                'z_dot': r'$\dot{z}$', 
                'r': r'$r$',
                'cos_theta': r'$\cos{\theta}$',
                'r_cos_theta': r'$r \cos{\theta}$',
                'r_dot_cos_theta': r'$\dot{r} \cos{\theta}$'
            }
            feature_df = pd.DataFrame(feature_data, columns=feature_names)

            target_data = z_ddot
            target_name = 'z_ddot'
            target_label = r'$\ddot{z}$'
            target_df = pd.DataFrame(target_data, columns=[target_name])
            
            # Store equation data
            equations_data['z_equation'] = {
                'feature_data': feature_data,
                'feature_names': feature_names,
                'feature_labels': feature_labels,
                'feature_df': feature_df,
                'target_data': target_data,
                'target_name': target_name,
                'target_label': target_label,
                'target_df': target_df,
                'equation_string': 'm*z_ddot = -mg - gamma*z_dot - k*(r-L0)*cos_theta + c*r_dot*cos_theta',
                'equation_label': r'$m \ddot{z} = -mg - \gamma \dot{z} - k (r - L_0) \cos(\theta) + c \dot{r} \cos(\theta)$'
            }
        
        else:
            raise ValueError(f"Unknown equation type: '{eq}'")
    
    return equations_data


def generate_spring_pendulum_candidates(coord_data):
    """
    Generate comprehensive feature and target candidates for exploration.
    
    Creates all possible coordinates, derivatives, and transformations that 
    could be used as features or targets in spring-pendulum regression.
    
    Parameters:
    -----------
    coord_data : dict
        Output from transform_spring_pendulum_coordinates()
    
    Returns:
    --------
    candidates_data : dict
        Feature and target candidate data with consistent structure
    """
    
    t = coord_data['t']
    
    # Collect feature candidates
    feature_candidates = {}
    feature_candidate_labels = {}
    
    # Original coordinates
    feature_candidates['x'] = coord_data['x']
    feature_candidates['y'] = coord_data['y'] 
    feature_candidates['z'] = coord_data['z']
    feature_candidate_labels.update({'x': r'$x$', 'y': r'$y$', 'z': r'$z$'})
    
    # Relative coordinates
    feature_candidates['x_rel'] = coord_data['x_rel']
    feature_candidates['y_rel'] = coord_data['y_rel']
    feature_candidates['z_rel'] = coord_data['z_rel']
    feature_candidate_labels.update({'x_rel': r'$x_{\mathrm{rel}}$', 'y_rel': r'$y_{\mathrm{rel}}$', 'z_rel': r'$z_{\mathrm{rel}}$'})
    
    # Spherical coordinates
    feature_candidates['r'] = coord_data['r']
    feature_candidates['theta'] = coord_data['theta']
    feature_candidates['phi'] = coord_data['phi']
    feature_candidate_labels.update({'r': r'$r$', 'theta': r'$\theta$', 'phi': r'$\phi$'})
    
    # Trigonometric terms (only those in equations of motion)
    feature_candidates['cos_theta'] = np.cos(coord_data['theta'])
    feature_candidates['sin_theta_cos_phi'] = np.sin(coord_data['theta']) * np.cos(coord_data['phi'])
    feature_candidates['sin_theta_sin_phi'] = np.sin(coord_data['theta']) * np.sin(coord_data['phi'])
    feature_candidate_labels.update({
        'cos_theta': r'$\cos{\theta}$',
        'sin_theta_cos_phi': r'$\sin{\theta}\cos{\phi}$',
        'sin_theta_sin_phi': r'$\sin{\theta}\sin{\phi}$'
    })
    
    # First derivatives
    for coord in ['x', 'y', 'z', 'r']:
        feature_candidates[f'{coord}_dot'] = finite_difference_derivative(t, coord_data[coord], order=1)
        feature_candidate_labels[f'{coord}_dot'] = rf'$\dot{{{coord}}}$'
    
    # Create feature data structure
    feature_candidate_names = list(feature_candidates.keys())
    feature_candidate_data = np.column_stack([feature_candidates[name] for name in feature_candidate_names])
    feature_candidate_df = pd.DataFrame(feature_candidate_data, columns=feature_candidate_names)
    
    # Collect target candidates (second derivatives)
    target_candidates = {}
    target_candidate_labels = {}
    for coord in ['x', 'y', 'z', 'r']:
        target_candidates[f'{coord}_ddot'] = finite_difference_derivative(t, coord_data[coord], order=2)
        target_candidate_labels[f'{coord}_ddot'] = rf'$\ddot{{{coord}}}$'
    
    # Create target data structure
    target_candidate_names = list(target_candidates.keys())
    target_candidate_data = np.column_stack([target_candidates[name] for name in target_candidate_names])
    target_candidate_df = pd.DataFrame(target_candidate_data, columns=target_candidate_names)
    
    return {
        'feature_candidate_data': feature_candidate_data,
        'feature_candidate_names': feature_candidate_names,
        'feature_candidate_labels': feature_candidate_labels,
        'feature_candidate_df': feature_candidate_df,
        'target_candidate_data': target_candidate_data,
        'target_candidate_names': target_candidate_names,
        'target_candidate_labels': target_candidate_labels,
        'target_candidate_df': target_candidate_df
    }



# %% Additional feature transformations

class FourierFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to create Fourier basis features, analogous to PolynomialFeatures.
    
    For a given input time vector `t`, this transformer generates a feature matrix 
    consisting of a constant term and pairs of sine and cosine terms up to the 
    specified number of harmonics.
    
    The basis expansion has the form:
        [1, cos(2πt/P), sin(2πt/P), ..., cos(2πKt/P), sin(2πKt/P)]
    where `P` is the period and `K = n_harmonics`.
    
    Parameters
    ----------
    n_harmonics : int, default=10
        Number of sine/cosine harmonic pairs to include in the basis. The total 
        number of output features will be `2 * n_harmonics + 1` (including the constant term).
    
    period : float or None, default=None
        Period of the underlying signal. If None, the period is set to 
        `X.max() - X.min()` during fitting.
    
    Attributes
    ----------
    period_ : float
        The period used for basis construction, inferred during fitting if not specified.
    """
    
    def __init__(self, n_harmonics=10, period=None):
        self.n_harmonics = n_harmonics
        self.period = period
    
    def fit(self, X, y=None):
        """
        Fit the transformer (mainly to determine period if not provided).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Input time points or 1D coordinates.
        
        y : Ignored
            Not used, present for compatibility with sklearn's API.
        
        Returns
        -------
        self : object
            Returns the fitted transformer.
        """
        if self.period is None:
            self.period_ = X.max() - X.min()
        else:
            self.period_ = self.period
        return self
    
    def transform(self, X):
        """
        Transform time points to Fourier basis features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Input time points or coordinates.
        
        Returns
        -------
        features : ndarray of shape (n_samples, 2 * n_harmonics + 1)
            Design matrix with constant term, cosines, and sines of increasing 
            harmonic frequency.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        t = X.flatten()
        
        n_features = 2 * self.n_harmonics + 1
        features = np.ones((len(t), n_features))  # Constant term
        
        col_idx = 1
        for k in range(1, self.n_harmonics + 1):
            features[:, col_idx] = np.cos(2 * np.pi * k * t / self.period_)
            features[:, col_idx + 1] = np.sin(2 * np.pi * k * t / self.period_)
            col_idx += 2
        
        return features