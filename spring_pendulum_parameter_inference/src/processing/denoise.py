"""
Denoising and smoothing classes for time series data.

This module provides various smoothing techniques for noisy time series data,
including local methods (moving average, Gaussian kernel, local polynomial)
and global methods (splines, frequency domain filtering).
"""
import numpy as np
import warnings
try:
    from numpy.exceptions import RankWarning  # numpy >= 1.25 / 2.0
except ImportError:
    from numpy import RankWarning             # numpy < 1.25
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from scipy.interpolate import UnivariateSpline

#%% Denoising classes
class MovingAverageSmoother(BaseEstimator, RegressorMixin):
    """
    Moving average smoother for 1D time series data.
    
    For each evaluation point, this smoother finds the closest `window_size` 
    training points and returns their arithmetic mean. This provides simple 
    local averaging that can reduce noise while preserving overall signal trends.
    
    Parameters
    ----------
    window_size : int, default=3
        Number of nearest neighbors to average. Larger values provide more 
        smoothing but may over-blur sharp features.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, 1)
        Training input data (time points).
    y_ : ndarray of shape (n_samples,)
        Training target data (signal values).
    """
    def __init__(self, window_size=3):
        self.window_size = window_size
    
    def fit(self, X, y):
        """
        Store training data for later prediction. "Memorizes" the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training time points.
        y : array-like of shape (n_samples,)
            Training signal values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self
    
    def predict(self, X):
        """
        Predict smoothed values using moving average.
        
        For each point in X, finds the `window_size` closest training points
        and returns their arithmetic mean.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Evaluation time points.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Smoothed signal values.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        
        y_pred = np.zeros_like(X_eval, dtype=float)
        X_train = self.X_.flatten()
        
        for i, x_i in enumerate(X_eval):
            # Find data points within the window
            distances = np.abs(X_train - x_i)
            # Sort and take the closest window_size points
            closest_indices = np.argsort(distances)[:self.window_size]
            # Simple average
            y_pred[i] = np.mean(self.y_[closest_indices])
        
        return y_pred
    
class GaussianKernelSmoother(BaseEstimator, RegressorMixin):
    """
    Gaussian kernel smoother for 1D time series data.
    
    For each evaluation point, this smoother computes Gaussian weights based on 
    distance to all training points, then returns a weighted average.
    
    The Gaussian weights are computed as: w_i = exp(-0.5 * ((x_i - x) / h)^2)
    where h is the bandwidth parameter.
    
    Parameters
    ----------
    bandwidth : float, default=0.1
        Width of the Gaussian kernel.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, 1)
        Training input data (time points).
    y_ : ndarray of shape (n_samples,)
        Training target data (signal values).
    """
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth
    
    def fit(self, X, y):
        """
        Store training data for later prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training time points.
        y : array-like of shape (n_samples,)
            Training signal values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self
    
    def predict(self, X):
        """
        Predict smoothed values using Gaussian kernel weighting.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Evaluation time points.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Smoothed signal values.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        
        y_pred = np.zeros_like(X_eval, dtype=float)
        X_train = self.X_.flatten()
        
        for i, x_i in enumerate(X_eval):
            # Compute Gaussian weights
            weights = np.exp(-0.5 * ((X_train - x_i) / self.bandwidth)**2)
            
            # Weighted average
            if np.sum(weights) == 0:
                y_pred[i] = 0
            else:
                y_pred[i] = np.sum(weights * self.y_) / np.sum(weights)
        
        return y_pred

class LocalPolynomialSmoother(BaseEstimator, RegressorMixin):
    """
    Local polynomial smoother using sliding window polynomial fits.
    
    For each evaluation point, this smoother finds the closest `window_size` 
    training points and fits a polynomial of degree `polyorder` to these points, 
    then evaluates the polynomial at the query point.

    This method preserves local signal features while reducing high-frequency noise.
    It generalizes moving average smoothing by fitting higher-order local models.
    
    Parameters
    ----------
    window_size : int, default=5
        Number of nearest neighbors to use for local polynomial fitting.
    polyorder : int, default=2
        Degree of polynomial to fit to each local neighborhood.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, 1)
        Training input data (time points).
    y_ : ndarray of shape (n_samples,)
        Training target data (signal values).
    """
    def __init__(self, window_size=5, polyorder=2):
        self.window_size = window_size
        self.polyorder = polyorder
    
    def fit(self, X, y):
        """
        Store training data for later prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training time points.
        y : array-like of shape (n_samples,)
            Training signal values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self
    
    def predict(self, X):
        """
        Predict smoothed values using local polynomial fits.
        
        For each evaluation point, selects the closest `window_size` training points,
        fits a polynomial of degree `polyorder`, and evaluates the polynomial at
        the query location. Falls back to the local mean if there are not enough
        distinct points to fit the requested polynomial degree.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Evaluation time points.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Smoothed signal values.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        X_train = self.X_.flatten()
        
        y_pred = np.zeros_like(X_eval, dtype=float)
        
        for i, x_i in enumerate(X_eval):
            # Find closest window_size points
            distances = np.abs(X_train - x_i)
            closest_indices = np.argsort(distances)[:self.window_size]
            
            # Get window data
            X_window = X_train[closest_indices]
            y_window = self.y_[closest_indices]
            
            # Fit polynomial to window
            if len(np.unique(X_window)) > self.polyorder:  # Ensure enough unique points
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RankWarning)
                    poly_coeffs = np.polyfit(X_window, y_window, self.polyorder)
                    y_pred[i] = np.polyval(poly_coeffs, x_i)
            else:
                # Fallback to mean if not enough unique points
                y_pred[i] = np.mean(y_window)
        
        return y_pred
    
class SplineSmoother(BaseEstimator, RegressorMixin):
    """
    Spline smoother using scipy's UnivariateSpline.
    
    Fits a smooth spline function globally to all training data, balancing 
    data fidelity against smoothness. The smoothing parameter controls the 
    trade-off between fitting the data exactly and creating a smooth curve.
    
    The fitted spline minimizes:
        \sum_i (y_i - f(x_i))^2 + s \int [f^{(m)}(x)]^2 dx
    where `s` is the smoothing parameter and `m` is typically 2.
    
    Parameters
    ----------
    s : float or None, default=None
        Smoothing parameter. If None, uses default smoothing. Larger values 
        create smoother curves but may underfit the data.
    k : int, default=3
        Degree of spline (1=linear, 2=quadratic, 3=cubic, etc.).
    
    Attributes
    ----------
    X_train_sorted_ : ndarray of shape (n_samples,)
        Training input data sorted in ascending order (required by spline).
    y_train_sorted_ : ndarray of shape (n_samples,)
        Corresponding target values sorted to match `X_train_sorted_`.
    spline_ : UnivariateSpline
        The fitted spline object.
    """
    def __init__(self, s=None, k=3):
        self.s = s  # Smoothing parameter
        self.k = k  # Degree of spline (1=linear, 2=quadratic, 3=cubic, etc.)
    
    def fit(self, X, y):
        """
        Fit spline to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training time points.
        y : array-like of shape (n_samples,)
            Training signal values.
        
        Returns
        -------
        self : object
            Fitted spline smoother instance.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_train = X.flatten()
        
        # Sort by X values for spline fitting
        sort_idx = np.argsort(X_train)
        self.X_train_sorted_ = X_train[sort_idx]
        self.y_train_sorted_ = y[sort_idx]
        
        # Fit univariate spline
        self.spline_ = UnivariateSpline(self.X_train_sorted_, self.y_train_sorted_, s=self.s, k=self.k)
        return self
    
    def predict(self, X):
        """
        Predict using fitted spline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Input locations at which to evaluate the spline.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted (smoothed) output values.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        
        # Evaluate spline at requested points
        y_pred = self.spline_(X_eval)
        return y_pred
    
    def derivative(self, X, order=1):
        """
        Compute analytic derivatives of the fitted spline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Points at which to evaluate the derivative.
        order : int, default=1
            Order of derivative to compute.
        
        Returns
        -------
        y_deriv : ndarray of shape (n_samples,)
            Derivative of the spline evaluated at the given points.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        
        # Evaluate derivative using scipy's built-in method
        y_deriv = self.spline_(X_eval, nu=order)
        return y_deriv

class FrequencyDomainSmoother(BaseEstimator, RegressorMixin):
    """
    Frequency-based smoother with multiple filtering strategies.
    Requires uniformly spaced training data.
    
    This method transforms the input signal to the frequency domain using the 
    Fast Fourier Transform (FFT), filters it according to a chosen strategy, and 
    reconstructs the signal via inverse Fourier synthesis.
    
    Supported filter types:
        - 'lowpass' : zeroes out all frequencies above `cutoff_freq`
        - 'lowpass_gaussian' : applies Gaussian rolloff around zero frequency
        - 'density_percentile' : keeps the top `percentile`% of Fourier components by power
        - 'density_threshold' : keeps components above a fixed power threshold (relative to max)
    
    Parameters
    ----------
    filter_type : {'lowpass', 'lowpass_gaussian', 'density_percentile', 'density_threshold'}, default='density_percentile'
        Type of filtering to apply in the frequency domain.
    cutoff_freq : float, default=10.0
        Cutoff frequency for lowpass filters (used for 'lowpass' and 'lowpass_gaussian').
    sigma : float, default=1.0
        Standard deviation of the Gaussian rolloff (used for 'lowpass_gaussian').
    percentile : float, default=95.0
        Percentile threshold for keeping strongest components (used for 'density_percentile').
    threshold : float, default=0.1
        Fraction of max power to use as a cutoff (used for 'density_threshold').
    
    Attributes
    ----------
    X_train_sorted_ : ndarray of shape (n_samples,)
        Sorted training input values (uniformly spaced).
    y_train_sorted_ : ndarray of shape (n_samples,)
        Sorted target values corresponding to `X_train_sorted_`.
    y_mean_ : float
        Mean of the training signal (added back after filtering).
    dt_ : float
        Average time step (used to infer frequency spacing).
    y_fft_filtered_ : ndarray of shape (n_samples,)
        Filtered complex-valued FFT coefficients.
    frequencies_ : ndarray of shape (n_samples,)
        Frequency values corresponding to FFT components.
    """
    
    def __init__(self, filter_type='density_percentile', cutoff_freq=10.0, sigma=1.0, 
                 percentile=95.0, threshold=0.1):
        self.filter_type = filter_type        # 'lowpass', 'lowpass_gaussian', 'density_percentile', 'density_threshold'
        self.cutoff_freq = cutoff_freq        # For lowpass filters
        self.sigma = sigma                    # Gaussian rolloff parameter
        self.percentile = percentile          # For percentile-based filtering
        self.threshold = threshold            # For threshold-based filtering
    
    def fit(self, X, y):
        """
        Fit FFT filter to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Training input time points (must be uniformly spaced).
        y : array-like of shape (n_samples,)
            Training signal values.
        
        Returns
        -------
        self : object
            Fitted frequency-domain smoother.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_train = X.flatten()
        
        # Sort by X values
        sort_idx = np.argsort(X_train)
        X_train_sorted = X_train[sort_idx]
        y_train_sorted = y[sort_idx]
        
        # Validate uniform spacing (required for FFT)
        if len(X_train_sorted) > 1:
            dt_values = np.diff(X_train_sorted)
            dt_mean = np.mean(dt_values)
            if not np.allclose(dt_values, dt_mean, rtol=1e-6):
                raise ValueError("FFTSmoother requires uniformly spaced training data. "
                               f"Time step variation: {np.std(dt_values):.2e}")
        
        # Store training data and parameters
        self.X_train_sorted_ = X_train_sorted
        self.y_train_sorted_ = y_train_sorted
        self.y_mean_ = np.mean(y_train_sorted)
        self.dt_ = dt_mean if len(X_train_sorted) > 1 else 1.0
        
        # Apply FFT filtering
        self._apply_fft_filter()
        return self
    
    def _apply_fft_filter(self):
        """
        Apply FFT filtering to the sorted training data.
        """
        # Take FFT (remove mean first)
        y_centered = self.y_train_sorted_ - self.y_mean_
        y_fft = np.fft.fft(y_centered)
        frequencies = np.fft.fftfreq(len(self.y_train_sorted_), d=self.dt_)
        
        # Create filter based on type
        if self.filter_type == 'lowpass':
            # Hard cutoff
            filter_mask = np.abs(frequencies) <= self.cutoff_freq
            
        elif self.filter_type == 'lowpass_gaussian':
            # Gaussian rolloff
            filter_mask = np.exp(-0.5 * (frequencies / self.cutoff_freq)**2 / self.sigma**2)
            
        elif self.filter_type == 'density_percentile':
            # Keep top percentile of components by power (energy)
            power = np.abs(y_fft)**2
            threshold_value = np.percentile(power, self.percentile)
            filter_mask = power >= threshold_value
            
        elif self.filter_type == 'density_threshold':
            # Keep components above fraction of max power
            power = np.abs(y_fft)**2
            max_power = np.max(power)
            threshold_value = self.threshold * max_power
            filter_mask = power >= threshold_value
        
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")
        
        # Apply filter
        self.y_fft_filtered_ = y_fft * filter_mask
        self.frequencies_ = frequencies
    
    def predict(self, X):
        """
        Predict using Fourier synthesis.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Time points at which to evaluate the reconstructed signal.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Reconstructed (denoised) signal values.
        """
        X = np.atleast_1d(X).reshape(-1, 1) if X.ndim == 1 else X
        X_eval = X.flatten()
        
        # Reconstruct signal using Fourier synthesis
        y_pred = self._reconstruct_fourier(X_eval)
        return y_pred
    
    def _reconstruct_fourier(self, t_query):
        """
        Reconstruct signal at arbitrary time points using Fourier synthesis.
        
        Parameters
        ----------
        t_query : array-like
            Time points at which to evaluate the reconstructed signal.
        
        Returns
        -------
        y_reconstructed : ndarray
            Reconstructed signal values.
        """
        t_query = np.atleast_1d(t_query)
        
        # Filter out effectively zero Fourier coefficients for efficiency
        tol = 1e-12 * np.max(np.abs(self.y_fft_filtered_))
        nonzero_mask = np.abs(self.y_fft_filtered_) > tol
        if not np.any(nonzero_mask):
            return np.full_like(t_query, self.y_mean_, dtype=float)
        
        frequencies_nonzero = self.frequencies_[nonzero_mask]
        y_fft_nonzero = self.y_fft_filtered_[nonzero_mask]
        
        # Fourier synthesis: sum of complex exponentials
        exponentials = np.exp(2j * np.pi * np.outer(t_query, frequencies_nonzero))
        y_reconstructed = (exponentials @ y_fft_nonzero).real / len(self.y_fft_filtered_)
        
        # Add back the mean
        return y_reconstructed + self.y_mean_


#%% Helper functions
def add_noise(data, noise_level):
    """Add Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise
