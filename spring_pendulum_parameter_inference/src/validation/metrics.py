"""
Metrics for validation and evaluation.
"""

import numpy as np

def mean_squared_relative_error(y_true, y_pred, epsilon=1e-8):
	"""
	Compute the mean squared relative error (MSRE) between y_true and y_pred.
	Excludes entries where abs(y_true) <= epsilon to avoid division by zero.
	Parameters
	----------
	y_true : array-like
		Ground truth (target) values.
	y_pred : array-like
		Predicted values.
	epsilon : float, optional
		Tolerance for excluding values near zero (default: 1e-8).
	Returns
	-------
	msre : float
		Mean squared relative error, or np.nan if all y_true values are near zero.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	mask = np.abs(y_true) > epsilon
	if not np.any(mask):
		return np.nan  # or raise an error if preferred
	rel_error = (y_true[mask] - y_pred[mask]) / y_true[mask]
	return np.mean(np.square(rel_error))
