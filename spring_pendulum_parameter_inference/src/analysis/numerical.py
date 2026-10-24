"""
Numerical methods and mathematical utilities for time series analysis.
"""
import numpy as np

def differentiate_finitediff(t, x, n):
    """
    Computes the nth derivative of a signal using finite difference methods, 
    up to n = 2.
    """
    dt = t[1] - t[0]  # Assuming uniform time steps

    if n == 0: 
        return x
    elif n == 1:
        dnx_dtn = np.zeros_like(x)
        dnx_dtn[1:-1] = (x[2:]-x[:-2])/(2*dt)    # interior points: central
        dnx_dtn[0] = (x[1]-x[0])/dt              # first point: forward
        dnx_dtn[-1] = (x[-1]-x[-2])/dt           # last point: backward
    elif n == 2:
        dnx_dtn = np.zeros_like(x)
        dnx_dtn[1:-1] = (x[2:]-2*x[1:-1]+x[:-2])/dt**2
        dnx_dtn[0] = (x[2]-2*x[1]+x[0])/dt**2
        dnx_dtn[-1] = (x[-1]-2*x[-2]+x[-3])/dt**2
    else:
        raise ValueError(f"Derivative order {n} not implemented")
    
    return dnx_dtn


def finite_difference_derivative(x, y, order=1):
    """Compute derivatives using finite differences. Independent variable `x`
    does not need to be uniformly spaced. Returns the derivative of `y` with 
    respect to `x` of the same length as `y`."""
    
    result = np.asarray(y, dtype=float)
    x = np.asarray(x)
    
    for _ in range(order):
        result = np.gradient(result, x)
    
    return result

def fourier_derivative(x, y, order=1):
    """Compute derivatives using FFT. Assumes periodic data with uniform spacing.
    Returns the derivative of `y` with respect to `x` of the same length as `y`."""
    
    x, y = np.asarray(x), np.asarray(y, dtype=complex)
    n = len(y)
    
    # Check for uniform spacing
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Fourier method requires uniform spacing")
    
    # FFT and frequency array
    y_fft = np.fft.fft(y)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    
    # Apply derivative in frequency domain
    for _ in range(order):
        y_fft *= 1j * k
    
    # Transform back and return real part
    result = np.fft.ifft(y_fft)
    return np.real(result)


# %% Time intervals
def generate_time_intervals(time_range, interval_sep, interval_len, sample = False, sample_interval = None, full = False, include_partial = True):
    """Generate overlapping time intervals for parameter inference. 
    If sample is True, returns only the sample interval. 
    If full is True, appends the full time range.
    If include_partial is True, includes partial intervals at the end that are shorter than interval_len."""

    # Generate time intervals
    time_intervals = np.column_stack([
        np.arange(time_range[0], time_range[1] - interval_len + 1, interval_sep),
        np.arange(time_range[0], time_range[1] - interval_len + 1, interval_sep) + interval_len
    ])

    # Add partial intervals at the end if include_partial is True
    if include_partial and time_intervals.shape[0] > 0:
        last_start = time_intervals[-1, 0] + interval_sep
        while last_start < time_range[1]:
            partial_end = min(last_start + interval_len, time_range[1])
            if partial_end > last_start:  # Only add if there's actual time span
                partial_interval = np.array([[last_start, partial_end]])
                time_intervals = np.vstack([time_intervals, partial_interval])
            last_start += interval_sep

    if time_intervals.shape[0] == 0: # Handle edge case where no intervals fit
        end_time = min(time_range[0] + interval_len, time_range[1])
        time_intervals = np.array([time_range[0], end_time]).reshape(1, 2)

    # If sample is True and sample_interval is provided, return only the sample interval
    if sample_interval is None: sample_interval = []
    if sample and len(sample_interval) == 2:
        sample_start, sample_end = sample_interval
        time_intervals = np.array([[sample_start, sample_end]])
    
    # If full is True, append the full time range
    if full:
        full_interval = np.array([[time_range[0], time_range[1]]])
        time_intervals = np.vstack([time_intervals, full_interval])
    
    return time_intervals.tolist()