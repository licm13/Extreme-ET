"""
Data processing functions for extreme evaporation analysis
"""

import numpy as np
import pandas as pd
from scipy import stats


def standardize_to_zscore(data, pentad=True):
    """
    Transform data to z-scores over pentads (5-day periods) or daily.
    
    Following Markonis (2025), removes seasonality by standardizing 
    within each pentad/day of year.
    
    Parameters
    ----------
    data : array-like
        Time series data (daily resolution)
    pentad : bool, default=True
        If True, standardize by pentad (5-day periods)
        If False, standardize by calendar day
    
    Returns
    -------
    z_scores : np.ndarray
        Standardized data
    
    Examples
    --------
    >>> data = np.random.randn(365 * 10)
    >>> z_scores = standardize_to_zscore(data, pentad=True)
    """
    data = np.asarray(data)
    n_days = len(data)
    z_scores = np.zeros_like(data, dtype=float)
    
    if pentad:
        # Group by pentad (72 or 73 pentads per year)
        pentads_per_year = 73
        for pentad_idx in range(pentads_per_year):
            # Get indices for this pentad across all years
            pentad_mask = np.arange(n_days) % pentads_per_year == pentad_idx
            pentad_data = data[pentad_mask]
            
            if len(pentad_data) > 1:
                mean = np.mean(pentad_data)
                std = np.std(pentad_data, ddof=1)
                if std > 0:
                    z_scores[pentad_mask] = (pentad_data - mean) / std
    else:
        # Group by calendar day (1-366)
        for day_of_year in range(1, 367):
            # Get indices for this calendar day across all years
            day_mask = (np.arange(n_days) % 365) == (day_of_year - 1)
            day_data = data[day_mask]
            
            if len(day_data) > 1:
                mean = np.mean(day_data)
                std = np.std(day_data, ddof=1)
                if std > 0:
                    z_scores[day_mask] = (day_data - mean) / std
    
    return z_scores


def calculate_hurst_exponent(data, max_lag=10):
    """
    Calculate Hurst coefficient using Maximum Likelihood Estimation.
    
    Following Markonis (2025), estimates long-term persistence/clustering.
    H ~ 0.5: no autocorrelation (white noise)
    H > 0.5: long-term persistence (clustering)
    H < 0.5: anti-persistence
    
    Parameters
    ----------
    data : array-like
        Time series data
    max_lag : int, default=10
        Maximum lag for autocorrelation calculation
    
    Returns
    -------
    hurst : float
        Hurst exponent estimate
    
    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> H = calculate_hurst_exponent(data)
    >>> print(f"Hurst exponent: {H:.3f}")
    """
    data = np.asarray(data)
    n_samples = len(data)
    
    # Calculate autocorrelation for different lags using vectorized operations
    lags = np.arange(1, min(max_lag + 1, n_samples // 4))
    
    # Pre-compute mean and standard deviation to avoid repeated calculations
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    if data_std == 0:
        return 0.5
    
    # Vectorized autocorrelation calculation (more efficient than list comprehension)
    autocorr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        # Use Pearson correlation formula directly for efficiency
        autocorr[i] = np.corrcoef(data[:-lag], data[lag:])[0, 1]
    
    # Remove NaN values
    valid_mask = ~np.isnan(autocorr)
    lags = lags[valid_mask]
    autocorr = autocorr[valid_mask]
    
    if len(lags) < 2:
        return 0.5
    
    # Fit log-log relationship
    # autocorr(k) ~ k^(2H-2) for large k
    log_lags = np.log(lags)
    log_autocorr = np.log(np.abs(autocorr) + 1e-10)
    
    slope, _ = np.polyfit(log_lags, log_autocorr, 1)
    hurst = (slope + 2) / 2
    
    # Constrain to [0, 1]
    hurst = np.clip(hurst, 0, 1)
    
    return hurst


def moving_average(data, window=7):
    """
    Calculate moving average with specified window size.
    
    Parameters
    ----------
    data : array-like
        Time series data
    window : int, default=7
        Window size for moving average
    
    Returns
    -------
    smoothed : np.ndarray
        Smoothed data
    
    Examples
    --------
    >>> data = np.random.randn(100)
    >>> smoothed = moving_average(data, window=7)
    """
    data = np.asarray(data)
    
    if window <= 1:
        return data.copy()
    
    # Use pandas for efficient rolling mean
    series = pd.Series(data)
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    
    return smoothed.values


def calculate_autocorrelation(data, max_lag=10):
    """
    Calculate autocorrelation function up to max_lag.
    
    Parameters
    ----------
    data : array-like
        Time series data
    max_lag : int, default=10
        Maximum lag
    
    Returns
    -------
    lags : np.ndarray
        Lag values
    autocorr : np.ndarray
        Autocorrelation coefficients
    
    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> lags, acf = calculate_autocorrelation(data, max_lag=10)
    """
    data = np.asarray(data)
    n_samples = len(data)
    
    lags = np.arange(0, min(max_lag + 1, n_samples // 2))
    autocorr = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag == 0:
            autocorr[i] = 1.0
        else:
            autocorr[i] = np.corrcoef(data[:-lag], data[lag:])[0, 1]
    
    return lags, autocorr


def calculate_climatological_means(*arrays):
    """
    Calculate climatological daily means for one or more time series.
    
    This helper function efficiently computes the mean value for each calendar 
    day (1-365) across all years in the dataset. It's optimized to process 
    multiple arrays simultaneously, avoiding redundant day_mask calculations.
    
    Parameters
    ----------
    *arrays : array-like
        One or more time series arrays of the same length (daily resolution).
        Each array should contain at least one full year of data.
    
    Returns
    -------
    climatological_means : np.ndarray or tuple of np.ndarray
        Climatological means for each input array, broadcast to the original 
        data length. If one array is provided, returns a single array. If 
        multiple arrays are provided, returns a tuple of arrays.
    
    Examples
    --------
    >>> # Single array
    >>> data = np.random.randn(365 * 10)
    >>> clim = calculate_climatological_means(data)
    
    >>> # Multiple arrays
    >>> temp = np.random.randn(365 * 10)
    >>> precip = np.random.randn(365 * 10)
    >>> temp_clim, precip_clim = calculate_climatological_means(temp, precip)
    
    Notes
    -----
    This function is more efficient than computing climatological means 
    separately for each variable, as it reuses the day_mask computation.
    """
    if not arrays:
        raise ValueError("At least one array must be provided")
    
    # Convert all inputs to arrays and validate they have the same length
    arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    n_days = len(arrays[0])
    
    if not all(len(arr) == n_days for arr in arrays):
        raise ValueError("All input arrays must have the same length")
    
    # Pre-allocate output arrays
    climatological_means = [np.zeros(n_days) for _ in arrays]
    
    # Compute day masks once and reuse for all arrays
    for day in range(365):
        day_mask = (np.arange(n_days) % 365) == day
        for i, arr in enumerate(arrays):
            climatological_means[i][day_mask] = np.mean(arr[day_mask])
    
    # Return single array or tuple based on input
    if len(arrays) == 1:
        return climatological_means[0]
    return tuple(climatological_means)


def deseasonalize_data(data, method='difference'):
    """
    Remove seasonal component from data.
    
    Parameters
    ----------
    data : array-like
        Time series data
    method : str, default='difference'
        Method for deseasonalization: 'difference', 'ratio', or 'zscore'
    
    Returns
    -------
    deseasonalized : np.ndarray
        Deseasonalized data
    seasonal : np.ndarray
        Seasonal component
    """
    data = np.asarray(data)
    n_days = len(data)
    
    # Calculate seasonal component (365-day period) using optimized helper
    seasonal = np.zeros(365)
    for day in range(365):
        day_mask = np.arange(n_days) % 365 == day
        seasonal[day] = np.mean(data[day_mask])
    
    # Repeat seasonal component to match data length
    seasonal_full = np.tile(seasonal, n_days // 365 + 1)[:n_days]
    
    if method == 'difference':
        deseasonalized = data - seasonal_full
    elif method == 'ratio':
        deseasonalized = data / (seasonal_full + 1e-10)
    elif method == 'zscore':
        deseasonalized = standardize_to_zscore(data, pentad=False)
        seasonal_full = deseasonalized  # Already standardized
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return deseasonalized, seasonal_full