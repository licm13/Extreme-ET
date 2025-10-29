"""
Non-Stationary Threshold Analysis
Based on Zhao et al. (2025) recommendations for adapting thresholds to changing climatology

This module implements methods to detect and account for non-stationarity in
extreme ET thresholds, separating forced trends from natural variability.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import warnings


def calculate_moving_percentile(
    data: np.ndarray,
    percentile: float = 95.0,
    window_years: int = 30,
    days_per_year: int = 365
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate moving percentile threshold using a rolling window.

    This method allows thresholds to adapt to changing climatology,
    useful for non-stationary time series under climate change.

    Parameters
    ----------
    data : np.ndarray
        Daily time series data
    percentile : float
        Percentile to calculate (default: 95.0)
    window_years : int
        Window size in years for moving calculation (default: 30)
    days_per_year : int
        Number of days per year (default: 365)

    Returns
    -------
    tuple of np.ndarray
        - moving_threshold: time-varying threshold values
        - moving_percentile_values: actual percentile values at each point

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 365*50)  # 50 years of ET0 data
    >>> threshold, percentiles = calculate_moving_percentile(data, percentile=95, window_years=30)
    >>> # threshold will adapt to trends in the data
    """
    n = len(data)
    window_size = window_years * days_per_year

    moving_threshold = np.full(n, np.nan)
    moving_percentile_values = np.full(n, np.nan)

    for i in range(n):
        # Define window centered on current point
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)

        # Calculate percentile within window
        window_data = data[start:end]
        moving_threshold[i] = np.percentile(window_data, percentile)
        moving_percentile_values[i] = percentile

    return moving_threshold, moving_percentile_values


def loess_smoothed_threshold(
    data: np.ndarray,
    dates: np.ndarray,
    percentile: float = 95.0,
    smoothing_factor: float = 0.3
) -> np.ndarray:
    """
    Calculate smoothed threshold using LOESS (LOcally Estimated Scatterplot Smoothing).

    This approach separates long-term trends from short-term variability,
    providing a robust baseline for non-stationary extreme identification.

    Parameters
    ----------
    data : np.ndarray
        Daily time series data
    dates : np.ndarray
        Time index (e.g., days since start)
    percentile : float
        Base percentile for threshold (default: 95.0)
    smoothing_factor : float
        Smoothing parameter (0-1, higher = smoother, default: 0.3)

    Returns
    -------
    np.ndarray
        Smoothed threshold values

    Examples
    --------
    >>> dates = np.arange(365*50)  # 50 years
    >>> data = 5 + 0.01*dates/365 + np.random.gamma(2, 1, len(dates))  # Trending data
    >>> threshold = loess_smoothed_threshold(data, dates, percentile=95, smoothing_factor=0.3)
    """
    # Calculate base threshold
    base_threshold = np.percentile(data, percentile)

    # Apply LOESS-like smoothing using splines
    # Higher s = more smoothing
    s_value = smoothing_factor * len(data)

    try:
        # Fit spline to data
        spline = UnivariateSpline(dates, data, s=s_value, k=3)
        smoothed_mean = spline(dates)

        # Calculate residuals
        residuals = data - smoothed_mean

        # Apply percentile to residuals
        residual_threshold = np.percentile(residuals, percentile)

        # Reconstruct threshold
        threshold = smoothed_mean + residual_threshold

    except Exception as e:
        warnings.warn(f"LOESS smoothing failed: {e}. Using constant threshold.")
        threshold = np.full(len(data), base_threshold)

    return threshold


def detect_trend_and_detrend(
    data: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Detect linear trend in data and return detrended series.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    dates : np.ndarray, optional
        Time index (if None, uses sequential indices)

    Returns
    -------
    tuple
        - slope: trend slope (units per time step)
        - p_value: significance of trend
        - detrended_data: data with trend removed
        - trend_stats: dictionary with trend statistics

    Examples
    --------
    >>> data = 5 + 0.02*np.arange(365*30) + np.random.randn(365*30)
    >>> slope, p_val, detrended, stats = detect_trend_and_detrend(data)
    >>> print(f"Trend: {slope:.4f} per day, p={p_val:.4e}")
    """
    if dates is None:
        dates = np.arange(len(data))

    # Remove NaN values
    valid_mask = ~np.isnan(data)
    valid_dates = dates[valid_mask]
    valid_data = data[valid_mask]

    if len(valid_data) < 10:
        return 0.0, 1.0, data, {'trend': 'insufficient_data'}

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_dates, valid_data)

    # Calculate trend line
    trend_line = slope * dates + intercept

    # Detrend
    detrended_data = data - trend_line

    # Compile statistics
    trend_stats = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_error': std_err,
        'trend_significance': 'significant' if p_value < 0.05 else 'not_significant',
        'trend_direction': 'increasing' if slope > 0 else 'decreasing'
    }

    return slope, p_value, detrended_data, trend_stats


def adaptive_threshold_by_decade(
    data: np.ndarray,
    years: np.ndarray,
    percentile: float = 95.0,
    days_per_year: int = 365
) -> np.ndarray:
    """
    Calculate adaptive thresholds that change by decade.

    This method allows thresholds to adapt to climate change while
    maintaining some temporal stability within each decade.

    Parameters
    ----------
    data : np.ndarray
        Daily time series data
    years : np.ndarray
        Year for each data point
    percentile : float
        Percentile for threshold (default: 95.0)
    days_per_year : int
        Number of days per year (default: 365)

    Returns
    -------
    np.ndarray
        Decade-adaptive threshold values

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 365*50)
    >>> years = np.repeat(np.arange(1980, 2030), 365)[:len(data)]
    >>> threshold = adaptive_threshold_by_decade(data, years, percentile=95)
    """
    # Group by decade
    decades = (years // 10) * 10
    unique_decades = np.unique(decades)

    threshold = np.full(len(data), np.nan)

    for decade in unique_decades:
        decade_mask = decades == decade
        decade_data = data[decade_mask]

        if len(decade_data) > 0:
            decade_threshold = np.percentile(decade_data, percentile)
            threshold[decade_mask] = decade_threshold

    return threshold


def separate_forced_variability(
    data: np.ndarray,
    window_size: int = 365 * 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate forced (long-term) trends from natural (short-term) variability.

    Uses Savitzky-Golay filter to extract low-frequency components.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    window_size : int
        Window size for separation (default: 10 years = 3650 days)

    Returns
    -------
    tuple of np.ndarray
        - forced_component: long-term forced signal
        - natural_variability: short-term natural variations

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 365*50)
    >>> forced, natural = separate_forced_variability(data, window_size=365*10)
    >>> # forced contains decadal trends, natural contains interannual variability
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Limit window size to data length
    window_size = min(window_size, len(data) - 1)
    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 5:
        return data, np.zeros_like(data)

    try:
        # Apply Savitzky-Golay filter for smoothing
        forced_component = savgol_filter(data, window_length=window_size, polyorder=3)
        natural_variability = data - forced_component
    except Exception as e:
        warnings.warn(f"Forced/natural separation failed: {e}")
        forced_component = data
        natural_variability = np.zeros_like(data)

    return forced_component, natural_variability


def quantile_regression_threshold(
    data: np.ndarray,
    dates: np.ndarray,
    percentile: float = 95.0,
    n_quantiles: int = 100
) -> np.ndarray:
    """
    Calculate time-varying threshold using quantile regression approach.

    This provides a non-stationary threshold that can capture changes
    in the upper tail of the distribution over time.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    dates : np.ndarray
        Time index
    percentile : float
        Target percentile (default: 95.0)
    n_quantiles : int
        Number of quantiles to calculate (default: 100)

    Returns
    -------
    np.ndarray
        Time-varying threshold from quantile regression

    Examples
    --------
    >>> dates = np.arange(365*50)
    >>> data = 5 + 0.01*dates/365 + np.random.gamma(2, 1, len(dates))
    >>> threshold = quantile_regression_threshold(data, dates, percentile=95)
    """
    # Simple implementation: bin data by time and calculate quantiles
    n_bins = min(n_quantiles, len(data) // 100)
    bin_edges = np.linspace(dates.min(), dates.max(), n_bins + 1)

    threshold = np.full(len(data), np.nan)

    for i in range(n_bins):
        bin_mask = (dates >= bin_edges[i]) & (dates < bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            bin_threshold = np.percentile(data[bin_mask], percentile)
            threshold[bin_mask] = bin_threshold

    # Fill any remaining NaN values with overall percentile
    if np.any(np.isnan(threshold)):
        overall_threshold = np.percentile(data, percentile)
        threshold[np.isnan(threshold)] = overall_threshold

    # Smooth the threshold
    threshold = savgol_filter(threshold, window_length=min(365, len(threshold)-1), polyorder=2)

    return threshold


def compare_stationary_vs_nonstationary(
    data: np.ndarray,
    dates: np.ndarray,
    percentile: float = 95.0
) -> Dict[str, any]:
    """
    Compare extreme event detection using stationary vs non-stationary thresholds.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    dates : np.ndarray
        Time index
    percentile : float
        Percentile for threshold (default: 95.0)

    Returns
    -------
    dict
        Comparison results including:
        - stationary_threshold: constant threshold
        - nonstationary_threshold: time-varying threshold
        - stationary_extremes: mask using constant threshold
        - nonstationary_extremes: mask using adaptive threshold
        - n_extremes_difference: difference in detected events
        - trend_stats: trend analysis results

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 365*50)
    >>> dates = np.arange(len(data))
    >>> comparison = compare_stationary_vs_nonstationary(data, dates, percentile=95)
    >>> print(f"Extra events detected: {comparison['n_extremes_difference']}")
    """
    # Stationary threshold
    stationary_threshold = np.percentile(data, percentile)
    stationary_extremes = data > stationary_threshold

    # Non-stationary threshold
    nonstationary_threshold = loess_smoothed_threshold(data, dates, percentile)
    nonstationary_extremes = data > nonstationary_threshold

    # Trend analysis
    slope, p_value, detrended, trend_stats = detect_trend_and_detrend(data, dates)

    results = {
        'stationary_threshold': stationary_threshold,
        'nonstationary_threshold': nonstationary_threshold,
        'stationary_extremes': stationary_extremes,
        'nonstationary_extremes': nonstationary_extremes,
        'n_stationary_extremes': np.sum(stationary_extremes),
        'n_nonstationary_extremes': np.sum(nonstationary_extremes),
        'n_extremes_difference': np.sum(nonstationary_extremes) - np.sum(stationary_extremes),
        'trend_stats': trend_stats,
        'detrended_data': detrended
    }

    return results
