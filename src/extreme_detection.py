"""
Extreme event detection methods
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from .data_processing import moving_average


def detect_extreme_events_hist(data, severity=0.005, return_details=False):
    """
    Detect extreme events using historical ERT method (ERT_hist).
    
    Following Zhao et al. (2025), identifies extreme days when values 
    exceed a threshold defined over the entire historical record.
    
    Parameters
    ----------
    data : array-like
        Time series data (e.g., daily ET0)
    severity : float, default=0.005
        Occurrence rate for extreme events (e.g., 0.005 = 0.5% = ~1.8 days/year)
    return_details : bool, default=False
        If True, return additional details about events
    
    Returns
    -------
    extreme_mask : np.ndarray (bool)
        Boolean array indicating extreme days
    threshold : float
        Threshold value used
    details : dict (optional)
        Additional event details if return_details=True
    
    Examples
    --------
    >>> data = np.random.gamma(2, 2, 365 * 40)
    >>> extreme_mask, threshold = detect_extreme_events_hist(data, severity=0.005)
    >>> print(f"Threshold: {threshold:.2f}")
    >>> print(f"Number of extreme days: {np.sum(extreme_mask)}")
    """
    data = np.asarray(data)
    n = len(data)
    
    # Calculate threshold based on severity (occurrence rate)
    quantile = 1 - severity
    threshold = np.quantile(data, quantile)
    
    # Identify extreme days
    extreme_mask = data > threshold
    
    if return_details:
        # Calculate event statistics
        n_extreme_days = np.sum(extreme_mask)
        occurrence_rate = n_extreme_days / n
        
        # Identify continuous events
        events = identify_events_from_mask(extreme_mask)
        
        details = {
            'threshold': threshold,
            'n_extreme_days': n_extreme_days,
            'occurrence_rate': occurrence_rate,
            'n_events': len(events),
            'events': events,
            'severity_level': severity,
        }
        
        return extreme_mask, threshold, details
    
    return extreme_mask, threshold


def detect_extreme_events_clim(data, severity=0.05, min_duration=3, 
                               window=7, return_details=False):
    """
    Detect extreme events using climatological ERT method (ERT_clim).
    
    Following Zhao et al. (2025), identifies anomalies relative to each 
    calendar day, similar to heatwave detection.
    
    Parameters
    ----------
    data : array-like
        Time series data (e.g., daily ET0)
    severity : float, default=0.05
        Occurrence rate for extreme events (e.g., 0.05 = 5%)
    min_duration : int, default=3
        Minimum number of consecutive days for an event
    window : int, default=7
        Window size for moving average preprocessing
    return_details : bool, default=False
        If True, return additional details
    
    Returns
    -------
    extreme_mask : np.ndarray (bool)
        Boolean array indicating extreme days
    thresholds : np.ndarray
        Daily threshold values (366 values for each calendar day)
    details : dict (optional)
        Additional event details if return_details=True
    
    Examples
    --------
    >>> data = np.random.gamma(2, 2, 365 * 40)
    >>> extreme_mask, thresholds = detect_extreme_events_clim(data, severity=0.05)
    >>> print(f"Mean threshold: {np.mean(thresholds):.2f}")
    """
    data = np.asarray(data)
    n = len(data)
    
    # Apply moving average preprocessing
    smoothed_data = moving_average(data, window=window)
    
    # Calculate thresholds for each calendar day using OPT method
    thresholds = optimal_path_threshold(
        smoothed_data, 
        target_occurrence_rate=severity,
        min_duration=min_duration
    )
    
    # Identify extreme events
    extreme_mask = identify_climatological_extremes(
        smoothed_data, 
        thresholds, 
        min_duration=min_duration
    )
    
    if return_details:
        # Calculate event statistics
        events = identify_events_from_mask(extreme_mask)
        
        details = {
            'thresholds': thresholds,
            'n_extreme_days': np.sum(extreme_mask),
            'occurrence_rate': np.sum(extreme_mask) / n,
            'n_events': len(events),
            'events': events,
            'severity_level': severity,
            'min_duration': min_duration,
            'window': window,
        }
        
        return extreme_mask, thresholds, details
    
    return extreme_mask, thresholds


def optimal_path_threshold(data, target_occurrence_rate=0.05, 
                           min_duration=3, max_iterations=50):
    """
    Calculate optimal thresholds for each calendar day using OPT method.
    
    Following Zhao et al. (2025), uses iterative optimization to find 
    thresholds that achieve the target occurrence rate.
    
    Parameters
    ----------
    data : array-like
        Time series data
    target_occurrence_rate : float
        Desired occurrence rate for extreme events
    min_duration : int
        Minimum duration for events
    max_iterations : int
        Maximum number of optimization iterations
    
    Returns
    -------
    thresholds : np.ndarray
        Optimal threshold for each calendar day (366 values)
    
    Examples
    --------
    >>> data = np.random.gamma(2, 2, 365 * 40)
    >>> thresholds = optimal_path_threshold(data, target_occurrence_rate=0.05)
    """
    data = np.asarray(data)
    n = len(data)
    
    # Calculate distribution for each calendar day
    day_distributions = {}
    for day in range(366):
        day_mask = (np.arange(n) % 365) == (day % 365)
        day_data = data[day_mask]
        if len(day_data) > 0:
            day_distributions[day] = day_data
    
    # Initialize with 90th percentile
    initial_quantile = 0.90
    thresholds = np.array([np.quantile(day_distributions.get(d, [0]), initial_quantile) 
                           for d in range(366)])
    
    # Iterative optimization
    for iteration in range(max_iterations):
        # Test current thresholds
        extreme_mask = identify_climatological_extremes(
            data, thresholds, min_duration=min_duration
        )
        
        current_rate = np.sum(extreme_mask) / n
        
        # Check convergence
        if np.abs(current_rate - target_occurrence_rate) < 0.001:
            break
        
        # Adjust thresholds
        if current_rate < target_occurrence_rate:
            # Lower thresholds
            adjustment = 0.95
        else:
            # Raise thresholds
            adjustment = 1.05
        
        thresholds *= adjustment
    
    return thresholds


def identify_climatological_extremes(data, thresholds, min_duration=3):
    """
    Identify extreme events based on daily thresholds and minimum duration.
    
    Parameters
    ----------
    data : array-like
        Time series data
    thresholds : array-like
        Threshold for each calendar day (366 values)
    min_duration : int
        Minimum consecutive days above threshold
    
    Returns
    -------
    extreme_mask : np.ndarray (bool)
        Boolean array indicating extreme days
    """
    data = np.asarray(data)
    thresholds = np.asarray(thresholds)
    n = len(data)
    
    # Get threshold for each day
    daily_thresholds = thresholds[np.arange(n) % 365]
    
    # Identify days above threshold
    above_threshold = data > daily_thresholds
    
    # Apply minimum duration criterion
    extreme_mask = np.zeros(n, dtype=bool)
    
    i = 0
    while i < n:
        if above_threshold[i]:
            # Count consecutive days
            j = i
            while j < n and above_threshold[j]:
                j += 1
            
            duration = j - i
            if duration >= min_duration:
                extreme_mask[i:j] = True
            
            i = j
        else:
            i += 1
    
    return extreme_mask


def identify_events_from_mask(mask):
    """
    Extract event periods from boolean mask.
    
    Parameters
    ----------
    mask : array-like (bool)
        Boolean array indicating extreme days
    
    Returns
    -------
    events : list of dict
        List of events with start, end, and duration
    
    Examples
    --------
    >>> mask = np.array([False, True, True, True, False, True, True, False])
    >>> events = identify_events_from_mask(mask)
    >>> for event in events:
    ...     print(f"Duration: {event['duration']} days")
    """
    mask = np.asarray(mask, dtype=bool)
    events = []
    
    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            end = i - 1
            
            events.append({
                'start': start,
                'end': end,
                'duration': end - start + 1,
            })
        else:
            i += 1
    
    return events


def calculate_event_statistics(data, events):
    """
    Calculate statistics for identified events.
    
    Parameters
    ----------
    data : array-like
        Time series data
    events : list of dict
        Events from identify_events_from_mask
    
    Returns
    -------
    event_stats : list of dict
        Enhanced event information with statistics
    """
    data = np.asarray(data)
    event_stats = []
    
    for event in events:
        start, end = event['start'], event['end']
        event_data = data[start:end+1]
        
        stats = {
            **event,
            'mean': np.mean(event_data),
            'max': np.max(event_data),
            'total': np.sum(event_data),
            'std': np.std(event_data),
        }
        
        event_stats.append(stats)
    
    return event_stats