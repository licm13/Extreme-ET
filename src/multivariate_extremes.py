"""
Multivariate Extreme Analysis
Based on recommendations from Zhao et al. (2025) and extreme value theory

This module implements copula-based methods for analyzing compound extremes,
such as simultaneous extreme ET and low precipitation (flash drought conditions).
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy import stats
from scipy.interpolate import interp1d
import warnings


def empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, callable]:
    """
    Calculate empirical cumulative distribution function.

    Parameters
    ----------
    data : np.ndarray
        Input data

    Returns
    -------
    tuple
        - sorted_data: sorted data values
        - ecdf_func: callable ECDF function

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 1000)
    >>> sorted_vals, ecdf = empirical_cdf(data)
    >>> prob = ecdf(5.0)  # Get probability for value 5.0
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf_values = np.arange(1, n + 1) / (n + 1)  # Avoid 0 and 1

    ecdf_func = interp1d(sorted_data, ecdf_values,
                         bounds_error=False,
                         fill_value=(ecdf_values[0], ecdf_values[-1]))

    return sorted_data, ecdf_func


def transform_to_uniform(data: np.ndarray) -> np.ndarray:
    """
    Transform data to uniform [0,1] using empirical CDF.

    This is the first step in copula analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data

    Returns
    -------
    np.ndarray
        Uniform-transformed data in [0,1]

    Examples
    --------
    >>> data = np.random.gamma(3, 1.5, 1000)
    >>> uniform_data = transform_to_uniform(data)
    >>> # uniform_data should be approximately uniform in [0,1]
    """
    _, ecdf = empirical_cdf(data)
    uniform = ecdf(data)
    # Clip to avoid exactly 0 or 1 for copula analysis
    uniform = np.clip(uniform, 0.001, 0.999)
    return uniform


def gaussian_copula_parameter(u: np.ndarray, v: np.ndarray) -> float:
    """
    Estimate Gaussian copula correlation parameter (rho).

    Parameters
    ----------
    u, v : np.ndarray
        Uniform-transformed marginal variables

    Returns
    -------
    float
        Correlation parameter rho

    Examples
    --------
    >>> u = transform_to_uniform(np.random.randn(1000))
    >>> v = transform_to_uniform(np.random.randn(1000))
    >>> rho = gaussian_copula_parameter(u, v)
    """
    # Transform uniform to standard normal
    z_u = stats.norm.ppf(u)
    z_v = stats.norm.ppf(v)

    # Calculate correlation
    rho = np.corrcoef(z_u, z_v)[0, 1]

    return rho


def clayton_copula_parameter(u: np.ndarray, v: np.ndarray) -> float:
    """
    Estimate Clayton copula parameter (theta) using method of moments.

    Clayton copula is useful for modeling lower tail dependence
    (e.g., simultaneous low precipitation and high ET).

    Parameters
    ----------
    u, v : np.ndarray
        Uniform-transformed marginal variables

    Returns
    -------
    float
        Clayton parameter theta (>= 0)

    Examples
    --------
    >>> u = transform_to_uniform(data1)
    >>> v = transform_to_uniform(data2)
    >>> theta = clayton_copula_parameter(u, v)
    """
    # Kendall's tau
    tau = stats.kendalltau(u, v)[0]

    # For Clayton copula: tau = theta / (theta + 2)
    # Solve for theta
    if tau > 0:
        theta = 2 * tau / (1 - tau)
    else:
        theta = 0.01  # Small positive value

    return max(theta, 0.01)


def gumbel_copula_parameter(u: np.ndarray, v: np.ndarray) -> float:
    """
    Estimate Gumbel copula parameter (theta) using method of moments.

    Gumbel copula is useful for modeling upper tail dependence
    (e.g., simultaneous extreme ET and extreme temperature).

    Parameters
    ----------
    u, v : np.ndarray
        Uniform-transformed marginal variables

    Returns
    -------
    float
        Gumbel parameter theta (>= 1)

    Examples
    --------
    >>> u = transform_to_uniform(et0_data)
    >>> v = transform_to_uniform(temp_data)
    >>> theta = gumbel_copula_parameter(u, v)
    """
    # Kendall's tau
    tau = stats.kendalltau(u, v)[0]

    # For Gumbel copula: tau = 1 - 1/theta
    # Solve for theta
    if tau > 0:
        theta = 1 / (1 - tau)
    else:
        theta = 1.01  # Just above 1

    return max(theta, 1.01)


def calculate_joint_return_period(
    var1: np.ndarray,
    var2: np.ndarray,
    threshold1: float,
    threshold2: float,
    copula_type: str = 'gaussian'
) -> Dict[str, float]:
    """
    Calculate joint return period for compound extremes using copulas.

    Parameters
    ----------
    var1, var2 : np.ndarray
        Two variables (e.g., ET0 and precipitation deficit)
    threshold1, threshold2 : float
        Thresholds for defining extremes
    copula_type : str
        Type of copula: 'gaussian', 'clayton', or 'gumbel' (default: 'gaussian')

    Returns
    -------
    dict
        Dictionary with:
        - marginal_return_period_1: return period for var1 alone
        - marginal_return_period_2: return period for var2 alone
        - joint_and_return_period: P(X > x AND Y > y)
        - joint_or_return_period: P(X > x OR Y > y)
        - copula_parameter: fitted copula parameter

    Examples
    --------
    >>> et0 = np.random.gamma(3, 1.5, 365*50)
    >>> precip_deficit = -np.random.gamma(2, 1, 365*50)
    >>> rp = calculate_joint_return_period(et0, precip_deficit,
    ...                                    threshold1=np.percentile(et0, 95),
    ...                                    threshold2=np.percentile(precip_deficit, 95),
    ...                                    copula_type='clayton')
    >>> print(f"Joint return period (AND): {rp['joint_and_return_period']:.1f} events")
    """
    n = len(var1)

    # Transform to uniform
    u = transform_to_uniform(var1)
    v = transform_to_uniform(var2)

    # Fit copula
    if copula_type == 'gaussian':
        param = gaussian_copula_parameter(u, v)
    elif copula_type == 'clayton':
        param = clayton_copula_parameter(u, v)
    elif copula_type == 'gumbel':
        param = gumbel_copula_parameter(u, v)
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")

    # Calculate marginal exceedance probabilities
    p1 = np.sum(var1 > threshold1) / n
    p2 = np.sum(var2 > threshold2) / n

    # Marginal return periods (in number of events, not years)
    rp1 = 1 / p1 if p1 > 0 else np.inf
    rp2 = 1 / p2 if p2 > 0 else np.inf

    # Joint probabilities
    # For simplicity, use empirical joint probability
    joint_and = np.sum((var1 > threshold1) & (var2 > threshold2)) / n
    joint_or = np.sum((var1 > threshold1) | (var2 > threshold2)) / n

    # Joint return periods
    rp_and = 1 / joint_and if joint_and > 0 else np.inf
    rp_or = 1 / joint_or if joint_or > 0 else np.inf

    return {
        'marginal_return_period_1': rp1,
        'marginal_return_period_2': rp2,
        'joint_and_return_period': rp_and,
        'joint_or_return_period': rp_or,
        'marginal_prob_1': p1,
        'marginal_prob_2': p2,
        'joint_and_prob': joint_and,
        'joint_or_prob': joint_or,
        'copula_type': copula_type,
        'copula_parameter': param
    }


def identify_compound_extreme_et_precipitation(
    et0: np.ndarray,
    precipitation: np.ndarray,
    et0_percentile: float = 95.0,
    precip_percentile: float = 5.0
) -> Dict[str, any]:
    """
    Identify compound extreme events: high ET0 + low precipitation (flash drought).

    Parameters
    ----------
    et0 : np.ndarray
        Daily reference evapotranspiration (mm/day)
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    et0_percentile : float
        Percentile for extreme high ET0 (default: 95.0)
    precip_percentile : float
        Percentile for extreme low precipitation (default: 5.0)

    Returns
    -------
    dict
        Dictionary with:
        - compound_events_mask: boolean mask for compound events
        - n_compound_events: number of compound extreme days
        - et0_only_mask: extreme ET0 but normal precipitation
        - precip_only_mask: extreme low precip but normal ET0
        - copula_analysis: results from copula-based analysis

    Examples
    --------
    >>> et0 = np.random.gamma(3, 1.5, 365*50)
    >>> precip = np.random.gamma(2, 2, 365*50)
    >>> compound = identify_compound_extreme_et_precipitation(et0, precip)
    >>> print(f"Compound extreme days: {compound['n_compound_events']}")
    """
    # Define thresholds
    et0_threshold = np.percentile(et0, et0_percentile)
    precip_threshold = np.percentile(precipitation, precip_percentile)

    # Identify extremes
    extreme_et0 = et0 > et0_threshold
    extreme_low_precip = precipitation < precip_threshold

    # Compound events
    compound_events = extreme_et0 & extreme_low_precip

    # Single-variable extremes
    et0_only = extreme_et0 & ~extreme_low_precip
    precip_only = extreme_low_precip & ~extreme_et0

    # Copula analysis (use negative precip to align tails)
    neg_precip = -precipitation
    copula_results = calculate_joint_return_period(
        et0, neg_precip,
        et0_threshold, -precip_threshold,
        copula_type='clayton'  # Clayton for lower tail dependence
    )

    return {
        'compound_events_mask': compound_events,
        'n_compound_events': np.sum(compound_events),
        'et0_only_mask': et0_only,
        'n_et0_only': np.sum(et0_only),
        'precip_only_mask': precip_only,
        'n_precip_only': np.sum(precip_only),
        'et0_threshold': et0_threshold,
        'precip_threshold': precip_threshold,
        'copula_analysis': copula_results,
        'compound_frequency': np.sum(compound_events) / len(et0)
    }


def calculate_drought_severity_index(
    et0: np.ndarray,
    precipitation: np.ndarray,
    soil_moisture: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate compound drought severity index based on ET demand and water supply.

    Integrates atmospheric evaporative demand (ET0) with water availability (P).
    Optionally includes soil moisture feedback.

    Parameters
    ----------
    et0 : np.ndarray
        Daily reference ET0 (mm/day)
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    soil_moisture : np.ndarray, optional
        Soil moisture content (if available)

    Returns
    -------
    np.ndarray
        Drought severity index (higher = more severe drought conditions)

    Examples
    --------
    >>> et0 = np.random.gamma(3, 1.5, 365*50)
    >>> precip = np.random.gamma(2, 1.5, 365*50)
    >>> dsi = calculate_drought_severity_index(et0, precip)
    >>> severe_drought_days = dsi > np.percentile(dsi, 95)
    """
    # Standardize variables
    et0_std = (et0 - np.mean(et0)) / np.std(et0)
    precip_std = (precipitation - np.mean(precipitation)) / np.std(precipitation)

    # Base index: high ET demand + low precipitation
    dsi = et0_std - precip_std

    # If soil moisture available, incorporate it
    if soil_moisture is not None:
        sm_std = (soil_moisture - np.mean(soil_moisture)) / np.std(soil_moisture)
        dsi = dsi - sm_std  # Low soil moisture worsens drought

    return dsi


def analyze_compound_event_characteristics(
    et0: np.ndarray,
    precipitation: np.ndarray,
    compound_mask: np.ndarray
) -> Dict[str, any]:
    """
    Analyze characteristics of identified compound extreme events.

    Parameters
    ----------
    et0 : np.ndarray
        Daily ET0 (mm/day)
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    compound_mask : np.ndarray
        Boolean mask for compound events

    Returns
    -------
    dict
        Event characteristics including:
        - mean_et0_during_events
        - mean_precip_during_events
        - mean_duration
        - total_water_deficit
        - intensity_metrics

    Examples
    --------
    >>> compound = identify_compound_extreme_et_precipitation(et0, precip)
    >>> characteristics = analyze_compound_event_characteristics(
    ...     et0, precip, compound['compound_events_mask'])
    >>> print(f"Mean ET0 during compound events: {characteristics['mean_et0_during_events']:.2f} mm/day")
    """
    if not np.any(compound_mask):
        return {'error': 'No compound events detected'}

    # Extract event periods
    events = []
    in_event = False
    start_idx = 0

    for i, is_extreme in enumerate(compound_mask):
        if is_extreme and not in_event:
            start_idx = i
            in_event = True
        elif not is_extreme and in_event:
            events.append((start_idx, i))
            in_event = False

    if in_event:  # Close last event
        events.append((start_idx, len(compound_mask)))

    # Analyze events
    durations = [end - start for start, end in events]
    mean_duration = np.mean(durations) if durations else 0

    # Metrics during compound events
    mean_et0 = np.mean(et0[compound_mask])
    mean_precip = np.mean(precipitation[compound_mask])
    water_deficit = mean_et0 - mean_precip

    # Intensity: cumulative water deficit per event
    event_intensities = []
    for start, end in events:
        event_deficit = np.sum(et0[start:end] - precipitation[start:end])
        event_intensities.append(event_deficit)

    return {
        'n_events': len(events),
        'mean_duration': mean_duration,
        'max_duration': max(durations) if durations else 0,
        'mean_et0_during_events': mean_et0,
        'mean_precip_during_events': mean_precip,
        'mean_water_deficit_rate': water_deficit,
        'mean_event_intensity': np.mean(event_intensities) if event_intensities else 0,
        'max_event_intensity': max(event_intensities) if event_intensities else 0,
        'total_water_deficit': np.sum(et0[compound_mask] - precipitation[compound_mask]),
        'event_list': events
    }
