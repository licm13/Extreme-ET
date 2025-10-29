"""
Water Cycle Acceleration Analysis
Based on Markonis (2025) - "On the Definition of Extreme Evaporation Events"

This module implements advanced water cycle metrics to understand the impact
of extreme ET events on regional water availability and cycle intensification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats


def calculate_water_availability(
    precipitation: np.ndarray,
    evaporation: np.ndarray
) -> np.ndarray:
    """
    Calculate water availability as P - E (precipitation minus evaporation).

    This metric represents the wetness or dryness of the region:
    - Positive values: net water surplus (wetter conditions)
    - Negative values: net water deficit (drier conditions)

    Parameters
    ----------
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    evaporation : np.ndarray
        Daily evaporation or ET0 (mm/day)

    Returns
    -------
    np.ndarray
        Water availability (P - E) in mm/day

    References
    ----------
    Markonis, Y. (2025). On the definition of extreme evaporation events.
    Geophysical Research Letters, 52, e2024GL113038.
    """
    return precipitation - evaporation


def calculate_water_cycle_intensity(
    precipitation: np.ndarray,
    evaporation: np.ndarray
) -> np.ndarray:
    """
    Calculate water cycle intensification as (P + E) / 2.

    This index quantifies the rate of land-atmosphere water exchange,
    following the formulation of Huntington et al. (2018) and extended
    by Markonis (2025).

    Parameters
    ----------
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    evaporation : np.ndarray
        Daily evaporation or ET0 (mm/day)

    Returns
    -------
    np.ndarray
        Water cycle intensity ((P + E) / 2) in mm/day

    References
    ----------
    Huntington, T. G., et al. (2018). A new indicator framework for
    quantifying the intensity of the terrestrial water cycle.
    Journal of Hydrology, 559, 361-372.
    """
    return (precipitation + evaporation) / 2


def decompose_water_cycle_by_extremes(
    precipitation: np.ndarray,
    evaporation: np.ndarray,
    extreme_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Decompose water cycle changes within and outside extreme ET events.

    This analysis reveals how extreme ET events differentially impact
    water availability and cycle acceleration compared to normal conditions.

    Parameters
    ----------
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    evaporation : np.ndarray
        Daily evaporation or ET0 (mm/day)
    extreme_mask : np.ndarray
        Boolean mask indicating extreme ET days

    Returns
    -------
    dict
        Dictionary with keys:
        - 'all_days': metrics for all days
        - 'extreme_days': metrics during extreme events
        - 'normal_days': metrics during non-extreme periods

        Each contains:
        - 'water_availability_mean': mean P-E
        - 'water_availability_std': std of P-E
        - 'water_cycle_intensity_mean': mean (P+E)/2
        - 'water_cycle_intensity_std': std of (P+E)/2
        - 'p_mean': mean precipitation
        - 'e_mean': mean evaporation
        - 'n_days': number of days

    Examples
    --------
    >>> import numpy as np
    >>> from extreme_et.water_cycle_analysis import decompose_water_cycle_by_extremes
    >>>
    >>> # Generate example data
    >>> np.random.seed(42)
    >>> days = 365 * 10
    >>> precip = np.random.gamma(2, 2, days)  # Rainfall
    >>> et0 = np.random.gamma(3, 1.5, days)  # ET0
    >>> extremes = et0 > np.percentile(et0, 95)
    >>>
    >>> # Analyze water cycle decomposition
    >>> results = decompose_water_cycle_by_extremes(precip, et0, extremes)
    >>> print(f"Water availability during extremes: {results['extreme_days']['water_availability_mean']:.2f} mm/day")
    >>> print(f"Water cycle intensity during extremes: {results['extreme_days']['water_cycle_intensity_mean']:.2f} mm/day")
    """
    water_avail = calculate_water_availability(precipitation, evaporation)
    water_intensity = calculate_water_cycle_intensity(precipitation, evaporation)

    normal_mask = ~extreme_mask

    results = {}

    # All days
    results['all_days'] = {
        'water_availability_mean': np.mean(water_avail),
        'water_availability_std': np.std(water_avail),
        'water_cycle_intensity_mean': np.mean(water_intensity),
        'water_cycle_intensity_std': np.std(water_intensity),
        'p_mean': np.mean(precipitation),
        'e_mean': np.mean(evaporation),
        'n_days': len(water_avail)
    }

    # Extreme days
    if np.any(extreme_mask):
        results['extreme_days'] = {
            'water_availability_mean': np.mean(water_avail[extreme_mask]),
            'water_availability_std': np.std(water_avail[extreme_mask]),
            'water_cycle_intensity_mean': np.mean(water_intensity[extreme_mask]),
            'water_cycle_intensity_std': np.std(water_intensity[extreme_mask]),
            'p_mean': np.mean(precipitation[extreme_mask]),
            'e_mean': np.mean(evaporation[extreme_mask]),
            'n_days': np.sum(extreme_mask)
        }
    else:
        results['extreme_days'] = None

    # Normal days
    if np.any(normal_mask):
        results['normal_days'] = {
            'water_availability_mean': np.mean(water_avail[normal_mask]),
            'water_availability_std': np.std(water_avail[normal_mask]),
            'water_cycle_intensity_mean': np.mean(water_intensity[normal_mask]),
            'water_cycle_intensity_std': np.std(water_intensity[normal_mask]),
            'p_mean': np.mean(precipitation[normal_mask]),
            'e_mean': np.mean(evaporation[normal_mask]),
            'n_days': np.sum(normal_mask)
        }
    else:
        results['normal_days'] = None

    return results


def analyze_temporal_changes(
    precipitation: np.ndarray,
    evaporation: np.ndarray,
    extreme_mask: np.ndarray,
    split_year_idx: int
) -> Dict[str, Dict[str, float]]:
    """
    Analyze temporal changes in water cycle metrics between two periods.

    This function compares water cycle characteristics before and after
    a specified split point (e.g., comparing 1981-2001 vs 2002-2022).

    Parameters
    ----------
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    evaporation : np.ndarray
        Daily evaporation or ET0 (mm/day)
    extreme_mask : np.ndarray
        Boolean mask indicating extreme ET days
    split_year_idx : int
        Index to split the time series (e.g., day index for Jan 1, 2002)

    Returns
    -------
    dict
        Dictionary with keys 'period1', 'period2', 'changes'
        Each period contains decomposed water cycle metrics
        'changes' contains the ratios: period2 / period1

    Examples
    --------
    >>> # Compare two 10-year periods
    >>> split_idx = 365 * 10  # After 10 years
    >>> changes = analyze_temporal_changes(precip, et0, extremes, split_idx)
    >>> print(f"Water availability change: {changes['changes']['water_availability_ratio']:.2%}")
    """
    # Split data into two periods
    p1_precip = precipitation[:split_year_idx]
    p1_evap = evaporation[:split_year_idx]
    p1_extreme = extreme_mask[:split_year_idx]

    p2_precip = precipitation[split_year_idx:]
    p2_evap = evaporation[split_year_idx:]
    p2_extreme = extreme_mask[split_year_idx:]

    # Analyze each period
    period1 = decompose_water_cycle_by_extremes(p1_precip, p1_evap, p1_extreme)
    period2 = decompose_water_cycle_by_extremes(p2_precip, p2_evap, p2_extreme)

    # Calculate changes
    changes = {}
    for condition in ['all_days', 'extreme_days', 'normal_days']:
        if period1[condition] is not None and period2[condition] is not None:
            changes[condition] = {
                'water_availability_ratio': (
                    period2[condition]['water_availability_mean'] /
                    period1[condition]['water_availability_mean']
                    if period1[condition]['water_availability_mean'] != 0 else np.nan
                ),
                'water_cycle_intensity_ratio': (
                    period2[condition]['water_cycle_intensity_mean'] /
                    period1[condition]['water_cycle_intensity_mean']
                ),
                'precipitation_ratio': (
                    period2[condition]['p_mean'] / period1[condition]['p_mean']
                ),
                'evaporation_ratio': (
                    period2[condition]['e_mean'] / period1[condition]['e_mean']
                ),
                'extreme_days_ratio': (
                    period2['extreme_days']['n_days'] / period1['extreme_days']['n_days']
                    if condition == 'extreme_days' else np.nan
                )
            }
        else:
            changes[condition] = None

    return {
        'period1': period1,
        'period2': period2,
        'changes': changes
    }


def classify_water_cycle_regime(
    water_availability_change: float,
    water_intensity_change: float
) -> str:
    """
    Classify water cycle regime based on changes in availability and intensity.

    Four regimes based on Markonis (2025):
    - Wetter-Accelerated: Increasing P-E and increasing (P+E)/2
    - Wetter-Decelerated: Increasing P-E but decreasing (P+E)/2
    - Drier-Accelerated: Decreasing P-E but increasing (P+E)/2
    - Drier-Decelerated: Decreasing P-E and decreasing (P+E)/2

    Parameters
    ----------
    water_availability_change : float
        Change in P-E (positive = wetter, negative = drier)
    water_intensity_change : float
        Change in (P+E)/2 (positive = accelerated, negative = decelerated)

    Returns
    -------
    str
        Regime classification

    Examples
    --------
    >>> regime = classify_water_cycle_regime(0.5, 0.3)
    >>> print(regime)  # "Wetter-Accelerated"
    """
    if water_availability_change > 0 and water_intensity_change > 0:
        return "Wetter-Accelerated"
    elif water_availability_change > 0 and water_intensity_change < 0:
        return "Wetter-Decelerated"
    elif water_availability_change < 0 and water_intensity_change > 0:
        return "Drier-Accelerated"
    else:
        return "Drier-Decelerated"


def analyze_seasonal_water_cycle(
    dates: pd.DatetimeIndex,
    precipitation: np.ndarray,
    evaporation: np.ndarray,
    extreme_mask: np.ndarray
) -> pd.DataFrame:
    """
    Analyze water cycle metrics by season.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index for the data
    precipitation : np.ndarray
        Daily precipitation (mm/day)
    evaporation : np.ndarray
        Daily evaporation or ET0 (mm/day)
    extreme_mask : np.ndarray
        Boolean mask indicating extreme ET days

    Returns
    -------
    pd.DataFrame
        Seasonal statistics with columns:
        - season
        - water_availability_mean
        - water_cycle_intensity_mean
        - extreme_frequency
        - p_mean, e_mean

    Examples
    --------
    >>> dates = pd.date_range('2000-01-01', periods=len(precip), freq='D')
    >>> seasonal_stats = analyze_seasonal_water_cycle(dates, precip, et0, extremes)
    >>> print(seasonal_stats)
    """
    df = pd.DataFrame({
        'date': dates,
        'precipitation': precipitation,
        'evaporation': evaporation,
        'extreme': extreme_mask
    })

    df['water_availability'] = calculate_water_availability(
        df['precipitation'].values, df['evaporation'].values
    )
    df['water_intensity'] = calculate_water_cycle_intensity(
        df['precipitation'].values, df['evaporation'].values
    )
    df['season'] = df['date'].dt.month % 12 // 3 + 1  # 1=DJF, 2=MAM, 3=JJA, 4=SON

    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}

    seasonal_stats = df.groupby('season').agg({
        'water_availability': ['mean', 'std'],
        'water_intensity': ['mean', 'std'],
        'precipitation': 'mean',
        'evaporation': 'mean',
        'extreme': ['sum', 'mean']
    }).reset_index()

    seasonal_stats.columns = [
        'season', 'water_availability_mean', 'water_availability_std',
        'water_intensity_mean', 'water_intensity_std',
        'p_mean', 'e_mean', 'n_extreme_days', 'extreme_frequency'
    ]
    seasonal_stats['season_name'] = seasonal_stats['season'].map(season_names)

    return seasonal_stats
