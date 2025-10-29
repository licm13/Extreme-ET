"""
Event Physical Evolution Analysis
Based on Markonis (2025) - Analysis of onset and termination conditions

This module analyzes the physical evolution of extreme ET events,
including energy balance components and moisture dynamics before,
during, and after events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def identify_event_periods(extreme_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify continuous periods of extreme events.

    Parameters
    ----------
    extreme_mask : np.ndarray
        Boolean mask indicating extreme days

    Returns
    -------
    list of tuples
        Each tuple is (start_index, end_index) for an event period

    Examples
    --------
    >>> mask = np.array([0,0,1,1,1,0,0,1,1,0], dtype=bool)
    >>> events = identify_event_periods(mask)
    >>> print(events)  # [(2, 5), (7, 9)]
    """
    events = []
    in_event = False
    start_idx = 0

    for i, is_extreme in enumerate(extreme_mask):
        if is_extreme and not in_event:
            start_idx = i
            in_event = True
        elif not is_extreme and in_event:
            events.append((start_idx, i))
            in_event = False

    if in_event:  # Close last event
        events.append((start_idx, len(extreme_mask)))

    return events


def analyze_onset_termination_conditions(
    et0: np.ndarray,
    meteorological_data: Dict[str, np.ndarray],
    extreme_mask: np.ndarray,
    window_days: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Analyze meteorological conditions during onset and termination of extreme events.

    Following Markonis (2025) methodology to understand physical triggers
    and termination mechanisms of ExEvE.

    Parameters
    ----------
    et0 : np.ndarray
        Daily ET0 (mm/day)
    meteorological_data : dict
        Dictionary with keys: 'temperature', 'radiation', 'wind_speed',
        'vapor_pressure' (or 'precipitation')
    extreme_mask : np.ndarray
        Boolean mask for extreme ET days
    window_days : int
        Days to analyze before/after onset/termination (default: 5)

    Returns
    -------
    dict
        Dictionary with keys 'onset', 'termination', 'during_event', 'normal'
        Each contains mean values of meteorological variables

    Examples
    --------
    >>> et0 = np.random.gamma(3, 1.5, 1000)
    >>> met_data = {
    ...     'temperature': np.random.gamma(20, 2, 1000),
    ...     'radiation': np.random.gamma(15, 3, 1000),
    ...     'wind_speed': np.random.gamma(2, 1, 1000),
    ...     'precipitation': np.random.gamma(2, 1.5, 1000)
    ... }
    >>> extremes = et0 > np.percentile(et0, 95)
    >>> conditions = analyze_onset_termination_conditions(et0, met_data, extremes)
    >>> print(f"Temperature at onset: {conditions['onset']['temperature_mean']:.2f} °C")
    """
    events = identify_event_periods(extreme_mask)

    if len(events) == 0:
        return {'error': 'No events detected'}

    # Initialize storage
    onset_indices = []
    termination_indices = []
    during_indices = []

    for start, end in events:
        # Onset period: window before and including start
        onset_start = max(0, start - window_days)
        onset_end = min(len(et0), start + 1)
        onset_indices.extend(range(onset_start, onset_end))

        # Termination period: including end and window after
        term_start = max(0, end)
        term_end = min(len(et0), end + window_days)
        termination_indices.extend(range(term_start, term_end))

        # During event
        during_indices.extend(range(start, end))

    # Normal conditions (non-extreme)
    normal_indices = np.where(~extreme_mask)[0]

    # Calculate statistics
    def calculate_stats(indices, data_dict):
        if len(indices) == 0:
            return {}

        stats = {}
        stats['et0_mean'] = np.mean(et0[indices])
        stats['et0_std'] = np.std(et0[indices])

        for var_name, var_data in data_dict.items():
            stats[f'{var_name}_mean'] = np.mean(var_data[indices])
            stats[f'{var_name}_std'] = np.std(var_data[indices])
            stats[f'{var_name}_min'] = np.min(var_data[indices])
            stats[f'{var_name}_max'] = np.max(var_data[indices])

        stats['n_samples'] = len(indices)
        return stats

    results = {
        'onset': calculate_stats(onset_indices, meteorological_data),
        'termination': calculate_stats(termination_indices, meteorological_data),
        'during_event': calculate_stats(during_indices, meteorological_data),
        'normal': calculate_stats(normal_indices, meteorological_data)
    }

    return results


def calculate_energy_balance_components(
    temperature: np.ndarray,
    radiation: np.ndarray,
    et0: np.ndarray,
    latent_heat_vaporization: float = 2.45  # MJ/kg
) -> Dict[str, np.ndarray]:
    """
    Calculate energy balance components following Markonis (2025).

    Components:
    - Latent heat flux: energy used for evaporation
    - Sensible heat: can be estimated from residual
    - Net radiation: available energy

    Parameters
    ----------
    temperature : np.ndarray
        Daily mean temperature (°C)
    radiation : np.ndarray
        Net radiation (MJ m⁻² day⁻¹)
    et0 : np.ndarray
        Daily ET0 (mm/day)
    latent_heat_vaporization : float
        Latent heat of vaporization (MJ/kg, default: 2.45)

    Returns
    -------
    dict
        Dictionary with energy components

    Examples
    --------
    >>> temp = np.random.gamma(20, 2, 1000)
    >>> rad = np.random.gamma(15, 3, 1000)
    >>> et0 = np.random.gamma(3, 1.5, 1000)
    >>> energy = calculate_energy_balance_components(temp, rad, et0)
    >>> print(f"Mean latent heat flux: {np.mean(energy['latent_heat_flux']):.2f} MJ/m²/day")
    """
    # Convert ET0 (mm/day) to kg/m²/day (assuming density of water = 1 kg/L)
    et0_kg = et0

    # Latent heat flux (MJ/m²/day)
    latent_heat_flux = et0_kg * latent_heat_vaporization

    # Sensible heat (residual from net radiation, simplified)
    # H = Rn - LE - G (where G is soil heat flux, assumed small for daily)
    sensible_heat = radiation - latent_heat_flux

    # Bowen ratio: H / LE
    bowen_ratio = np.where(latent_heat_flux > 0,
                           sensible_heat / latent_heat_flux,
                           np.nan)

    return {
        'latent_heat_flux': latent_heat_flux,
        'sensible_heat': sensible_heat,
        'net_radiation': radiation,
        'bowen_ratio': bowen_ratio
    }


def analyze_energy_partitioning(
    temperature: np.ndarray,
    radiation: np.ndarray,
    et0: np.ndarray,
    extreme_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Analyze energy partitioning during extreme vs normal conditions.

    Parameters
    ----------
    temperature : np.ndarray
        Daily mean temperature (°C)
    radiation : np.ndarray
        Net radiation (MJ m⁻² day⁻¹)
    et0 : np.ndarray
        Daily ET0 (mm/day)
    extreme_mask : np.ndarray
        Boolean mask for extreme events

    Returns
    -------
    dict
        Energy partitioning for 'extreme' and 'normal' conditions

    Examples
    --------
    >>> temp = np.random.gamma(20, 2, 1000)
    >>> rad = np.random.gamma(15, 3, 1000)
    >>> et0 = np.random.gamma(3, 1.5, 1000)
    >>> extremes = et0 > np.percentile(et0, 95)
    >>> partitioning = analyze_energy_partitioning(temp, rad, et0, extremes)
    >>> print(f"Bowen ratio during extremes: {partitioning['extreme']['bowen_ratio_mean']:.3f}")
    """
    energy_components = calculate_energy_balance_components(temperature, radiation, et0)

    results = {}

    for condition, mask in [('extreme', extreme_mask), ('normal', ~extreme_mask)]:
        if np.any(mask):
            valid_bowen = energy_components['bowen_ratio'][mask]
            valid_bowen = valid_bowen[~np.isnan(valid_bowen)]

            results[condition] = {
                'latent_heat_flux_mean': np.mean(energy_components['latent_heat_flux'][mask]),
                'sensible_heat_mean': np.mean(energy_components['sensible_heat'][mask]),
                'net_radiation_mean': np.mean(energy_components['net_radiation'][mask]),
                'bowen_ratio_mean': np.mean(valid_bowen) if len(valid_bowen) > 0 else np.nan,
                'latent_heat_fraction': (
                    np.mean(energy_components['latent_heat_flux'][mask]) /
                    np.mean(energy_components['net_radiation'][mask])
                    if np.mean(energy_components['net_radiation'][mask]) > 0 else np.nan
                )
            }

    return results


def identify_event_triggers(
    meteorological_data: Dict[str, np.ndarray],
    extreme_mask: np.ndarray,
    lookback_days: int = 7
) -> Dict[str, any]:
    """
    Identify which meteorological conditions act as triggers for extreme events.

    Analyzes anomalies in the days leading up to event onset.

    Parameters
    ----------
    meteorological_data : dict
        Dictionary with meteorological variables
    extreme_mask : np.ndarray
        Boolean mask for extreme events
    lookback_days : int
        Days to analyze before event onset (default: 7)

    Returns
    -------
    dict
        Trigger analysis results

    Examples
    --------
    >>> met_data = {
    ...     'temperature': np.random.gamma(20, 2, 1000),
    ...     'radiation': np.random.gamma(15, 3, 1000)
    ... }
    >>> extremes = np.random.binomial(1, 0.05, 1000).astype(bool)
    >>> triggers = identify_event_triggers(met_data, extremes, lookback_days=7)
    """
    events = identify_event_periods(extreme_mask)

    if len(events) == 0:
        return {'error': 'No events detected'}

    # Calculate anomalies
    anomalies = {}
    for var_name, var_data in meteorological_data.items():
        var_mean = np.mean(var_data)
        var_std = np.std(var_data)
        anomalies[var_name] = (var_data - var_mean) / var_std

    # Analyze pre-event conditions
    pre_event_anomalies = {var: [] for var in meteorological_data.keys()}

    for start, end in events:
        lookback_start = max(0, start - lookback_days)
        for var_name in anomalies.keys():
            pre_event_anomaly = np.mean(anomalies[var_name][lookback_start:start])
            pre_event_anomalies[var_name].append(pre_event_anomaly)

    # Summarize triggers
    triggers = {}
    for var_name, anomaly_list in pre_event_anomalies.items():
        if len(anomaly_list) > 0:
            mean_anomaly = np.mean(anomaly_list)
            triggers[var_name] = {
                'mean_pre_event_anomaly': mean_anomaly,
                'is_positive_trigger': mean_anomaly > 0.5,  # Threshold for "trigger"
                'consistency': np.sum(np.array(anomaly_list) > 0) / len(anomaly_list)
            }

    return triggers


def analyze_event_intensity_evolution(
    et0: np.ndarray,
    extreme_mask: np.ndarray
) -> List[Dict[str, any]]:
    """
    Analyze how event intensity evolves from onset to termination.

    Parameters
    ----------
    et0 : np.ndarray
        Daily ET0 (mm/day)
    extreme_mask : np.ndarray
        Boolean mask for extreme events

    Returns
    -------
    list of dict
        Each dict contains evolution metrics for one event

    Examples
    --------
    >>> et0 = np.random.gamma(3, 1.5, 1000)
    >>> extremes = et0 > np.percentile(et0, 95)
    >>> evolution = analyze_event_intensity_evolution(et0, extremes)
    >>> for i, event in enumerate(evolution):
    ...     print(f"Event {i}: peak intensity = {event['peak_intensity']:.2f} mm/day")
    """
    events = identify_event_periods(extreme_mask)

    event_evolutions = []

    for start, end in events:
        event_et0 = et0[start:end]

        if len(event_et0) == 0:
            continue

        evolution = {
            'start_index': start,
            'end_index': end,
            'duration': end - start,
            'mean_intensity': np.mean(event_et0),
            'peak_intensity': np.max(event_et0),
            'peak_day_relative': np.argmax(event_et0),
            'initial_intensity': event_et0[0],
            'final_intensity': event_et0[-1],
            'intensity_trend': np.polyfit(range(len(event_et0)), event_et0, 1)[0] if len(event_et0) > 1 else 0,
            'cumulative_et': np.sum(event_et0)
        }

        event_evolutions.append(evolution)

    return event_evolutions


def compare_seasonal_event_characteristics(
    dates: pd.DatetimeIndex,
    et0: np.ndarray,
    meteorological_data: Dict[str, np.ndarray],
    extreme_mask: np.ndarray
) -> pd.DataFrame:
    """
    Compare event characteristics across seasons.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index
    et0 : np.ndarray
        Daily ET0
    meteorological_data : dict
        Meteorological variables
    extreme_mask : np.ndarray
        Boolean mask for extremes

    Returns
    -------
    pd.DataFrame
        Seasonal comparison of event characteristics

    Examples
    --------
    >>> dates = pd.date_range('2000-01-01', periods=1000, freq='D')
    >>> et0 = np.random.gamma(3, 1.5, 1000)
    >>> met = {'temperature': np.random.gamma(20, 2, 1000)}
    >>> extremes = et0 > np.percentile(et0, 95)
    >>> seasonal_comp = compare_seasonal_event_characteristics(dates, et0, met, extremes)
    """
    df = pd.DataFrame({
        'date': dates,
        'et0': et0,
        'extreme': extreme_mask
    })

    for var_name, var_data in meteorological_data.items():
        df[var_name] = var_data

    df['season'] = df['date'].dt.month % 12 // 3 + 1  # 1=DJF, 2=MAM, 3=JJA, 4=SON

    # Analyze only extreme days
    extreme_df = df[df['extreme']]

    if len(extreme_df) == 0:
        return pd.DataFrame()

    season_stats = extreme_df.groupby('season').agg({
        'et0': ['mean', 'std', 'max', 'count'],
        **{var: ['mean', 'std'] for var in meteorological_data.keys()}
    })

    return season_stats
