"""
Contribution analysis for meteorological drivers of extreme evaporation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .penman_monteith import calculate_et0
from .data_processing import calculate_climatological_means


def calculate_contributions(T_mean, T_max, T_min, Rs, u2, ea, 
                           extreme_mask, z=50.0, latitude=40.0, doy=None):
    """
    Calculate relative contribution of each meteorological forcing to extreme ET0.
    
    Following Zhao et al. (2025), uses sensitivity analysis by replacing each
    forcing with its climatological value while keeping others unchanged.
    
    Parameters
    ----------
    T_mean, T_max, T_min : array-like
        Temperature data (°C)
    Rs : array-like
        Solar radiation (MJ m-2 day-1)
    u2 : array-like
        Wind speed at 2m (m s-1)
    ea : array-like
        Actual vapor pressure (kPa)
    extreme_mask : array-like (bool)
        Boolean mask indicating extreme days
    z : float
        Elevation (m)
    latitude : float
        Latitude (degrees)
    doy : array-like, optional
        Day of year
    
    Returns
    -------
    contributions : dict
        Relative contribution (%) of each forcing:
        {'temperature': float, 'radiation': float, 'wind': float, 'humidity': float}
    
    Examples
    --------
    >>> # Generate sample data
    >>> n = 365 * 10
    >>> T_mean = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365)
    >>> T_max = T_mean + 5
    >>> T_min = T_mean - 5
    >>> Rs = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365)
    >>> u2 = 2 + np.random.randn(n) * 0.5
    >>> ea = 1.5 + np.random.randn(n) * 0.2
    >>> extreme_mask = np.random.rand(n) < 0.01  # 1% extremes
    >>> 
    >>> contributions = calculate_contributions(
    ...     T_mean, T_max, T_min, Rs, u2, ea, extreme_mask
    ... )
    >>> print(f"Temperature contribution: {contributions['temperature']:.1f}%")
    """
    # Convert to arrays
    T_mean = np.asarray(T_mean)
    T_max = np.asarray(T_max)
    T_min = np.asarray(T_min)
    Rs = np.asarray(Rs)
    u2 = np.asarray(u2)
    ea = np.asarray(ea)
    extreme_mask = np.asarray(extreme_mask, dtype=bool)
    
    # Calculate original ET0
    ET0_orig = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z, latitude, doy)
    
    # Calculate climatological values efficiently using optimized helper
    T_mean_clim, T_max_clim, T_min_clim, Rs_clim, u2_clim, ea_clim = \
        calculate_climatological_means(T_mean, T_max, T_min, Rs, u2, ea)
    
    # Calculate ET0 with each forcing replaced by climatology
    ET0_temp_clim = calculate_et0(T_mean_clim, T_max_clim, T_min_clim, 
                                   Rs, u2, ea, z, latitude, doy)
    ET0_rad_clim = calculate_et0(T_mean, T_max, T_min, 
                                  Rs_clim, u2, ea, z, latitude, doy)
    ET0_wind_clim = calculate_et0(T_mean, T_max, T_min, 
                                   Rs, u2_clim, ea, z, latitude, doy)
    ET0_hum_clim = calculate_et0(T_mean, T_max, T_min, 
                                  Rs, u2, ea_clim, z, latitude, doy)
    
    # Calculate contributions for extreme days only
    # Following Equation 4-5 from Zhao et al. (2025)
    diff_temp = np.sum((ET0_orig - ET0_temp_clim)[extreme_mask])
    diff_rad = np.sum((ET0_orig - ET0_rad_clim)[extreme_mask])
    diff_wind = np.sum((ET0_orig - ET0_wind_clim)[extreme_mask])
    diff_hum = np.sum((ET0_orig - ET0_hum_clim)[extreme_mask])
    
    total_diff = diff_temp + diff_rad + diff_wind + diff_hum
    
    if total_diff > 0:
        contributions = {
            'temperature': (diff_temp / total_diff) * 100,
            'radiation': (diff_rad / total_diff) * 100,
            'wind': (diff_wind / total_diff) * 100,
            'humidity': (diff_hum / total_diff) * 100,
        }
    else:
        # No contribution detected
        contributions = {
            'temperature': 25.0,
            'radiation': 25.0,
            'wind': 25.0,
            'humidity': 25.0,
        }
    
    return contributions


def sensitivity_analysis(T_mean, T_max, T_min, Rs, u2, ea, 
                        perturbation=0.1, z=50.0, latitude=40.0):
    """
    Perform sensitivity analysis by perturbing each forcing.
    
    Parameters
    ----------
    T_mean, T_max, T_min : float or array-like
        Temperature (°C)
    Rs : float or array-like
        Solar radiation (MJ m-2 day-1)
    u2 : float or array-like
        Wind speed (m s-1)
    ea : float or array-like
        Vapor pressure (kPa)
    perturbation : float, default=0.1
        Relative perturbation magnitude (10%)
    z, latitude : float
        Site parameters
    
    Returns
    -------
    sensitivity : dict
        Sensitivity coefficients for each forcing
    """
    # Baseline ET0
    ET0_base = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z, latitude)
    
    # Perturb each variable
    T_pert = perturbation * T_mean
    ET0_temp = calculate_et0(T_mean + T_pert, T_max + T_pert, T_min + T_pert, 
                             Rs, u2, ea, z, latitude)
    
    Rs_pert = perturbation * Rs
    ET0_rad = calculate_et0(T_mean, T_max, T_min, Rs + Rs_pert, u2, ea, z, latitude)
    
    u2_pert = perturbation * u2
    ET0_wind = calculate_et0(T_mean, T_max, T_min, Rs, u2 + u2_pert, ea, z, latitude)
    
    ea_pert = perturbation * ea
    ET0_hum = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea + ea_pert, z, latitude)
    
    # Calculate sensitivity (dET0 / dX)
    sensitivity = {
        'temperature': (ET0_temp - ET0_base) / T_pert,
        'radiation': (ET0_rad - ET0_base) / Rs_pert,
        'wind': (ET0_wind - ET0_base) / u2_pert,
        'humidity': (ET0_hum - ET0_base) / ea_pert,
    }
    
    return sensitivity


def analyze_seasonal_contributions(T_mean, T_max, T_min, Rs, u2, ea, 
                                  extreme_mask, seasons=None):
    """
    Calculate contributions for different seasons.
    
    Parameters
    ----------
    T_mean, T_max, T_min, Rs, u2, ea : array-like
        Meteorological forcings
    extreme_mask : array-like (bool)
        Extreme event mask
    seasons : dict, optional
        Dictionary mapping season names to month indices
        Default: {'winter': [12,1,2], 'spring': [3,4,5], ...}
    
    Returns
    -------
    seasonal_contributions : dict
        Contributions for each season
    """
    n = len(T_mean)
    
    if seasons is None:
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11],
        }
    
    # Determine month for each day (approximate)
    doy = np.arange(n) % 365
    month = (doy / 30.4).astype(int) + 1
    month = np.clip(month, 1, 12)
    
    seasonal_contributions = {}
    
    for season_name, months in seasons.items():
        # Create seasonal mask
        season_mask = np.isin(month, months)
        seasonal_extreme_mask = extreme_mask & season_mask
        
        if np.sum(seasonal_extreme_mask) > 0:
            contributions = calculate_contributions(
                T_mean, T_max, T_min, Rs, u2, ea, 
                seasonal_extreme_mask
            )
            seasonal_contributions[season_name] = contributions
    
    return seasonal_contributions


def identify_dominant_driver(contributions):
    """
    Identify the dominant meteorological driver.
    
    Parameters
    ----------
    contributions : dict
        Contribution percentages for each forcing
    
    Returns
    -------
    dominant : str
        Name of dominant forcing
    primary_contribution : float
        Contribution percentage
    """
    dominant = max(contributions, key=contributions.get)
    primary_contribution = contributions[dominant]

    return dominant, primary_contribution


def dynamic_perturbation_response(
    T_mean,
    T_max,
    T_min,
    Rs,
    u2,
    ea,
    extreme_mask,
    amplitude_scale: float = 0.2,
    period_days: int = 30,
    n_phase_samples: int = 8,
    z: float = 50.0,
    latitude: float = 40.0,
    doy=None,
):
    """Assess oscillatory perturbation responses for each meteorological driver."""

    T_mean = np.asarray(T_mean, dtype=float)
    T_max = np.asarray(T_max, dtype=float)
    T_min = np.asarray(T_min, dtype=float)
    Rs = np.asarray(Rs, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    ea = np.asarray(ea, dtype=float)
    extreme_mask = np.asarray(extreme_mask, dtype=bool)

    n = T_mean.size
    if n == 0:
        raise ValueError("输入序列为空 / Input arrays must be non-empty")

    if not extreme_mask.any():
        extreme_mask = np.ones(n, dtype=bool)

    days = np.arange(n)
    base_et0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z, latitude, doy)

    phases = np.linspace(0, 2 * np.pi, n_phase_samples, endpoint=False)
    responses: Dict[str, Dict[str, Union[float, List[float]]]] = {}

    def _aggregate_response(et0_series):
        delta = et0_series - base_et0
        return float(np.mean(delta[extreme_mask]))

    driver_configs = {
        'temperature': [T_mean, T_max, T_min],
        'radiation': [Rs],
        'wind': [u2],
        'humidity': [ea],
    }

    for driver, arrays in driver_configs.items():
        stack = np.vstack(arrays)
        amplitude = amplitude_scale * np.std(stack, axis=1).max()
        if amplitude == 0:
            responses[driver] = {
                'mean_response': 0.0,
                'phase_responses': [0.0] * len(phases),
                'amplitude': 0.0,
                'phases_radians': phases.tolist(),
            }
            continue

        phase_effects: List[float] = []
        for phase in phases:
            perturbation = amplitude * np.sin(2 * np.pi * days / period_days + phase)

            if driver == 'temperature':
                T_mean_mod = T_mean + perturbation
                T_max_mod = T_max + perturbation
                T_min_mod = T_min + perturbation
                et0 = calculate_et0(T_mean_mod, T_max_mod, T_min_mod, Rs, u2, ea, z, latitude, doy)
            elif driver == 'radiation':
                Rs_mod = np.clip(Rs + perturbation, a_min=0.0, a_max=None)
                et0 = calculate_et0(T_mean, T_max, T_min, Rs_mod, u2, ea, z, latitude, doy)
            elif driver == 'wind':
                reference = np.maximum(np.mean(u2), 1e-6)
                u2_mod = np.clip(u2 * (1 + perturbation / reference), 0.1, None)
                et0 = calculate_et0(T_mean, T_max, T_min, Rs, u2_mod, ea, z, latitude, doy)
            else:  # humidity
                ea_mod = np.clip(ea + perturbation, 1e-4, None)
                et0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea_mod, z, latitude, doy)

            phase_effects.append(_aggregate_response(et0))

        responses[driver] = {
            'mean_response': float(np.mean(phase_effects)),
            'phase_responses': phase_effects,
            'amplitude': float(amplitude),
            'phases_radians': phases.tolist(),
        }

    return responses


def compute_perturbation_pathway(
    T_mean,
    T_max,
    T_min,
    Rs,
    u2,
    ea,
    perturbations: Optional[Dict[str, float]] = None,
    extreme_mask=None,
    order: Optional[List[str]] = None,
    z: float = 50.0,
    latitude: float = 40.0,
    doy=None,
):
    """Propagate sequential perturbations and capture non-linear synergy."""

    T_mean = np.asarray(T_mean, dtype=float)
    T_max = np.asarray(T_max, dtype=float)
    T_min = np.asarray(T_min, dtype=float)
    Rs = np.asarray(Rs, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    ea = np.asarray(ea, dtype=float)

    n = T_mean.size
    if n == 0:
        raise ValueError("输入序列为空 / Input arrays must be non-empty")

    if extreme_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(extreme_mask, dtype=bool)
        if mask.size != n:
            raise ValueError("extreme_mask 长度必须与强迫时间序列一致")

    if perturbations is None:
        perturbations = {
            'temperature': 1.0,
            'radiation': 0.1,
            'wind': 0.15,
            'humidity': -0.1,
        }

    driver_order = order or ['temperature', 'radiation', 'wind', 'humidity']

    base_state = {
        'T_mean': T_mean.copy(),
        'T_max': T_max.copy(),
        'T_min': T_min.copy(),
        'Rs': Rs.copy(),
        'u2': u2.copy(),
        'ea': ea.copy(),
    }

    base_et0 = calculate_et0(**base_state, z=z, latitude=latitude, doy=doy)

    def _apply(state, driver, magnitude):
        updated = {key: val.copy() for key, val in state.items()}
        if driver == 'temperature':
            for key in ('T_mean', 'T_max', 'T_min'):
                updated[key] = updated[key] + magnitude
        elif driver == 'radiation':
            factor = 1 + magnitude
            updated['Rs'] = np.clip(updated['Rs'] * factor, 0.0, None)
        elif driver == 'wind':
            factor = 1 + magnitude
            updated['u2'] = np.clip(updated['u2'] * factor, 0.05, None)
        elif driver == 'humidity':
            updated['ea'] = np.clip(updated['ea'] + magnitude, 1e-4, None)
        else:
            raise ValueError(f"未知驱动因子: {driver}")
        return updated

    def _mean_change(et0_series, reference):
        delta = et0_series - reference
        return float(np.mean(delta[mask]))

    independent_effects: Dict[str, Dict[str, float]] = {}
    for driver, magnitude in perturbations.items():
        state = _apply(base_state, driver, magnitude)
        et0 = calculate_et0(**state, z=z, latitude=latitude, doy=doy)
        independent_effects[driver] = {'mean_change': _mean_change(et0, base_et0)}

    incremental_changes: List[float] = []
    cumulative_changes: List[float] = []
    current_state = base_state
    current_et0 = base_et0

    for driver in driver_order:
        if driver not in perturbations:
            continue
        current_state = _apply(current_state, driver, perturbations[driver])
        new_et0 = calculate_et0(**current_state, z=z, latitude=latitude, doy=doy)
        increment = _mean_change(new_et0, current_et0)
        incremental_changes.append(increment)
        current_et0 = new_et0
        cumulative_changes.append(_mean_change(current_et0, base_et0))

    total_change = cumulative_changes[-1] if cumulative_changes else 0.0
    linear_sum = sum(effect['mean_change'] for effect in independent_effects.values())
    residual_synergy = total_change - linear_sum

    return {
        'order': driver_order,
        'incremental_changes': incremental_changes,
        'cumulative_changes': cumulative_changes,
        'independent_effects': independent_effects,
        'total_change': total_change,
        'residual_synergy': residual_synergy,
    }