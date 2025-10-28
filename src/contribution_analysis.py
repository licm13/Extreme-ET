"""
Contribution analysis for meteorological drivers of extreme evaporation
"""

import numpy as np
from typing import Dict, List, Tuple
from .penman_monteith import calculate_et0


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
    
    # Calculate climatological values (mean for each calendar day)
    n = len(T_mean)
    T_mean_clim = np.zeros(n)
    T_max_clim = np.zeros(n)
    T_min_clim = np.zeros(n)
    Rs_clim = np.zeros(n)
    u2_clim = np.zeros(n)
    ea_clim = np.zeros(n)
    
    for day in range(365):
        day_mask = (np.arange(n) % 365) == day
        T_mean_clim[day_mask] = np.mean(T_mean[day_mask])
        T_max_clim[day_mask] = np.mean(T_max[day_mask])
        T_min_clim[day_mask] = np.mean(T_min[day_mask])
        Rs_clim[day_mask] = np.mean(Rs[day_mask])
        u2_clim[day_mask] = np.mean(u2[day_mask])
        ea_clim[day_mask] = np.mean(ea[day_mask])
    
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