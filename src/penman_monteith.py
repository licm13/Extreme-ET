"""
Penman-Monteith equation for reference evapotranspiration calculation
"""

import numpy as np


def calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z=50.0, 
                  latitude=40.0, doy=None):
    """
    Calculate reference evapotranspiration using ASCE-PM equation.
    
    Following Zhao et al. (2025), implements the ASCE Standardized 
    Penman-Monteith equation for grass reference surface.
    
    Parameters
    ----------
    T_mean : float or array-like
        Mean daily air temperature (°C)
    T_max : float or array-like
        Maximum daily air temperature (°C)
    T_min : float or array-like
        Minimum daily air temperature (°C)
    Rs : float or array-like
        Incoming solar radiation (MJ m-2 day-1)
    u2 : float or array-like
        Wind speed at 2m height (m s-1)
    ea : float or array-like
        Actual vapor pressure (kPa)
    z : float, default=50.0
        Elevation above sea level (m)
    latitude : float, default=40.0
        Latitude (degrees)
    doy : int or array-like, optional
        Day of year (1-365) for net radiation calculation
    
    Returns
    -------
    ET0 : float or np.ndarray
        Reference evapotranspiration (mm day-1)
    
    Examples
    --------
    >>> T_mean, T_max, T_min = 20, 25, 15
    >>> Rs, u2, ea = 20, 2.0, 1.5
    >>> ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea)
    >>> print(f"ET0 = {ET0:.2f} mm/day")
    
    References
    ----------
    Allen et al. (1998). Crop evapotranspiration - Guidelines for computing 
    crop water requirements. FAO Irrigation and Drainage Paper 56.
    """
    # Convert inputs to arrays
    T_mean = np.asarray(T_mean)
    T_max = np.asarray(T_max)
    T_min = np.asarray(T_min)
    Rs = np.asarray(Rs)
    u2 = np.asarray(u2)
    ea = np.asarray(ea)
    
    # Constants for grass reference (from ASCE-PM)
    Cn = 900  # Numerator constant (K mm s3 Mg-1 day-1)
    Cd = 0.34  # Denominator constant (s m-1)
    
    # Psychrometric constant (kPa °C-1)
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26  # Atmospheric pressure
    gamma = 0.000665 * P
    
    # Slope of saturation vapor pressure curve (kPa °C-1)
    es_max = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3))
    es_min = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
    es = (es_max + es_min) / 2  # Mean saturation vapor pressure
    
    Delta = (4098 * es) / ((T_mean + 237.3) ** 2)
    
    # Net radiation (simplified - assumes G = 0 for daily step)
    if doy is not None:
        # More accurate with extraterrestrial radiation
        Rn = calculate_net_radiation(Rs, T_max, T_min, ea, latitude, doy)
    else:
        # Simplified: assume Rn ≈ 0.77 * Rs
        Rn = 0.77 * Rs
    
    G = 0  # Soil heat flux (assumed zero for daily calculation)
    
    # ASCE-PM equation (Equation 3 from Zhao et al. 2025)
    numerator = 0.408 * Delta * (Rn - G) + gamma * (Cn / (T_mean + 273)) * u2 * (es - ea)
    denominator = Delta + gamma * (1 + Cd * u2)
    
    ET0 = numerator / denominator
    
    # Ensure non-negative values
    ET0 = np.maximum(ET0, 0)
    
    return ET0


def calculate_net_radiation(Rs, T_max, T_min, ea, latitude, doy):
    """
    Calculate net radiation for ET0 calculation.
    
    Parameters
    ----------
    Rs : float or array-like
        Solar radiation (MJ m-2 day-1)
    T_max, T_min : float or array-like
        Max and min air temperature (°C)
    ea : float or array-like
        Actual vapor pressure (kPa)
    latitude : float
        Latitude (degrees)
    doy : int or array-like
        Day of year (1-365)
    
    Returns
    -------
    Rn : float or np.ndarray
        Net radiation (MJ m-2 day-1)
    """
    # Convert to arrays
    Rs = np.asarray(Rs)
    T_max = np.asarray(T_max)
    T_min = np.asarray(T_min)
    ea = np.asarray(ea)
    doy = np.asarray(doy)
    
    # Albedo for grass reference crop
    alpha = 0.23
    
    # Net shortwave radiation
    Rns = (1 - alpha) * Rs
    
    # Net longwave radiation
    sigma = 4.903e-9  # Stefan-Boltzmann constant (MJ K-4 m-2 day-1)
    
    # Convert temperatures to Kelvin
    T_max_K = T_max + 273.16
    T_min_K = T_min + 273.16
    
    # Calculate extraterrestrial radiation for cloudiness
    Ra = calculate_extraterrestrial_radiation(latitude, doy)
    
    # Cloudiness function
    Rs_so = (0.75 + 2e-5 * 0) * Ra  # Clear sky radiation (simplified)
    cloudiness = 1.35 * (Rs / Rs_so) - 0.35
    cloudiness = np.clip(cloudiness, 0.05, 1.0)
    
    # Longwave radiation
    Rnl = sigma * ((T_max_K**4 + T_min_K**4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * cloudiness
    
    # Net radiation
    Rn = Rns - Rnl
    
    return Rn


def calculate_extraterrestrial_radiation(latitude, doy):
    """
    Calculate extraterrestrial radiation.
    
    Parameters
    ----------
    latitude : float
        Latitude (degrees)
    doy : int or array-like
        Day of year
    
    Returns
    -------
    Ra : float or np.ndarray
        Extraterrestrial radiation (MJ m-2 day-1)
    """
    doy = np.asarray(doy)
    
    # Solar constant
    Gsc = 0.0820  # MJ m-2 min-1
    
    # Convert latitude to radians
    lat_rad = np.deg2rad(latitude)
    
    # Solar declination
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    
    # Sunset hour angle
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    
    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    
    # Extraterrestrial radiation
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta) +
        np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )
    
    return Ra


def calculate_vapor_pressure_from_vpd(vpd, T_mean):
    """
    Calculate actual vapor pressure from VPD and temperature.
    
    Following Zhao et al. (2025) Equation 1.
    
    Parameters
    ----------
    vpd : float or array-like
        Vapor pressure deficit (kPa)
    T_mean : float or array-like
        Mean air temperature (°C)
    
    Returns
    -------
    ea : float or np.ndarray
        Actual vapor pressure (kPa)
    """
    # Saturated vapor pressure
    es = 0.6108 * np.exp(17.27 * T_mean / (T_mean + 237.3))
    
    # Actual vapor pressure
    ea = es - vpd
    
    return ea


def adjust_wind_speed(u_z, z=10.0):
    """
    Adjust wind speed from height z to 2m.
    
    Following Zhao et al. (2025) Equation 2.
    
    Parameters
    ----------
    u_z : float or array-like
        Wind speed at height z (m s-1)
    z : float, default=10.0
        Measurement height (m)
    
    Returns
    -------
    u2 : float or np.ndarray
        Wind speed at 2m height (m s-1)
    """
    u2 = u_z * 4.87 / np.log(67.8 * z - 5.42)
    
    return u2