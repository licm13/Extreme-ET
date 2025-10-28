"""
Optional IO utilities for reading gridded ET products (NetCDF/xarray) and
sampling at station locations using nearest-neighbor or bilinear interpolation.

These functions are lightweight wrappers: if xarray is not installed, they
raise a clear ImportError. For bilinear interpolation on regular lon/lat grids
we also provide a NumPy-only implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _require_xarray():
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "xarray is required for NetCDF IO. Install xarray and netCDF4."
        ) from exc
    return xr


def read_netcdf_variable(
    path: str,
    var_name: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
    chunks: Optional[dict] = None,
):
    """Read a variable and its coordinates from a NetCDF file via xarray.

    Returns a tuple (data, lats, lons, times) where data is a DataArray.
    """
    xr = _require_xarray()
    ds = xr.open_dataset(path, chunks=chunks)
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in {path}")
    da = ds[var_name]
    lats = ds[lat_name]
    lons = ds[lon_name]
    times = ds[time_name] if time_name in ds.coords else None
    return da, lats, lons, times


def sample_series_at_point(
    da,
    lat: float,
    lon: float,
    method: str = "nearest",
    lat_name: str = "lat",
    lon_name: str = "lon",
):
    """Extract a time series at a lat/lon location from an xarray DataArray.

    method: 'nearest' (reindex) or 'bilinear' (xarray.interp).
    """
    xr = _require_xarray()
    if method == "nearest":
        series = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
    elif method == "bilinear":
        series = da.interp({lat_name: lat, lon_name: lon})
    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")
    # Drop spatial dims and return as 1D array aligned with time
    for dim in [lat_name, lon_name]:
        if dim in series.dims:
            series = series.isel({dim: 0})
    return series


def bilinear_on_regular_grid(
    field2d: np.ndarray,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    """Bilinear interpolate a 2D field on a regular 1D lon/lat grid at (lat, lon).

    field2d shape must be (nlat, nlon). lats_1d and lons_1d must be monotonous.
    """
    lat_idx = np.searchsorted(lats_1d, lat) - 1
    lon_idx = np.searchsorted(lons_1d, lon) - 1
    lat_idx = np.clip(lat_idx, 0, len(lats_1d) - 2)
    lon_idx = np.clip(lon_idx, 0, len(lons_1d) - 2)

    lat0, lat1 = lats_1d[lat_idx], lats_1d[lat_idx + 1]
    lon0, lon1 = lons_1d[lon_idx], lons_1d[lon_idx + 1]
    f00 = field2d[lat_idx, lon_idx]
    f01 = field2d[lat_idx, lon_idx + 1]
    f10 = field2d[lat_idx + 1, lon_idx]
    f11 = field2d[lat_idx + 1, lon_idx + 1]

    # Weights
    t = 0.0 if lat1 == lat0 else (lat - lat0) / (lat1 - lat0)
    u = 0.0 if lon1 == lon0 else (lon - lon0) / (lon1 - lon0)

    return (1 - t) * (1 - u) * f00 + (1 - t) * u * f01 + t * (1 - u) * f10 + t * u * f11

