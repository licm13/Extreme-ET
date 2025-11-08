"""SREX / AR6 region helpers.

用于基于区域的趋势和检测分析。
"""

import numpy as np
import xarray as xr

from .config import SREX_MASK_PATH


def load_srex_mask() -> xr.DataArray:
    if not SREX_MASK_PATH.exists():
        raise FileNotFoundError(
            f"SREX mask not found at {SREX_MASK_PATH}. Please provide it."
        )
    ds = xr.open_dataset(SREX_MASK_PATH)
    # pick first integer-like variable
    for v in ds.data_vars:
        if ds[v].dtype.kind in ("i", "u"):
            return ds[v]
    raise ValueError("No integer region variable found in SREX mask file.")


def regional_mean(da: xr.DataArray, region_mask: xr.DataArray, region_id: int) -> xr.DataArray:
    """Area-weighted mean over specified region_id."""
    mask = xr.where(region_mask == region_id, 1.0, np.nan)
    lat_weights = np.cos(np.deg2rad(da["lat"]))
    w = mask * lat_weights
    num = (da * w).sum(("lat", "lon"))
    den = w.sum(("lat", "lon"))
    return num / den
