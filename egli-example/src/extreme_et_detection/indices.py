"""Core ET indices:
- ETx7d: annual max N-day cumulative ET
- Seasonal mean ET

基础蒸散发极端与季节性指标。
"""

import numpy as np
import xarray as xr


def compute_etx7d_annual_max(et_mm_day: xr.DataArray, window: int = 7) -> xr.DataArray:
    """Compute ETxNd annual maxima (ETx7d by default).

    Parameters
    ----------
    et_mm_day : xr.DataArray
        Daily ET [mm/day], dims: (time, lat, lon)
    window : int
        Rolling window in days (default 7).

    Returns
    -------
    xr.DataArray
        Annual max rolling-sum ET, dims: (year, lat, lon)
    """
    et_roll = et_mm_day.rolling(time=window, center=True).sum()
    year = et_mm_day["time"].dt.year

    def _annual_max(block: xr.DataArray) -> xr.DataArray:
        return block.max("time")

    etx = et_roll.groupby(year).map(_annual_max)
    etx = etx.rename({"group": "year"})
    etx["year"] = sorted(set(year.values))
    return etx


def compute_seasonal_mean(et_mm_day: xr.DataArray, season: str = "JJA") -> xr.DataArray:
    """Compute annual seasonal mean ET.

    season:
    - "JJA": 北半球夏季
    - "DJF": 跨年夏季（归属下一年）

    Returns
    -------
    xr.DataArray(year, lat, lon)
    """
    season = season.upper()
    if season == "JJA":
        sel = et_mm_day.where(et_mm_day["time"].dt.month.isin([6, 7, 8]))
        da = sel.groupby(sel["time"].dt.year).mean("time")
        return da.rename({"group": "year"})
    elif season == "DJF":
        et = et_mm_day
        month = et["time"].dt.month
        year = et["time"].dt.year
        year_for_group = xr.where(month == 12, year + 1, year)
        sel = et.where(month.isin([12, 1, 2]))
        da = sel.groupby(year_for_group).mean("time")
        return da.rename({"group": "year"})
    else:
        raise ValueError("Unsupported season. Use 'JJA' or 'DJF'.")
