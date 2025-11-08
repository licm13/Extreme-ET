"""Generic statistical utilities.

通用统计函数：距平、去趋势等。
"""

from typing import Tuple
import numpy as np
import xarray as xr
from scipy import stats


def detrend_linear_1d(y):
    x = np.arange(len(y))
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return y - (slope * x + intercept)


def to_anomaly(da: xr.DataArray, base_years: Tuple[int, int]) -> xr.DataArray:
    """Compute anomalies relative to baseline period.

    相对基准期 (base_years) 的距平。
    """
    if "year" in da.dims:
        years = da["year"].values
    else:
        years = da["time"].dt.year.values
    mask = (years >= base_years[0]) & (years <= base_years[1])
    clim = da.isel(year=mask).mean("year") if "year" in da.dims else da.sel(time=mask).mean("time")
    return da - clim
