"""Trend estimation and detection tests.

趋势估计与检验：
- OLS 线性趋势
- Theil-Sen 鲁棒趋势
- Mann-Kendall 趋势检验
"""

from typing import Dict, Tuple
import numpy as np
import xarray as xr
from scipy import stats
import pymannkendall as mk


def linear_trend(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    slope, _, _, p, _ = stats.linregress(x, y)
    return slope, p


def theil_sen_trend(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    res = stats.theilslopes(y, x)
    slope = res[0]
    _, p = stats.kendalltau(x, y)
    return slope, p


def mann_kendall_test(y: np.ndarray) -> Dict[str, float]:
    res = mk.original_test(y)
    return {
        "trend": res.trend,
        "h": int(res.h),
        "p": res.p,
        "z": res.z,
        "Tau": res.Tau,
        "s": res.s,
        "var_s": res.var_s,
    }


def compute_trend_da(da: xr.DataArray, years: np.ndarray,
                     method: str = "theil-sen") -> xr.DataArray:
    """Grid-wise trend along 'year' dimension.

    在 year 维度上逐格计算趋势，用于全球/区域趋势图。
    """
    def _trend_1d(y):
        mask = np.isfinite(y)
        if mask.sum() < 5:
            return np.nan
        if method == "theil-sen":
            slope, _ = theil_sen_trend(y[mask], years[mask])
        else:
            slope, _ = linear_trend(y[mask], years[mask])
        return slope

    return xr.apply_ufunc(
        _trend_1d,
        da,
        input_core_dims=[["year"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
