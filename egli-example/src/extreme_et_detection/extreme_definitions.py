"""Flexible extreme definitions for ET and PET.

支持 WRR / NCC 计划中的多种极端指标定义：
- 年最大块极值 (block maxima)
- 分位数阈值 (q95, q99 等)
- 考虑季节性的本地阈值定义

All functions operate on xarray objects for easy chaining.
"""

from typing import Tuple
import numpy as np
import xarray as xr


def annual_max_block(da: xr.DataArray) -> xr.DataArray:
    """Annual block maxima of input variable.

    对任意日尺度变量计算逐年块极值（如日 ET、日 PET、7 日累积等）。
    """
    years = da["time"].dt.year
    return da.groupby(years).max("time").rename({"group": "year"})


def percentile_extreme_threshold(
    da: xr.DataArray,
    q: float = 0.95,
    base_years: Tuple[int, int] = None,
) -> xr.DataArray:
    """Compute grid-wise percentile threshold.

    在基准期上计算每个格点的 q 分位值，用于定义极端事件阈值。

    If base_years is None, use full period.
    """
    if base_years is not None:
        years = da["time"].dt.year
        mask = (years >= base_years[0]) & (years <= base_years[1])
        da_base = da.where(mask, drop=True)
    else:
        da_base = da
    thr = da_base.quantile(q, dim="time")
    return thr


def seasonal_local_extreme(
    da: xr.DataArray,
    q: float = 0.95,
    window_days: int = 31,
) -> xr.DataArray:
    """Seasonally varying local extreme threshold.

    类似气温极端的“本地百分位”方法：
    - 对每一天（或滑动窗口）分别计算历史分位数
    - 消除强季节循环后识别异常极端 ET/PET

    返回与原始时间轴一致的阈值字段。
    """
    # day-of-year based moving window quantile
    doy = da["time"].dt.dayofyear
    # 简化实现：按 DOY 分组求分位数，可再扩展为滑动窗口平滑
    thr = (
        da.groupby(doy)
        .quantile(q, dim="time")
        .reindex(doy=doy)
        .rename({"doy": "time"})
    )
    # thr 现在是一个与时间对齐的阈值序列 (但是需要广播 lat,lon)
    thr = thr.assign_coords(time=da["time"])
    return thr
