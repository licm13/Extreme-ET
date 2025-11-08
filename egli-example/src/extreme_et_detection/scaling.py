"""Scaling diagnostics between resolutions.

用于分析：
- 0.1° 与 流域 / 100m 极端 ET 的差异和比例
- 支撑 NCC 文章中的 “scale dependence of ET extremes”。

这里只写出基本接口，用户可按需求丰富统计。
"""

import numpy as np
import xarray as xr


def compare_scales(metric_coarse: xr.DataArray, metric_fine: xr.DataArray) -> xr.Dataset:
    """Compute simple diagnostics between two scales of the same metric.

    返回差值、比值、相关系数等。

    Assumes metric_coarse and metric_fine already on comparable support
    (e.g. both basin-scale or mapped to same grid).
    """
    ds = xr.Dataset()
    ds["diff"] = metric_fine - metric_coarse
    ds["ratio"] = metric_fine / metric_coarse
    # correlation along year/time if dimension exists
    for dim in ["year", "time"]:
        if dim in metric_coarse.dims and dim in metric_fine.dims:
            # flatten spatial, compute Pearson r for each location
            def _r(a, b):
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 5:
                    return np.nan
                return np.corrcoef(a[m], b[m])[0, 1]
            rmap = xr.apply_ufunc(
                _r,
                metric_coarse,
                metric_fine,
                input_core_dims=[[dim], [dim]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            ds["corr_" + dim] = rmap
            break
    return ds
