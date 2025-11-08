"""Basin-scale aggregation utilities.

流域尺度分析工具：
- 使用离散流域栅格 mask 或 shapefile 转栅格的 mask
- 聚合 ET / extreme ET / ET/PET 至流域尺度

适配 WRR/NCC 方案中的：0.1° ↔ 流域 ↔ 高分辨率城市子流域。
"""

import numpy as np
import xarray as xr


def aggregate_by_basin_mask(
    da: xr.DataArray,
    basin_mask: xr.DataArray,
    basin_dim: str = "basin",
) -> xr.DataArray:
    """Aggregate to basin means given a basin ID mask.

    Parameters
    ----------
    da : (time or year, lat, lon)
    basin_mask : (lat, lon) integer IDs
    """
    # align
    basin_mask = basin_mask.broadcast_like(da.isel(time=0) if "time" in da.dims else da.isel(year=0))
    ids = np.unique(basin_mask.values[~np.isnan(basin_mask.values)])
    ids = ids.astype(int)

    results = []
    for bid in ids:
        mask = xr.where(basin_mask == bid, 1.0, np.nan)
        num = (da * mask).sum(("lat", "lon"))
        den = mask.sum(("lat", "lon"))
        results.append((bid, num / den))

    data_vars = []
    coords = {basin_dim: ids}
    # stack along new basin dimension
    arr = xr.concat([x for _, x in results], dim=basin_dim)
    arr = arr.assign_coords({basin_dim: ids})
    return arr
