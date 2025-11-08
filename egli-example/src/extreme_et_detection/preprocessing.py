"""Preprocessing utilities.

预处理模块：
- 单位转换
- 时间处理
- 简化的重网格（建议真实研究中使用 xESMF 等更严谨方法）
"""

from typing import Tuple
import numpy as np
import xarray as xr


def to_daily_et_mm(da: xr.DataArray, flux_is_kg_m2_s: bool = True) -> xr.DataArray:
    """Convert ET flux to daily total in mm/day.

    将 (kg m-2 s-1) 通量转换为 mm/day。
    若数据已为 mm/day，可设置 flux_is_kg_m2_s=False 跳过。

    Notes
    -----
    这里使用均匀时间步长近似，真实数据建议显式使用时间差。
    """
    if not flux_is_kg_m2_s:
        return da

    time = da["time"].values
    if len(time) < 2:
        raise ValueError("Time dimension too short to infer timestep.")
    dt_seconds = (time[1] - time[0]) / np.timedelta64(1, "s")
    factor = dt_seconds  # 1 kg/m2 = 1 mm
    return da * factor


def regrid_coarsen_boxmean(
    da: xr.DataArray, dlat: float, dlon: float
) -> xr.DataArray:
    """Simple box-mean coarsening to target resolution.

    简化重网格：适用于原始为规则网格的测试和快速实验。
    严肃计算建议使用 xESMF (conservative/bilinear)。

    Parameters
    ----------
    da : xr.DataArray(time, lat, lon)
    dlat, dlon : target resolution in degrees
    """
    lat = da["lat"].values
    lon = da["lon"].values
    dlat0 = float(np.abs(lat[1] - lat[0]))
    dlon0 = float(np.abs(lon[1] - lon[0]))
    nlat = max(1, int(round(dlat / dlat0)))
    nlon = max(1, int(round(dlon / dlon0)))
    return da.coarsen(lat=nlat, lon=nlon, boundary="trim").mean()
