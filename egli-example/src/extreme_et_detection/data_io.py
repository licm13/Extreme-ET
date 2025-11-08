"""Data I/O utilities.

数据读取工具函数：
- 针对 CMIP6 / ERA5-Land / GLEAM / X-BASE / 高分辨率产品的接口
- 与分析逻辑解耦，便于替换与扩展
"""

from pathlib import Path
from typing import List, Optional
import xarray as xr

from .config import RAW_CMIP6_DIR, RAW_ERA5_DIR, RAW_GLEAM_DIR, RAW_XBASE_DIR, ET_VAR_NAME


def open_mfds_sorted(files: List[Path]) -> xr.Dataset:
    """Open multiple NetCDF files as one Dataset, sorted by filename.

    多文件按文件名排序后合并，适用于按年/成员拆分的数据。
    """
    files = sorted(map(str, files))
    if not files:
        raise FileNotFoundError("No input files found for open_mfds_sorted.")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds


def load_cmip6_et(model: str, experiment: str, member: str,
                  varname: str = ET_VAR_NAME) -> xr.DataArray:
    """Load CMIP6 ET for given model/experiment/member.

    加载指定 CMIP6 模型的蒸散发数据（用户需预先下载）。

    Returns
    -------
    xr.DataArray
        dims: (time, lat, lon)
    """
    pattern = RAW_CMIP6_DIR / model / experiment / f"*{member}*{varname}*.nc"
    files = list(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    ds = open_mfds_sorted(files)
    return ds[varname]


def load_era5_land_et(varname: str = "e") -> xr.DataArray:
    """Load ERA5-Land ET (latent heat/flux already converted if needed).

    加载 ERA5-Land 蒸散发数据（需提前统一单位）。
    """
    files = sorted(RAW_ERA5_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError("No ERA5-Land files found.")
    ds = open_mfds_sorted(files)
    return ds[varname]


def load_gleam_et(varname: str = "ET") -> xr.DataArray:
    """Load GLEAM ET.

    加载 GLEAM 蒸散发产品。
    """
    files = sorted(RAW_GLEAM_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError("No GLEAM files found.")
    ds = open_mfds_sorted(files)
    return ds[varname]


def load_xbase_et(varname: str = "ET") -> xr.DataArray:
    """Load X-BASE ET or other observation-based blended products.

    加载 X-BASE 或其他基于观测融合的 ET 产品。
    """
    files = sorted(RAW_XBASE_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError("No X-BASE files found.")
    ds = open_mfds_sorted(files)
    return ds[varname]
