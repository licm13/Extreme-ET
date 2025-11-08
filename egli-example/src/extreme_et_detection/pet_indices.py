"""PET-related helpers.

PET 指标与 ET/PET 极端比值计算：
- 支持从外部 PET 数据集中读取
- 或提供接口计算简单的基于温度/辐射的 PET（占位，可按需要补全）

这里只给出框架，方便后续拓展。
"""

from typing import Tuple
import xarray as xr


def load_pet_from_dataset(ds: xr.Dataset, varname: str = "pet") -> xr.DataArray:
    """Extract PET variable from dataset.

    简单封装，便于替换不同数据源（CRU/ERA5/自算等）。
    """
    if varname not in ds:
        raise KeyError(f"PET variable '{varname}' not found in dataset.")
    return ds[varname]


def et_pet_ratio(et: xr.DataArray, pet: xr.DataArray) -> xr.DataArray:
    """Compute ET/PET ratio with masking.

    ET/PET 比值是极端干旱/蒸发受限条件的重要诊断量。
    """
    ratio = et / pet
    return ratio.where(pet > 0)
