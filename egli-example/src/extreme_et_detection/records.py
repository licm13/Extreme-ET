"""Record year analysis.

记录年统计：每个格点的极端指标最大值出现在何年。
"""

import xarray as xr


def record_years(etx: xr.DataArray) -> xr.DataArray:
    """Return year of record maximum along 'year' dimension."""
    years = etx["year"].values
    idx = etx.argmax("year")
    out = etx["year"].isel(year=idx)
    out.name = "record_year"
    return out
