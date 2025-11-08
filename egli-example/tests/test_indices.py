import numpy as np
import xarray as xr
from extreme_et_detection.indices import compute_etx7d_annual_max

def test_etx7d_basic_shape():
    time = np.arange(365 * 3)
    lat = [0, 1]
    lon = [10, 20]
    da = xr.DataArray(
        np.random.rand(len(time), len(lat), len(lon)),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
    )
    etx = compute_etx7d_annual_max(da)
    assert "year" in etx.dims
    assert etx.sizes["year"] == 3
