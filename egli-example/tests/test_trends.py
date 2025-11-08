import numpy as np
import xarray as xr
from extreme_et_detection.trends import compute_trend_da

def test_compute_trend_da():
    years = np.arange(2000, 2010)
    ny = len(years)
    lat = [0,]
    lon = [0,]
    data = np.tile(np.linspace(0, 9, ny)[:, None, None], (1, 1, 1))
    da = xr.DataArray(data, coords={"year": years, "lat": lat, "lon": lon},
                      dims=("year", "lat", "lon"))
    slope = compute_trend_da(da, years.values)
    assert slope.squeeze() > 0.5
