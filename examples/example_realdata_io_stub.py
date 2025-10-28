"""Minimal real-data IO stub using xarray + sampling.

Edit the path/variable/coords below to run against a NetCDF product.
Requires: xarray, netCDF4.
"""

from __future__ import annotations

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    import xarray as xr  # noqa: F401
except Exception as exc:
    print("xarray/netCDF4 not installed. Install them to run this demo.")
    raise SystemExit(0)

from src.io_utils import read_netcdf_variable, sample_series_at_point


def main():
    # TODO: edit these
    path = "path/to/product.nc"  # <-- change to your NetCDF file
    var = "et"                    # <-- variable name in file
    lat, lon = 35.0, -120.0       # <-- station location

    try:
        da, lats, lons, times = read_netcdf_variable(path, var)
    except Exception as e:
        print("Failed to open dataset:", e)
        return

    series_nn = sample_series_at_point(da, lat, lon, method="nearest")
    series_bl = sample_series_at_point(da, lat, lon, method="bilinear")

    # Print quick stats
    print("Nearest-neighbor:", float(series_nn.mean().values))
    print("Bilinear:", float(series_bl.mean().values))


if __name__ == "__main__":
    main()

