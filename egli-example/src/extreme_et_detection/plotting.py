"""Lightweight plotting helpers.

注意：这里只提供基础接口，不强制配色和风格，便于按期刊标准自定义。
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr


def plot_global_map(da: xr.DataArray, title: str = ""):
    proj = ccrs.Robinson()
    fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(10, 5))
    da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"shrink": 0.6},
    )
    ax.coastlines()
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
