"""Multi-resolution utilities.

多分辨率工具：
- 0.25/0.1° 等规则网格之间聚合
- 高分辨率 (例如 30–100 m) 与粗分辨率之间的上/下采样接口
- 用于构造 WRR / NCC 文章中多尺度极端 ET 对比

这里只提供结构化实现，核心思想：
- 使用面积权重 / 像元计数进行聚合
- 使用 nearest/mean 等方式从粗网格映射回高分辨率分析模式
"""

import numpy as np
import xarray as xr


def upscale_mean(hi: xr.DataArray, factor_lat: int, factor_lon: int) -> xr.DataArray:
    """Simple upscale by mean over factor_lat x factor_lon blocks.

    用于从高分辨率到粗分辨率的测试。
    """
    return hi.coarsen(lat=factor_lat, lon=factor_lon, boundary="trim").mean()


def downscale_nearest(coarse: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Downscale by nearest-neighbor mapping to target grid.

    将粗网格变量插值到高分辨率目标网格（简单最近邻）。
    """
    return coarse.interp(lat=target["lat"], lon=target["lon"], method="nearest")
