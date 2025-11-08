"""NCC-style pipeline sketch (multi-resolution, urban vs natural).

用于 Nature Climate Change 方向构思：
- 使用多源高分辨率不透水面 / NDVI / LST / 城市边界数据
- 将 0.1° / 2.5° 极端 ET 与 100 m 城市/流域格点进行匹配
- 利用 scaling.compare_scales 等接口，量化 scale-dependence
- 提取结构-ET 因果关系图谱（此处留空接口，便于后续加入因果/ML）

该脚本同样为结构草图，不强绑数据源。
"""

from extreme_et_detection.multi_res import upscale_mean, downscale_nearest
from extreme_et_detection.scaling import compare_scales
from extreme_et_detection.basin_scale import aggregate_by_basin_mask

# TODO:
# 1. 读取高分辨率 ET 或能量平衡反演结果
# 2. 上聚合至 0.1°，与 coarse 产品对比 (upscale_mean)
# 3. 利用建筑/不透水比例等特征解释极端差异（后续可加入 ML 模块）
