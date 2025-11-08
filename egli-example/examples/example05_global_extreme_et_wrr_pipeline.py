"""WRR-style pipeline sketch (0.1° global extreme ET & ET/PET).

该示例脚本给出完整思路（伪代码接口），你只需补充真实数据加载路径：
1. 读取 0.1° 全球 ET & PET 产品（多源融合或自建）
2. 计算多种极端定义 (ETx7d, block maxima, percentile-based)
3. 建立 ForcedResponseDetector 与 CMIP6 强迫响应
4. 检验 1950–2025 趋势、对比 piControl / historical 分布
"""

# 占位导入，用于展示逻辑
from extreme_et_detection.indices import compute_etx7d_annual_max
from extreme_et_detection.extreme_definitions import (
    annual_max_block,
    percentile_extreme_threshold,
    seasonal_local_extreme,
)
from extreme_et_detection.pet_indices import et_pet_ratio
from extreme_et_detection.ridge_detection import ForcedResponseDetector
from extreme_et_detection.trends import compute_trend_da
from extreme_et_detection.basin_scale import aggregate_by_basin_mask
from extreme_et_detection.scaling import compare_scales

# 实际实现时：
# - 利用 data_io 加载 0.1° ET/PET 数据集
# - 用上述函数构造：
#   * 多定义极端强度与频率地图
#   * 流域尺度 vs 网格尺度对比
#   * 强迫响应趋势在多指标下的一致性
