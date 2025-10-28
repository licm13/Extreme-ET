"""
Extreme-ET: 极端蒸发事件分析工具包
Extreme Evaporation Events Analysis Toolkit

本工具包实现了两篇顶级期刊论文中提出的极端蒸发事件检测和分析方法。
This toolkit implements extreme evaporation event detection and analysis methods
from two top-tier journal papers.

参考文献 / References:
-----------------------
1. Markonis, Y. (2025). On the Definition of Extreme Evaporation Events (ExEvEs).
   Geophysical Research Letters.

2. Zhao et al. (2025). Regional variations in drivers of extreme reference
   evapotranspiration across the contiguous United States. Water Resources Research.

主要功能 / Main Features:
--------------------------
1. 极端事件检测 / Extreme Event Detection
   - ERT_hist: 历史相对阈值法 / Historical relative threshold
   - ERT_clim: 气候学方法 / Climatological method
   - OPT: 最优路径阈值法 / Optimal path threshold

2. ET0 计算 / ET0 Calculation
   - ASCE 标准化 Penman-Monteith 方程
   - ASCE Standardized Penman-Monteith equation

3. 驱动因子分析 / Driver Attribution Analysis
   - 气象驱动因子贡献分析
   - Meteorological driver contribution analysis
   - 季节变化分析 / Seasonal variation analysis

4. 数据处理 / Data Processing
   - Z-score 标准化 / Z-score standardization
   - Hurst 指数计算 / Hurst exponent calculation
   - 移动平均 / Moving average

使用示例 / Usage Examples:
--------------------------
示例 1: 基本极端事件检测
Example 1: Basic extreme event detection
>>> from extreme_et import detect_extreme_events_hist, calculate_et0
>>> import numpy as np
>>>
>>> # 生成合成气象数据 / Generate synthetic meteorological data
>>> n_days = 365 * 40  # 40 years
>>> T_mean = 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
>>> T_max = T_mean + 5
>>> T_min = T_mean - 5
>>> Rs = 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
>>> u2 = np.full(n_days, 2.0)
>>> ea = np.full(n_days, 1.5)
>>>
>>> # 计算 ET0 / Calculate ET0
>>> ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea)
>>>
>>> # 检测极端事件 / Detect extreme events
>>> extreme_mask, threshold = detect_extreme_events_hist(ET0, severity=0.005)
>>> print(f"检测到 {np.sum(extreme_mask)} 个极端日")
>>> print(f"Detected {np.sum(extreme_mask)} extreme days")

示例 2: 驱动因子贡献分析
Example 2: Driver contribution analysis
>>> from extreme_et import calculate_contributions
>>>
>>> # 计算各驱动因子的贡献 / Calculate driver contributions
>>> contributions = calculate_contributions(
...     T_mean, T_max, T_min, Rs, u2, ea, extreme_mask
... )
>>> print("驱动因子贡献 / Driver contributions:")
>>> for driver, contrib in contributions.items():
...     print(f"  {driver}: {contrib:.1f}%")

示例 3: 数据预处理
Example 3: Data preprocessing
>>> from extreme_et import standardize_to_zscore, calculate_hurst_exponent
>>>
>>> # Z-score 标准化（去季节性）/ Z-score standardization (deseasonalize)
>>> z_scores = standardize_to_zscore(ET0, pentad=True)
>>>
>>> # 计算 Hurst 指数（长期持续性）/ Calculate Hurst exponent (long-term persistence)
>>> H = calculate_hurst_exponent(z_scores)
>>> print(f"Hurst 指数 / Hurst exponent: {H:.3f}")
>>> if H > 0.5:
...     print("检测到事件聚类特征 / Event clustering detected")

包结构 / Package Structure:
---------------------------
extreme_et/
├── __init__.py                 # 本文件 / This file
├── extreme_detection.py        # 极端事件检测 / Extreme event detection
├── penman_monteith.py          # ET0 计算 / ET0 calculation
├── contribution_analysis.py    # 驱动因子分析 / Driver attribution
├── data_processing.py          # 数据处理 / Data processing
└── utils.py                    # 工具函数 / Utility functions

安装 / Installation:
--------------------
使用 pip 安装 / Install using pip:
    pip install -e .

或直接安装依赖 / Or install dependencies directly:
    pip install -r requirements.txt

依赖项 / Dependencies:
---------------------
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

许可证 / License:
-----------------
MIT License - 详见 LICENSE 文件 / See LICENSE file for details

作者 / Authors:
---------------
Extreme-ET Development Team

版本历史 / Version History:
---------------------------
- 0.1.0 (2025-01): 初始发布 / Initial release
- 1.0.0 (2025-10): 代码重构、添加详细注释和文档 / Code refactoring, detailed comments and docs

联系方式 / Contact:
-------------------
- GitHub: https://github.com/your-org/extreme-et
- Issues: https://github.com/your-org/extreme-et/issues
- Documentation: https://extreme-et.readthedocs.io
"""

# ============================================================================
# 版本信息 / Version Information
# ============================================================================

__version__ = "1.0.0"
__author__ = "Extreme-ET Development Team"
__license__ = "MIT"
__maintainer__ = "Extreme-ET Team"
__email__ = "extreme-et@example.com"
__status__ = "Production"

# ============================================================================
# 模块导入 / Module Imports
# ============================================================================

# 极端事件检测模块 / Extreme Event Detection Module
# 提供三种主要检测方法
# Provides three main detection methods
from .extreme_detection import (
    detect_extreme_events_hist,      # 历史相对阈值法 / Historical threshold
    detect_extreme_events_clim,      # 气候学方法 / Climatological method
    detect_compound_extreme_events,  # 复合多尺度检测 / Compound multi-scale detection
    optimal_path_threshold,          # 最优路径阈值 / Optimal path threshold
    identify_events_from_mask,       # 事件提取 / Event extraction
    calculate_event_statistics,      # 事件统计 / Event statistics
)

# 驱动因子分析模块 / Driver Attribution Analysis Module
# 量化气象变量对极端 ET0 的贡献
# Quantifies meteorological variable contributions to extreme ET0
from .contribution_analysis import (
    calculate_contributions,         # 贡献分析 / Contribution analysis
    sensitivity_analysis,            # 敏感性分析 / Sensitivity analysis
    analyze_seasonal_contributions,  # 季节分析 / Seasonal analysis
    identify_dominant_driver,        # 主导因子识别 / Dominant driver identification
    dynamic_perturbation_response,   # 动态扰动响应 / Dynamic perturbation response
    compute_perturbation_pathway,    # 扰动传播链 / Perturbation pathway solver
)

# Penman-Monteith ET0 计算模块 / Penman-Monteith ET0 Calculation Module
# 实现 ASCE 标准化方程
# Implements ASCE standardized equation
from .penman_monteith import (
    calculate_et0,                   # 主计算函数 / Main calculation function
    calculate_net_radiation,         # 净辐射 / Net radiation
    calculate_vapor_pressure_from_vpd,  # 水汽压 / Vapor pressure
    adjust_wind_speed,               # 风速调整 / Wind speed adjustment
)

# 数据处理模块 / Data Processing Module
# 统计分析和数据变换工具
# Statistical analysis and data transformation tools
from .data_processing import (
    standardize_to_zscore,           # Z-score 标准化 / Z-score standardization
    calculate_hurst_exponent,        # Hurst 指数 / Hurst exponent
    moving_average,                  # 移动平均 / Moving average
    calculate_autocorrelation,       # 自相关 / Autocorrelation
    deseasonalize_data,              # 去季节性 / Deseasonalization
)

# 工具函数模块 / Utility Functions Module
# 数据生成和可视化
# Data generation and visualization
from .utils import (
    generate_synthetic_data,         # 合成数据生成 / Synthetic data generation
    plot_extreme_events,             # 极端事件可视化 / Extreme event visualization
    plot_contribution_pie,           # 贡献饼图 / Contribution pie chart
    plot_seasonal_contributions,     # 季节贡献图 / Seasonal contribution plot
    plot_autocorrelation,            # 自相关图 / Autocorrelation plot
    calculate_event_metrics,         # 事件度量 / Event metrics
    summary_statistics,              # 统计摘要 / Summary statistics
)

# ============================================================================
# 公共 API / Public API
# ============================================================================

# 定义公共接口（控制 "from extreme_et import *" 导入的内容）
# Define public interface (controls what "from extreme_et import *" imports)
__all__ = [
    # 极端事件检测 / Extreme Event Detection
    "detect_extreme_events_hist",
    "detect_extreme_events_clim",
    "optimal_path_threshold",
    "identify_events_from_mask",
    "calculate_event_statistics",

    # 驱动因子分析 / Driver Attribution
    "calculate_contributions",
    "sensitivity_analysis",
    "analyze_seasonal_contributions",
    "identify_dominant_driver",

    # ET0 计算 / ET0 Calculation
    "calculate_et0",
    "calculate_net_radiation",
    "calculate_vapor_pressure_from_vpd",
    "adjust_wind_speed",

    # 数据处理 / Data Processing
    "standardize_to_zscore",
    "calculate_hurst_exponent",
    "moving_average",
    "calculate_autocorrelation",
    "deseasonalize_data",

    # 工具函数 / Utility Functions
    "generate_synthetic_data",
    "plot_extreme_events",
    "plot_contribution_pie",
    "plot_seasonal_contributions",
    "plot_autocorrelation",
    "calculate_event_metrics",
    "summary_statistics",
]

# ============================================================================
# 包初始化 / Package Initialization
# ============================================================================

def _check_dependencies():
    """
    检查必要的依赖项是否安装
    Check if required dependencies are installed

    如果缺少依赖项，显示友好的错误消息
    Show friendly error message if dependencies are missing
    """
    try:
        import numpy
        import pandas
        import scipy
        import matplotlib
    except ImportError as e:
        missing_package = str(e).split("'")[1]
        raise ImportError(
            f"缺少必要的依赖包: {missing_package}\n"
            f"Missing required dependency: {missing_package}\n"
            f"请运行: pip install -r requirements.txt\n"
            f"Please run: pip install -r requirements.txt"
        )


# 执行依赖项检查
# Execute dependency check
_check_dependencies()

# 显示欢迎信息（仅在交互模式下）
# Show welcome message (only in interactive mode)
if __name__ != "__main__":
    import sys
    if hasattr(sys, 'ps1'):  # 检查是否为交互模式 / Check if interactive mode
        print(f"Extreme-ET v{__version__} loaded successfully!")
        print("使用 help(extreme_et) 查看帮助信息")
        print("Use help(extreme_et) for more information")
