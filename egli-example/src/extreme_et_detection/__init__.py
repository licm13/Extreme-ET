"""Extreme ET Detection Package

Implements indices, preprocessing, ridge-based forced response detection,
trend tests, regional diagnostics, and multi-resolution analysis for
evapotranspiration extremes.

本包用于复刻 Egli et al. (2025) 的蒸散发极端与季节变化检测框架，
并向更高分辨率与多源数据拓展，支持 WRR / Nature Climate Change
级别工作的完整科研工作流。
"""

from .indices import compute_etx7d_annual_max, compute_seasonal_mean
from .extreme_definitions import (
    annual_max_block,
    percentile_extreme_threshold,
    seasonal_local_extreme,
)
from .ridge_detection import ForcedResponseDetector
from .trends import linear_trend, theil_sen_trend, mann_kendall_test
