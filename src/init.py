"""
Extreme Evaporation Events Analysis Toolkit

Implementations of methods from:
- Markonis (2025): Extreme Evaporation Events (ExEvEs)
- Zhao et al. (2025): Regional variations in ETo drivers
"""

__version__ = "0.1.0"

from .extreme_detection import (
    detect_extreme_events_hist,
    detect_extreme_events_clim,
    optimal_path_threshold
)
from .contribution_analysis import (
    calculate_contributions,
    sensitivity_analysis
)
from .penman_monteith import calculate_et0
from .data_processing import (
    standardize_to_zscore,
    calculate_hurst_exponent,
    moving_average
)

__all__ = [
    "detect_extreme_events_hist",
    "detect_extreme_events_clim",
    "optimal_path_threshold",
    "calculate_contributions",
    "sensitivity_analysis",
    "calculate_et0",
    "standardize_to_zscore",
    "calculate_hurst_exponent",
    "moving_average",
]