"""
极端事件检测方法模块 (Extreme Event Detection Module)

本模块实现了三种极端蒸发事件检测方法，基于 Zhao et al. (2025) 论文。
This module implements three extreme evaporation event detection methods
based on Zhao et al. (2025).

主要方法 (Main Methods):
-----------------------
1. ERT_hist: 历史相对阈值法 (Historical Relative Threshold)
   - 使用整个历史记录的百分位数作为阈值
   - Uses percentile from entire historical record as threshold

2. ERT_clim: 气候学方法 (Climatological Method)
   - 基于每个日历日的气候学异常检测
   - Detects anomalies relative to each calendar day climatology

3. OPT: 最优路径阈值法 (Optimal Path Threshold)
   - 通过迭代优化确定最优阈值
   - Determines optimal thresholds through iterative optimization

参考文献 (References):
----------------------
Zhao et al. (2025). Regional variations in drivers of extreme reference
evapotranspiration across the contiguous United States. Water Resources Research.

作者: Extreme-ET Team
日期: 2025
版本: 1.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Iterable
from scipy import stats
try:
    # Prefer package-relative import when imported as src.extreme_detection
    from .data_processing import moving_average
except Exception:
    # Fallback if imported without package context
    from src.data_processing import moving_average


# ============================================================================
# 模块常量定义 (Module Constants)
# ============================================================================

# 默认严重性阈值 (Default Severity Thresholds)
DEFAULT_SEVERITY_HIST = 0.005  # 历史法: 0.5% (~1.8天/年)
DEFAULT_SEVERITY_CLIM = 0.05   # 气候学法: 5%

# 事件持续时间参数 (Event Duration Parameters)
DEFAULT_MIN_DURATION = 3       # 最小连续天数 (minimum consecutive days)
DEFAULT_SMOOTHING_WINDOW = 7   # 平滑窗口大小 (smoothing window size)

# 优化参数 (Optimization Parameters)
DEFAULT_MAX_ITERATIONS = 50    # 最大迭代次数 (max iterations)
CONVERGENCE_TOLERANCE = 0.001  # 收敛容差 (convergence tolerance)
INITIAL_QUANTILE = 0.90        # 初始百分位数 (initial quantile)

# 阈值调整因子 (Threshold Adjustment Factors)
THRESHOLD_DECREASE_FACTOR = 0.95  # 降低阈值因子 (to increase event rate)
THRESHOLD_INCREASE_FACTOR = 1.05  # 提高阈值因子 (to decrease event rate)

# 日历参数 (Calendar Parameters)
DAYS_PER_YEAR = 365
DAYS_IN_LEAP_YEAR = 366


# ============================================================================
# 主要检测函数 (Main Detection Functions)
# ============================================================================


def _estimate_tail_threshold(
    data: Union[np.ndarray, List, pd.Series],
    severity: float,
    model: str = 'empirical',
    base_quantile: float = 0.9,
    min_exceedances: int = 20
) -> Tuple[float, Dict[str, Union[str, float, int]]]:
    """Estimate an upper-tail threshold for a given severity level.

    This helper consolidates the empirical percentile approach used in Zhao
    et al. (2025) with an optional extreme value extension based on the
    Generalized Pareto Distribution (GPD).  The GPD branch follows the peak
    over threshold framework and extrapolates the quantile corresponding to
    the target occurrence rate when sufficient exceedances are available.

    Parameters
    ----------
    data : array-like
        Sample values.
    severity : float
        Tail probability (e.g., ``0.005`` for the top 0.5% of the
        distribution).
    model : {"empirical", "gpd"}, default="empirical"
        Tail estimator to use.  ``"empirical"`` relies on the direct sample
        quantile.  ``"gpd"`` fits a generalized Pareto distribution to the
        exceedances above ``base_quantile`` and extrapolates the requested
        severity.  The function automatically falls back to the empirical
        estimate if the tail sample is too small or the fitted parameters are
        numerically unstable.
    base_quantile : float, default=0.9
        Quantile used as the lower bound of the tail model when ``model`` is
        ``"gpd"``.  Must satisfy ``base_quantile > 1 - severity`` so that the
        extrapolation targets a rarer percentile than the fitting threshold.
    min_exceedances : int, default=20
        Minimum number of exceedances required to activate the GPD branch.

    Returns
    -------
    threshold : float
        Estimated threshold value.
    info : dict
        Dictionary describing the tail model, fitted parameters, and
        diagnostics that can be surfaced to the calling function.
    """

    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        data = data.ravel()

    quantile = 1 - severity
    info: Dict[str, Union[str, float, int]] = {
        'tail_model': model,
        'severity': float(severity),
        'target_quantile': float(quantile),
    }

    if model.lower() != 'gpd':
        threshold = float(np.quantile(data, quantile))
        info['tail_model'] = 'empirical'
        return threshold, info

    # Ensure base quantile is sensible and targets a less extreme percentile
    # than the requested severity.
    base_quantile = float(np.clip(base_quantile, 0.0, 0.999))
    if base_quantile <= quantile:
        base_quantile = min(0.95, max(quantile + 1e-4, base_quantile + 1e-4))

    threshold_base = float(np.quantile(data, base_quantile))
    exceedances = data[data > threshold_base] - threshold_base
    n_exceedances = exceedances.size
    p_exceed_base = 1 - base_quantile

    info.update({
        'tail_model': 'gpd',
        'base_quantile': base_quantile,
        'base_threshold': threshold_base,
        'n_exceedances': int(n_exceedances),
        'p_exceed_base': p_exceed_base,
    })

    # Guard conditions for the GPD fit.
    if (
        n_exceedances < max(min_exceedances, 5)
        or severity >= p_exceed_base
        or np.allclose(np.std(exceedances), 0.0)
    ):
        info['tail_model'] = 'empirical_fallback'
        threshold = float(np.quantile(data, quantile))
        info['threshold_fallback'] = threshold
        return threshold, info

    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    except (RuntimeError, ValueError) as exc:
        info['tail_model'] = 'empirical_fallback'
        info['fit_error'] = str(exc)
        threshold = float(np.quantile(data, quantile))
        info['threshold_fallback'] = threshold
        return threshold, info

    info.update({
        'gpd_shape': float(shape),
        'gpd_scale': float(scale),
    })

    if scale <= 0:
        info['tail_model'] = 'empirical_fallback'
        threshold = float(np.quantile(data, quantile))
        info['threshold_fallback'] = threshold
        return threshold, info

    # Compute quantile using the survival function of the fitted GPD.
    ratio = severity / p_exceed_base
    ratio = max(ratio, 1e-12)

    if abs(shape) < 1e-6:
        # Limit as shape -> 0 (exponential tail)
        tail_increment = scale * np.log(1 / ratio)
    else:
        tail_increment = (scale / shape) * (ratio ** (-shape) - 1)

    threshold = threshold_base + tail_increment
    info['threshold_extrapolated'] = float(threshold)

    # Safety: ensure extrapolated threshold is at least as large as the base.
    if threshold < threshold_base:
        threshold = float(np.quantile(data, quantile))
        info['tail_model'] = 'empirical_fallback'
        info['threshold_fallback'] = threshold

    return float(threshold), info

def detect_extreme_events_hist(
    data: Union[np.ndarray, List, pd.Series],
    severity: float = DEFAULT_SEVERITY_HIST,
    tail_model: str = 'empirical',
    gpd_base_quantile: float = 0.9,
    return_details: bool = False
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict]]:
    """
    使用历史相对阈值法检测极端事件 (ERT_hist)
    Detect extreme events using historical ERT method (ERT_hist).

    算法原理 (Algorithm):
    -------------------
    1. 计算整个时间序列的高百分位数阈值
       Calculate high percentile threshold from entire time series
    2. 标记所有超过该阈值的日子为极端事件
       Mark all days exceeding threshold as extreme events
    3. 适用于识别最大量级事件
       Suitable for identifying largest magnitude events

    方法特点 (Characteristics):
    -------------------------
    - 优点: 简单快速，计算效率高
      Advantages: Simple, fast, computationally efficient
    - 缺点: 不考虑季节性变化
      Disadvantages: Does not account for seasonality
    - 适用场景: 识别记录破纪录的极端值
      Use case: Identifying record-breaking extremes

    Parameters
    ----------
    data : array-like
        时间序列数据（如日蒸发量 ET0）
        Time series data (e.g., daily ET0 in mm/day)
        要求: 至少包含 365 天数据
        Requirement: At least 365 days of data

    severity : float, default=0.005
        事件发生率（极端程度）
        Occurrence rate for extreme events
        例如: 0.005 = 0.5% ≈ 1.8 天/年
        Example: 0.005 = 0.5% ≈ 1.8 days/year
        范围: (0, 1)
        Range: (0, 1)

    tail_model : {"empirical", "gpd"}, default="empirical"
        上尾阈值估计方法选择
        Tail model selector. ``"empirical"`` keeps the historical percentile
        approach, while ``"gpd"`` enables a Generalized Pareto extrapolation
        for very rare events.

    gpd_base_quantile : float, default=0.9
        当使用 GPD 扩展时的基准分位数
        Base quantile used for fitting the tail model when
        ``tail_model='gpd'``. 必须高于目标分位数以保证外推有效。

    return_details : bool, default=False
        是否返回详细事件信息
        Whether to return detailed event information
        包括: 事件数量、持续时间、统计信息等
        Includes: event count, duration, statistics, etc.

    Returns
    -------
    extreme_mask : np.ndarray (bool)
        布尔数组，指示极端日
        Boolean array indicating extreme days
        True = 极端日, False = 正常日
        True = extreme day, False = normal day

    threshold : float
        使用的阈值（与 data 单位相同）
        Threshold value used (same units as data)

    details : dict (可选, optional)
        详细事件信息字典 (当 return_details=True 时)
        Detailed event information (when return_details=True)
        键值对 (Keys):
            - 'threshold': 阈值 (threshold value)
            - 'tail_model': 阈值拟合方法 (tail fitting method)
            - 'threshold_diagnostics': 阈值诊断信息 (diagnostic dict)
            - 'n_extreme_days': 极端日数量 (number of extreme days)
            - 'occurrence_rate': 实际发生率 (actual occurrence rate)
            - 'n_events': 事件数量 (number of events)
            - 'events': 事件列表 (list of events)
            - 'severity_level': 输入的严重性参数 (input severity)

    Raises
    ------
    ValueError
        如果 severity 不在 (0, 1) 范围内
        If severity is not in range (0, 1)
        如果数据长度小于 365 天
        If data length is less than 365 days
    TypeError
        如果输入数据类型不正确
        If input data type is incorrect

    Examples
    --------
    示例 1: 基本用法 (Basic usage)
    >>> import numpy as np
    >>> # 生成 40 年的合成数据 (Generate 40 years of synthetic data)
    >>> data = np.random.gamma(2, 2, 365 * 40)  # Gamma 分布模拟 ET0
    >>>
    >>> # 检测极端事件 (Detect extreme events)
    >>> extreme_mask, threshold = detect_extreme_events_hist(data, severity=0.005)
    >>>
    >>> print(f"阈值 Threshold: {threshold:.2f} mm/day")
    >>> print(f"极端日数量 Extreme days: {np.sum(extreme_mask)}")
    >>> print(f"发生率 Rate: {np.sum(extreme_mask)/len(data)*100:.2f}%")

    示例 2: 获取详细信息 (Get detailed information)
    >>> extreme_mask, threshold, details = detect_extreme_events_hist(
    ...     data, severity=0.005, return_details=True
    ... )
    >>> print(f"检测到 {details['n_events']} 个事件")
    >>> print(f"Detected {details['n_events']} events")
    >>>
    >>> # 分析第一个事件 (Analyze first event)
    >>> if details['events']:
    ...     first_event = details['events'][0]
    ...     print(f"事件持续时间: {first_event['duration']} 天")
    ...     print(f"Event duration: {first_event['duration']} days")

    Notes
    -----
    算法复杂度 (Computational Complexity):
    - 时间复杂度: O(n log n) 由于百分位数计算
      Time complexity: O(n log n) due to quantile calculation
    - 空间复杂度: O(n)
      Space complexity: O(n)

    与其他方法的对比 (Comparison with other methods):
    - vs ERT_clim: ERT_hist 不考虑季节性，适合全年统一标准
      vs ERT_clim: ERT_hist ignores seasonality, uniform threshold
    - vs OPT: ERT_hist 更简单但缺乏优化
      vs OPT: ERT_hist is simpler but lacks optimization

    See Also
    --------
    detect_extreme_events_clim : 气候学方法 (Climatological method)
    optimal_path_threshold : 最优路径阈值法 (OPT method)
    identify_events_from_mask : 从掩码提取事件 (Extract events from mask)
    """
    # ========================================================================
    # 输入验证 (Input Validation)
    # ========================================================================

    # 类型检查和转换 (Type checking and conversion)
    try:
        data = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"输入数据无法转换为数值数组 (Cannot convert data to numeric array): {e}"
        )

    # 检查数据有效性 (Check data validity)
    if len(data) < DAYS_PER_YEAR:
        raise ValueError(
            f"数据长度 ({len(data)} 天) 少于最小要求 ({DAYS_PER_YEAR} 天)\n"
            f"Data length ({len(data)} days) is less than minimum ({DAYS_PER_YEAR} days)"
        )

    # 检查严重性参数范围 (Check severity parameter range)
    if not 0 < severity < 1:
        raise ValueError(
            f"严重性参数 severity ({severity}) 必须在 (0, 1) 范围内\n"
            f"Severity parameter ({severity}) must be in range (0, 1)"
        )

    # 规范化尾部模型名称并验证 (Normalize and validate tail model)
    tail_model_normalized = tail_model.lower()
    if tail_model_normalized not in {'empirical', 'gpd'}:
        raise ValueError(
            "tail_model 必须为 'empirical' 或 'gpd'\n"
            "tail_model must be either 'empirical' or 'gpd'"
        )

    if not 0 < gpd_base_quantile < 1:
        raise ValueError(
            f"gpd_base_quantile ({gpd_base_quantile}) 必须在 (0, 1) 范围内\n"
            f"gpd_base_quantile ({gpd_base_quantile}) must fall within (0, 1)"
        )

    # 检查是否有 NaN 值 (Check for NaN values)
    if np.any(np.isnan(data)):
        raise ValueError(
            "输入数据包含 NaN 值，请先进行数据清洗\n"
            "Input data contains NaN values, please clean data first"
        )

    # ========================================================================
    # 阈值计算 (Threshold Calculation)
    # ========================================================================

    n_days = len(data)

    # 计算上尾阈值，可选择经验分位数或 GPD 外推
    # Estimate the tail threshold using empirical quantiles or a GPD fit
    threshold, tail_info = _estimate_tail_threshold(
        data,
        severity=severity,
        model=tail_model_normalized,
        base_quantile=gpd_base_quantile,
    )

    # ========================================================================
    # 极端日识别 (Extreme Day Identification)
    # ========================================================================

    # 创建布尔掩码：数据 > 阈值的日子标记为极端日
    # Create boolean mask: days with data > threshold are marked as extreme
    extreme_mask = data > threshold

    # ========================================================================
    # 返回结果 (Return Results)
    # ========================================================================

    if not return_details:
        # 简单返回：掩码和阈值
        # Simple return: mask and threshold
        return extreme_mask, threshold

    # ========================================================================
    # 计算详细统计信息 (Calculate Detailed Statistics)
    # ========================================================================

    # 统计极端日数量 (Count extreme days)
    n_extreme_days = np.sum(extreme_mask)

    # 计算实际发生率 (Calculate actual occurrence rate)
    occurrence_rate = n_extreme_days / n_days

    # 识别连续极端事件 (Identify continuous events)
    # 连续的极端日被视为一个事件
    # Consecutive extreme days are considered as one event
    events = identify_events_from_mask(extreme_mask)

    # 组装详细信息字典 (Assemble details dictionary)
    details = {
        'threshold': float(threshold),
        'tail_model': tail_info.get('tail_model', tail_model_normalized),
        'threshold_diagnostics': tail_info,
        'n_extreme_days': int(n_extreme_days),
        'occurrence_rate': float(occurrence_rate),
        'n_events': len(events),
        'events': events,
        'severity_level': severity,
        'method': 'ERT_hist',
        'data_length_days': n_days,
        'data_length_years': n_days / DAYS_PER_YEAR,
    }

    return extreme_mask, threshold, details


def detect_extreme_events_clim(
    data: Union[np.ndarray, List, pd.Series],
    severity: float = DEFAULT_SEVERITY_CLIM,
    min_duration: int = DEFAULT_MIN_DURATION,
    window: int = DEFAULT_SMOOTHING_WINDOW,
    return_details: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    使用气候学相对阈值法检测极端事件 (ERT_clim)
    Detect extreme events using climatological ERT method (ERT_clim).

    算法原理 (Algorithm):
    -------------------
    1. 对每个日历日计算气候学平均值
       Calculate climatological mean for each calendar day
    2. 对数据进行移动平均平滑处理
       Apply moving average smoothing to data
    3. 使用 OPT 方法确定每日阈值
       Determine daily thresholds using OPT method
    4. 识别超过相应日期阈值且持续足够长的事件
       Identify events exceeding daily threshold with sufficient duration

    方法特点 (Characteristics):
    -------------------------
    - 优点: 考虑季节性变化，类似热浪检测
      Advantages: Accounts for seasonality, similar to heatwave detection
    - 缺点: 计算复杂度较高
      Disadvantages: Higher computational complexity
    - 适用场景: 识别相对于季节气候的异常事件
      Use case: Identifying anomalies relative to seasonal climatology

    Parameters
    ----------
    data : array-like
        时间序列数据
        Time series data (e.g., daily ET0)
        要求: 至少包含 2 年数据以获得可靠的气候学统计
        Requirement: At least 2 years for reliable climatology

    severity : float, default=0.05
        事件发生率（5% ≈ 18 天/年）
        Occurrence rate (5% ≈ 18 days/year)
        范围: (0, 1)
        Range: (0, 1)

    min_duration : int, default=3
        事件最小持续天数
        Minimum consecutive days for an event
        典型值: 2-5 天
        Typical values: 2-5 days

    window : int, default=7
        移动平均窗口大小
        Window size for moving average preprocessing
        用于平滑短期波动
        Used to smooth short-term fluctuations

    return_details : bool, default=False
        是否返回详细事件信息
        Whether to return detailed event information

    Returns
    -------
    extreme_mask : np.ndarray (bool)
        布尔数组，指示极端日
        Boolean array indicating extreme days

    thresholds : np.ndarray
        每个日历日的阈值（366 个值）
        Threshold for each calendar day (366 values)
        索引 0 = 1月1日, 索引 365 = 12月31日/闰年
        Index 0 = Jan 1, Index 365 = Dec 31/leap day

    details : dict (可选, optional)
        详细事件信息字典
        Detailed event information

    Raises
    ------
    ValueError
        如果参数不在有效范围内
        If parameters are not in valid range

    Examples
    --------
    示例 1: 基本用法 (Basic usage)
    >>> data = np.random.gamma(2, 2, 365 * 40)
    >>> extreme_mask, thresholds = detect_extreme_events_clim(
    ...     data, severity=0.05, min_duration=3
    ... )
    >>> print(f"平均阈值 Mean threshold: {np.mean(thresholds):.2f} mm/day")
    >>> print(f"阈值范围 Threshold range: {np.min(thresholds):.2f} to {np.max(thresholds):.2f}")

    示例 2: 分析季节性差异 (Analyze seasonal differences)
    >>> # 可视化每日阈值的季节变化
    >>> # Visualize seasonal variation of daily thresholds
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.arange(1, 367), thresholds)
    >>> plt.xlabel('Day of Year')
    >>> plt.ylabel('Threshold (mm/day)')
    >>> plt.title('Seasonal Variation in Extreme Event Thresholds')

    Notes
    -----
    算法复杂度 (Computational Complexity):
    - 时间复杂度: O(n * k) 其中 k 是迭代次数
      Time complexity: O(n * k) where k is number of iterations
    - 空间复杂度: O(n)
      Space complexity: O(n)

    最小持续时间的作用 (Role of minimum duration):
    - 过滤短暂的波动
      Filters out brief fluctuations
    - 识别持续性极端事件
      Identifies persistent extreme events
    - 类似热浪定义（通常要求 ≥3 天）
      Similar to heatwave definition (typically ≥3 days)

    See Also
    --------
    detect_extreme_events_hist : 历史阈值法 (Historical threshold method)
    optimal_path_threshold : 优化阈值计算 (Optimized threshold calculation)
    identify_climatological_extremes : 气候学极端识别
    """
    # ========================================================================
    # 输入验证 (Input Validation)
    # ========================================================================

    # 数据类型转换 (Data type conversion)
    try:
        data = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert data to numeric array: {e}")

    n_days = len(data)

    # 验证数据长度 (Validate data length)
    if n_days < 2 * DAYS_PER_YEAR:
        raise ValueError(
            f"ERT_clim 方法需要至少 2 年数据以获得可靠的气候学统计\n"
            f"ERT_clim method requires at least 2 years of data for reliable climatology\n"
            f"当前数据: {n_days} 天 ({n_days/DAYS_PER_YEAR:.1f} 年)\n"
            f"Current data: {n_days} days ({n_days/DAYS_PER_YEAR:.1f} years)"
        )

    # 验证参数范围 (Validate parameter ranges)
    if not 0 < severity < 1:
        raise ValueError(f"Severity must be in range (0, 1), got {severity}")

    if min_duration < 1:
        raise ValueError(f"min_duration must be >= 1, got {min_duration}")

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # 检查 NaN 值 (Check for NaN values)
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")

    # ========================================================================
    # 数据预处理 (Data Preprocessing)
    # ========================================================================

    # 应用移动平均平滑，减少高频噪声
    # Apply moving average smoothing to reduce high-frequency noise
    # 这有助于识别持续性事件而不是单日突发
    # This helps identify persistent events rather than single-day spikes
    smoothed_data = moving_average(data, window=window)

    # ========================================================================
    # 计算每日阈值 (Calculate Daily Thresholds)
    # ========================================================================

    # 使用 OPT 方法为每个日历日优化阈值
    # Use OPT method to optimize threshold for each calendar day
    # 这确保了全年事件发生率接近目标严重性
    # This ensures year-round occurrence rate matches target severity
    thresholds = optimal_path_threshold(
        smoothed_data, # type: ignore
        target_occurrence_rate=severity,
        min_duration=min_duration
    )

    # ========================================================================
    # 识别极端事件 (Identify Extreme Events)
    # ========================================================================

    # 应用最小持续时间标准识别极端事件
    # Identify extreme events applying minimum duration criterion
    extreme_mask = identify_climatological_extremes(
        smoothed_data, # type: ignore
        thresholds,
        min_duration=min_duration
    )

    # ========================================================================
    # 返回结果 (Return Results)
    # ========================================================================

    if not return_details:
        return extreme_mask, thresholds

    # ========================================================================
    # 计算详细统计 (Calculate Detailed Statistics)
    # ========================================================================

    # 识别连续事件 (Identify continuous events)
    events = identify_events_from_mask(extreme_mask)

    # 计算增强的事件统计 (Calculate enhanced event statistics)
    events_with_stats = calculate_event_statistics(data, events)

    # 组装详细信息 (Assemble details)
    details = {
        'thresholds': thresholds,
        'threshold_mean': float(np.mean(thresholds)),
        'threshold_std': float(np.std(thresholds)),
        'threshold_min': float(np.min(thresholds)),
        'threshold_max': float(np.max(thresholds)),
        'n_extreme_days': int(np.sum(extreme_mask)),
        'occurrence_rate': float(np.sum(extreme_mask) / n_days),
        'n_events': len(events),
        'events': events_with_stats,
        'severity_level': severity,
        'min_duration': min_duration,
        'window': window,
        'method': 'ERT_clim',
        'data_length_days': n_days,
        'data_length_years': n_days / DAYS_PER_YEAR,
    }

    return extreme_mask, thresholds, details


def detect_compound_extreme_events(
    data: Union[np.ndarray, List, pd.Series],
    scales: Iterable[int] = (7, 30),
    severity_levels: Optional[Iterable[float]] = None,
    aggregator: str = 'all',
    weights: Optional[Iterable[float]] = None,
    tail_model: str = 'empirical',
    gpd_base_quantile: float = 0.9,
    return_details: bool = False
) -> Union[Tuple[np.ndarray, Dict[int, float]], Tuple[np.ndarray, Dict[int, float], Dict]]:
    """多尺度复合极端事件检测 (Compound multi-scale extreme detection).

    Inspired by the "复合阈值与尺度依赖检测" extension, this function applies
    multiple moving-average filters and enforces thresholds at each scale.  It
    supports logical AND/OR aggregations as well as a weighted exceedance score
    to highlight events that simultaneously amplify across thermal and
    dynamical time horizons.

    Parameters
    ----------
    data : array-like
        时间序列数据（通常为日 ET0）
        Time series data, typically daily ET0 values.
    scales : iterable of int, default=(7, 30)
        移动平均窗口（天）
        Moving-average windows in days.  Each window defines a temporal scale.
    severity_levels : iterable of float, optional
        每个尺度的极端发生率（0-1）。如果只提供一个值，则对所有尺度复用。
        Per-scale severity levels.  If ``None`` a default of 5% is used.  A
        single value will be broadcast to all scales.
    aggregator : {"all", "any", "weighted"}, default="all"
        复合策略：``"all"`` 要求所有尺度同时超过阈值；``"any"`` 为并集；
        ``"weighted"`` 根据权重加权正向超阈量。
        Compound rule controlling how individual scale masks are combined.
    weights : iterable of float, optional
        当 ``aggregator='weighted'`` 时使用的权重。未提供则等权。
        Positive weights for the weighted aggregator.  Ignored otherwise.
    tail_model : {"empirical", "gpd"}, default="empirical"
        阈值估计方法，同 :func:`detect_extreme_events_hist`。
        Tail model for each scale's threshold.
    gpd_base_quantile : float, default=0.9
        GPD 拟合基准分位数，仅在 ``tail_model='gpd'`` 时生效。
    return_details : bool, default=False
        是否返回详细诊断信息。

    Returns
    -------
    compound_mask : np.ndarray (bool)
        表示复合极端日的布尔数组。
        Boolean mask of compound extreme days.
    scale_thresholds : dict
        键为窗口长度，值为对应阈值。

    details : dict, optional
        当 ``return_details`` 为 ``True`` 时返回，包含各尺度掩码、阈值诊断、
        事件统计及加权得分（若适用）。

    Raises
    ------
    ValueError
        当 ``scales`` 或 ``severity_levels`` 无法对齐，或 aggregator/权重无效时抛出。
    """

    data = np.asarray(data, dtype=float)
    if np.any(np.isnan(data)):
        raise ValueError("输入数据包含 NaN，请先清洗 / Input data contains NaN values")

    n_days = data.size
    if n_days < DAYS_PER_YEAR:
        raise ValueError("复合检测建议至少使用一年数据 / At least one year of data is recommended")

    windows = [int(w) for w in scales]
    if any(w <= 0 for w in windows):
        raise ValueError("所有窗口必须为正整数 / All windows must be positive integers")

    if severity_levels is None:
        severity_list = [DEFAULT_SEVERITY_CLIM] * len(windows)
    else:
        severity_list = list(severity_levels)
        if len(severity_list) == 1 and len(windows) > 1:
            severity_list = severity_list * len(windows)
        if len(severity_list) != len(windows):
            raise ValueError(
                "severity_levels 长度必须与 scales 相同，或仅提供一个值\n"
                "severity_levels must match number of scales or be a single value"
            )

    for severity_value in severity_list:
        if not 0 < severity_value < 1:
            raise ValueError("severity_levels 中的值必须位于 (0,1) 范围内")

    aggregator_normalized = aggregator.lower()
    if aggregator_normalized not in {'all', 'any', 'weighted'}:
        raise ValueError("aggregator 必须为 'all'、'any' 或 'weighted'")

    tail_model_normalized = tail_model.lower()
    if tail_model_normalized not in {'empirical', 'gpd'}:
        raise ValueError("tail_model 必须为 'empirical' 或 'gpd'")

    if not 0 < gpd_base_quantile < 1:
        raise ValueError("gpd_base_quantile 必须位于 (0,1)")

    if aggregator_normalized == 'weighted':
        if weights is None:
            weight_array = np.ones(len(windows), dtype=float)
        else:
            weight_array = np.asarray(list(weights), dtype=float)
            if weight_array.size != len(windows):
                raise ValueError("weights 长度必须与 scales 一致")
        if np.any(weight_array < 0):
            raise ValueError("weights 必须为非负数")
        if np.allclose(weight_array.sum(), 0):
            weight_array = np.ones(len(windows), dtype=float)
        weight_array = weight_array / weight_array.sum()
    else:
        weight_array = np.ones(len(windows), dtype=float) / len(windows)

    scale_thresholds: Dict[int, float] = {}
    scale_diagnostics: Dict[int, Dict[str, Union[str, float, int]]] = {}
    scale_masks: Dict[int, np.ndarray] = {}
    smoothed_series: Dict[int, np.ndarray] = {}

    for window, severity in zip(windows, severity_list):
        smoothed = moving_average(data, window=window)
        threshold, diagnostics = _estimate_tail_threshold(
            smoothed, # type: ignore
            severity=severity,
            model=tail_model_normalized,
            base_quantile=gpd_base_quantile,
        )

        mask = smoothed > threshold # type: ignore
        scale_thresholds[window] = float(threshold)
        scale_diagnostics[window] = diagnostics
        scale_masks[window] = mask
        smoothed_series[window] = smoothed # type: ignore

    masks = list(scale_masks.values())
    if aggregator_normalized == 'all':
        compound_mask = np.logical_and.reduce(masks)
        composite_score = None
    elif aggregator_normalized == 'any':
        compound_mask = np.logical_or.reduce(masks)
        composite_score = None
    else:
        composite_score = np.zeros(n_days, dtype=float)
        for weight, window in zip(weight_array, windows):
            smoothed = smoothed_series[window]
            threshold = scale_thresholds[window]
            threshold_denominator = abs(threshold) + 1e-9
            excess = np.clip(smoothed - threshold, a_min=0.0, a_max=None)
            composite_score += weight * (excess / threshold_denominator)
        compound_mask = composite_score > 0

    if not return_details:
        return compound_mask, scale_thresholds

    events = identify_events_from_mask(compound_mask)
    enhanced_events = calculate_event_statistics(data, events)

    details = {
        'aggregator': aggregator_normalized,
        'weights': weight_array,
        'scale_thresholds': scale_thresholds,
        'scale_severity_levels': dict(zip(windows, severity_list)),
        'scale_threshold_diagnostics': scale_diagnostics,
        'scale_occurrence_rates': {
            window: float(mask.mean()) for window, mask in scale_masks.items()
        },
        'scale_masks': scale_masks,
        'n_extreme_days': int(compound_mask.sum()),
        'occurrence_rate': float(compound_mask.mean()),
        'n_events': len(enhanced_events),
        'events': enhanced_events,
        'tail_model': tail_model_normalized,
        'gpd_base_quantile': gpd_base_quantile,
        'method': 'compound_multiscale',
    }

    if composite_score is not None:
        details['composite_score'] = composite_score

    return compound_mask, scale_thresholds, details


def optimal_path_threshold(
    data: np.ndarray,
    target_occurrence_rate: float = DEFAULT_SEVERITY_CLIM,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> np.ndarray:
    """
    使用 OPT 方法计算最优阈值
    Calculate optimal thresholds for each calendar day using OPT method.

    算法流程 (Algorithm Flow):
    ------------------------
    1. 初始化: 为每个日历日设置初始阈值（第90百分位数）
       Initialize: Set initial thresholds for each calendar day (90th percentile)
    2. 迭代优化:
       Iterative optimization:
       a. 测试当前阈值下的事件发生率
          Test occurrence rate under current thresholds
       b. 如果发生率低于目标，降低阈值
          If rate is below target, decrease thresholds
       c. 如果发生率高于目标，提高阈值
          If rate is above target, increase thresholds
       d. 重复直到收敛或达到最大迭代次数
          Repeat until convergence or max iterations

    收敛标准 (Convergence Criteria):
    -------------------------------
    当实际发生率与目标发生率的差异 < 0.1% 时收敛
    Converges when |actual_rate - target_rate| < 0.1%

    Parameters
    ----------
    data : array-like
        时间序列数据
        Time series data

    target_occurrence_rate : float, default=0.05
        目标事件发生率
        Target event occurrence rate

    min_duration : int, default=3
        最小事件持续时间
        Minimum event duration

    max_iterations : int, default=50
        最大优化迭代次数
        Maximum optimization iterations
        防止无限循环
        Prevents infinite loops

    Returns
    -------
    thresholds : np.ndarray
        每个日历日的最优阈值（366 个值）
        Optimal threshold for each calendar day (366 values)

    Raises
    ------
    ValueError
        如果无法收敛到目标发生率
        If unable to converge to target occurrence rate

    Examples
    --------
    >>> data = np.random.gamma(2, 2, 365 * 40)
    >>> thresholds = optimal_path_threshold(data, target_occurrence_rate=0.05)
    >>> print(f"计算得到 {len(thresholds)} 个日阈值")
    >>> print(f"Computed {len(thresholds)} daily thresholds")

    Notes
    -----
    优化特性 (Optimization Characteristics):
    - 使用全局调整策略而非逐日优化
      Uses global adjustment rather than day-by-day optimization
    - 保持各日阈值的相对关系
      Maintains relative relationships between daily thresholds
    - 收敛速度快，通常 10-20 次迭代
      Fast convergence, typically 10-20 iterations

    See Also
    --------
    identify_climatological_extremes : 应用阈值识别事件
    detect_extreme_events_clim : 使用此方法的主函数
    """
    # ========================================================================
    # 输入验证 (Input Validation)
    # ========================================================================

    data = np.asarray(data, dtype=float)
    n_days = len(data)

    if n_days < DAYS_PER_YEAR:
        raise ValueError(
            f"数据长度 ({n_days}) 必须至少为 1 年 ({DAYS_PER_YEAR} 天)\n"
            f"Data length ({n_days}) must be at least 1 year ({DAYS_PER_YEAR} days)"
        )

    if not 0 < target_occurrence_rate < 1:
        raise ValueError(
            f"目标发生率必须在 (0, 1) 范围内，当前值: {target_occurrence_rate}\n"
            f"Target occurrence rate must be in (0, 1), got: {target_occurrence_rate}"
        )

    # ========================================================================
    # 计算每日数据分布 (Calculate Daily Data Distributions)
    # ========================================================================

    # 为每个日历日收集数据
    # Collect data for each calendar day across all years
    day_distributions = {}
    for day in range(DAYS_IN_LEAP_YEAR):
        # 创建日历日掩码（处理闰年）
        # Create calendar day mask (handle leap years)
        day_mask = (np.arange(n_days) % DAYS_PER_YEAR) == (day % DAYS_PER_YEAR)
        day_data = data[day_mask]

        if len(day_data) > 0:
            day_distributions[day] = day_data

    # ========================================================================
    # 初始化阈值 (Initialize Thresholds)
    # ========================================================================

    # 使用第 90 百分位数作为初始阈值
    # Use 90th percentile as initial threshold
    # 这提供了一个合理的起点，接近常见的极端定义
    # This provides a reasonable starting point near common extreme definitions
    initial_quantile = INITIAL_QUANTILE

    thresholds = np.array([
        np.quantile(day_distributions.get(d, [0]), initial_quantile)
        for d in range(DAYS_IN_LEAP_YEAR)
    ])

    # 确保阈值非负 (Ensure non-negative thresholds)
    thresholds = np.maximum(thresholds, 0)

    # ========================================================================
    # 迭代优化 (Iterative Optimization)
    # ========================================================================

    for iteration in range(max_iterations):
        # 使用当前阈值测试 (Test with current thresholds)
        extreme_mask = identify_climatological_extremes(
            data, thresholds, min_duration=min_duration
        )

        # 计算当前发生率 (Calculate current occurrence rate)
        current_rate = np.sum(extreme_mask) / n_days

        # 检查收敛 (Check convergence)
        rate_diff = abs(current_rate - target_occurrence_rate)
        if rate_diff < CONVERGENCE_TOLERANCE:
            # 成功收敛
            # Successfully converged
            break

        # 调整阈值 (Adjust thresholds)
        if current_rate < target_occurrence_rate:
            # 当前发生率太低，需要降低阈值以增加事件
            # Current rate too low, decrease thresholds to increase events
            adjustment = THRESHOLD_DECREASE_FACTOR
        else:
            # 当前发生率太高，需要提高阈值以减少事件
            # Current rate too high, increase thresholds to decrease events
            adjustment = THRESHOLD_INCREASE_FACTOR

        # 应用调整（保持相对比例）
        # Apply adjustment (maintain relative proportions)
        thresholds *= adjustment

    else:
        # 达到最大迭代次数但未收敛
        # Reached max iterations without convergence
        print(
            f"警告: OPT 方法在 {max_iterations} 次迭代后未完全收敛\n"
            f"Warning: OPT method did not fully converge after {max_iterations} iterations\n"
            f"目标发生率: {target_occurrence_rate:.4f}\n"
            f"Target rate: {target_occurrence_rate:.4f}\n"
            f"实际发生率: {current_rate:.4f}\n" # type: ignore
            f"Actual rate: {current_rate:.4f}\n" # type: ignore
            f"差异: {rate_diff:.4f}\n" # type: ignore
            f"Difference: {rate_diff:.4f}" # type: ignore
        )

    return thresholds


# ============================================================================
# 辅助函数 (Helper Functions)
# ============================================================================

def identify_climatological_extremes(
    data: np.ndarray,
    thresholds: np.ndarray,
    min_duration: int = DEFAULT_MIN_DURATION
) -> np.ndarray:
    """
    基于日阈值和最小持续时间识别极端事件
    Identify extreme events based on daily thresholds and minimum duration.

    算法步骤 (Algorithm Steps):
    -------------------------
    1. 为每一天分配对应日历日的阈值
       Assign threshold for each day based on calendar day
    2. 标记所有超过阈值的日子
       Mark all days exceeding threshold
    3. 识别连续超过阈值的时段
       Identify continuous periods exceeding threshold
    4. 过滤出持续时间 >= min_duration 的事件
       Filter events with duration >= min_duration

    参数说明 (Parameters):
    --------------------
    data : array-like
        时间序列数据
        Time series data

    thresholds : array-like
        每个日历日的阈值（366 个值）
        Threshold for each calendar day (366 values)

    min_duration : int, default=3
        事件最小持续天数
        Minimum consecutive days above threshold
        用于过滤短暂波动
        Used to filter brief fluctuations

    返回值 (Returns):
    ---------------
    extreme_mask : np.ndarray (bool)
        布尔数组，True 表示极端日
        Boolean array, True indicates extreme day

    示例 (Examples):
    --------------
    >>> data = np.random.randn(365 * 10)
    >>> thresholds = np.random.randn(366) + 1.0
    >>> extreme_mask = identify_climatological_extremes(data, thresholds, min_duration=3)
    >>> print(f"检测到 {np.sum(extreme_mask)} 个极端日")

    算法复杂度 (Complexity):
    -----------------------
    - 时间复杂度: O(n) 单次扫描
      Time complexity: O(n) single pass
    - 空间复杂度: O(n)
      Space complexity: O(n)
    """
    # 数据验证和转换 (Data validation and conversion)
    data = np.asarray(data, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n_days = len(data)

    # 为每一天获取对应的阈值 (Get threshold for each day)
    # 使用模运算处理跨年数据 (Use modulo to handle multi-year data)
    daily_thresholds = thresholds[np.arange(n_days) % DAYS_PER_YEAR]

    # 标记超过阈值的日子 (Mark days exceeding threshold)
    above_threshold = data > daily_thresholds

    # 初始化极端事件掩码 (Initialize extreme event mask)
    extreme_mask = np.zeros(n_days, dtype=bool)

    # ========================================================================
    # 应用最小持续时间标准 (Apply Minimum Duration Criterion)
    # ========================================================================

    # 使用双指针算法识别连续时段
    # Use two-pointer algorithm to identify continuous periods
    i = 0
    while i < n_days:
        if above_threshold[i]:
            # 找到超过阈值的起点 (Found start of period above threshold)
            start = i

            # 向前扫描找到连续时段的终点
            # Scan forward to find end of continuous period
            while i < n_days and above_threshold[i]:
                i += 1
            end = i

            # 计算持续时间 (Calculate duration)
            duration = end - start

            # 如果持续时间足够长，标记为极端事件
            # If duration is sufficient, mark as extreme event
            if duration >= min_duration:
                extreme_mask[start:end] = True
        else:
            # 当前日未超过阈值，继续扫描
            # Current day not above threshold, continue scanning
            i += 1

    return extreme_mask


def identify_events_from_mask(mask: Union[np.ndarray, List]) -> List[Dict]:
    """
    从布尔掩码中提取事件时段
    Extract event periods from boolean mask.

    功能说明 (Functionality):
    -----------------------
    将连续的 True 值识别为一个事件，返回每个事件的:
    Identifies consecutive True values as one event, returns for each event:
    - 起始索引 (start index)
    - 结束索引 (end index)
    - 持续时间 (duration)

    参数 (Parameters):
    -----------------
    mask : array-like (bool)
        布尔数组，True 表示极端日
        Boolean array, True indicates extreme day

    返回值 (Returns):
    ---------------
    events : list of dict
        事件列表，每个事件包含:
        List of events, each containing:
        {
            'start': int,      # 起始索引 (0-based)
            'end': int,        # 结束索引 (inclusive)
            'duration': int,   # 持续天数
        }

    示例 (Examples):
    --------------
    >>> mask = np.array([False, True, True, True, False, True, True, False])
    >>> events = identify_events_from_mask(mask)
    >>> for event in events:
    ...     print(f"事件: 第 {event['start']}-{event['end']} 天, "
    ...           f"持续 {event['duration']} 天")
    Event: Days 1-3, duration 3 days
    Event: Days 5-6, duration 2 days

    应用场景 (Use Cases):
    -------------------
    - 统计极端事件的数量和特征
      Count and characterize extreme events
    - 分析事件的时间分布
      Analyze temporal distribution of events
    - 计算事件间隔
      Calculate inter-event periods
    """
    # 类型转换 (Type conversion)
    mask = np.asarray(mask, dtype=bool)

    # 初始化事件列表 (Initialize event list)
    events = []

    # 扫描掩码以识别事件 (Scan mask to identify events)
    i = 0
    n_days = len(mask)

    while i < n_days:
        if mask[i]:
            # 事件开始 (Event starts)
            start = i

            # 向前扫描直到事件结束 (Scan forward until event ends)
            while i < n_days and mask[i]:
                i += 1

            # 记录事件 (Record event)
            end = i - 1  # end 是最后一个 True 的索引 (inclusive)
            events.append({
                'start': int(start),
                'end': int(end),
                'duration': int(end - start + 1),
            })
        else:
            # 非极端日，继续扫描 (Not extreme day, continue)
            i += 1

    return events


def calculate_event_statistics(
    data: np.ndarray,
    events: List[Dict]
) -> List[Dict]:
    """
    计算事件的详细统计信息
    Calculate detailed statistics for identified events.

    统计指标 (Statistical Metrics):
    -----------------------------
    为每个事件计算以下统计量:
    Calculates the following statistics for each event:
    - 平均值 (mean): 事件期间的平均强度
    - 最大值 (max): 事件期间的峰值强度
    - 总和 (total): 事件期间的累积值
    - 标准差 (std): 事件期间的变异性

    参数 (Parameters):
    -----------------
    data : array-like
        时间序列数据
        Time series data

    events : list of dict
        事件列表（来自 identify_events_from_mask）
        Events list (from identify_events_from_mask)
        必须包含 'start' 和 'end' 键
        Must contain 'start' and 'end' keys

    返回值 (Returns):
    ---------------
    event_stats : list of dict
        增强的事件信息，包含原始字段加上统计量
        Enhanced event information with original fields plus statistics

    示例 (Examples):
    --------------
    >>> data = np.array([1, 5, 6, 7, 2, 8, 9, 3])
    >>> events = [{'start': 1, 'end': 3, 'duration': 3}]
    >>> stats = calculate_event_statistics(data, events)
    >>> print(f"平均强度: {stats[0]['mean']:.2f}")
    >>> print(f"峰值强度: {stats[0]['max']:.2f}")

    应用场景 (Use Cases):
    -------------------
    - 比较不同事件的强度
      Compare intensity across events
    - 识别最严重的事件
      Identify most severe events
    - 分析事件特征的统计分布
      Analyze statistical distribution of event characteristics
    """
    data = np.asarray(data, dtype=float)
    event_stats = []

    for event in events:
        # 提取事件期间的数据 (Extract data during event)
        start = event['start']
        end = event['end']
        event_data = data[start:end+1]  # end 是 inclusive

        # 计算统计量 (Calculate statistics)
        stats = {
            **event,  # 保留原始字段 (keep original fields)
            'mean': float(np.mean(event_data)),
            'max': float(np.max(event_data)),
            'total': float(np.sum(event_data)),
            'std': float(np.std(event_data)),
        }

        event_stats.append(stats)

    return event_stats


# ============================================================================
# 模块元信息 (Module Metadata)
# ============================================================================

__all__ = [
    'detect_extreme_events_hist',
    'detect_extreme_events_clim',
    'detect_compound_extreme_events',
    'optimal_path_threshold',
    'identify_climatological_extremes',
    'identify_events_from_mask',
    'calculate_event_statistics',
]
