"""
检测与归因 (Detection and Attribution, D&A) 模块
Detection and Attribution (D&A) Module

本模块实现了 Egli et al. (2025) 提出的检测与归因框架，用于从观测数据中
分离出人为强迫的气候变化信号。
This module implements the Detection and Attribution (D&A) framework proposed by
Egli et al. (2025) to separate anthropogenic forcing signals from observational data.

核心方法 (Core Methods):
-----------------------
1. 岭回归检测器 (Ridge Regression Detector)
   - 训练统计模型从单成员模拟中提取强迫响应
   - Train statistical model to extract forced response from single-member simulations

2. 信号应用 (Signal Application)
   - 将训练好的模型应用于观测数据
   - Apply trained model to observational data

3. 完整 D&A 工作流 (Complete D&A Workflow)
   - ETx7d 计算 + 岭回归训练 + 趋势分析
   - ETx7d calculation + Ridge training + Trend analysis

算法原理 (Algorithm Principle):
-----------------------------
Egli et al. (2025) 的 D&A 框架基于以下假设:
Egli et al. (2025) D&A framework is based on the following assumption:

    观测信号 = 强迫响应 + 内部变率
    Observed signal = Forced response + Internal variability

通过训练岭回归模型学习"强迫响应"的空间-时间特征，然后应用于观测数据，
从而估算出观测中的人为强迫信号，最后通过趋势分析进行检测与归因。
By training a ridge regression model to learn the spatial-temporal characteristics
of "forced response", then applying it to observational data to estimate the
anthropogenic forcing signal in observations, and finally performing detection
and attribution through trend analysis.

参考文献 (References):
---------------------
Egli et al. (2025). Detecting Anthropogenically Induced Changes in Extreme
and Seasonal Evapotranspiration Observations.

作者: Extreme-ET Team
日期: 2025
版本: 1.2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler

# 重用现有模块的函数 (Reuse functions from existing modules)
try:
    # 优先使用包内相对导入 (Prefer package-relative import)
    from .nonstationary_threshold import detect_trend_and_detrend
    from .extreme_detection import detect_extremes_etx7d
except ImportError:
    # 后备导入方案 (Fallback import)
    from src.nonstationary_threshold import detect_trend_and_detrend
    from src.extreme_detection import detect_extremes_etx7d


# ============================================================================
# 模块常量定义 (Module Constants)
# ============================================================================

# 岭回归默认参数 (Ridge Regression Default Parameters)
DEFAULT_ALPHAS = np.logspace(-3, 3, 100)  # 候选正则化参数 (Candidate regularization parameters)
DEFAULT_CV_FOLDS = 5                       # 交叉验证折数 (Cross-validation folds)

# 趋势分析默认参数 (Trend Analysis Default Parameters)
DEFAULT_TREND_RANGE = (1980, 2023)        # 默认趋势分析时段 (Default trend period)
DEFAULT_BLOCK_SIZE = 44                    # piControl 趋势块大小 (piControl trend block size)


# ============================================================================
# 核心函数 (Core Functions)
# ============================================================================


def fit_ridge_detector(
    model_members: np.ndarray,
    model_ensemble_mean: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    cv: int = DEFAULT_CV_FOLDS,
    **ridge_kwargs
) -> Tuple[Ridge, StandardScaler, StandardScaler]:
    """
    训练岭回归检测器以提取强迫响应
    Train ridge regression detector to extract forced response.

    算法原理 (Algorithm Principle):
    -----------------------------
    该函数实现了 Egli et al. (2025) 的核心统计学习方法：
    This function implements the core statistical learning method from Egli et al. (2025):

    目标 (Objective):
        学习一个映射关系: X (单成员) → y (集合平均)
        Learn a mapping: X (single member) → y (ensemble mean)

    训练数据 (Training Data):
        - X: 多个 CMIP6 模式的单成员时间序列（包含强迫响应 + 内部变率）
          X: Single-member time series from multiple CMIP6 models (forced + internal variability)
        - y: 同一模式的集合平均（主要是强迫响应，内部变率被平均消除）
          y: Ensemble mean from the same model (mainly forced response, internal variability averaged out)

    学习策略 (Learning Strategy):
        使用岭回归 (Ridge Regression) 而非普通最小二乘 (OLS)，因为:
        Use Ridge Regression instead of OLS because:
        1. 防止过拟合 (Prevent overfitting)
        2. 处理多重共线性 (Handle multicollinearity)
        3. 正则化参数通过交叉验证自动选择
           Regularization parameter automatically selected via cross-validation

    参数 (Parameters):
    -----------------
    model_members : np.ndarray
        CMIP6 模式的单成员数据
        Single-member data from CMIP6 models
        形状 (shape): (n_samples, n_features) 或 (n_samples,)
        例如 (Example): 多个模式的多个成员展平后的年ETx7d值
        Flattened annual ETx7d values from multiple members of multiple models

    model_ensemble_mean : np.ndarray
        对应的集合平均数据（强迫响应）
        Corresponding ensemble mean data (forced response)
        形状 (shape): (n_samples,) 或 (n_samples, 1)
        注意 (Note): 长度必须与 model_members 的第一维相同
        Length must match first dimension of model_members

    alphas : np.ndarray, optional
        岭回归正则化参数候选值
        Candidate regularization parameters for ridge regression
        默认 (default): np.logspace(-3, 3, 100)
        范围 (range): 建议 [0.001, 1000]

    cv : int, default=5
        交叉验证折数
        Number of cross-validation folds
        典型值 (typical): 3-10

    **ridge_kwargs : dict
        传递给 Ridge 的其他参数
        Additional parameters for Ridge
        例如 (e.g.): fit_intercept, normalize

    返回值 (Returns):
    ---------------
    detector_model : Ridge
        训练好的岭回归模型
        Trained ridge regression model
        可用于预测新数据的强迫响应
        Can be used to predict forced response for new data

    scaler_X : StandardScaler
        输入数据的标准化器
        Scaler for input data
        用于标准化新数据
        Used to standardize new data

    scaler_y : StandardScaler
        输出数据的标准化器
        Scaler for output data
        用于逆标准化预测结果
        Used to inverse-transform predictions

    引发异常 (Raises):
    -----------------
    ValueError
        如果 model_members 和 model_ensemble_mean 长度不匹配
        If model_members and model_ensemble_mean have incompatible lengths
        如果数据包含 NaN 或 Inf
        If data contains NaN or Inf

    示例 (Examples):
    --------------
    >>> # 假设我们有 3 个模式，每个 5 个成员，共 170 年
    >>> # Assume 3 models, 5 members each, 170 years
    >>> n_members = 15
    >>> n_years = 170
    >>> X_train = np.random.randn(n_members * n_years)  # 单成员数据
    >>> y_train = np.random.randn(n_members * n_years)  # 集合平均
    >>>
    >>> # 训练检测器 (Train detector)
    >>> model, scaler_X, scaler_y = fit_ridge_detector(X_train, y_train)
    >>> print(f"最优 alpha: {model.alpha}")
    >>> print(f"模型系数: {model.coef_}")

    注释 (Notes):
    -----------
    标准化的必要性 (Why Standardization):
    - 岭回归对尺度敏感，标准化确保所有特征平等对待
      Ridge regression is scale-sensitive, standardization ensures fair treatment
    - 正则化参数的解释更加直观
      Makes regularization parameter interpretation more intuitive

    See Also
    --------
    apply_ridge_detector : 应用训练好的检测器 (Apply trained detector)
    run_egli_attribution_workflow : 完整 D&A 工作流 (Complete D&A workflow)
    """
    # ========================================================================
    # 输入验证 (Input Validation)
    # ========================================================================

    # 转换为 numpy 数组 (Convert to numpy arrays)
    model_members = np.asarray(model_members, dtype=float)
    model_ensemble_mean = np.asarray(model_ensemble_mean, dtype=float)

    # 确保 model_members 是 2D (Ensure model_members is 2D)
    if model_members.ndim == 1:
        model_members = model_members.reshape(-1, 1)
    elif model_members.ndim > 2:
        raise ValueError(
            f"model_members 必须是 1D 或 2D 数组，当前维度: {model_members.ndim}\n"
            f"model_members must be 1D or 2D array, current ndim: {model_members.ndim}"
        )

    # 确保 model_ensemble_mean 是 1D (Ensure model_ensemble_mean is 1D)
    if model_ensemble_mean.ndim == 2 and model_ensemble_mean.shape[1] == 1:
        model_ensemble_mean = model_ensemble_mean.ravel()
    elif model_ensemble_mean.ndim > 1:
        raise ValueError(
            f"model_ensemble_mean 必须是 1D 数组，当前维度: {model_ensemble_mean.ndim}\n"
            f"model_ensemble_mean must be 1D array, current ndim: {model_ensemble_mean.ndim}"
        )

    # 检查长度匹配 (Check length compatibility)
    n_samples = model_members.shape[0]
    if len(model_ensemble_mean) != n_samples:
        raise ValueError(
            f"样本数量不匹配: model_members ({n_samples}) vs model_ensemble_mean ({len(model_ensemble_mean)})\n"
            f"Sample size mismatch: model_members ({n_samples}) vs model_ensemble_mean ({len(model_ensemble_mean)})"
        )

    # 检查 NaN 和 Inf (Check for NaN and Inf)
    if np.any(~np.isfinite(model_members)) or np.any(~np.isfinite(model_ensemble_mean)):
        raise ValueError(
            "输入数据包含 NaN 或 Inf 值，请先清洗数据\n"
            "Input data contains NaN or Inf values, please clean data first"
        )

    # 设置默认 alphas (Set default alphas)
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # ========================================================================
    # 数据标准化 (Data Standardization)
    # ========================================================================

    # 初始化标准化器 (Initialize scalers)
    # 岭回归对尺度敏感，必须标准化
    # Ridge regression is scale-sensitive, standardization is required
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 标准化输入和输出 (Standardize inputs and outputs)
    X_scaled = scaler_X.fit_transform(model_members)
    y_scaled = scaler_y.fit_transform(model_ensemble_mean.reshape(-1, 1)).ravel()

    # ========================================================================
    # 岭回归训练 (Ridge Regression Training)
    # ========================================================================

    # 步骤 1: 使用 RidgeCV 寻找最优 alpha
    # Step 1: Use RidgeCV to find optimal alpha
    ridge_cv = RidgeCV(
        alphas=alphas,
        cv=cv,
        **ridge_kwargs
    )

    # 拟合 RidgeCV (Fit RidgeCV)
    ridge_cv.fit(X_scaled, y_scaled)

    # 获取最优 alpha (Get optimal alpha)
    best_alpha = ridge_cv.alpha_

    # 步骤 2: 使用最优 alpha 训练最终模型
    # Step 2: Train final model with optimal alpha
    detector_model = Ridge(alpha=best_alpha, **ridge_kwargs)
    detector_model.fit(X_scaled, y_scaled)

    # ========================================================================
    # 返回结果 (Return Results)
    # ========================================================================

    return detector_model, scaler_X, scaler_y


def apply_ridge_detector(
    detector_model: Ridge,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    new_data: np.ndarray
) -> np.ndarray:
    """
    应用训练好的岭回归检测器估算强迫响应
    Apply trained ridge detector to estimate forced response.

    算法原理 (Algorithm Principle):
    -----------------------------
    使用训练好的模型从新数据（如观测数据）中提取强迫响应信号:
    Use trained model to extract forced response signal from new data (e.g., observations):

    工作流程 (Workflow):
        1. 使用 scaler_X 标准化新数据
           Standardize new data using scaler_X
        2. 应用岭回归模型预测
           Apply ridge regression model for prediction
        3. 使用 scaler_y 逆标准化预测结果
           Inverse-transform predictions using scaler_y

    物理意义 (Physical Meaning):
        预测结果代表新数据中可归因于人为强迫的部分
        Predictions represent the component attributable to anthropogenic forcing

    参数 (Parameters):
    -----------------
    detector_model : Ridge
        训练好的岭回归模型（来自 fit_ridge_detector）
        Trained ridge regression model (from fit_ridge_detector)

    scaler_X : StandardScaler
        输入数据标准化器（来自 fit_ridge_detector）
        Input data scaler (from fit_ridge_detector)

    scaler_y : StandardScaler
        输出数据标准化器（来自 fit_ridge_detector）
        Output data scaler (from fit_ridge_detector)

    new_data : np.ndarray
        新数据（如观测数据）
        New data (e.g., observational data)
        形状 (shape): (n_samples,) 或 (n_samples, n_features)
        必须与训练数据特征数一致
        Must have same number of features as training data

    返回值 (Returns):
    ---------------
    forced_response : np.ndarray
        估算的强迫响应（1D 数组）
        Estimated forced response (1D array)
        形状 (shape): (n_samples,)
        单位与输入数据相同 (same units as input data)

    引发异常 (Raises):
    -----------------
    ValueError
        如果特征数量不匹配
        If number of features mismatch
        如果数据包含 NaN 或 Inf
        If data contains NaN or Inf

    示例 (Examples):
    --------------
    >>> # 假设已经训练好模型 (Assume model already trained)
    >>> # model, scaler_X, scaler_y = fit_ridge_detector(...)
    >>>
    >>> # 新的观测数据 (New observational data)
    >>> obs_data = np.random.randn(100)  # 100 年的观测
    >>>
    >>> # 估算强迫响应 (Estimate forced response)
    >>> forced = apply_ridge_detector(model, scaler_X, scaler_y, obs_data)
    >>> print(f"观测强迫响应: {forced}")
    >>> print(f"平均强迫趋势: {np.polyfit(range(len(forced)), forced, 1)[0]:.4f} /年")

    注释 (Notes):
    -----------
    逆标准化的重要性 (Importance of Inverse Transformation):
    - 确保预测结果与原始数据在同一物理尺度上
      Ensures predictions are on the same physical scale as original data
    - 便于后续趋势分析和可视化
      Facilitates subsequent trend analysis and visualization

    See Also
    --------
    fit_ridge_detector : 训练岭回归检测器 (Train ridge detector)
    run_egli_attribution_workflow : 完整 D&A 工作流 (Complete D&A workflow)
    """
    # ========================================================================
    # 输入验证 (Input Validation)
    # ========================================================================

    # 转换为 numpy 数组 (Convert to numpy array)
    new_data = np.asarray(new_data, dtype=float)

    # 确保 new_data 是 2D (Ensure new_data is 2D)
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, 1)
    elif new_data.ndim > 2:
        raise ValueError(
            f"new_data 必须是 1D 或 2D 数组，当前维度: {new_data.ndim}\n"
            f"new_data must be 1D or 2D array, current ndim: {new_data.ndim}"
        )

    # 检查 NaN 和 Inf (Check for NaN and Inf)
    if np.any(~np.isfinite(new_data)):
        raise ValueError(
            "新数据包含 NaN 或 Inf 值，请先清洗数据\n"
            "New data contains NaN or Inf values, please clean data first"
        )

    # 检查特征数量是否匹配 (Check feature count compatibility)
    expected_features = scaler_X.n_features_in_
    actual_features = new_data.shape[1]
    if actual_features != expected_features:
        raise ValueError(
            f"特征数量不匹配: 期望 {expected_features}，实际 {actual_features}\n"
            f"Feature count mismatch: expected {expected_features}, got {actual_features}"
        )

    # ========================================================================
    # 应用检测器 (Apply Detector)
    # ========================================================================

    # 步骤 1: 标准化新数据 (Step 1: Standardize new data)
    X_scaled = scaler_X.transform(new_data)

    # 步骤 2: 应用模型预测 (Step 2: Apply model for prediction)
    y_pred_scaled = detector_model.predict(X_scaled)

    # 步骤 3: 逆标准化预测结果 (Step 3: Inverse-transform predictions)
    # 确保 y_pred_scaled 是 2D 以便逆标准化
    # Ensure y_pred_scaled is 2D for inverse transformation
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    forced_response = scaler_y.inverse_transform(y_pred_scaled).ravel()

    # ========================================================================
    # 返回结果 (Return Results)
    # ========================================================================

    return forced_response


def run_egli_attribution_workflow(
    historical_data_members: Union[np.ndarray, List[np.ndarray]],
    picontrol_data_members: Union[np.ndarray, List[np.ndarray]],
    observation_data: np.ndarray,
    dates_hist: Union[np.ndarray, pd.DatetimeIndex],
    dates_pi: Union[np.ndarray, pd.DatetimeIndex],
    dates_obs: Union[np.ndarray, pd.DatetimeIndex],
    trend_range: Tuple[int, int] = DEFAULT_TREND_RANGE,
    detection_window: int = 7,
    block_size: int = DEFAULT_BLOCK_SIZE,
    **kwargs
) -> Dict[str, Union[float, np.ndarray, pd.Series, List[pd.Series], Ridge, StandardScaler]]:
    """
    完整的 Egli et al. (2025) 检测与归因工作流
    Complete Egli et al. (2025) Detection and Attribution (D&A) workflow.

    算法原理 (Algorithm Principle):
    -----------------------------
    该函数整合了完整的 D&A 分析流程，复现 Egli et al. (2025) Figure 3d/e/f:
    This function integrates the complete D&A analysis pipeline, reproducing
    Egli et al. (2025) Figure 3d/e/f:

    完整流程 (Complete Workflow):
    1. 计算 ETx7d 指标
       Calculate ETx7d index
       - historical 成员 → historical ETx7d
       - piControl 成员 → piControl ETx7d
       - observation → observation ETx7d

    2. 训练岭回归检测器
       Train ridge regression detector
       - 输入: historical 单成员 ETx7d
       - 目标: historical 集合平均 ETx7d

    3. 应用检测器估算强迫响应
       Apply detector to estimate forced response
       - observation ETx7d → forced response
       - historical ETx7d → forced response
       - piControl ETx7d → forced response

    4. 趋势分析
       Trend analysis
       - 计算 observation 强迫响应的趋势
       - 计算 historical 强迫响应的趋势分布
       - 计算 piControl 强迫响应的趋势分布（非重叠块）

    5. 检测与归因
       Detection and Attribution
       - 比较 observation 趋势与 piControl 分布 → 检测
       - 比较 observation 趋势与 historical 分布 → 归因

    参数 (Parameters):
    -----------------
    historical_data_members : np.ndarray or list of np.ndarray
        CMIP6 历史模拟的单成员日 ET 数据
        Single-member daily ET data from CMIP6 historical simulations
        形状 (shape): (n_members, n_days) 或 list of (n_days,)
        单位 (units): mm/day

    picontrol_data_members : np.ndarray or list of np.ndarray
        CMIP6 piControl 模拟的单成员日 ET 数据
        Single-member daily ET data from CMIP6 piControl simulations
        形状 (shape): (n_members, n_days) 或 list of (n_days,)
        单位 (units): mm/day

    observation_data : np.ndarray
        观测日 ET 数据
        Observational daily ET data
        形状 (shape): (n_days,)
        单位 (units): mm/day

    dates_hist : array-like or pd.DatetimeIndex
        historical 数据对应的日期
        Dates for historical data

    dates_pi : array-like or pd.DatetimeIndex
        piControl 数据对应的日期
        Dates for piControl data

    dates_obs : array-like or pd.DatetimeIndex
        observation 数据对应的日期
        Dates for observation data

    trend_range : tuple of (int, int), default=(1980, 2023)
        趋势分析的年份范围
        Year range for trend analysis
        例如 (Example): (1980, 2023) 分析 1980-2023 年趋势

    detection_window : int, default=7
        ETx7d 计算的滑动窗口大小（天）
        Rolling window size for ETx7d calculation (days)

    block_size : int, default=44
        piControl 趋势估计的块大小（年）
        Block size for piControl trend estimation (years)
        用于生成多个非重叠块的趋势分布
        Used to generate trend distribution from non-overlapping blocks

    **kwargs : dict
        传递给 fit_ridge_detector 的其他参数
        Additional parameters for fit_ridge_detector

    返回值 (Returns):
    ---------------
    results : dict
        包含以下键值对的字典:
        Dictionary containing the following key-value pairs:

        - 'obs_trend' : float
            观测强迫响应的趋势 (Trend of observed forced response)
            单位: mm/year (units: mm/year)

        - 'hist_trends' : np.ndarray
            historical 强迫响应的趋势分布
            Trend distribution of historical forced response
            形状: (n_members,)

        - 'picontrol_trends' : np.ndarray
            piControl 强迫响应的趋势分布（非重叠块）
            Trend distribution of piControl forced response (non-overlapping blocks)
            形状: (n_blocks,)

        - 'forced_response_obs' : pd.Series
            观测强迫响应时间序列
            Time series of observed forced response
            索引: 年份 (index: year)

        - 'forced_response_hist' : list of pd.Series
            historical 强迫响应时间序列列表
            List of historical forced response time series

        - 'forced_response_pi' : list of pd.Series
            piControl 强迫响应时间序列列表
            List of piControl forced response time series

        - 'detector_model' : Ridge
            训练好的岭回归模型
            Trained ridge regression model

    引发异常 (Raises):
    -----------------
    ValueError
        如果数据格式不正确
        If data format is incorrect
        如果 trend_range 超出数据范围
        If trend_range exceeds data range

    示例 (Examples):
    --------------
    >>> # 准备合成数据 (Prepare synthetic data)
    >>> # 详见 examples/example_egli_2025.py
    >>> # See examples/example_egli_2025.py for details
    >>>
    >>> # 运行 D&A 工作流 (Run D&A workflow)
    >>> results = run_egli_attribution_workflow(
    ...     historical_data_members=hist_members,
    ...     picontrol_data_members=pi_members,
    ...     observation_data=obs_data,
    ...     dates_hist=dates_hist,
    ...     dates_pi=dates_pi,
    ...     dates_obs=dates_obs,
    ...     trend_range=(1980, 2020)
    ... )
    >>>
    >>> # 分析结果 (Analyze results)
    >>> print(f"观测趋势: {results['obs_trend']:.4f} mm/year")
    >>> print(f"历史趋势范围: {results['hist_trends'].min():.4f} - {results['hist_trends'].max():.4f}")
    >>> print(f"piControl 趋势范围: {results['picontrol_trends'].min():.4f} - {results['picontrol_trends'].max():.4f}")

    注释 (Notes):
    -----------
    检测与归因的判断标准 (Detection and Attribution Criteria):
    - 检测 (Detection): 观测趋势显著超出 piControl 趋势分布的范围
      Detection: Observed trend significantly exceeds piControl trend distribution
    - 归因 (Attribution): 观测趋势与 historical 趋势分布一致
      Attribution: Observed trend consistent with historical trend distribution

    See Also
    --------
    fit_ridge_detector : 训练岭回归检测器 (Train ridge detector)
    apply_ridge_detector : 应用岭回归检测器 (Apply ridge detector)
    detect_extremes_etx7d : 计算 ETx7d 指标 (Calculate ETx7d index)
    """
    # ========================================================================
    # 步骤 1: 计算 ETx7d (Step 1: Calculate ETx7d)
    # ========================================================================

    # 处理 historical 数据 (Process historical data)
    if isinstance(historical_data_members, list):
        hist_etx7d_list = [
            detect_extremes_etx7d(member, dates_hist, window_days=detection_window)
            for member in historical_data_members
        ]
    else:
        # 假设是 (n_members, n_days) 的 2D 数组
        # Assume 2D array of shape (n_members, n_days)
        hist_etx7d_list = [
            detect_extremes_etx7d(historical_data_members[i], dates_hist, window_days=detection_window)
            for i in range(historical_data_members.shape[0])
        ]

    # 处理 piControl 数据 (Process piControl data)
    if isinstance(picontrol_data_members, list):
        pi_etx7d_list = [
            detect_extremes_etx7d(member, dates_pi, window_days=detection_window)
            for member in picontrol_data_members
        ]
    else:
        # 假设是 (n_members, n_days) 的 2D 数组
        # Assume 2D array of shape (n_members, n_days)
        pi_etx7d_list = [
            detect_extremes_etx7d(picontrol_data_members[i], dates_pi, window_days=detection_window)
            for i in range(picontrol_data_members.shape[0])
        ]

    # 处理 observation 数据 (Process observation data)
    obs_etx7d = detect_extremes_etx7d(observation_data, dates_obs, window_days=detection_window)

    # ========================================================================
    # 步骤 2: 准备训练数据 (Step 2: Prepare Training Data)
    # ========================================================================

    # 计算 historical 集合平均 (Calculate historical ensemble mean)
    # 对齐所有成员的年份 (Align years for all members)
    common_years = hist_etx7d_list[0].index
    for etx in hist_etx7d_list[1:]:
        common_years = common_years.intersection(etx.index)

    # 重新索引并计算集合平均 (Reindex and calculate ensemble mean)
    hist_etx7d_aligned = [etx.reindex(common_years) for etx in hist_etx7d_list]
    hist_ensemble_mean = pd.concat(hist_etx7d_aligned, axis=1).mean(axis=1)

    # 准备 X (单成员) 和 y (集合平均) (Prepare X (single members) and y (ensemble mean))
    X_train = np.concatenate([etx.values for etx in hist_etx7d_aligned])
    y_train = np.tile(hist_ensemble_mean.values, len(hist_etx7d_aligned))

    # ========================================================================
    # 步骤 3: 训练检测器 (Step 3: Train Detector)
    # ========================================================================

    detector_model, scaler_X, scaler_y = fit_ridge_detector(
        X_train, y_train, **kwargs
    )

    # ========================================================================
    # 步骤 4: 应用检测器估算强迫响应 (Step 4: Apply Detector to Estimate Forced Response)
    # ========================================================================

    # 观测数据 (Observation data)
    obs_forced = apply_ridge_detector(
        detector_model, scaler_X, scaler_y, obs_etx7d.values
    )
    obs_forced_series = pd.Series(obs_forced, index=obs_etx7d.index)

    # historical 数据 (Historical data)
    hist_forced_list = [
        pd.Series(
            apply_ridge_detector(detector_model, scaler_X, scaler_y, etx.values),
            index=etx.index
        )
        for etx in hist_etx7d_list
    ]

    # piControl 数据 (piControl data)
    pi_forced_list = [
        pd.Series(
            apply_ridge_detector(detector_model, scaler_X, scaler_y, etx.values),
            index=etx.index
        )
        for etx in pi_etx7d_list
    ]

    # ========================================================================
    # 步骤 5: 趋势分析 (Step 5: Trend Analysis)
    # ========================================================================

    # 观测趋势 (Observation trend)
    obs_trend_data = obs_forced_series.loc[trend_range[0]:trend_range[1]]
    obs_trend, _ = detect_trend_and_detrend(
        obs_trend_data.values,
        return_detrended=False
    )

    # historical 趋势分布 (Historical trend distribution)
    hist_trends = []
    for forced in hist_forced_list:
        trend_data = forced.loc[trend_range[0]:trend_range[1]]
        trend, _ = detect_trend_and_detrend(
            trend_data.values,
            return_detrended=False
        )
        hist_trends.append(trend)
    hist_trends = np.array(hist_trends)

    # piControl 趋势分布（非重叠块）(piControl trend distribution with non-overlapping blocks)
    pi_trends = []
    for forced in pi_forced_list:
        # 将长时间序列分割为多个非重叠块
        # Split long time series into multiple non-overlapping blocks
        n_years = len(forced)
        n_blocks = n_years // block_size
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            block_data = forced.iloc[start_idx:end_idx].values
            if len(block_data) >= block_size:
                trend, _ = detect_trend_and_detrend(
                    block_data,
                    return_detrended=False
                )
                pi_trends.append(trend)
    pi_trends = np.array(pi_trends)

    # ========================================================================
    # 返回结果 (Return Results)
    # ========================================================================

    results = {
        'obs_trend': float(obs_trend),
        'hist_trends': hist_trends,
        'picontrol_trends': pi_trends,
        'forced_response_obs': obs_forced_series,
        'forced_response_hist': hist_forced_list,
        'forced_response_pi': pi_forced_list,
        'detector_model': detector_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
    }

    return results


# ============================================================================
# 模块元信息 (Module Metadata)
# ============================================================================

__all__ = [
    'fit_ridge_detector',
    'apply_ridge_detector',
    'run_egli_attribution_workflow',
]
