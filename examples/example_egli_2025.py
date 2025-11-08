"""
Egli et al. (2025) 检测与归因 (D&A) 示例
Example of Detection and Attribution (D&A) from Egli et al. (2025)

本脚本演示如何使用 Extreme-ET 工具包中的 D&A 模块来:
This script demonstrates how to use the D&A module in Extreme-ET toolkit to:

1. 模拟 CMIP6 模式和观测数据
   Simulate CMIP6 model and observational data
2. 计算 ETx7d 极端指标
   Calculate ETx7d extreme index
3. 训练岭回归检测器提取强迫响应
   Train ridge regression detector to extract forced response
4. 进行趋势分析和检测归因
   Perform trend analysis and detection-attribution
5. 可视化结果（复现 Figure 3d）
   Visualize results (reproduce Figure 3d)

参考文献 (Reference):
Egli et al. (2025). Detecting Anthropogenically Induced Changes in Extreme
and Seasonal Evapotranspiration Observations.

作者: Extreme-ET Team
日期: 2025
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加 src 到路径 (Add src to path)
# 这允许我们导入本地开发的模块
# This allows us to import locally developed modules
sys.path.insert(0, '../src')
sys.path.insert(0, './src')

# 导入 Extreme-ET 模块 (Import Extreme-ET modules)
from src.detection_attribution import run_egli_attribution_workflow

# ============================================================================
# 数据模拟函数 (Data Simulation Functions)
# ============================================================================


def simulate_picontrol_data(
    n_members: int = 5,
    n_years: int = 500,
    base_mean: float = 3.0,
    noise_std: float = 0.5,
    seed: int = 42
) -> tuple:
    """
    模拟 piControl 数据（纯自然变率，无趋势）
    Simulate piControl data (pure natural variability, no trend).

    算法原理 (Algorithm):
    -------------------
    piControl 模拟代表工业化前的气候条件，没有人为强迫:
    piControl simulations represent pre-industrial climate with no anthropogenic forcing:
    - 无长期趋势 (No long-term trend)
    - 仅包含内部气候变率 (Only internal climate variability)
    - 用于评估自然变率的背景噪声水平
      Used to assess background noise level of natural variability

    参数 (Parameters):
    -----------------
    n_members : int
        成员数量 (Number of ensemble members)
    n_years : int
        模拟年数 (Number of simulated years)
    base_mean : float
        基准平均日 ET (mm/day) (Base mean daily ET in mm/day)
    noise_std : float
        噪声标准差 (Noise standard deviation)
    seed : int
        随机种子 (Random seed for reproducibility)

    返回值 (Returns):
    ---------------
    data : list of np.ndarray
        每个成员的日 ET 数据 (Daily ET data for each member)
        长度 (length): n_members
        每个数组形状 (each array shape): (n_years * 365,)
    dates : pd.DatetimeIndex
        对应的日期序列 (Corresponding date sequence)
    """
    np.random.seed(seed)

    # 生成日期序列 (Generate date sequence)
    # 从 1500 年开始（工业化前）
    # Starting from year 1500 (pre-industrial)
    start_date = datetime(1500, 1, 1)
    dates = pd.date_range(start_date, periods=n_years * 365, freq='D')

    # 为每个成员生成数据 (Generate data for each member)
    data = []
    for i in range(n_members):
        # 纯随机噪声，无趋势 (Pure random noise, no trend)
        # 使用 Gamma 分布模拟日 ET 的正偏分布
        # Use Gamma distribution to simulate positive-skewed daily ET
        daily_et = np.random.gamma(
            shape=base_mean**2 / noise_std**2,
            scale=noise_std**2 / base_mean,
            size=n_years * 365
        )
        # 添加日内变率 (Add daily variability)
        daily_et = np.maximum(0, daily_et + np.random.normal(0, noise_std * 0.5, size=n_years * 365))

        data.append(daily_et)

    return data, dates


def simulate_historical_data(
    n_members: int = 10,
    n_years: int = 165,
    base_mean: float = 3.0,
    noise_std: float = 0.5,
    trend_strength: float = 0.015,
    seed: int = 100
) -> tuple:
    """
    模拟 historical 数据（自然变率 + 人为强迫趋势）
    Simulate historical data (natural variability + anthropogenic forcing trend).

    算法原理 (Algorithm):
    -------------------
    Historical 模拟包含两部分:
    Historical simulations contain two components:
    1. 内部变率（与 piControl 类似）
       Internal variability (similar to piControl)
    2. 人为强迫导致的长期趋势
       Long-term trend due to anthropogenic forcing

    集合平均可以消除内部变率，保留强迫信号:
    Ensemble mean can cancel out internal variability and retain forced signal:
        集合平均 ≈ 强迫响应
        Ensemble mean ≈ Forced response

    参数 (Parameters):
    -----------------
    n_members : int
        成员数量 (Number of ensemble members)
    n_years : int
        模拟年数 (Number of simulated years)
        典型: 1850-2014 (165 years)
    base_mean : float
        基准平均日 ET (mm/day)
    noise_std : float
        噪声标准差
    trend_strength : float
        趋势强度 (mm/day per year)
        典型值: 0.01-0.02 mm/day/year
    seed : int
        随机种子

    返回值 (Returns):
    ---------------
    data : list of np.ndarray
        每个成员的日 ET 数据
    dates : pd.DatetimeIndex
        对应的日期序列
    """
    np.random.seed(seed)

    # 生成日期序列 (Generate date sequence)
    # 1850-2014 年（历史模拟典型时段）
    # 1850-2014 (typical historical simulation period)
    start_date = datetime(1850, 1, 1)
    dates = pd.date_range(start_date, periods=n_years * 365, freq='D')

    # 为每个成员生成数据 (Generate data for each member)
    data = []
    for i in range(n_members):
        # 基准值 + 线性趋势 + 随机噪声
        # Base value + linear trend + random noise
        time_years = np.arange(n_years * 365) / 365.0
        forced_trend = trend_strength * time_years  # 强迫信号 (Forced signal)

        # 基准 ET (Base ET)
        daily_et = np.random.gamma(
            shape=base_mean**2 / noise_std**2,
            scale=noise_std**2 / base_mean,
            size=n_years * 365
        )

        # 添加强迫趋势 (Add forced trend)
        daily_et += forced_trend

        # 添加内部变率噪声 (Add internal variability noise)
        daily_et += np.random.normal(0, noise_std * 0.5, size=n_years * 365)

        data.append(daily_et)

    return data, dates


def simulate_observation_data(
    n_years: int = 70,
    base_mean: float = 3.2,
    noise_std: float = 0.6,
    trend_strength: float = 0.025,
    seed: int = 200
) -> tuple:
    """
    模拟观测数据（强趋势 + 观测不确定性）
    Simulate observational data (strong trend + observational uncertainty).

    算法原理 (Algorithm):
    -------------------
    观测数据特点:
    Characteristics of observational data:
    1. 通常时段较短（如 1950-2020）
       Usually shorter period (e.g., 1950-2020)
    2. 包含更强的变暖趋势（相对于 historical）
       Contains stronger warming trend (relative to historical)
    3. 观测不确定性可能更大
       Observational uncertainty may be larger

    参数 (Parameters):
    -----------------
    n_years : int
        观测年数 (Number of observed years)
        典型: 50-70 years
    base_mean : float
        基准平均日 ET (mm/day)
        通常略高于模式（观测偏差）
        Usually slightly higher than models (observational bias)
    noise_std : float
        噪声标准差（观测不确定性）
        Noise std (observational uncertainty)
    trend_strength : float
        观测趋势强度 (mm/day per year)
        通常强于 historical 模拟
        Usually stronger than historical simulations
    seed : int
        随机种子

    返回值 (Returns):
    ---------------
    data : np.ndarray
        日 ET 数据 (Daily ET data)
    dates : pd.DatetimeIndex
        对应的日期序列
    """
    np.random.seed(seed)

    # 生成日期序列 (Generate date sequence)
    # 1950-2019 年（典型观测时段）
    # 1950-2019 (typical observational period)
    start_date = datetime(1950, 1, 1)
    dates = pd.date_range(start_date, periods=n_years * 365, freq='D')

    # 生成观测数据 (Generate observational data)
    time_years = np.arange(n_years * 365) / 365.0
    forced_trend = trend_strength * time_years  # 强观测趋势 (Strong observed trend)

    # 基准 ET (Base ET)
    daily_et = np.random.gamma(
        shape=base_mean**2 / noise_std**2,
        scale=noise_std**2 / base_mean,
        size=n_years * 365
    )

    # 添加趋势 (Add trend)
    daily_et += forced_trend

    # 添加观测噪声 (Add observational noise)
    daily_et += np.random.normal(0, noise_std * 0.5, size=n_years * 365)

    return daily_et, dates


# ============================================================================
# 主程序 (Main Program)
# ============================================================================


def main():
    """
    主函数：运行完整的 Egli et al. (2025) D&A 示例
    Main function: Run complete Egli et al. (2025) D&A example.
    """

    print("=" * 80)
    print("Egli et al. (2025) 检测与归因 (D&A) 示例")
    print("Detection and Attribution Example from Egli et al. (2025)")
    print("=" * 80)
    print()

    # ========================================================================
    # 步骤 1: 模拟数据 (Step 1: Simulate Data)
    # ========================================================================

    print("步骤 1: 模拟 CMIP6 模式和观测数据")
    print("Step 1: Simulating CMIP6 model and observational data")
    print("-" * 80)

    # 模拟 piControl 数据 (Simulate piControl data)
    print("  - 模拟 piControl (5 成员, 500 年)...")
    print("  - Simulating piControl (5 members, 500 years)...")
    pi_data, pi_dates = simulate_picontrol_data(
        n_members=5,
        n_years=500,
        base_mean=3.0,
        noise_std=0.5,
        seed=42
    )

    # 模拟 historical 数据 (Simulate historical data)
    print("  - 模拟 historical (10 成员, 165 年)...")
    print("  - Simulating historical (10 members, 165 years)...")
    hist_data, hist_dates = simulate_historical_data(
        n_members=10,
        n_years=165,
        base_mean=3.0,
        noise_std=0.5,
        trend_strength=0.015,
        seed=100
    )

    # 模拟观测数据 (Simulate observational data)
    print("  - 模拟观测数据 (70 年)...")
    print("  - Simulating observational data (70 years)...")
    obs_data, obs_dates = simulate_observation_data(
        n_years=70,
        base_mean=3.2,
        noise_std=0.6,
        trend_strength=0.025,
        seed=200
    )

    print("  ✓ 数据模拟完成")
    print("  ✓ Data simulation completed")
    print()

    # ========================================================================
    # 步骤 2: 运行 D&A 工作流 (Step 2: Run D&A Workflow)
    # ========================================================================

    print("步骤 2: 运行 Egli et al. (2025) D&A 工作流")
    print("Step 2: Running Egli et al. (2025) D&A workflow")
    print("-" * 80)

    print("  - 计算 ETx7d 指标...")
    print("  - Calculating ETx7d index...")
    print("  - 训练岭回归检测器...")
    print("  - Training ridge regression detector...")
    print("  - 估算强迫响应...")
    print("  - Estimating forced response...")
    print("  - 计算趋势分布...")
    print("  - Computing trend distributions...")

    # 运行完整 D&A 工作流
    # Run complete D&A workflow
    results = run_egli_attribution_workflow(
        historical_data_members=hist_data,
        picontrol_data_members=pi_data,
        observation_data=obs_data,
        dates_hist=hist_dates,
        dates_pi=pi_dates,
        dates_obs=obs_dates,
        trend_range=(1980, 2020),  # 分析 1980-2020 年趋势
        detection_window=7,         # ETx7d
        block_size=40               # piControl 块大小
    )

    print("  ✓ D&A 分析完成")
    print("  ✓ D&A analysis completed")
    print()

    # ========================================================================
    # 步骤 3: 分析和显示结果 (Step 3: Analyze and Display Results)
    # ========================================================================

    print("步骤 3: 分析结果")
    print("Step 3: Analyzing results")
    print("-" * 80)

    # 提取关键结果 (Extract key results)
    obs_trend = results['obs_trend']
    hist_trends = results['hist_trends']
    pi_trends = results['picontrol_trends']

    # 计算统计量 (Calculate statistics)
    hist_mean = np.mean(hist_trends)
    hist_std = np.std(hist_trends)
    pi_mean = np.mean(pi_trends)
    pi_std = np.std(pi_trends)

    # 计算百分位数 (Calculate percentiles)
    pi_95th = np.percentile(pi_trends, 95)
    pi_5th = np.percentile(pi_trends, 5)

    # 打印结果 (Print results)
    print(f"\n趋势分析结果 (1980-2020):")
    print(f"Trend analysis results (1980-2020):")
    print(f"  观测趋势 (Observed trend):         {obs_trend:.6f} mm/year")
    print(f"  Historical 平均趋势 (mean):        {hist_mean:.6f} mm/year")
    print(f"  Historical 标准差 (std):           {hist_std:.6f} mm/year")
    print(f"  piControl 平均趋势 (mean):         {pi_mean:.6f} mm/year")
    print(f"  piControl 标准差 (std):            {pi_std:.6f} mm/year")
    print(f"  piControl 5th-95th 百分位数:       [{pi_5th:.6f}, {pi_95th:.6f}]")
    print()

    # 检测判断 (Detection assessment)
    print("检测与归因判断:")
    print("Detection and Attribution Assessment:")
    if obs_trend > pi_95th:
        print(f"  ✓ 检测成功: 观测趋势 ({obs_trend:.6f}) > piControl 95th 百分位数 ({pi_95th:.6f})")
        print(f"  ✓ Detection: Observed trend ({obs_trend:.6f}) > piControl 95th percentile ({pi_95th:.6f})")
    else:
        print(f"  ✗ 检测失败: 观测趋势未显著超出 piControl 范围")
        print(f"  ✗ No detection: Observed trend not significantly above piControl range")

    # 归因判断 (Attribution assessment)
    hist_5th = np.percentile(hist_trends, 5)
    hist_95th = np.percentile(hist_trends, 95)
    if hist_5th <= obs_trend <= hist_95th:
        print(f"  ✓ 归因成功: 观测趋势位于 Historical 5th-95th 百分位数范围内")
        print(f"  ✓ Attribution: Observed trend within Historical 5th-95th percentile range")
    else:
        print(f"  ? 归因不确定: 观测趋势超出 Historical 范围")
        print(f"  ? Attribution uncertain: Observed trend outside Historical range")
    print()

    # ========================================================================
    # 步骤 4: 可视化 (Step 4: Visualization)
    # ========================================================================

    print("步骤 4: 可视化结果（复现 Egli et al. 2025 Figure 3d）")
    print("Step 4: Visualizing results (reproducing Egli et al. 2025 Figure 3d)")
    print("-" * 80)

    # 设置绘图风格 (Set plot style)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11

    # 创建图形 (Create figure)
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制 piControl 趋势分布（绿色）
    # Plot piControl trend distribution (green)
    sns.kdeplot(
        pi_trends,
        color='green',
        linewidth=2.5,
        label=f'piControl (n={len(pi_trends)})',
        ax=ax,
        alpha=0.6
    )

    # 绘制 historical 趋势分布（蓝色）
    # Plot historical trend distribution (blue)
    sns.kdeplot(
        hist_trends,
        color='blue',
        linewidth=2.5,
        label=f'Historical (n={len(hist_trends)})',
        ax=ax,
        alpha=0.6
    )

    # 绘制观测趋势（垂直黄线）
    # Plot observed trend (vertical yellow line)
    ax.axvline(
        obs_trend,
        color='orange',
        linewidth=3,
        linestyle='--',
        label=f'Observation ({obs_trend:.6f} mm/yr)'
    )

    # 添加 piControl 95% 置信区间阴影
    # Add piControl 95% confidence interval shading
    ax.axvspan(pi_5th, pi_95th, alpha=0.1, color='green', label='piControl 5th-95th percentile')

    # 设置标签和标题 (Set labels and title)
    ax.set_xlabel('Trend in Forced Response (mm/year)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title(
        'Detection & Attribution of Extreme ET Trends (Egli et al. 2025 Style)\n'
        'ETx7d Forced Response Trend Distribution (1980-2020)',
        fontsize=14,
        fontweight='bold'
    )

    # 添加图例 (Add legend)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

    # 添加网格 (Add grid)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局 (Adjust layout)
    plt.tight_layout()

    # 保存图形 (Save figure)
    output_path = 'egli_2025_detection_attribution_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 图形已保存: {output_path}")
    print(f"  ✓ Figure saved: {output_path}")

    # 显示图形 (Display figure)
    plt.show()

    print()
    print("=" * 80)
    print("分析完成!")
    print("Analysis completed!")
    print("=" * 80)


# ============================================================================
# 脚本入口 (Script Entry Point)
# ============================================================================

if __name__ == "__main__":
    main()
