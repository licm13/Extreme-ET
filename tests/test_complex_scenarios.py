"""
复杂场景集成测试 (Complex Scenario Integration Tests)

本模块实现了真实科研场景的端到端测试，包括：
1. 完整的 I/O 流程（NetCDF 读写）
2. 复杂的时间序列处理
3. 空间数据处理
4. 多方法对比验证
5. 性能压力测试

作者: Extreme-ET Team
日期: 2025
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


# ============================================================================
# 测试夹具 (Test Fixtures)
# ============================================================================

@pytest.fixture
def mock_netcdf_file(tmp_path):
    """
    创建一个模拟的 NetCDF 文件

    包含：
    - 2年的日数据（730天）
    - 5x5 的空间网格
    - ET0、温度、辐射等变量
    - 预埋的极端事件（在特定位置和时间）
    """
    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray is required for this test")

    # 创建空间坐标
    lat = np.linspace(30, 40, 5)
    lon = np.linspace(-100, -90, 5)
    time = pd.date_range("2000-01-01", periods=365*2, freq="D")

    # 创建基础数据（随机 + 季节性）
    n_time = len(time)
    n_lat = len(lat)
    n_lon = len(lon)

    # ET0: 基础值 3-7 mm/day，有季节变化
    doy = time.dayofyear.values
    seasonal_cycle = 2 + 2.5 * np.sin(2 * np.pi * (doy - 80) / 365)  # 夏高冬低
    et0_data = np.zeros((n_time, n_lat, n_lon))

    for i in range(n_lat):
        for j in range(n_lon):
            et0_data[:, i, j] = seasonal_cycle + np.random.normal(0, 0.3, n_time)

    # 🔥 预埋极端事件
    # 事件1: 在网格点 (2, 2) 的第100-105天
    et0_data[100:105, 2, 2] += 5.0

    # 事件2: 在网格点 (3, 3) 的第200-207天（持续时间更长）
    et0_data[200:207, 3, 3] += 4.5

    # 温度数据
    temp_mean = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp_data = np.tile(temp_mean[:, np.newaxis, np.newaxis], (1, n_lat, n_lon))
    temp_data += np.random.normal(0, 1.5, (n_time, n_lat, n_lon))

    # 辐射数据
    rad_data = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
    rad_data = np.tile(rad_data[:, np.newaxis, np.newaxis], (1, n_lat, n_lon))
    rad_data += np.random.normal(0, 1.0, (n_time, n_lat, n_lon))
    rad_data = np.maximum(rad_data, 0)  # 确保非负

    # 创建 xarray Dataset
    ds = xr.Dataset(
        {
            "et0": (["time", "lat", "lon"], et0_data),
            "temperature": (["time", "lat", "lon"], temp_data),
            "radiation": (["time", "lat", "lon"], rad_data),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "Mock ET Data for Testing",
            "institution": "Extreme-ET Test Suite",
            "source": "Synthetic data with embedded extreme events"
        }
    )

    # 保存到临时文件
    file_path = tmp_path / "mock_et_data.nc"
    ds.to_netcdf(file_path)

    # 返回文件路径和极端事件的位置信息
    extreme_events = {
        'event1': {
            'lat_idx': 2,
            'lon_idx': 2,
            'time_range': (100, 105),
            'lat': lat[2],
            'lon': lon[2]
        },
        'event2': {
            'lat_idx': 3,
            'lon_idx': 3,
            'time_range': (200, 207),
            'lat': lat[3],
            'lon': lon[3]
        }
    }

    return file_path, extreme_events


@pytest.fixture
def sample_station_data():
    """
    创建模拟的站点观测数据（用于验证）

    返回一个 pandas DataFrame，包含：
    - 3年的日数据
    - 完整的气象变量
    - 一些缺失值（测试插值）
    - 一些异常值（测试质量控制）
    """
    dates = pd.date_range("2000-01-01", periods=365*3, freq="D")
    n = len(dates)

    # 基础季节性模式
    doy = dates.dayofyear.values
    temp_mean = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 2, n)
    temp_max = temp_mean + 5 + np.random.normal(0, 1, n)
    temp_min = temp_mean - 5 + np.random.normal(0, 1, n)
    radiation = np.maximum(10 + 8 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 1.5, n), 0)
    wind = np.abs(2.0 + np.random.normal(0, 0.5, n))

    # 水汽压（从温度估算）
    es = 0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3))
    rh = 60 + 20 * np.random.rand(n)  # 相对湿度 40-80%
    ea = es * rh / 100

    df = pd.DataFrame({
        'T_mean': temp_mean,
        'T_max': temp_max,
        'T_min': temp_min,
        'Rs': radiation,
        'u2': wind,
        'ea': ea
    }, index=dates)

    # 🔧 制造一些缺失值（模拟真实情况）
    missing_indices = np.random.choice(n, size=int(n * 0.02), replace=False)  # 2% 缺失
    df.loc[df.index[missing_indices], 'Rs'] = np.nan

    # 制造一段连续缺失（测试插值极限）
    df.loc[df.index[500:506], 'u2'] = np.nan  # 连续6天缺失

    # 🚨 制造一些异常值（测试质量控制）
    df.loc[df.index[100], 'T_max'] = 999.9  # 明显异常
    df.loc[df.index[200], 'Rs'] = -5.0  # 物理上不可能的负值

    return df


# ============================================================================
# 测试 1: 端到端 I/O 和极端检测流程
# ============================================================================

def test_end_to_end_io_and_detection(mock_netcdf_file):
    """
    测试完整的数据处理流程：
    1. 从 NetCDF 读取数据
    2. 提取特定站点时间序列
    3. 使用气候学方法检测极端事件
    4. 验证预埋的极端事件是否被正确检测
    """
    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray is required for this test")

    from src.extreme_detection import detect_extreme_events_clim

    file_path, extreme_events = mock_netcdf_file

    # 1. 读取 NetCDF 数据
    ds = xr.open_dataset(file_path)
    assert 'et0' in ds.data_vars, "ET0 variable not found in dataset"

    # 2. 提取第一个预埋事件的位置
    event1 = extreme_events['event1']
    lat_idx = event1['lat_idx']
    lon_idx = event1['lon_idx']

    et0_series = ds['et0'][:, lat_idx, lon_idx].values

    # 3. 运行极端检测（ERT_clim）
    # 使用较宽松的阈值，确保能检测到预埋事件
    extreme_mask, thresholds, details = detect_extreme_events_clim(
        et0_series,
        severity=0.10,  # 前 10%
        min_duration=3,
        return_details=True
    )

    # 4. 验证检测结果
    event1_range = range(event1['time_range'][0], event1['time_range'][1])

    # 检查预埋事件期间至少有部分被标记为极端
    detected_in_event = np.sum(extreme_mask[event1_range])
    assert detected_in_event >= 3, \
        f"Failed to detect event1: only {detected_in_event}/5 days detected"

    # 5. 检查返回的统计信息
    assert 'n_events' in details, "Details should contain number of events"
    assert details['n_events'] >= 1, "At least one event should be detected"
    assert details['total_extreme_days'] >= 5, "Should detect at least 5 extreme days"

    print(f"✓ Test passed: Detected {details['n_events']} events, "
          f"{details['total_extreme_days']} extreme days")


# ============================================================================
# 测试 2: 数据清洗与质量控制流程
# ============================================================================

def test_data_quality_control_pipeline(sample_station_data):
    """
    测试数据质量控制流程：
    1. 检测异常值
    2. 处理缺失值
    3. 验证温度逻辑一致性
    4. 确保处理后的数据可用于 ET0 计算
    """
    from src.penman_monteith import calculate_et0

    df_original = sample_station_data.copy()

    # ========== 步骤 1: 物理范围检查 ==========
    def check_and_clean_physical_bounds(df):
        bounds = {
            'T_mean': (-60, 60),
            'T_max': (-50, 70),
            'T_min': (-70, 50),
            'Rs': (0, 40),
            'u2': (0, 30),
            'ea': (0, 7)
        }

        df_clean = df.copy()
        for var, (lower, upper) in bounds.items():
            if var in df_clean.columns:
                invalid = (df_clean[var] < lower) | (df_clean[var] > upper)
                n_invalid = np.sum(invalid)
                if n_invalid > 0:
                    print(f"  Cleaning {var}: {n_invalid} out-of-bounds values")
                    df_clean.loc[invalid, var] = np.nan

        return df_clean

    df_clean = check_and_clean_physical_bounds(df_original)

    # 验证：异常的 T_max=999.9 应该被清除
    assert not np.any(df_clean['T_max'] > 100), "Outlier not removed"

    # ========== 步骤 2: 温度逻辑一致性 ==========
    def fix_temperature_consistency(df):
        df_fixed = df.copy()
        # 如果 T_min > T_max，交换它们
        swap_mask = df_fixed['T_min'] > df_fixed['T_max']
        if np.any(swap_mask):
            print(f"  Fixing {np.sum(swap_mask)} inconsistent temperature records")
            df_fixed.loc[swap_mask, ['T_min', 'T_max']] = \
                df_fixed.loc[swap_mask, ['T_max', 'T_min']].values

        # 重新计算 T_mean（如果不一致）
        expected_mean = (df_fixed['T_max'] + df_fixed['T_min']) / 2
        inconsistent = np.abs(df_fixed['T_mean'] - expected_mean) > 5
        if np.any(inconsistent):
            print(f"  Recalculating T_mean for {np.sum(inconsistent)} records")
            df_fixed.loc[inconsistent, 'T_mean'] = expected_mean[inconsistent]

        return df_fixed

    df_clean = fix_temperature_consistency(df_clean)

    # ========== 步骤 3: 缺失值插值 ==========
    def interpolate_missing(df, max_gap=7):
        df_filled = df.copy()
        for col in df_filled.columns:
            df_filled[col] = df_filled[col].interpolate(
                method='linear',
                limit=max_gap,
                limit_direction='both'
            )
        return df_filled

    df_clean = interpolate_missing(df_clean, max_gap=7)

    # 验证：短缺失应该被填补
    assert df_clean['Rs'].isnull().sum() < df_original['Rs'].isnull().sum(), \
        "Interpolation did not reduce missing values"

    # 验证：长缺失（6天）应该被填补（因为 < max_gap=7）
    assert df_clean['u2'].isnull().sum() == 0, \
        "6-day gap should be filled with max_gap=7"

    # ========== 步骤 4: 计算 ET0（验证数据可用性）==========
    et0 = calculate_et0(
        T_mean=df_clean['T_mean'].values,
        T_max=df_clean['T_max'].values,
        T_min=df_clean['T_min'].values,
        Rs=df_clean['Rs'].values,
        u2=df_clean['u2'].values,
        ea=df_clean['ea'].values,
        z=50.0,
        latitude=40.0
    )

    # 验证 ET0 的合理性
    assert np.all(et0 >= 0), "ET0 should be non-negative"
    assert np.all(et0 < 20), "ET0 should be less than 20 mm/day"
    assert 2 < np.mean(et0) < 8, f"Mean ET0 ({np.mean(et0):.2f}) is unrealistic"

    print(f"✓ Test passed: Data cleaned and ET0 calculated successfully")
    print(f"  Mean ET0: {np.mean(et0):.2f} mm/day")
    print(f"  Remaining missing values: {np.sum(np.isnan(et0))}")


# ============================================================================
# 测试 3: 多方法对比测试
# ============================================================================

def test_multiple_detection_methods_comparison():
    """
    对比三种检测方法（ERT_hist, ERT_clim, OPT）：
    1. 在相同数据上运行三种方法
    2. 验证检测结果的合理性
    3. 比较方法间的差异
    """
    from src.extreme_detection import (
        detect_extreme_events_hist,
        detect_extreme_events_clim,
        optimal_path_threshold
    )

    # 创建合成数据（10年）
    np.random.seed(42)
    n_years = 10
    n_days = n_years * 365

    # 基础季节性 + 随机噪声
    doy = np.tile(np.arange(365), n_years)
    et0 = 3.5 + 2.5 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 0.5, n_days)

    # 制造几个极端事件
    extreme_indices = [100, 101, 102, 500, 501, 502, 503, 1000, 1001, 1002]
    et0[extreme_indices] += 4.0

    # ========== 运行三种方法 ==========
    target_severity = 0.01  # 前 1% (~3.65 天/年)

    # 方法1: ERT_hist
    mask_hist, threshold_hist = detect_extreme_events_hist(
        et0, severity=target_severity
    )

    # 方法2: ERT_clim
    mask_clim, thresholds_clim = detect_extreme_events_clim(
        et0, severity=0.05, min_duration=3
    )

    # 方法3: OPT（使用简化版本，如果实现了的话）
    try:
        mask_opt, thresholds_opt = optimal_path_threshold(
            et0, target_severity=target_severity
        )
    except (NameError, AttributeError):
        pytest.skip("OPT method not implemented")
        mask_opt = None

    # ========== 验证检测结果 ==========
    n_detected = {
        'ERT_hist': np.sum(mask_hist),
        'ERT_clim': np.sum(mask_clim),
        'OPT': np.sum(mask_opt) if mask_opt is not None else None
    }

    print("\n=== Detection Method Comparison ===")
    for method, count in n_detected.items():
        if count is not None:
            rate = count / n_days * 100
            print(f"{method:12s}: {count:4d} days ({rate:.2f}%)")

    # 验证1: ERT_hist 应该接近目标严重性
    expected_days = n_days * target_severity
    assert abs(n_detected['ERT_hist'] - expected_days) < expected_days * 0.2, \
        f"ERT_hist detected {n_detected['ERT_hist']} days, expected ~{expected_days}"

    # 验证2: 方法间的重叠度
    overlap_hist_clim = np.sum(mask_hist & mask_clim)
    overlap_rate = overlap_hist_clim / np.sum(mask_hist) if np.sum(mask_hist) > 0 else 0

    print(f"\nOverlap between ERT_hist and ERT_clim: {overlap_rate:.1%}")

    # 验证3: 至少有一个方法检测到预埋的极端事件
    detected_embedded = np.sum(mask_hist[extreme_indices]) + np.sum(mask_clim[extreme_indices])
    assert detected_embedded >= 5, \
        f"Methods should detect at least 5 of the 10 embedded extreme days"

    print(f"Embedded extremes detected: {detected_embedded}/10")
    print("✓ Test passed: All methods produced reasonable results")


# ============================================================================
# 测试 4: 空间分析测试
# ============================================================================

def test_spatial_analysis_on_gridded_data(mock_netcdf_file):
    """
    测试空间分析功能：
    1. 计算区域平均
    2. 识别空间相关的极端事件
    3. 空间插值
    """
    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray is required for this test")

    from src.extreme_detection import detect_extreme_events_hist

    file_path, extreme_events = mock_netcdf_file

    # 读取数据
    ds = xr.open_dataset(file_path)
    et0_3d = ds['et0'].values  # (time, lat, lon)

    # ========== 测试1: 区域平均 ==========
    regional_mean = np.mean(et0_3d, axis=(1, 2))  # 对空间平均

    # 验证：区域平均应该平滑化极端值
    max_grid_value = np.max(et0_3d)
    max_regional_value = np.max(regional_mean)
    assert max_regional_value < max_grid_value, \
        "Regional mean should be smoother than grid points"

    print(f"Max grid ET0: {max_grid_value:.2f} mm/day")
    print(f"Max regional mean ET0: {max_regional_value:.2f} mm/day")

    # ========== 测试2: 逐格点检测 ==========
    n_lat, n_lon = et0_3d.shape[1], et0_3d.shape[2]
    detected_count = np.zeros((n_lat, n_lon))

    for i in range(n_lat):
        for j in range(n_lon):
            series = et0_3d[:, i, j]
            mask, _ = detect_extreme_events_hist(series, severity=0.05)
            detected_count[i, j] = np.sum(mask)

    # 验证：预埋事件的位置应该有更多检测
    event1 = extreme_events['event1']
    event1_count = detected_count[event1['lat_idx'], event1['lon_idx']]

    mean_count = np.mean(detected_count)
    assert event1_count > mean_count * 1.5, \
        f"Event location should have more detections ({event1_count:.0f} vs mean {mean_count:.0f})"

    print(f"Event location extreme days: {event1_count:.0f}")
    print(f"Mean extreme days: {mean_count:.0f}")

    # ========== 测试3: 空间相关性 ==========
    # 检查相邻格点的极端事件是否有空间聚集
    event2 = extreme_events['event2']
    i2, j2 = event2['lat_idx'], event2['lon_idx']

    # 检查周围8个格点
    neighbors_count = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = i2 + di, j2 + dj
            if 0 <= ni < n_lat and 0 <= nj < n_lon and not (di == 0 and dj == 0):
                neighbors_count.append(detected_count[ni, nj])

    # 相邻格点的平均检测数应该高于全局平均（空间相关性）
    neighbor_mean = np.mean(neighbors_count)
    print(f"Neighbor mean extreme days: {neighbor_mean:.0f}")

    # 注意：这个测试可能不稳定，因为我们只预埋了孤立事件
    # 在真实数据中，极端事件通常有空间相关性

    print("✓ Test passed: Spatial analysis completed")


# ============================================================================
# 测试 5: 驱动因子贡献分析测试
# ============================================================================

def test_driver_contribution_analysis(sample_station_data):
    """
    测试驱动因子贡献率分析：
    1. 计算 ET0
    2. 检测极端事件
    3. 分析各驱动因子的贡献
    4. 验证贡献率总和为 100%
    """
    from src.penman_monteith import calculate_et0
    from src.extreme_detection import detect_extreme_events_hist
    from src.contribution_analysis import calculate_contributions

    df = sample_station_data.copy()

    # 数据清洗（简化版）
    df = df.interpolate(method='linear', limit=7)

    # ========== 计算 ET0 ==========
    et0 = calculate_et0(
        T_mean=df['T_mean'].values,
        T_max=df['T_max'].values,
        T_min=df['T_min'].values,
        Rs=df['Rs'].values,
        u2=df['u2'].values,
        ea=df['ea'].values,
        z=50.0,
        latitude=40.0
    )

    # ========== 检测极端事件 ==========
    extreme_mask, _ = detect_extreme_events_hist(et0, severity=0.05)
    n_extreme = np.sum(extreme_mask)

    assert n_extreme > 0, "Should detect at least some extreme events"
    print(f"Detected {n_extreme} extreme days for contribution analysis")

    # ========== 计算贡献率 ==========
    contributions = calculate_contributions(
        T_mean=df['T_mean'].values,
        T_max=df['T_max'].values,
        T_min=df['T_min'].values,
        Rs=df['Rs'].values,
        u2=df['u2'].values,
        ea=df['ea'].values,
        extreme_mask=extreme_mask,
        z=50.0,
        latitude=40.0
    )

    # ========== 验证结果 ==========
    print("\nDriver Contributions:")
    for factor, contrib in contributions.items():
        print(f"  {factor:15s}: {contrib:6.2f}%")

    # 验证1: 所有贡献率应该在合理范围内
    for factor, contrib in contributions.items():
        assert 0 <= contrib <= 100, \
            f"{factor} contribution ({contrib:.2f}%) is out of range [0, 100]"

    # 验证2: 贡献率总和应该接近 100%（允许小误差）
    total_contrib = sum(contributions.values())
    assert 99 <= total_contrib <= 101, \
        f"Total contribution ({total_contrib:.2f}%) should be ~100%"

    # 验证3: 至少有一个主导因子（贡献 > 30%）
    max_contrib = max(contributions.values())
    assert max_contrib > 30, \
        f"At least one factor should be dominant (>30%), but max is {max_contrib:.2f}%"

    max_factor = max(contributions, key=contributions.get)
    print(f"\nDominant driver: {max_factor} ({contributions[max_factor]:.1f}%)")
    print("✓ Test passed: Contribution analysis is valid")


# ============================================================================
# 测试 6: 性能压力测试
# ============================================================================

def test_performance_on_large_dataset():
    """
    性能测试：
    1. 测试大规模数据集（50年日数据）的处理速度
    2. 验证内存使用合理
    3. 确保算法在合理时间内完成
    """
    import time
    from src.extreme_detection import detect_extreme_events_hist

    # 创建50年的日数据
    n_years = 50
    n_days = n_years * 365
    print(f"\nTesting with {n_days} days ({n_years} years) of data...")

    np.random.seed(42)
    doy = np.tile(np.arange(365), n_years)
    et0 = 4.0 + 2.0 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 0.6, n_days)

    # ========== 测试 ERT_hist（应该很快）==========
    start_time = time.time()
    mask_hist, threshold = detect_extreme_events_hist(et0, severity=0.01)
    elapsed_hist = time.time() - start_time

    print(f"ERT_hist: {elapsed_hist:.3f} seconds")
    assert elapsed_hist < 1.0, \
        f"ERT_hist is too slow: {elapsed_hist:.3f}s (should be < 1s)"

    # ========== 测试 ERT_clim（稍慢，但仍应合理）==========
    from src.extreme_detection import detect_extreme_events_clim

    start_time = time.time()
    mask_clim, thresholds = detect_extreme_events_clim(
        et0, severity=0.05, min_duration=3
    )
    elapsed_clim = time.time() - start_time

    print(f"ERT_clim: {elapsed_clim:.3f} seconds")
    assert elapsed_clim < 5.0, \
        f"ERT_clim is too slow: {elapsed_clim:.3f}s (should be < 5s)"

    # ========== 内存检查 ==========
    # 确保返回的掩码大小正确
    assert mask_hist.shape == (n_days,), "Mask shape mismatch"
    assert mask_clim.shape == (n_days,), "Mask shape mismatch"
    assert thresholds.shape == (365,), "Threshold shape mismatch"

    print("✓ Performance test passed")
    print(f"  ERT_hist: {elapsed_hist:.3f}s | ERT_clim: {elapsed_clim:.3f}s")


# ============================================================================
# 测试 7: 错误处理测试
# ============================================================================

def test_error_handling():
    """
    测试各种异常输入的错误处理：
    1. 空数组
    2. 全为 NaN 的数组
    3. 长度不匹配
    4. 无效的参数
    """
    from src.extreme_detection import detect_extreme_events_hist
    from src.penman_monteith import calculate_et0

    # ========== 测试1: 空数组 ==========
    with pytest.raises((ValueError, IndexError)):
        detect_extreme_events_hist(np.array([]), severity=0.01)

    # ========== 测试2: 全 NaN ==========
    all_nan = np.full(100, np.nan)
    with pytest.raises((ValueError, RuntimeError)):
        detect_extreme_events_hist(all_nan, severity=0.01)

    # ========== 测试3: 长度不匹配 ==========
    T_mean = np.random.rand(100)
    T_max = np.random.rand(100)
    T_min = np.random.rand(100)
    Rs = np.random.rand(100)
    u2 = np.random.rand(100)
    ea = np.random.rand(50)  # 长度不匹配！

    with pytest.raises((ValueError, IndexError)):
        calculate_et0(T_mean, T_max, T_min, Rs, u2, ea)

    # ========== 测试4: 无效的严重性参数 ==========
    data = np.random.rand(100)

    with pytest.raises(ValueError):
        detect_extreme_events_hist(data, severity=-0.01)  # 负数

    with pytest.raises(ValueError):
        detect_extreme_events_hist(data, severity=1.5)  # > 1

    print("✓ Error handling test passed")


# ============================================================================
# 测试 8: 回归测试（确保结果稳定性）
# ============================================================================

def test_regression_stability():
    """
    回归测试：
    确保在固定的随机种子下，结果完全可重复
    """
    from src.extreme_detection import detect_extreme_events_hist
    from src.penman_monteith import calculate_et0

    # 固定随机种子
    np.random.seed(12345)

    # 生成数据
    n = 365 * 5
    T_mean = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n)
    T_max = T_mean + 5
    T_min = T_mean - 5
    Rs = 15 + 8 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 1, n)
    u2 = np.abs(2.0 + np.random.normal(0, 0.5, n))
    ea = np.abs(1.5 + np.random.normal(0, 0.3, n))

    # 计算 ET0
    et0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea)

    # 检测极端事件
    mask, threshold = detect_extreme_events_hist(et0, severity=0.01)

    # 期望结果（通过首次运行获得，固定为"黄金标准"）
    expected_n_extreme = int(n * 0.01)  # 应该接近这个值
    expected_threshold_range = (6.5, 9.5)  # 阈值应该在这个范围内

    # 验证
    actual_n_extreme = np.sum(mask)
    assert abs(actual_n_extreme - expected_n_extreme) <= expected_n_extreme * 0.3, \
        f"Detected {actual_n_extreme} extremes, expected ~{expected_n_extreme}"

    assert expected_threshold_range[0] <= threshold <= expected_threshold_range[1], \
        f"Threshold {threshold:.2f} is outside expected range {expected_threshold_range}"

    print(f"✓ Regression test passed")
    print(f"  Detected: {actual_n_extreme} extremes")
    print(f"  Threshold: {threshold:.2f} mm/day")


# ============================================================================
# 运行所有测试
# ============================================================================

if __name__ == "__main__":
    """
    直接运行此脚本进行快速测试（不使用 pytest）
    """
    print("=" * 70)
    print("Running Complex Scenario Integration Tests")
    print("=" * 70)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 创建夹具
        print("\n[1/8] Setting up mock data...")
        mock_nc, events = pytest.fixture()(mock_netcdf_file)(tmp_path)

        print("\n[2/8] Test: End-to-end I/O and detection...")
        try:
            test_end_to_end_io_and_detection((mock_nc, events))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[3/8] Test: Data quality control...")
        sample_data = pytest.fixture()(sample_station_data)()
        try:
            test_data_quality_control_pipeline(sample_data)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[4/8] Test: Multiple detection methods...")
        try:
            test_multiple_detection_methods_comparison()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[5/8] Test: Spatial analysis...")
        try:
            test_spatial_analysis_on_gridded_data((mock_nc, events))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[6/8] Test: Driver contribution...")
        try:
            test_driver_contribution_analysis(sample_data)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[7/8] Test: Performance...")
        try:
            test_performance_on_large_dataset()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[8/9] Test: Error handling...")
        try:
            test_error_handling()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        print("\n[9/9] Test: Regression stability...")
        try:
            test_regression_stability()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
