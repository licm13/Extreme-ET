# 数据准备指南 (Data Preparation Guide)

## 目录 (Table of Contents)

1. [数据格式要求](#1-数据格式要求)
2. [NetCDF 文件处理](#2-netcdf-文件处理)
3. [数据清洗与质量控制](#3-数据清洗与质量控制)
4. [缺失值处理策略](#4-缺失值处理策略)
5. [时间序列对齐与重采样](#5-时间序列对齐与重采样)
6. [空间数据处理](#6-空间数据处理)
7. [常见数据源接入](#7-常见数据源接入)
8. [完整工作流示例](#8-完整工作流示例)

---

## 1. 数据格式要求

### 1.1 核心输入数据

Extreme-ET 工具包需要以下气象变量来计算 ET₀：

| 变量 | 符号 | 单位 | 必需性 | 备注 |
|------|------|------|--------|------|
| 平均气温 | T_mean | °C | ✅ 必需 | 可由 T_max 和 T_min 计算 |
| 最高气温 | T_max | °C | ✅ 必需 | 用于计算饱和水汽压 |
| 最低气温 | T_min | °C | ✅ 必需 | 用于计算饱和水汽压 |
| 太阳辐射 | Rs | MJ m⁻² day⁻¹ | ✅ 必需 | 短波入射辐射 |
| 风速 | u₂ | m s⁻¹ | ✅ 必需 | 2米高度风速 |
| 实际水汽压 | ea | kPa | ✅ 必需 | 或通过相对湿度计算 |
| 海拔高度 | z | m | 🔶 推荐 | 影响气压修正 |
| 纬度 | latitude | °N | 🔶 推荐 | 用于辐射计算 |
| 日序数 | DOY | 1-365 | 🔶 推荐 | 用于精确辐射计算 |

### 1.2 数据结构

**一维时间序列（站点数据）：**

```python
# NumPy 数组格式
data = {
    'time': np.array(['2000-01-01', ..., '2020-12-31'], dtype='datetime64'),
    'T_mean': np.array([...]),  # 形状: (n_days,)
    'T_max': np.array([...]),
    'T_min': np.array([...]),
    'Rs': np.array([...]),
    'u2': np.array([...]),
    'ea': np.array([...])
}

# Pandas DataFrame 格式（推荐）
import pandas as pd
df = pd.DataFrame({
    'T_mean': [...],
    'T_max': [...],
    'T_min': [...],
    'Rs': [...],
    'u2': [...],
    'ea': [...]
}, index=pd.date_range('2000-01-01', '2020-12-31', freq='D'))
```

**多维网格数据（NetCDF）：**

```python
# xarray.Dataset 格式
import xarray as xr
ds = xr.Dataset({
    'T_mean': (['time', 'lat', 'lon'], data_3d),
    'Rs': (['time', 'lat', 'lon'], data_3d),
    # ...
}, coords={
    'time': pd.date_range('2000-01-01', '2020-12-31', freq='D'),
    'lat': np.arange(25, 50, 0.25),
    'lon': np.arange(-125, -66, 0.25)
})
```

---

## 2. NetCDF 文件处理

### 2.1 基础读取操作

**使用 xarray（推荐）：**

```python
import xarray as xr

# 读取单个文件
ds = xr.open_dataset('path/to/data.nc')

# 查看变量列表
print(ds.data_vars)

# 读取多个文件（按时间合并）
ds = xr.open_mfdataset('data/*.nc', combine='by_coords')

# 选择特定变量
da = ds['temperature']  # DataArray

# 查看属性
print(da.attrs)  # 元数据
print(da.coords)  # 坐标信息
```

### 2.2 工具包提供的 I/O 函数

**文件**: `src/io_utils.py`

```python
from src.io_utils import read_netcdf_variable, sample_series_at_point

# 读取 NetCDF 变量
da, lats, lons, times = read_netcdf_variable(
    filepath='era5_land_et.nc',
    varname='evaporation',
    lat_slice=(30, 45),  # 纬度范围
    lon_slice=(-120, -100),  # 经度范围
    time_slice=('2000-01-01', '2020-12-31')  # 时间范围
)

print(f"数据形状: {da.shape}")  # (时间, 纬度, 经度)
print(f"时间范围: {times[0]} 到 {times[-1]}")
```

### 2.3 提取特定位置的时间序列

**最近邻插值（Nearest Neighbor）：**

```python
# 提取洛杉矶的时间序列
lat_target, lon_target = 34.05, -118.24

series_nn = sample_series_at_point(
    da,
    lat_target,
    lon_target,
    method='nearest'
)

print(f"提取的序列长度: {len(series_nn)}")
print(f"前5个值: {series_nn[:5].values}")
```

**双线性插值（Bilinear）：**

```python
# 更平滑的插值结果
series_bl = sample_series_at_point(
    da,
    lat_target,
    lon_target,
    method='bilinear'
)

# 对比两种方法
import matplotlib.pyplot as plt
plt.plot(series_nn, label='Nearest', alpha=0.7)
plt.plot(series_bl, label='Bilinear', alpha=0.7)
plt.legend()
plt.show()
```

### 2.4 批量提取多个站点

```python
def extract_multiple_stations(filepath, varname, stations):
    """
    批量提取多个站点的时间序列

    Parameters
    ----------
    filepath : str
        NetCDF 文件路径
    varname : str
        变量名
    stations : dict
        站点字典，格式: {'站点名': (纬度, 经度)}

    Returns
    -------
    station_data : dict
        每个站点的时间序列
    """
    da, lats, lons, times = read_netcdf_variable(filepath, varname)

    station_data = {}
    for station_name, (lat, lon) in stations.items():
        series = sample_series_at_point(da, lat, lon, method='bilinear')
        station_data[station_name] = pd.Series(
            series.values,
            index=times,
            name=station_name
        )

    return pd.DataFrame(station_data)

# 使用示例
stations = {
    'Los Angeles': (34.05, -118.24),
    'Chicago': (41.88, -87.63),
    'New York': (40.71, -74.01)
}

station_df = extract_multiple_stations('data.nc', 'temperature', stations)
print(station_df.head())
```

---

## 3. 数据清洗与质量控制

### 3.1 检测异常值

**物理范围检查：**

```python
def check_physical_bounds(data, varname):
    """
    检查变量是否在物理合理范围内

    Parameters
    ----------
    data : np.ndarray
        变量数据
    varname : str
        变量名称

    Returns
    -------
    valid_mask : np.ndarray (bool)
        有效数据掩码
    """
    bounds = {
        'T_mean': (-60, 60),     # °C
        'T_max': (-50, 70),
        'T_min': (-70, 50),
        'Rs': (0, 40),           # MJ/m²/day
        'u2': (0, 50),           # m/s
        'ea': (0, 7),            # kPa
        'ET0': (0, 20)           # mm/day
    }

    if varname not in bounds:
        return np.ones(len(data), dtype=bool)

    lower, upper = bounds[varname]
    valid_mask = (data >= lower) & (data <= upper)

    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"警告: {varname} 有 {n_invalid} 个值超出范围 [{lower}, {upper}]")

    return valid_mask

# 使用示例
T_mean = np.array([15, 20, -999, 25, 30])  # -999 是缺失值标识
valid = check_physical_bounds(T_mean, 'T_mean')
T_mean_cleaned = np.where(valid, T_mean, np.nan)
```

**统计异常检测（3σ 法则）：**

```python
def detect_outliers_zscore(data, threshold=3.0):
    """
    使用 Z-score 方法检测异常值

    Parameters
    ----------
    data : np.ndarray
        时间序列数据
    threshold : float, default=3.0
        Z-score 阈值（通常 3 表示 99.7% 置信）

    Returns
    -------
    outlier_mask : np.ndarray (bool)
        异常值掩码（True = 异常）
    """
    # 移除 NaN 后计算统计量
    valid_data = data[~np.isnan(data)]
    mean = np.mean(valid_data)
    std = np.std(valid_data)

    z_scores = np.abs((data - mean) / std)
    outlier_mask = z_scores > threshold

    return outlier_mask

# 使用示例
outliers = detect_outliers_zscore(T_mean_cleaned)
print(f"检测到 {np.sum(outliers)} 个异常值")
```

**基于 IQR 的稳健检测：**

```python
def detect_outliers_iqr(data, factor=1.5):
    """
    使用四分位距 (IQR) 方法检测异常值

    Parameters
    ----------
    data : np.ndarray
        时间序列数据
    factor : float, default=1.5
        IQR 倍数（1.5 是标准，3.0 是极端）

    Returns
    -------
    outlier_mask : np.ndarray (bool)
        异常值掩码
    """
    valid_data = data[~np.isnan(data)]
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask
```

### 3.2 数据一致性检查

**温度逻辑检查：**

```python
def check_temperature_consistency(T_mean, T_max, T_min):
    """
    检查温度数据的逻辑一致性

    要求: T_min <= T_mean <= T_max
    """
    inconsistent = (T_min > T_mean) | (T_mean > T_max) | (T_min > T_max)

    n_errors = np.sum(inconsistent)
    if n_errors > 0:
        print(f"警告: 发现 {n_errors} 个温度不一致的记录")

        # 尝试自动修复（使用平均值）
        T_mean_fixed = np.where(
            inconsistent,
            (T_max + T_min) / 2,
            T_mean
        )
        return T_mean_fixed, inconsistent

    return T_mean, inconsistent

# 使用示例
T_mean_fixed, errors = check_temperature_consistency(T_mean, T_max, T_min)
```

---

## 4. 缺失值处理策略

### 4.1 缺失值诊断

```python
def diagnose_missing_data(df):
    """
    诊断数据框中的缺失情况

    Parameters
    ----------
    df : pd.DataFrame
        包含气象变量的数据框

    Returns
    -------
    report : pd.DataFrame
        缺失值诊断报告
    """
    report = pd.DataFrame({
        'n_missing': df.isnull().sum(),
        'pct_missing': df.isnull().sum() / len(df) * 100,
        'n_consecutive_max': [
            df[col].isnull().astype(int).groupby(
                df[col].notnull().astype(int).cumsum()
            ).sum().max()
            for col in df.columns
        ]
    })

    print("=== 缺失值诊断报告 ===")
    print(report)

    # 可视化缺失模式
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 4))
    sns.heatmap(df.isnull().T, cbar=False, cmap='viridis', yticklabels=df.columns)
    plt.title('Missing Data Pattern')
    plt.xlabel('Time Index')
    plt.show()

    return report
```

### 4.2 线性插值填补

**简单线性插值：**

```python
def fill_missing_linear(data, max_gap=7):
    """
    使用线性插值填补缺失值

    Parameters
    ----------
    data : np.ndarray or pd.Series
        时间序列数据
    max_gap : int, default=7
        允许插值的最大连续缺失天数

    Returns
    -------
    data_filled : np.ndarray or pd.Series
        填补后的数据
    """
    if isinstance(data, pd.Series):
        # Pandas Series 方法
        data_filled = data.interpolate(
            method='linear',
            limit=max_gap,
            limit_direction='both'
        )
    else:
        # NumPy 方法
        from scipy.interpolate import interp1d

        valid_indices = ~np.isnan(data)
        if np.sum(valid_indices) < 2:
            return data  # 无法插值

        x = np.arange(len(data))[valid_indices]
        y = data[valid_indices]

        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        data_filled = f(np.arange(len(data)))

        # 只填补小于 max_gap 的缺失
        gap_sizes = _calculate_gap_sizes(data)
        data_filled = np.where(gap_sizes <= max_gap, data_filled, np.nan)

    return data_filled

def _calculate_gap_sizes(data):
    """计算每个缺失值所在缺失段的长度"""
    is_nan = np.isnan(data)
    gap_id = (~is_nan).cumsum()  # 给每个缺失段分配ID
    gap_sizes = np.zeros(len(data))

    for gid in np.unique(gap_id[is_nan]):
        gap_mask = (gap_id == gid) & is_nan
        gap_sizes[gap_mask] = np.sum(gap_mask)

    return gap_sizes
```

**示例（来自 `examples/example_zhao_2025.py`）：**

```python
def _fill_nan_linear(arr, max_gap=7):
    """
    线性插值填补NaN，但跳过过长的缺失段

    这是 Zhao et al. (2025) 论文中使用的方法
    """
    arr = np.asarray(arr, dtype=float)
    idx = np.arange(len(arr))
    valid = ~np.isnan(arr)

    if np.sum(valid) < 2:
        return arr

    # 使用 scipy 进行插值
    from scipy.interpolate import interp1d
    f = interp1d(idx[valid], arr[valid], kind='linear',
                 bounds_error=False, fill_value=np.nan)
    arr_interp = f(idx)

    # 识别缺失段
    nan_mask = np.isnan(arr)
    nan_segments = np.split(np.arange(len(arr)),
                           np.where(np.diff(nan_mask.astype(int)) != 0)[0] + 1)

    # 只填补短缺失段
    arr_filled = arr.copy()
    for segment in nan_segments:
        if len(segment) > 0 and nan_mask[segment[0]]:
            if len(segment) <= max_gap:
                arr_filled[segment] = arr_interp[segment]

    return arr_filled
```

### 4.3 气候学填补

**使用多年平均值：**

```python
def fill_missing_climatology(data, dates):
    """
    使用气候学平均值填补缺失值

    Parameters
    ----------
    data : pd.Series
        时间序列（索引为日期）
    dates : pd.DatetimeIndex
        对应的日期索引

    Returns
    -------
    data_filled : pd.Series
        填补后的数据
    """
    # 计算每个日历日（DOY）的多年平均
    doy = dates.dayofyear
    climatology = data.groupby(doy).mean()

    # 用气候学值填补缺失
    data_filled = data.copy()
    missing_mask = data.isnull()
    data_filled[missing_mask] = climatology[doy[missing_mask]].values

    return data_filled

# 使用示例
df = pd.DataFrame({
    'ET0': [3.5, np.nan, 4.2, np.nan, 5.1],
}, index=pd.date_range('2020-01-01', periods=5))

df['ET0_filled'] = fill_missing_climatology(df['ET0'], df.index)
```

### 4.4 多变量插补（高级）

**使用相关变量预测：**

```python
from sklearn.linear_model import LinearRegression

def fill_missing_multivariate(df, target_var, predictor_vars):
    """
    使用多元线性回归填补缺失值

    例如：用温度和辐射预测缺失的风速

    Parameters
    ----------
    df : pd.DataFrame
        包含所有变量的数据框
    target_var : str
        需要填补的目标变量
    predictor_vars : list of str
        用于预测的变量列表

    Returns
    -------
    df_filled : pd.DataFrame
        填补后的数据框
    """
    # 分离训练集（完整数据）和待填补集
    complete_mask = df[predictor_vars + [target_var]].notnull().all(axis=1)
    missing_mask = df[target_var].isnull() & df[predictor_vars].notnull().all(axis=1)

    if np.sum(missing_mask) == 0:
        return df

    # 训练模型
    X_train = df.loc[complete_mask, predictor_vars]
    y_train = df.loc[complete_mask, target_var]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测缺失值
    X_missing = df.loc[missing_mask, predictor_vars]
    y_pred = model.predict(X_missing)

    # 填补
    df_filled = df.copy()
    df_filled.loc[missing_mask, target_var] = y_pred

    print(f"使用 {predictor_vars} 填补了 {np.sum(missing_mask)} 个 {target_var} 的缺失值")
    print(f"模型 R² = {model.score(X_train, y_train):.3f}")

    return df_filled

# 使用示例
df_filled = fill_missing_multivariate(
    df,
    target_var='u2',  # 填补风速
    predictor_vars=['T_mean', 'Rs']  # 使用温度和辐射
)
```

---

## 5. 时间序列对齐与重采样

### 5.1 时间对齐

**处理不同时区：**

```python
import pandas as pd

def align_to_utc(df, source_timezone='US/Pacific'):
    """
    将本地时间转换为 UTC

    Parameters
    ----------
    df : pd.DataFrame
        索引为日期时间的数据框
    source_timezone : str
        源时区（如 'US/Pacific', 'Europe/London'）

    Returns
    -------
    df_utc : pd.DataFrame
        UTC 时间的数据框
    """
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_localize(source_timezone).tz_convert('UTC')
    return df_utc
```

**对齐到日界线（UTC 0:00）：**

```python
def align_to_daily(df, method='mean'):
    """
    将子日尺度数据聚合为日数据

    Parameters
    ----------
    df : pd.DataFrame
        高频数据（如小时尺度）
    method : str
        聚合方法：'mean', 'sum', 'max', 'min'

    Returns
    -------
    df_daily : pd.DataFrame
        日尺度数据
    """
    if method == 'mean':
        df_daily = df.resample('D').mean()
    elif method == 'sum':
        df_daily = df.resample('D').sum()
    elif method == 'max':
        df_daily = df.resample('D').max()
    elif method == 'min':
        df_daily = df.resample('D').min()
    else:
        raise ValueError(f"Unknown method: {method}")

    return df_daily

# 使用示例
# 假设有小时数据
hourly_data = pd.DataFrame({
    'T': np.random.rand(24*365),
}, index=pd.date_range('2020-01-01', periods=24*365, freq='H'))

daily_data = align_to_daily(hourly_data, method='mean')
```

### 5.2 重采样到不同时间分辨率

**上采样（日 → 小时）：**

```python
def upsample_with_diurnal_cycle(daily_temp, method='sine'):
    """
    将日数据上采样为小时数据（考虑日变化）

    Parameters
    ----------
    daily_temp : pd.Series
        日平均温度
    method : str
        日变化模型：'sine', 'linear'

    Returns
    -------
    hourly_temp : pd.Series
        小时温度
    """
    # 创建小时索引
    hourly_index = pd.date_range(
        daily_temp.index[0],
        daily_temp.index[-1] + pd.Timedelta(days=1),
        freq='H',
        inclusive='left'
    )

    # 线性插值
    hourly_temp = daily_temp.reindex(hourly_index).interpolate(method='linear')

    if method == 'sine':
        # 叠加日变化（简化模型）
        hour_of_day = hourly_index.hour
        diurnal_cycle = 5 * np.sin((hour_of_day - 6) * np.pi / 12)  # 峰值在14:00
        hourly_temp += diurnal_cycle

    return hourly_temp
```

**下采样（小时 → 日）：**

```python
# 见上面的 align_to_daily 函数
```

### 5.3 闰年处理

```python
def remove_leap_days(df):
    """
    移除闰年的2月29日

    Parameters
    ----------
    df : pd.DataFrame
        包含闰年的数据框

    Returns
    -------
    df_no_leap : pd.DataFrame
        移除2月29日后的数据框
    """
    df_no_leap = df[~((df.index.month == 2) & (df.index.day == 29))]
    print(f"移除了 {len(df) - len(df_no_leap)} 个闰年日期")
    return df_no_leap

def expand_to_366days(data_365):
    """
    将365天数据扩展为366天（复制2月28日）

    Parameters
    ----------
    data_365 : np.ndarray or list
        长度为365的数据

    Returns
    -------
    data_366 : np.ndarray
        长度为366的数据
    """
    data_365 = np.asarray(data_365)
    # 在索引58（2月28日）后插入一个重复值
    data_366 = np.insert(data_365, 59, data_365[58])
    return data_366
```

---

## 6. 空间数据处理

### 6.1 坐标系统转换

```python
def convert_longitude_convention(lon):
    """
    转换经度表示法

    0-360° ↔ -180-180°

    Parameters
    ----------
    lon : float or np.ndarray
        经度值

    Returns
    -------
    lon_converted : float or np.ndarray
        转换后的经度
    """
    lon = np.asarray(lon)

    # 0-360 -> -180-180
    lon_converted = np.where(lon > 180, lon - 360, lon)

    # 如果需要反向转换：
    # lon_converted = np.where(lon < 0, lon + 360, lon)

    return lon_converted
```

### 6.2 网格重插值

**双线性插值到新网格：**

```python
from scipy.interpolate import RegularGridInterpolator

def regrid_data(data_old, lat_old, lon_old, lat_new, lon_new):
    """
    将数据从旧网格插值到新网格

    Parameters
    ----------
    data_old : np.ndarray (n_lat_old, n_lon_old)
        旧网格数据
    lat_old, lon_old : np.ndarray
        旧网格的坐标
    lat_new, lon_new : np.ndarray
        新网格的坐标

    Returns
    -------
    data_new : np.ndarray (n_lat_new, n_lon_new)
        新网格数据
    """
    # 创建插值函数
    interp_func = RegularGridInterpolator(
        (lat_old, lon_old),
        data_old,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # 创建新网格的坐标网格
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
    points_new = np.column_stack([
        lat_new_grid.ravel(),
        lon_new_grid.ravel()
    ])

    # 插值
    data_new = interp_func(points_new).reshape(len(lat_new), len(lon_new))

    return data_new
```

**使用 xarray 重插值（更简单）：**

```python
def regrid_xarray(ds, target_grid):
    """
    使用 xarray 重插值

    Parameters
    ----------
    ds : xr.Dataset
        源数据集
    target_grid : xr.Dataset
        目标网格（提供 lat/lon 坐标）

    Returns
    -------
    ds_regridded : xr.Dataset
        重插值后的数据集
    """
    ds_regridded = ds.interp(
        lat=target_grid.lat,
        lon=target_grid.lon,
        method='linear'
    )
    return ds_regridded
```

### 6.3 空间聚合

**计算区域平均：**

```python
def calculate_regional_mean(da, lat_bounds, lon_bounds, weights='cosine'):
    """
    计算区域平均（考虑纬度权重）

    Parameters
    ----------
    da : xr.DataArray
        数据数组（维度: time, lat, lon）
    lat_bounds : tuple
        纬度范围 (lat_min, lat_max)
    lon_bounds : tuple
        经度范围 (lon_min, lon_max)
    weights : str
        权重方案：'cosine'（余弦纬度权重）或 'equal'（等权）

    Returns
    -------
    regional_mean : xr.DataArray
        区域平均时间序列
    """
    # 选择区域
    da_region = da.sel(
        lat=slice(*lat_bounds),
        lon=slice(*lon_bounds)
    )

    if weights == 'cosine':
        # 计算纬度权重（因为网格单元面积随纬度变化）
        lat_weights = np.cos(np.deg2rad(da_region.lat))
        lat_weights = lat_weights / lat_weights.sum()

        # 加权平均
        regional_mean = (da_region * lat_weights).sum(dim=['lat', 'lon'])
    else:
        # 简单平均
        regional_mean = da_region.mean(dim=['lat', 'lon'])

    return regional_mean

# 使用示例
# 计算美国大平原的平均 ET0
great_plains_et = calculate_regional_mean(
    da,
    lat_bounds=(35, 45),
    lon_bounds=(-105, -95),
    weights='cosine'
)
```

---

## 7. 常见数据源接入

### 7.1 ERA5-Land

```python
def load_era5_land(filepath, variables, lat_range=None, lon_range=None):
    """
    加载 ERA5-Land 数据

    ERA5-Land 数据特点：
    - 分辨率：0.1° × 0.1°
    - 时间频率：小时
    - 变量命名：参考 Copernicus CDS

    Parameters
    ----------
    filepath : str
        ERA5-Land NetCDF 文件路径
    variables : list of str
        需要的变量列表，如：
        - 't2m': 2米温度 (K)
        - 'u10', 'v10': 10米风速分量 (m/s)
        - 'ssrd': 短波辐射 (J/m²)
        - 'd2m': 2米露点温度 (K)
    lat_range : tuple, optional
        纬度范围 (lat_min, lat_max)
    lon_range : tuple, optional
        经度范围 (lon_min, lon_max)

    Returns
    -------
    df : pd.DataFrame
        处理后的数据框（SI单位）
    """
    import xarray as xr

    # 读取数据
    ds = xr.open_dataset(filepath)

    # 空间子集
    if lat_range:
        ds = ds.sel(latitude=slice(*lat_range))
    if lon_range:
        ds = ds.sel(longitude=slice(*lon_range))

    # 提取变量并转换单位
    data = {}

    for var in variables:
        if var == 't2m':
            # 开尔文 -> 摄氏度
            data['T_mean'] = ds[var] - 273.15
        elif var == 'd2m':
            # 露点温度 -> 实际水汽压
            data['ea'] = 0.6108 * np.exp(17.27 * (ds[var] - 273.15) / ((ds[var] - 273.15) + 237.3))
        elif var == 'ssrd':
            # 累积辐射 (J/m²) -> 日平均 (MJ/m²/day)
            # 注意：ERA5 的 ssrd 是累积值，需要差分
            data['Rs'] = ds[var].diff('time') / 1e6  # J -> MJ
        elif var in ['u10', 'v10']:
            # 10米风速 -> 2米风速
            if 'u10' in variables and 'v10' in variables:
                u10 = ds['u10']
                v10 = ds['v10']
                wind_10m = np.sqrt(u10**2 + v10**2)
                # 使用对数风廓线调整
                from src.penman_monteith import adjust_wind_speed
                data['u2'] = adjust_wind_speed(wind_10m, 10, 2)

    # 转换为 DataFrame
    df = xr.Dataset(data).to_dataframe()

    return df
```

### 7.2 gridMET

```python
def load_gridmet(base_path, year_start, year_end, lat, lon):
    """
    加载 gridMET 数据（美国本土高分辨率数据）

    gridMET 数据特点：
    - 分辨率：约 4 km
    - 覆盖范围：CONUS（美国本土）
    - 时间范围：1979-present
    - 已计算好 ET0！

    Parameters
    ----------
    base_path : str
        gridMET 数据目录
    year_start, year_end : int
        时间范围
    lat, lon : float
        目标位置

    Returns
    -------
    df : pd.DataFrame
        包含所有气象变量和 ET0
    """
    import xarray as xr
    from pathlib import Path

    variables = {
        'tmmx': 'T_max',
        'tmmn': 'T_min',
        'srad': 'Rs',
        'vs': 'u2',
        'etr': 'ET0'  # gridMET 已经计算好的ET0
    }

    data = {}

    for var_gridmet, var_name in variables.items():
        files = [
            f"{base_path}/{var_gridmet}_{year}.nc"
            for year in range(year_start, year_end + 1)
        ]

        # 打开多文件
        ds = xr.open_mfdataset(files, combine='by_coords')

        # 提取时间序列
        series = ds[var_gridmet].sel(
            lat=lat, lon=lon, method='nearest'
        ).values

        data[var_name] = series

    # 创建 DataFrame
    time_index = pd.date_range(
        f"{year_start}-01-01",
        f"{year_end}-12-31",
        freq='D'
    )
    df = pd.DataFrame(data, index=time_index)

    # 单位转换
    df['T_max'] = df['T_max'] - 273.15  # K -> °C
    df['T_min'] = df['T_min'] - 273.15
    df['Rs'] = df['Rs'] * 0.0864  # W/m² -> MJ/m²/day

    return df
```

### 7.3 站点观测数据

```python
def load_station_data(filepath, format='csv'):
    """
    加载站点观测数据

    支持格式：
    - CSV（通用）
    - GHCN-Daily（NOAA 全球历史气候学网络）
    - CIMIS（加州灌溉管理信息系统）

    Parameters
    ----------
    filepath : str
        数据文件路径
    format : str
        数据格式

    Returns
    -------
    df : pd.DataFrame
        标准化的数据框
    """
    if format == 'csv':
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

        # 假设列名映射
        column_mapping = {
            'temp_max': 'T_max',
            'temp_min': 'T_min',
            'solar_rad': 'Rs',
            'wind_speed': 'u2',
            'rel_humidity': 'RH'
        }
        df = df.rename(columns=column_mapping)

        # 如果有相对湿度，转换为水汽压
        if 'RH' in df.columns and 'T_mean' in df.columns:
            es = 0.6108 * np.exp(17.27 * df['T_mean'] / (df['T_mean'] + 237.3))
            df['ea'] = es * df['RH'] / 100

    elif format == 'ghcn':
        # GHCN-Daily 格式较复杂，需要特殊解析
        # 参考：https://www.ncei.noaa.gov/data/ghcn-daily/doc/
        pass  # 实现略

    return df
```

---

## 8. 完整工作流示例

### 8.1 从原始 ERA5-Land 到极端检测

```python
import numpy as np
import pandas as pd
import xarray as xr
from src.penman_monteith import calculate_et0
from src.extreme_detection import detect_extreme_events_clim
from src.contribution_analysis import calculate_contributions

# ========== 步骤 1: 加载数据 ==========
print("正在加载 ERA5-Land 数据...")
ds = xr.open_dataset('era5_land_2000_2020.nc')

# 提取洛杉矶的时间序列
lat_target, lon_target = 34.05, -118.24
ds_site = ds.sel(lat=lat_target, lon=lon_target, method='nearest')

# ========== 步骤 2: 单位转换 ==========
print("转换单位...")
T_max = ds_site['t2m_max'].values - 273.15  # K -> °C
T_min = ds_site['t2m_min'].values - 273.15
T_mean = (T_max + T_min) / 2
Rs = ds_site['ssrd'].values / 1e6  # J/m² -> MJ/m²
u10 = np.sqrt(ds_site['u10']**2 + ds_site['v10']**2).values

# 风速高度调整
from src.penman_monteith import adjust_wind_speed
u2 = adjust_wind_speed(u10, z_measurement=10, z_target=2)

# 露点温度 -> 水汽压
Td = ds_site['d2m'].values - 273.15
ea = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))

# ========== 步骤 3: 数据清洗 ==========
print("数据清洗...")
# 检查物理范围
valid = (
    (T_mean >= -50) & (T_mean <= 50) &
    (Rs >= 0) & (Rs <= 40) &
    (u2 >= 0) & (u2 <= 30) &
    (ea >= 0) & (ea <= 7)
)

print(f"移除了 {np.sum(~valid)} 个无效数据点")

T_mean[~valid] = np.nan
T_max[~valid] = np.nan
T_min[~valid] = np.nan
Rs[~valid] = np.nan
u2[~valid] = np.nan
ea[~valid] = np.nan

# 线性插值填补小缺失
from scipy.interpolate import interp1d
for var in [T_mean, T_max, T_min, Rs, u2, ea]:
    valid_idx = ~np.isnan(var)
    if np.sum(valid_idx) > 10:
        f = interp1d(
            np.arange(len(var))[valid_idx],
            var[valid_idx],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        var[:] = f(np.arange(len(var)))

# ========== 步骤 4: 计算 ET0 ==========
print("计算 ET0...")
et0 = calculate_et0(
    T_mean=T_mean,
    T_max=T_max,
    T_min=T_min,
    Rs=Rs,
    u2=u2,
    ea=ea,
    z=50.0,
    latitude=lat_target
)

# ========== 步骤 5: 极端事件检测 ==========
print("检测极端事件...")
extreme_mask, thresholds = detect_extreme_events_clim(
    et0,
    severity=0.05,
    min_duration=3
)

print(f"检测到 {np.sum(extreme_mask)} 个极端天数")

# ========== 步骤 6: 驱动因子分析 ==========
print("分析驱动因子...")
contributions = calculate_contributions(
    T_mean=T_mean,
    T_max=T_max,
    T_min=T_min,
    Rs=Rs,
    u2=u2,
    ea=ea,
    extreme_mask=extreme_mask,
    z=50.0,
    latitude=lat_target
)

print("\n驱动因子贡献率：")
for factor, contrib in contributions.items():
    print(f"  {factor:15s}: {contrib:5.1f}%")

# ========== 步骤 7: 可视化 ==========
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1: ET0 时间序列
axes[0].plot(et0, color='steelblue', linewidth=0.5, alpha=0.7)
axes[0].scatter(np.where(extreme_mask)[0], et0[extreme_mask],
               color='red', s=10, zorder=5, label='Extreme Events')
axes[0].set_ylabel('ET₀ (mm/day)')
axes[0].set_title('Los Angeles: ET₀ Time Series (2000-2020)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 子图2: 气候学阈值
doy = np.arange(365)
axes[1].plot(doy, thresholds, color='orange', linewidth=2)
axes[1].set_xlabel('Day of Year')
axes[1].set_ylabel('Threshold (mm/day)')
axes[1].set_title('Climatological Threshold (95th Percentile)')
axes[1].grid(alpha=0.3)

# 子图3: 贡献率饼图
axes[2].pie(
    contributions.values(),
    labels=contributions.keys(),
    autopct='%1.1f%%',
    startangle=90
)
axes[2].set_title('Driver Contributions to Extreme ET Events')

plt.tight_layout()
plt.savefig('extreme_et_analysis.png', dpi=300)
print("\n图表已保存至 extreme_et_analysis.png")
```

---

## 9. 常见问题 (FAQ)

### Q1: 如何处理亚日尺度（小时）数据？

**A**: 首先聚合到日尺度：

```python
df_daily = df_hourly.resample('D').agg({
    'T': 'mean',
    'T_max': 'max',
    'T_min': 'min',
    'Rs': 'sum',  # 辐射需要累积
    'u2': 'mean',
    'ea': 'mean'
})

# 注意：辐射单位需要从 W/m² 转换为 MJ/m²/day
df_daily['Rs'] = df_daily['Rs'] * 3600 / 1e6  # W·h/m² -> MJ/m²
```

### Q2: 如何从相对湿度计算实际水汽压？

**A**: 使用以下公式：

```python
def rh_to_ea(T, RH):
    """
    从相对湿度计算实际水汽压

    Parameters
    ----------
    T : float or array-like
        气温 (°C)
    RH : float or array-like
        相对湿度 (%)

    Returns
    -------
    ea : float or array-like
        实际水汽压 (kPa)
    """
    # 饱和水汽压（Tetens 公式）
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))

    # 实际水汽压
    ea = es * RH / 100

    return ea
```

### Q3: 数据量太大，内存不够怎么办？

**A**: 使用分块处理：

```python
def process_large_dataset_chunked(filepath, chunk_size=365*5):
    """
    分块处理大型 NetCDF 文件

    Parameters
    ----------
    filepath : str
        NetCDF 文件路径
    chunk_size : int
        每块的时间步数（默认5年）

    Yields
    ------
    chunk_result : dict
        每块的处理结果
    """
    ds = xr.open_dataset(filepath, chunks={'time': chunk_size})

    n_chunks = len(ds.time) // chunk_size + 1

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(ds.time))

        # 加载当前块到内存
        chunk = ds.isel(time=slice(start_idx, end_idx)).load()

        # 处理...
        et0_chunk = calculate_et0(...)
        extreme_mask_chunk = detect_extreme_events_hist(et0_chunk)

        yield {
            'time': chunk.time.values,
            'et0': et0_chunk,
            'extreme_mask': extreme_mask_chunk
        }

        # 清理内存
        del chunk
        import gc
        gc.collect()
```

### Q4: 如何验证我的数据处理是否正确？

**A**: 使用以下检查清单：

```python
def validate_processed_data(df):
    """
    验证处理后的数据质量

    Returns
    -------
    is_valid : bool
        数据是否通过所有检查
    """
    checks = []

    # 1. 温度逻辑性
    checks.append(
        np.all(df['T_min'] <= df['T_mean']) and
        np.all(df['T_mean'] <= df['T_max'])
    )

    # 2. 物理范围
    checks.append(np.all((df['Rs'] >= 0) & (df['Rs'] <= 40)))
    checks.append(np.all((df['u2'] >= 0) & (df['u2'] <= 30)))
    checks.append(np.all((df['ea'] >= 0) & (df['ea'] <= 7)))

    # 3. 缺失值比例
    missing_rate = df.isnull().sum() / len(df)
    checks.append(np.all(missing_rate < 0.1))  # <10% 缺失

    # 4. 时间连续性
    time_diff = df.index.to_series().diff()
    checks.append(np.all(time_diff[1:] == pd.Timedelta(days=1)))

    # 5. 合理的季节性
    monthly_mean = df['T_mean'].groupby(df.index.month).mean()
    seasonal_range = monthly_mean.max() - monthly_mean.min()
    checks.append(seasonal_range > 5)  # 至少5°C季节差异

    is_valid = all(checks)

    if not is_valid:
        print("数据验证失败！")
        for i, check in enumerate(checks, 1):
            status = "✓" if check else "✗"
            print(f"  检查 {i}: {status}")

    return is_valid
```

---

## 10. 参考资源

### 数据源

1. **ERA5-Land**: https://cds.climate.copernicus.eu/
2. **gridMET**: https://www.climatologylab.org/gridmet.html
3. **PRISM**: https://prism.oregonstate.edu/
4. **Daymet**: https://daymet.ornl.gov/

### 工具文档

1. **xarray**: https://docs.xarray.dev/
2. **pandas**: https://pandas.pydata.org/docs/
3. **netCDF4-python**: https://unidata.github.io/netcdf4-python/

### 推荐阅读

1. **Hersbach et al. (2020)**. The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*.
2. **Abatzoglou, J. T. (2013)**. Development of gridded surface meteorological data for ecological applications and modelling. *International Journal of Climatology*.
