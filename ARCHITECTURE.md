# Extreme-ET 代码架构文档 / Code Architecture Documentation

## 目录 / Table of Contents
1. [项目概述 / Project Overview](#项目概述--project-overview)
2. [系统架构 / System Architecture](#系统架构--system-architecture)
3. [模块说明 / Module Description](#模块说明--module-description)
4. [数据流 / Data Flow](#数据流--data-flow)
5. [设计模式 / Design Patterns](#设计模式--design-patterns)
6. [性能考虑 / Performance Considerations](#性能考虑--performance-considerations)

---

## 项目概述 / Project Overview

### 项目名称 / Project Name
**Extreme-ET: 极端蒸发事件分析工具包**
**Extreme-ET: Extreme Evaporation Events Analysis Toolkit**

### 版本 / Version
v0.1.0

### 目标 / Purpose
实现两篇顶级期刊论文中提出的极端蒸发事件检测和分析方法：
Implements extreme evaporation event detection and analysis methods from two top-tier journal papers:

1. **Markonis (2025)** - "On the Definition of Extreme Evaporation Events" (GRL)
2. **Zhao et al. (2025)** - "Regional variations in drivers of extreme reference evapotranspiration" (WRR)

### 技术栈 / Tech Stack
- **语言 / Language**: Python 3.8+
- **核心依赖 / Core Dependencies**:
  - NumPy (数值计算 / numerical computing)
  - Pandas (数据处理 / data processing)
  - SciPy (科学计算 / scientific computing)
  - Matplotlib/Seaborn (可视化 / visualization)

---

## 系统架构 / System Architecture

### 整体架构图 / Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Extreme-ET Toolkit                          │
│                  (极端蒸发事件分析工具包)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├── src/  (核心包 / Core Package)
                              │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   extreme_   │    │   penman_    │    │ contribution │
│  detection   │    │   monteith   │    │  _analysis   │
│              │    │              │    │              │
│ 极端事件检测   │    │  ET0计算     │    │  驱动因子分析  │
│              │    │              │    │              │
│  • ERT_hist  │    │ • ASCE-PM    │    │ • 敏感性分析  │
│  • ERT_clim  │    │ • 净辐射计算  │    │ • 季节贡献    │
│  • OPT       │    │ • 气压修正    │    │ • 主导因子    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    data_     │    │    utils     │    │   examples   │
│  processing  │    │              │    │              │
│              │    │ 数据生成工具   │    │  使用示例     │
│ 数据预处理    │    │              │    │              │
│              │    │ • 合成数据    │    │ • Zhao 2025  │
│ • Z标准化    │    │ • 可视化      │    │ • Markonis   │
│ • Hurst指数  │    │ • 事件统计    │    │   2025       │
│ • 移动平均    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 层次结构 / Hierarchical Structure

```
1. 表示层 (Presentation Layer)
   ├── examples/ - 示例脚本 / Example scripts
   └── 可视化函数 / Visualization functions

2. 应用层 (Application Layer)
   ├── 极端事件检测 / Extreme event detection
   ├── 驱动因子分析 / Driver contribution analysis
   └── 数据处理 / Data processing

3. 核心层 (Core Layer)
   ├── Penman-Monteith 方程 / Penman-Monteith equation
   ├── 统计分析 / Statistical analysis
   └── 数学计算 / Mathematical calculations

4. 基础层 (Foundation Layer)
   └── NumPy, Pandas, SciPy
```

---

## 模块说明 / Module Description

### 1. extreme_detection.py (347 → 990 行)

**功能 / Functionality:**
实现三种极端事件检测方法
Implements three extreme event detection methods

**核心类/函数 / Core Classes/Functions:**

| 函数名 | 功能 | 复杂度 | 用途 |
|--------|------|--------|------|
| `detect_extreme_events_hist()` | 历史相对阈值法 | O(n log n) | 识别破纪录极值 |
| `detect_extreme_events_clim()` | 气候学方法 | O(n × k) | 识别季节异常 |
| `optimal_path_threshold()` | 最优路径阈值 | O(n × k) | 优化阈值选择 |
| `identify_climatological_extremes()` | 应用日阈值 | O(n) | 识别持续事件 |
| `identify_events_from_mask()` | 提取事件时段 | O(n) | 事件分割 |
| `calculate_event_statistics()` | 事件统计 | O(m × d) | 强度分析 |

**设计模式 / Design Patterns:**
- 策略模式 (Strategy Pattern): 三种检测方法可互换使用
- 模板方法模式 (Template Method): 共享的验证和后处理流程

**关键算法 / Key Algorithms:**

1. **ERT_hist 算法:**
```
Input: data[n], severity
1. threshold ← quantile(data, 1 - severity)
2. extreme_mask ← data > threshold
3. return extreme_mask, threshold
```

2. **ERT_clim 算法:**
```
Input: data[n], severity, min_duration
1. smoothed_data ← moving_average(data)
2. FOR each day d in [0, 365]:
       thresholds[d] ← optimize_threshold(d)
3. extreme_mask ← apply_thresholds(smoothed_data, thresholds)
4. filter by min_duration
5. return extreme_mask, thresholds
```

3. **OPT 优化算法:**
```
Input: data, target_rate
1. Initialize: thresholds ← 90th percentile for each day
2. REPEAT until converge or max_iterations:
       current_rate ← test(thresholds)
       IF |current_rate - target_rate| < tolerance:
           break
       ELSE IF current_rate < target_rate:
           thresholds ← thresholds × 0.95  # 降低
       ELSE:
           thresholds ← thresholds × 1.05  # 提高
3. return thresholds
```

---

### 2. penman_monteith.py (246 行)

**功能 / Functionality:**
实现 ASCE 标准化 Penman-Monteith 方程计算参考蒸散发
Implements ASCE Standardized Penman-Monteith equation for ET0

**核心方程 / Core Equations:**

```
ET0 = [0.408 × Δ × (Rn - G) + γ × (Cn/(T+273)) × u2 × (es - ea)]
      ─────────────────────────────────────────────────────────
                    Δ + γ × (1 + Cd × u2)

其中 / Where:
- Δ: 饱和水汽压曲线斜率 / slope of saturation vapor pressure curve (kPa/°C)
- Rn: 净辐射 / net radiation (MJ m⁻² day⁻¹)
- G: 土壤热通量 / soil heat flux (≈0 for daily)
- γ: 干湿表常数 / psychrometric constant (kPa/°C)
- T: 平均温度 / mean temperature (°C)
- u2: 2米高风速 / wind speed at 2m height (m/s)
- es: 饱和水汽压 / saturation vapor pressure (kPa)
- ea: 实际水汽压 / actual vapor pressure (kPa)
- Cn: 分子常数 / numerator constant (900 for grass)
- Cd: 分母常数 / denominator constant (0.34 for grass)
```

**函数关系图 / Function Dependency:**

```
calculate_et0()
    ├── calculate_net_radiation()
    │   ├── calculate_extraterrestrial_radiation()
    │   └── Stefan-Boltzmann radiation
    ├── calculate_vapor_pressure_from_vpd()
    └── adjust_wind_speed()
```

**输入要求 / Input Requirements:**

| 变量 | 单位 | 典型范围 | 必需性 |
|------|------|----------|--------|
| T_mean | °C | -10 to 40 | 必需 |
| T_max | °C | -5 to 45 | 必需 |
| T_min | °C | -15 to 35 | 必需 |
| Rs | MJ/m²/day | 0 to 35 | 必需 |
| u2 | m/s | 0.5 to 10 | 必需 |
| ea | kPa | 0.1 to 5 | 必需 |
| z | m | 0 to 5000 | 可选 (默认50) |
| latitude | degrees | -90 to 90 | 可选 (默认40) |
| doy | 1-365 | 1 to 365 | 可选 |

**物理意义 / Physical Meaning:**
- ET0 表示充足供水条件下草地参考面的蒸发潜力
- ET0 represents evaporation potential of well-watered grass reference surface
- 受能量供应（辐射、温度）和蒸发能力（风速、湿度）共同控制
- Controlled by energy supply (radiation, temperature) and evaporation capacity (wind, humidity)

---

### 3. contribution_analysis.py (251 行)

**功能 / Functionality:**
量化各气象驱动因子对极端ET0的相对贡献
Quantifies relative contribution of meteorological drivers to extreme ET0

**核心方法 / Core Method:**

**单因子敏感性实验 (Single-factor sensitivity experiment):**

```
方法原理 / Methodology:
1. 计算原始 ET0: ET0_orig = f(T, Rs, u2, ea)
2. 逐个替换为气候态:
   ET0_temp_clim  = f(T_clim, Rs,      u2,     ea)      # 温度替换
   ET0_rad_clim   = f(T,      Rs_clim, u2,     ea)      # 辐射替换
   ET0_wind_clim  = f(T,      Rs,      u2_clim, ea)     # 风速替换
   ET0_hum_clim   = f(T,      Rs,      u2,     ea_clim) # 湿度替换

3. 计算差异:
   diff_temp = ∑(ET0_orig - ET0_temp_clim)[extreme_days]
   diff_rad  = ∑(ET0_orig - ET0_rad_clim)[extreme_days]
   diff_wind = ∑(ET0_orig - ET0_wind_clim)[extreme_days]
   diff_hum  = ∑(ET0_orig - ET0_hum_clim)[extreme_days]

4. 计算贡献率:
   RC_i = diff_i / ∑(all diffs) × 100%
```

**典型贡献比例 / Typical Contribution Ratios:**
- 温度 / Temperature: 30-60% (通常最高 / usually dominant)
- 辐射 / Radiation: 20-40%
- 风速 / Wind: 10-25%
- 湿度 / Humidity: 5-20%

**季节差异 / Seasonal Differences:**
- 夏季 / Summer: 辐射和温度主导 / radiation and temperature dominate
- 冬季 / Winter: 风速和湿度相对重要 / wind and humidity relatively important
- 春秋 / Spring/Fall: 过渡特征 / transitional characteristics

---

### 4. data_processing.py (235 行)

**功能 / Functionality:**
数据预处理和统计分析工具
Data preprocessing and statistical analysis tools

**核心功能 / Core Functions:**

#### 4.1 Z-score 标准化

**目的 / Purpose:**
去除季节性，使数据可比
Remove seasonality to make data comparable

**方法 / Method:**
```
对于每个 pentad/day:
For each pentad/day:
    z[i] = (x[i] - mean[pentad]) / std[pentad]
```

**应用场景 / Use Cases:**
- Markonis (2025) 方法的预处理步骤
- 比较不同季节的极端程度
- 长期趋势分析

#### 4.2 Hurst 指数计算

**定义 / Definition:**
描述时间序列的长程相关性
Describes long-range dependence of time series

**物理意义 / Physical Meaning:**
```
H ≈ 0.5: 白噪声，无记忆 / white noise, no memory
H > 0.5: 长期持续性，事件聚类 / long-term persistence, event clustering
H < 0.5: 反持续性 / anti-persistence
```

**计算方法 / Calculation Method:**
```
1. 计算自相关函数 ACF(k)
2. 拟合: log(ACF(k)) ~ (2H - 2) × log(k)
3. 求解: H = (slope + 2) / 2
```

#### 4.3 移动平均

**目的 / Purpose:**
平滑短期波动，保留趋势
Smooth short-term fluctuations, preserve trends

**实现 / Implementation:**
```python
# 使用 pandas 高效实现
series.rolling(window=window, center=True, min_periods=1).mean()
```

---

### 5. utils.py (360 行)

**功能 / Functionality:**
数据生成、可视化和辅助工具
Data generation, visualization, and utility tools

**主要组件 / Main Components:**

#### 5.1 合成数据生成

**生成特征 / Generated Features:**
```python
data[i] = baseline + seasonal(i) + trend(i) + noise(i) + extreme_events(i)

其中 / Where:
- seasonal(i) = A × sin(2π × i / 365 + φ)
- trend(i) = k × year(i)
- noise(i) ~ N(0, σ²)
- extreme_events: 1% 的日子有极端值增强
```

**用途 / Use Cases:**
- 方法测试和验证
- 教学演示
- 敏感性测试

#### 5.2 可视化函数

| 函数名 | 绘图类型 | 用途 |
|--------|----------|------|
| `plot_extreme_events()` | 时间序列 + 标记 | 展示极端事件分布 |
| `plot_contribution_pie()` | 饼图 | 显示贡献比例 |
| `plot_seasonal_contributions()` | 柱状图 | 季节对比 |
| `plot_autocorrelation()` | 茎叶图 | ACF 分析 |

#### 5.3 事件度量

**计算指标 / Calculated Metrics:**
- 事件数量 / Number of events
- 平均持续时间 / Mean duration
- 最长持续时间 / Max duration
- 平均强度 / Mean intensity
- 峰值强度 / Peak intensity

---

## 数据流 / Data Flow

### 完整工作流程 / Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 数据准备 / Stage 1: Data Preparation                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        输入气象数据 / Input Meteorological Data
        ├── T_mean, T_max, T_min (°C)
        ├── Rs (MJ m⁻² day⁻¹)
        ├── u2 (m s⁻¹)
        └── ea (kPa)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: ET0 计算 / Stage 2: ET0 Calculation                 │
└─────────────────────────────────────────────────────────────┘
                              │
                      penman_monteith.py
                              │
                              ▼
                    ET0 时间序列 / ET0 Time Series
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3A: 极端事件检测 / Stage 3A: Extreme Event Detection   │
└─────────────────────────────────────────────────────────────┘
                              │
                    extreme_detection.py
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
     ERT_hist           ERT_clim              OPT
          │                  │                  │
          └──────────────────┴──────────────────┘
                              │
                              ▼
                极端事件掩码 / Extreme Event Mask
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3B: 数据预处理 (可选) / Stage 3B: Preprocessing (Optional) │
└─────────────────────────────────────────────────────────────┘
                              │
                    data_processing.py
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    Z-score 标准化      Hurst 指数          移动平均
    Z-score Norm      Hurst Exponent     Moving Avg
          │                  │                  │
          └──────────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 4: 驱动因子分析 / Stage 4: Driver Attribution          │
└─────────────────────────────────────────────────────────────┘
                              │
                  contribution_analysis.py
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      总体贡献           季节贡献            敏感性分析
    Overall Contrib    Seasonal Contrib    Sensitivity
          │                  │                  │
          └──────────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 5: 结果输出 / Stage 5: Results Output                   │
└─────────────────────────────────────────────────────────────┘
                              │
                         utils.py
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      时间序列图           贡献饼图            统计表格
      Time Series        Contribution       Statistics
        Plot               Pie Chart           Table
```

### 数据类型流转 / Data Type Flow

```
Raw Data (list/array/Series)
    ↓ [type conversion]
np.ndarray (float64)
    ↓ [calculation]
ET0: np.ndarray (float64, unit: mm/day)
    ↓ [detection]
extreme_mask: np.ndarray (bool)
    ↓ [event identification]
events: List[Dict]
    ↓ [statistics]
event_stats: List[Dict]
    ↓ [visualization]
matplotlib.Figure
```

---

## 设计模式 / Design Patterns

### 1. 策略模式 (Strategy Pattern)

**应用场景 / Application:**
极端事件检测方法的选择

```python
# 三种策略可互换
# Three strategies are interchangeable
strategy = "ERT_hist"  # or "ERT_clim" or "OPT"

if strategy == "ERT_hist":
    mask, thresh = detect_extreme_events_hist(data)
elif strategy == "ERT_clim":
    mask, thresh = detect_extreme_events_clim(data)
else:
    thresh = optimal_path_threshold(data)
    mask = identify_climatological_extremes(data, thresh)
```

### 2. 模板方法模式 (Template Method Pattern)

**应用场景 / Application:**
共享的输入验证流程

```python
def detection_method(data, ...):
    # 1. 输入验证 (共享步骤)
    # Input validation (shared step)
    data = validate_input(data)

    # 2. 核心算法 (子类特定)
    # Core algorithm (subclass-specific)
    result = _core_algorithm(data, ...)

    # 3. 后处理 (共享步骤)
    # Post-processing (shared step)
    return post_process(result)
```

### 3. 工厂模式 (Factory Pattern)

**应用场景 / Application:**
合成数据生成

```python
def generate_synthetic_data(n_days, add_trend, add_seasonality):
    # 工厂方法根据参数生成不同特征的数据
    # Factory method generates data with different characteristics
    data = {}
    data['T_mean'] = _generate_temperature(n_days, add_trend, add_seasonality)
    data['Rs'] = _generate_radiation(n_days, add_trend, add_seasonality)
    # ...
    return data
```

### 4. 装饰器模式 (Decorator Pattern)

**应用场景 / Application:**
详细信息返回选项

```python
@optional_details
def detect_extreme_events(data, return_details=False):
    # 核心计算
    mask, threshold = _core_detection(data)

    # 装饰器根据 return_details 决定是否添加详细信息
    if return_details:
        details = calculate_details(mask)
        return mask, threshold, details
    return mask, threshold
```

---

## 性能考虑 / Performance Considerations

### 计算复杂度总结 / Computational Complexity Summary

| 操作 / Operation | 复杂度 / Complexity | 瓶颈 / Bottleneck |
|------------------|---------------------|-------------------|
| ET0 calculation | O(n) | 向量化 NumPy 操作 |
| ERT_hist | O(n log n) | np.quantile() |
| ERT_clim | O(n × k × 366) | OPT 迭代 |
| OPT | O(n × k) | 收敛迭代 |
| Contribution analysis | O(n × m) | 多次 ET0 计算 |
| Z-score | O(n) | 向量化操作 |
| Hurst exponent | O(n × lag) | 相关性计算 |

**说明 / Notes:**
- n: 数据点数量 / number of data points
- k: OPT 迭代次数 / OPT iterations (~10-20)
- m: 气象变量数量 / number of meteorological variables (4-6)
- lag: 最大滞后步数 / maximum lag steps (~10)

### 优化建议 / Optimization Recommendations

#### 1. 向量化优先 / Vectorization First
```python
# ❌ 避免循环 / Avoid loops
for i in range(len(data)):
    result[i] = calculate(data[i])

# ✅ 使用向量化 / Use vectorization
result = calculate(data)  # NumPy vectorized operation
```

#### 2. 内存管理 / Memory Management
```python
# 大数据集：分块处理
# Large datasets: process in chunks
chunk_size = 365 * 10  # 10 years
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
```

#### 3. 缓存重复计算 / Cache Repeated Calculations
```python
# 缓存气候态值
# Cache climatological values
@lru_cache(maxsize=366)
def get_climatology(day_of_year):
    return calculate_climatology(day_of_year)
```

#### 4. 并行处理 / Parallel Processing
```python
# 对于独立的季节分析
# For independent seasonal analysis
from multiprocessing import Pool

with Pool(processes=4) as pool:
    seasonal_results = pool.map(analyze_season, seasons)
```

### 内存使用估算 / Memory Usage Estimation

**典型 40 年数据集 / Typical 40-year dataset:**

```
数据点数 / Data points: 14,600 (40 × 365)

内存占用 / Memory footprint:
- 输入气象数据 (6 变量): 6 × 14,600 × 8 bytes ≈ 0.7 MB
- ET0 时间序列: 14,600 × 8 bytes ≈ 0.1 MB
- 极端事件掩码: 14,600 × 1 byte ≈ 0.01 MB
- 阈值数组 (366 天): 366 × 8 bytes ≈ 0.003 MB

总计 / Total: < 1 MB (非常轻量 / very lightweight)
```

### 运行时间估算 / Runtime Estimation

**标准工作站 (Standard workstation) - Intel i7, 16GB RAM:**

| 操作 / Operation | 数据规模 / Data Size | 估计时间 / Estimated Time |
|------------------|---------------------|-------------------------|
| ET0 calculation | 40 years | < 0.1 s |
| ERT_hist | 40 years | < 0.1 s |
| ERT_clim | 40 years | 1-5 s |
| Contribution analysis | 40 years | 0.5-1 s |
| Complete workflow | 40 years | < 10 s |

---

## 扩展性考虑 / Extensibility Considerations

### 1. 添加新检测方法 / Adding New Detection Methods

```python
# 步骤 / Steps:
# 1. 在 extreme_detection.py 中添加新函数
# 2. 遵循相同的输入验证模式
# 3. 返回统一的输出格式
# 4. 添加到 __all__ 导出列表

def detect_extreme_events_custom(data, **params):
    """自定义检测方法 / Custom detection method"""
    # 验证输入 / Validate input
    data = validate_input(data)

    # 核心算法 / Core algorithm
    extreme_mask = custom_algorithm(data, **params)

    # 返回统一格式 / Return uniform format
    return extreme_mask, details
```

### 2. 支持新气象变量 / Supporting New Meteorological Variables

```python
# 在 contribution_analysis.py 中:
# In contribution_analysis.py:

def calculate_contributions_extended(data, extreme_mask,
                                    additional_vars=None):
    """扩展版本支持额外变量 / Extended version supports extra variables"""
    base_contrib = calculate_contributions(data, extreme_mask)

    if additional_vars:
        for var_name, var_data in additional_vars.items():
            contrib = calculate_single_contribution(var_name, var_data)
            base_contrib[var_name] = contrib

    return normalize_contributions(base_contrib)
```

### 3. 添加新可视化类型 / Adding New Visualization Types

```python
# 在 utils.py 中遵循命名约定:
# Follow naming convention in utils.py:

def plot_NEWTYPE(data, **kwargs):
    """
    创建 NEWTYPE 可视化 / Create NEWTYPE visualization

    Parameters: 遵循一致的参数风格
    Returns: fig, ax (matplotlib objects)
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    # ... 绘图逻辑
    return fig, ax
```

---

## 测试策略 / Testing Strategy

### 单元测试覆盖 / Unit Test Coverage

```python
tests/test_methods.py 包含:
tests/test_methods.py contains:

1. test_data_processing()
   - Z-score 标准化正确性
   - Hurst 指数边界情况
   - 移动平均边界处理

2. test_extreme_detection()
   - 三种方法的发生率准确性
   - 边界条件处理
   - 事件连续性识别

3. test_penman_monteith()
   - ET0 计算准确性
   - 物理合理性检查
   - 极端输入处理

4. test_contribution_analysis()
   - 贡献总和 = 100%
   - 季节差异合理性
   - 极端情况处理

5. test_integration()
   - 完整工作流可执行
   - 结果一致性
   - 输出格式正确性
```

### 验证方法 / Validation Methods

1. **理论验证 / Theoretical Validation:**
   - 对比已发表论文的结果
   - 检查物理合理性

2. **数值验证 / Numerical Validation:**
   - 使用标准测试数据集
   - 对比其他软件包结果 (如 PyETo)

3. **统计验证 / Statistical Validation:**
   - 检查事件发生率
   - 验证贡献比例范围

---

## 版本历史 / Version History

| 版本 / Version | 日期 / Date | 主要变更 / Major Changes |
|----------------|-------------|-------------------------|
| 0.1.0 | 2025-01 | 初始发布 / Initial release |
| 1.0.0 | 2025-10 | 代码重构，添加详细注释和文档 / Code refactoring, detailed comments and documentation |

---

## 贡献指南 / Contributing Guidelines

### 代码风格 / Code Style
- 遵循 PEP 8
- 使用类型提示 (Type hints)
- 编写详细的 docstrings (NumPy 格式)
- 中英文双语注释

### 文档要求 / Documentation Requirements
- 每个函数必须有完整的 docstring
- 包含算法说明和示例
- 说明计算复杂度
- 标注参考文献

### Pull Request 流程 / Pull Request Process
1. Fork 仓库
2. 创建功能分支
3. 添加测试
4. 更新文档
5. 提交 PR

---

## 联系信息 / Contact Information

- **项目主页 / Project Home**: https://github.com/your-org/Extreme-ET
- **问题反馈 / Issue Tracker**: https://github.com/your-org/Extreme-ET/issues
- **文档 / Documentation**: https://extreme-et.readthedocs.io

---

## 许可证 / License

MIT License - 详见 LICENSE 文件 / See LICENSE file for details

---

## 致谢 / Acknowledgments

本工具包实现了以下论文中提出的方法：
This toolkit implements methods proposed in the following papers:

1. Markonis, Y. (2025). On the Definition of Extreme Evaporation Events. *Geophysical Research Letters*.

2. Zhao et al. (2025). Regional variations in drivers of extreme reference evapotranspiration across the contiguous United States. *Water Resources Research*.

---

**文档版本 / Document Version**: 1.0
**最后更新 / Last Updated**: 2025-10-28
**维护者 / Maintainer**: Extreme-ET Team
