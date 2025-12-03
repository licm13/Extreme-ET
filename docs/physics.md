# 物理原理篇 (Physics Principles)

## 目录 (Table of Contents)

1. [Penman-Monteith 方程详解](#1-penman-monteith-方程详解)
2. [辐射平衡与净辐射](#2-辐射平衡与净辐射)
3. [饱和水汽压与大气湿度](#3-饱和水汽压与大气湿度)
4. [空气动力学阻抗](#4-空气动力学阻抗)
5. [风速高度调整](#5-风速高度调整)
6. [土壤热通量](#6-土壤热通量)
7. [实际应用中的简化与假设](#7-实际应用中的简化与假设)

---

## 1. Penman-Monteith 方程详解

### 1.1 完整形式

ASCE标准化的 Penman-Monteith 方程是计算参考蒸散发 (ET₀) 的黄金标准。其完整形式为：

$$
ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{C_n}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma(1 + C_d \cdot u_2)}
$$

**变量说明：**

| 符号 | 名称 | 单位 | 物理意义 |
|------|------|------|----------|
| ET₀ | 参考蒸散发 | mm/day | 标准草地表面的蒸散发量 |
| Δ | 饱和水汽压曲线斜率 | kPa/°C | 温度对饱和水汽压的敏感性 |
| Rₙ | 净辐射 | MJ m⁻² day⁻¹ | 地表净吸收的辐射能量 |
| G | 土壤热通量 | MJ m⁻² day⁻¹ | 传入土壤的热量（日尺度≈0） |
| γ | 干湿表常数 | kPa/°C | 反映大气压和潜热的综合参数 |
| T | 平均气温 | °C | 近地表空气温度 |
| u₂ | 2米风速 | m/s | 标准测量高度的风速 |
| eₛ | 饱和水汽压 | kPa | 当前温度下空气能容纳的最大水汽压 |
| eₐ | 实际水汽压 | kPa | 当前空气的实际水汽含量 |
| Cₙ | 分子常数 | K mm s³ Mg⁻¹ day⁻¹ | 草地: 900, 苜蓿: 1600 |
| Cₐ | 分母常数 | s/m | 草地: 0.34, 苜蓿: 0.38 |

### 1.2 方程的物理意义

该方程由两部分组成：

#### **能量项 (Energy Term)**

$$
\frac{0.408 \Delta (R_n - G)}{\Delta + \gamma(1 + C_d \cdot u_2)}
$$

- 表示**辐射驱动**的蒸发
- 系数 0.408 将能量单位 (MJ m⁻²) 转换为水深单位 (mm)
- Δ/(Δ+γ) 是"能量分配比"，决定了有多少净辐射用于蒸发而非显热

#### **空气动力项 (Aerodynamic Term)**

$$
\frac{\gamma \frac{C_n}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma(1 + C_d \cdot u_2)}
$$

- 表示**风速和湿度梯度驱动**的蒸发
- (eₛ - eₐ) 是**饱和差 (Vapor Pressure Deficit, VPD)**，空气的"干渴程度"
- u₂ 加速了水汽从表面向大气的输送

### 1.3 代码实现解析

在 `src/penman_monteith.py` 中：

```python
def calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z=50.0, latitude=40.0, doy=None):
    # 1. 大气压力修正（考虑海拔）
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26  # kPa

    # 2. 干湿表常数
    gamma = 0.000665 * P  # kPa/°C

    # 3. 饱和水汽压（使用Tetens公式）
    es_max = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3))
    es_min = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
    es = (es_max + es_min) / 2

    # 4. 饱和水汽压曲线斜率
    Delta = (4098 * es) / ((T_mean + 237.3) ** 2)

    # 5. 净辐射（简化或精确计算）
    if doy is not None:
        Rn = calculate_net_radiation(Rs, T_max, T_min, ea, latitude, doy)
    else:
        Rn = 0.77 * Rs  # 简化假设

    # 6. ASCE-PM方程
    numerator = 0.408 * Delta * (Rn - G) + gamma * (900 / (T_mean + 273)) * u2 * (es - ea)
    denominator = Delta + gamma * (1 + 0.34 * u2)

    ET0 = numerator / denominator
    return np.maximum(ET0, 0)  # 确保非负
```

**关键点：**
- 使用 **Tetens 公式** 计算饱和水汽压（精度优于简化公式）
- 日尺度计算假设 **G = 0**（土壤热通量在24小时内净贡献为零）
- 如果提供 `doy`（日序数），会进行更精确的天文辐射计算

---

## 2. 辐射平衡与净辐射

### 2.1 净辐射的构成

$$
R_n = R_{ns} - R_{nl}
$$

- **短波净辐射 (Rₙₛ)**：来自太阳的可见光和紫外线
- **长波净辐射 (Rₙₗ)**：地表和大气间的红外辐射交换

### 2.2 短波辐射

$$
R_{ns} = (1 - \alpha) \cdot R_s
$$

- **α**: 反照率 (albedo)，草地取 0.23
- **Rₛ**: 入射太阳辐射

**代码实现 (lines 98-109):**

```python
def calculate_net_radiation(Rs, T_max, T_min, ea, latitude, doy):
    # 假设反照率为 0.23（标准草地）
    albedo = 0.23
    Rns = (1 - albedo) * Rs  # 短波净辐射
    ...
```

### 2.3 长波辐射

长波辐射损失遵循 **Stefan-Boltzmann 定律**：

$$
R_{nl} = \sigma \left( \frac{T_{max,K}^4 + T_{min,K}^4}{2} \right) \left(0.34 - 0.14\sqrt{e_a}\right) \left(1.35\frac{R_s}{R_{so}} - 0.35\right)
$$

- **σ**: Stefan-Boltzmann 常数 (4.903 × 10⁻⁹ MJ K⁻⁴ m⁻² day⁻¹)
- **第一项**：地表黑体辐射（基于温度）
- **第二项**：大气发射率（基于水汽压）
- **第三项**：云量修正（基于实际/晴空辐射比）

**代码实现 (lines 121-130):**

```python
# 长波辐射
sigma = 4.903e-9  # MJ K-4 m-2 day-1
T_max_K = T_max + 273.16
T_min_K = T_min + 273.16

# Stefan-Boltzmann 方程，考虑大气发射率
Rnl = sigma * ((T_max_K**4 + T_min_K**4) / 2) * \
      (0.34 - 0.14 * np.sqrt(ea)) * \
      (1.35 * Rs / Rso - 0.35)

Rn = Rns - Rnl
```

### 2.4 晴空辐射的计算

晴空辐射 (Rₛₒ) 需要考虑**天文因素**：

$$
R_{so} = (0.75 + 2 \times 10^{-5} z) \cdot R_a
$$

其中 Rₐ 是**地外辐射**，取决于：
1. **太阳常数** (1367 W/m²)
2. **日地距离变化** (随日序数变化)
3. **太阳高度角** (随纬度和季节变化)

**代码实现 (lines 111-119):**

```python
def calculate_extraterrestrial_radiation(latitude, doy):
    lat_rad = np.deg2rad(latitude)

    # 太阳赤纬（天文因素）
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # 日落时角
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))

    # 地外辐射
    Ra = (24 * 60 / np.pi) * 0.082 * dr * \
         (omega_s * np.sin(lat_rad) * np.sin(delta) +
          np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s))
    return Ra
```

**物理解释：**
- **dr**: 日地距离修正因子（1月近日点，7月远日点）
- **δ**: 太阳赤纬（夏至最大 +23.45°，冬至最小 -23.45°）
- **ωₛ**: 日落时角（决定白昼长度）

---

## 3. 饱和水汽压与大气湿度

### 3.1 Tetens 公式

饱和水汽压随温度指数增长：

$$
e_s(T) = 0.6108 \exp\left(\frac{17.27 T}{T + 237.3}\right) \quad \text{[kPa]}
$$

**为什么不用 Clausius-Clapeyron 方程？**
- Tetens 公式是经验拟合，在 0-50°C 范围内误差 < 0.1%
- 计算速度快，避免了复杂的积分

### 3.2 饱和水汽压曲线斜率

$$
\Delta = \frac{d e_s}{d T} = \frac{4098 \cdot e_s}{(T + 237.3)^2}
$$

**物理意义：**
- 高温时 Δ 更大 → 温度对蒸发的影响更显著
- 这就是为什么热浪容易引发极端蒸散发

**代码实现 (lines 69-74):**

```python
es_max = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3))
es_min = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
es = (es_max + es_min) / 2

Delta = (4098 * es) / ((T_mean + 237.3) ** 2)
```

### 3.3 饱和差 (VPD) 的生态意义

$$
\text{VPD} = e_s - e_a
$$

- **VPD > 2 kPa**: 植物气孔关闭，抑制蒸腾
- **VPD < 0.5 kPa**: 高湿环境，病害风险增加
- **极端蒸散发常伴随 VPD > 3 kPa**

---

## 4. 空气动力学阻抗

### 4.1 阻抗的物理概念

空气动力学阻抗 (rₐ) 表示水汽从表面传输到大气的"困难程度"：

$$
r_a = \frac{\ln\left(\frac{z - d}{z_0}\right) \ln\left(\frac{z - d}{z_h}\right)}{k^2 u_z}
$$

- **z**: 测量高度
- **d**: 零平面位移（植被高度的 2/3）
- **z₀**: 动量粗糙度长度
- **zₕ**: 热量粗糙度长度
- **k**: von Kármán 常数 (0.41)

**在 ASCE-PM 中的简化：**
- 标准化为 2米高度
- 阻抗隐含在常数 Cₙ 和 Cₐ 中

---

## 5. 风速高度调整

### 5.1 对数风廓线

风速随高度遵循**对数律**：

$$
u_2 = u_z \cdot \frac{\ln\left(\frac{2 - d}{z_0}\right)}{\ln\left(\frac{z - d}{z_0}\right)}
$$

**代码实现 (lines 155-175):**

```python
def adjust_wind_speed(u_z, z_measurement, z_target=2.0, surface='grass'):
    """
    将风速从测量高度调整到目标高度

    Parameters
    ----------
    u_z : float or array-like
        测量高度的风速 (m/s)
    z_measurement : float
        实际测量高度 (m)
    z_target : float, default=2.0
        目标高度 (m)
    surface : {'grass', 'crop', 'bare'}, default='grass'
        表面类型（决定粗糙度参数）
    """
    if surface == 'grass':
        z0 = 0.012  # 粗糙度长度 (m)
        d = 0.08    # 零平面位移 (m)
    elif surface == 'crop':
        z0 = 0.05
        d = 0.5
    else:  # bare soil
        z0 = 0.001
        d = 0.0

    # 对数风廓线公式
    u_target = u_z * (np.log((z_target - d) / z0) /
                      np.log((z_measurement - d) / z0))
    return u_target
```

### 5.2 为什么要调整到2米？

1. **标准化**：WMO 规定的气象观测标准高度
2. **避免近地层影响**：地表摩擦层内风速变化剧烈
3. **模型一致性**：ASCE-PM 方程的参数基于2米观测

---

## 6. 土壤热通量

### 6.1 日尺度假设

$$
G_{\text{day}} \approx 0
$$

**物理依据：**
- 白天土壤吸热（G > 0）
- 夜晚土壤放热（G < 0）
- 24小时积分近似为零

### 6.2 小时尺度修正

如果需要小时尺度 ET，必须考虑 G：

$$
G = c_s \cdot \frac{T(t) - T(t-\Delta t)}{\Delta z} \cdot \Delta z
$$

- **cₛ**: 土壤比热容 (约 2.1 MJ m⁻³ °C⁻¹)
- **Δz**: 土壤深度 (0.1-0.2 m)

---

## 7. 实际应用中的简化与假设

### 7.1 常见简化方案

| 简化项 | 完整计算 | 简化假设 | 误差范围 |
|--------|----------|----------|----------|
| 净辐射 | Stefan-Boltzmann | Rₙ = 0.77 Rₛ | ±10% |
| 土壤热通量 | 温度梯度 | G = 0 | 日尺度忽略不计 |
| 相对湿度 | 露点温度 | RH = 70% 常数 | ±15% |
| 风速 | 实测 u₁₀ | u₂ = 2 m/s 常数 | ±20% |

### 7.2 误差传播分析

使用 **蒙特卡洛模拟** 评估各变量的不确定性对 ET₀ 的影响：

```python
def et0_uncertainty_analysis(T_mean, Rs, u2, ea, n_samples=1000):
    # 假设各变量的测量误差
    T_error = np.random.normal(0, 1.0, n_samples)  # ±1°C
    Rs_error = np.random.normal(0, 1.5, n_samples)  # ±1.5 MJ/m²/day
    u2_error = np.random.normal(0, 0.3, n_samples)  # ±0.3 m/s
    ea_error = np.random.normal(0, 0.1, n_samples)  # ±0.1 kPa

    et0_samples = []
    for i in range(n_samples):
        et0 = calculate_et0(
            T_mean + T_error[i],
            T_mean + 5,
            T_mean - 5,
            Rs + Rs_error[i],
            u2 + u2_error[i],
            ea + ea_error[i]
        )
        et0_samples.append(et0)

    return {
        'mean': np.mean(et0_samples),
        'std': np.std(et0_samples),
        'ci_95': np.percentile(et0_samples, [2.5, 97.5])
    }
```

### 7.3 不同气候区的参数优化

| 气候带 | 反照率 α | 粗糙度 z₀ (m) | 推荐 Cₙ | 备注 |
|--------|---------|--------------|---------|------|
| 湿润热带 | 0.20 | 0.015 | 900 | 高蒸发需求 |
| 温带大陆 | 0.23 | 0.012 | 900 | ASCE 标准 |
| 干旱区 | 0.25 | 0.008 | 850 | 稀疏植被 |
| 寒带 | 0.30 | 0.010 | 750 | 低温限制 |

---

## 8. 参考文献

1. **Allen, R. G., et al. (1998)**. *Crop evapotranspiration - Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. Rome: FAO.

2. **ASCE-EWRI (2005)**. *The ASCE standardized reference evapotranspiration equation*. Report by the ASCE-EWRI Task Committee on Standardization of Reference Evapotranspiration.

3. **Penman, H. L. (1948)**. Natural evaporation from open water, bare soil and grass. *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 193(1032), 120-145.

4. **Monteith, J. L. (1965)**. Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.

5. **Zhao, W., et al. (2025)**. Regional variations in drivers of extreme reference evapotranspiration across the contiguous United States. *Water Resources Research*.

---

## 附录：符号表

| 符号 | 名称（中文） | 名称（英文） | 单位 |
|------|------------|------------|------|
| ET₀ | 参考蒸散发 | Reference evapotranspiration | mm/day |
| Rₙ | 净辐射 | Net radiation | MJ m⁻² day⁻¹ |
| G | 土壤热通量 | Soil heat flux | MJ m⁻² day⁻¹ |
| T | 气温 | Air temperature | °C |
| u₂ | 2米风速 | Wind speed at 2m | m/s |
| eₛ | 饱和水汽压 | Saturation vapor pressure | kPa |
| eₐ | 实际水汽压 | Actual vapor pressure | kPa |
| Δ | 饱和水汽压曲线斜率 | Slope of saturation vapor pressure curve | kPa/°C |
| γ | 干湿表常数 | Psychrometric constant | kPa/°C |
| α | 反照率 | Albedo | - |
| z₀ | 粗糙度长度 | Roughness length | m |
| rₐ | 空气动力学阻抗 | Aerodynamic resistance | s/m |
