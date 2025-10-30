# Extreme-ET 框架与数学原理说明

本说明文档基于仓库 `src/` 目录下的全部核心模块，系统梳理 Extreme-ET 工具包如何支撑“极端蒸散/势蒸散（ET/PET）”研究的四个关键任务：

1. **哪些产品能抓 Extreme、如何定义？**
2. **中国的 Extreme 与全球的 Extreme 如何对照？**
3. **Extreme 的归因路径？**
4. **PET（ETo）的 Extreme 如何量化？**

随后给出逐模块、逐函数的功能与数学原理说明，便于在写作或扩展代码时快速定位所需能力。

---

## 1. 产品与极端定义：方法到模块的映射

| 研究需求 | 对应模块/函数 | 核心思想 |
|-----------|----------------|----------|
| 长期逐日 ET/PET 及驱动因子 | `src.penman_monteith.calculate_et0`、`src.penman_monteith.calculate_vapor_pressure_from_vpd`、`src.penman_monteith.adjust_wind_speed` | ASCE 标准化 Penman–Monteith 方程，将温度、辐射、风、湿度转换为参考作物 ET0。|
| 阈值型极端识别 | `src.extreme_detection.detect_extreme_events_hist`、`detect_extreme_events_clim`、`optimal_path_threshold` | 历史百分位、季节气候学、目标发生率优化三套阈值体系，支撑日尺度极端识别。|
| 事件尺度聚合 | `src.extreme_detection.identify_events_from_mask`、`calculate_event_statistics`、`src.event_evolution.identify_event_periods` | 将逐日极端聚合为事件，计算持续时间、峰值、累计量。|
| 复合极端 | `src.multivariate_extremes.identify_compound_extreme_et_precipitation`、`calculate_joint_return_period` | Copula & 返回期分析刻画“高 ET + 低 P”等复合事件。|
| 非平稳阈值 | `src.nonstationary_threshold.detect_trend_and_detrend`、`compare_stationary_vs_nonstationary` | 线性趋势检验 + LOESS/分段百分位实现阈值随时间演变。|

**建议策略：** 先用 `detect_extreme_events_hist`/`detect_extreme_events_clim` 给出基准极端掩膜，再结合 `identify_events_from_mask` 与事件统计完成 ExEvEs 级别的分析；必要时使用 `compare_stationary_vs_nonstationary` 检查阈值随年代漂移。

---

## 2. 中国 vs 全球：可复用的区域化流程

| 步骤 | 中国尺度 | 全球尺度 | 关键代码 |
|------|-----------|-----------|----------|
| 数据预处理 | `src.data_processing.deseasonalize_data`、`standardize_to_zscore` | 同 | 季节性剥离 + Z 分数标准化，便于跨区域阈值统一。|
| 区域阈值 | `detect_extreme_events_clim`（考虑纬度差异） | 同 | 利用日历日阈值或 OPT 方法在不同气候带保持发生率一致。|
| 事件合成 | `src.event_evolution.analyze_onset_termination_conditions`、`analyze_energy_partitioning` | 同 | 事件前后 ±N 天复合，呈现温度/辐射/水汽演化。|
| 空间传播 | `src.spatial_analysis.detect_event_propagation`、`calculate_spatial_correlation` | 可拓展到全球格点 | 计算空间相关系数与传播速度，识别区域簇。|
| 水循环响应 | `src.water_cycle_analysis.decompose_water_cycle_by_extremes`、`analyze_seasonal_water_cycle` | 同 | 对比东部湿区、西北干旱区等在极端期 P−E、(P+E)/2 的变化。|

通过上述模块，既可在中国典型气候区构建极端谱系，也可换成全球再分析/模式产品做同构分析。

---

## 3. Extreme 的归因工具链

| 归因问题 | 支撑函数 | 数学原理 |
|-----------|----------|-----------|
| Penman–Monteith 驱动贡献 | `src.contribution_analysis.calculate_contributions` | **方程：** 对于极端日集合 \(\mathcal{E}\)，对每个驱动 \(X\in\{T, R_s, u_2, e_a\}\)，计算原始 ET0 与替换为气候态 \(X_{clim}\) 后的差值；相对贡献 $$RC_X = \frac{\sum_{t\in\mathcal{E}} [ET0(t)-ET0_{X\rightarrow clim}(t)]}{\sum_{Y} \sum_{t\in\mathcal{E}} [ET0(t)-ET0_{Y\rightarrow clim}(t)]}.$$ |
| 灵敏度分析 | `src.contribution_analysis.sensitivity_analysis`、`dynamic_perturbation_response` | 对每个驱动施加相对扰动 \(\Delta X = \epsilon X\)，计算 $$S_X = \frac{ET0(X+\Delta X)-ET0(X)}{\Delta X}.$$ |
| 过程能量闭合 | `src.event_evolution.analyze_energy_partitioning`、`calculate_energy_balance_components` | 基于 \(\lambda E = L_v \cdot ET\) 与感热、Bowen 比等公式评估能量分配。|
| 触发因子识别 | `src.event_evolution.identify_event_triggers` | 计算事件前滚动窗口内的标准化异常，区分正触发/负触发。|
| 水循环响应 | `src.water_cycle_analysis.analyze_temporal_changes` | 对比两个时期的 P−E、(P+E)/2 均值比率，给出加速或干化分类。|
| 非平稳检测与信号分离 | `src.nonstationary_threshold.detect_trend_and_detrend`、`separate_forced_variability` | 线性回归 + 残差分解，将外强迫（趋势）与内部变率拆分。|
| 复合事件归因 | `src.multivariate_extremes.calculate_joint_return_period`、`analyze_compound_event_characteristics` | 使用高斯/Clayton/Gumbel copula 得到联合分布，返回期公式 $$T_{AND} = \frac{1}{1 - C(u,v)}$$。|

---

## 4. PET（ETo）Extreme：公式与业务指标

- **核心方程：** `calculate_et0` 实现 ASCE 标准化 Penman–Monteith：
  $$ET_0 = \frac{0.408\,\Delta (R_n - G) + \gamma \frac{C_n}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + C_d u_2)}.$$ 其中 \(\Delta\) 为饱和水汽压曲线斜率，\(R_n\) 净辐射，\(G\) 土壤热通量（逐日假设为 0），\(\gamma\) 心理常数。
- **VPD→实际水汽压：** `calculate_vapor_pressure_from_vpd` 根据 \(e_a = e_s(T) - \mathrm{VPD}\) 转换湿度口径。
- **风速折算：** `adjust_wind_speed` 采用对数风廓线将 \(u_z\) 统一到 2 m。
- **贡献与季节差异：** `calculate_contributions` + `analyze_seasonal_contributions` 提供中国不同季节/区域的主导因子；`identify_dominant_driver` 将贡献结果映射为主导驱动标签。
- **极端频度对业务含义：**
  - `src.water_cycle_analysis.decompose_water_cycle_by_extremes` 得出极端日平均 P−E、(P+E)/2 → 水资源调度压力。
  - `src.event_evolution.analyze_event_intensity_evolution` 输出持续时间、峰值、累计耗水量 → 灌溉用水预估。

---

## 模块与函数详解

### src/extreme_detection.py

| 函数 | 功能 | 数学/算法原理 |
|------|------|---------------|
| `_estimate_tail_threshold` | 估计给定严重度的上尾阈值 | 支持经验分位与 GPD 外推；当超越样本数不足时自动回退。|
| `detect_extreme_events_hist` | 历史相对阈值法 ERT_hist | 阈值 = \(Q_{1-\alpha}\)，极端掩膜 `data > threshold`；`return_details=True` 时调用 `identify_events_from_mask` 计算事件统计。|
| `detect_extreme_events_clim` | 气候学阈值 ERT_clim | 以日历日窗口平滑百分位；结合最小持续天数过滤短脉冲。|
| `detect_compound_extreme_events` | 同步检测两个变量的极端 | 将各变量标准化后取交集，输出联合事件统计。|
| `optimal_path_threshold` | 目标发生率优化 | 迭代缩放日阈值（×0.95/1.05）直至达到指定严重度。|
| `identify_climatological_extremes` | 应用气候学阈值至时间序列 | 使用 DOY 阈值数组映射到逐日数据。|
| `identify_events_from_mask` | 将布尔掩膜分解为事件列表 | 连续 True 合并为事件，返回起止索引、峰值、持续等。|
| `calculate_event_statistics` | 汇总事件强度指标 | 计算持续时间、峰值、累计量、间隔、严重度等统计分布。|

### src/penman_monteith.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `calculate_et0` | 计算参考作物 ET0 | ASCE 标准化 Penman–Monteith 方程（草地参考面），默认日尺度取 \(G=0\)。|
| `calculate_net_radiation` | 净辐射 | $R_n = (1-\alpha)R_s - R_{nl}$，长波项依赖 Stefan–Boltzmann 定律与云量校正。|
| `calculate_extraterrestrial_radiation` | 地外辐射 | 依据太阳常数、地球日距、太阳赤纬与日落时角积分。|
| `calculate_vapor_pressure_from_vpd` | VPD→实际水汽压 | $e_a = e_s(T) - \mathrm{VPD}$。|
| `adjust_wind_speed` | 风速高度折算 | 对数风速廓线 $u_2 = u_z \cdot 4.87 / \ln(67.8z - 5.42)$。|

### src/contribution_analysis.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `calculate_contributions` | 逐驱动贡献率 | 方程见上表：极端日 ET0 与替换气候态后的差值比。|
| `sensitivity_analysis` | 一阶灵敏度 | $S_X = \frac{\Delta ET0}{\Delta X}$，默认 10% 扰动。|
| `analyze_seasonal_contributions` | 季节贡献 | 将极端日按月份映射到季节集合，调用 `calculate_contributions`。|
| `identify_dominant_driver` | 主导因子判定 | 取贡献最大者，若近似并列则返回 `'mixed'`。|
| `dynamic_perturbation_response` | 动态扰动轨迹 | 以多个扰动幅度生成响应曲线。|
| `compute_perturbation_pathway` | 多变量协同扰动 | 沿给定路径比例地调节驱动，输出 ET0 响应。|

### src/data_processing.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `standardize_to_zscore` | 气候态 Z 分数 | 可按五日/日尺度计算 $z = (x - \mu)/\sigma$。|
| `calculate_hurst_exponent` | Hurst 指数 | R/S 分析估计持久性，最大滞后可调。|
| `moving_average` | 滑动平均 | 简单卷积平滑。|
| `calculate_autocorrelation` | 自相关函数 | 逐滞后计算 $\rho(k)$。|
| `deseasonalize_data` | 去季节化 | 差分或 Z-score 方法移除季节周期。|

### src/event_evolution.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `identify_event_periods` | 从掩膜提取事件 | 连续区间 -> 事件列表。|
| `analyze_onset_termination_conditions` | 事件起止合成 | 对起始/终止窗口做均值、标准差对比。|
| `calculate_energy_balance_components` | 能量分量 | 依据 $R_n = \lambda E + H + G$ 估算潜热、感热。|
| `analyze_energy_partitioning` | 极端 vs 正常能量分配 | 输出潜热分量、Bowen 比、潜热分数。|
| `identify_event_triggers` | 触发因子 | 事件前滚动窗口 Z-score 异常 + 一致性率。|
| `analyze_event_intensity_evolution` | 强度演化 | 计算事件持续、峰值、累积耗水等指标随时间演化。|
| `compare_seasonal_event_characteristics` | 季节对比 | 逐季节统计事件持续/峰值差异。|

### src/water_cycle_analysis.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `calculate_water_availability` | 水分可用量 | $P - E$。|
| `calculate_water_cycle_intensity` | 水循环强度 | $(P + E)/2$。|
| `decompose_water_cycle_by_extremes` | 极端 vs 正常拆分 | 对全体/极端/正常集合分别计算均值与标准差。|
| `analyze_temporal_changes` | 时段对比 | 以分界年索引切分，计算二期比率。|
| `classify_water_cycle_regime` | 水循环分类 | 根据 P−E 与 (P+E)/2 的相对变化划分 `accelerating_wet` 等类型。|
| `analyze_seasonal_water_cycle` | 季节统计 | 以气象季集合计算水量和极端频度。|

### src/nonstationary_threshold.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `calculate_moving_percentile` | 滑动窗口百分位 | 在给定窗口宽度内求高分位数。|
| `loess_smoothed_threshold` | LOESS 平滑阈值 | 局部多项式回归拟合日序列阈值曲线。|
| `detect_trend_and_detrend` | 趋势检验与去趋势 | 线性回归 + t 检验 + R²；返回残差序列。|
| `adaptive_threshold_by_decade` | 十年尺度阈值 | 分年代计算百分位形成阶梯式非平稳阈值。|
| `separate_forced_variability` | 强迫 vs 内部变率 | 使用线性趋势或指定信号拆分序列。|
| `quantile_regression_threshold` | 分位数回归 | 对 `data ~ time` 进行 τ 分位回归。|
| `compare_stationary_vs_nonstationary` | 阈值方案对比 | 结合上面方法给出极端日数量差异。|

### src/multivariate_extremes.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `empirical_cdf` | 经验分布函数 | 排序 + 等概率分配。|
| `transform_to_uniform` | 变量统一到 [0,1] | 通过经验 CDF 得到概率积分变换。|
| `gaussian_copula_parameter` / `clayton_copula_parameter` / `gumbel_copula_parameter` | Copula 参数估计 | 分别使用相关系数/极值理论方法估计依赖结构。|
| `calculate_joint_return_period` | 联合返回期 | 基于选定 Copula \(C(u,v)\) 计算 `AND/OR` 返回期。|
| `identify_compound_extreme_et_precipitation` | 复合极端识别 | 对 ET0、P 分别取百分位阈值后用 Copula 描述联合统计。|
| `calculate_drought_severity_index` | 干旱严重度指数 | 综合标准化降水缺口与高 ET 异常。|
| `analyze_compound_event_characteristics` | 复合事件特征 | 统计持续时间、间隔、联合严重度。|

### src/spatial_analysis.py

| 函数 | 功能 | 数学原理 |
|------|------|-----------|
| `calculate_spatial_correlation` | 空间相关 | 依据站点距离分箱后计算极端发生的一致性。|
| `detect_event_propagation` | 事件传播 | 最大滞后相关 + 平均传播速度。|
| `ordinary_kriging` | 克里金插值 | 半方差函数 + 克里金系统求权重。|
| `calculate_regional_synchrony` | 区域同步度 | 统计多个站点同时发生极端的概率。|
| `identify_spatial_clusters` | 空间聚类 | 基于相关矩阵和阈值筛选团簇。|
| `calculate_spatial_extent_metrics` | 空间范围指标 | 极端覆盖面积、质心等。|

### src/evaluation.py

| 函数 | 功能 |
|------|------|
| `aggregate_hourly_to_daily` | 将小时序列转换为日尺度（和型/平均等）。|
| `nearest_grid_point` | 最近邻格点索引。|
| `compute_daywise_skill` | 逐日 POD/FAR/CSI 技能评估。|
| `event_timing_errors` | 事件起止时间偏差。|
| `severity_sweep_skill` | 多严重度阈值下的技能曲线。|
| `matched_event_intensity_bias` | 匹配事件强度偏差。|
| `serialize_skill_summary` | 将技能统计转成 Markdown 表。|

### src/io_utils.py

| 函数 | 功能 |
|------|------|
| `_require_xarray` | 检查 `xarray` 依赖。|
| `read_netcdf_variable` | 读取 NetCDF 变量为 `xarray.DataArray`。|
| `sample_series_at_point` | 最近邻或双线性抽取栅格时间序列。|
| `bilinear_on_regular_grid` | 双线性插值实现。|

### src/utils.py

| 函数 | 功能 |
|------|------|
| `set_paper_style` | 统一论文风格的 `matplotlib` 主题。|
| `label_subplots` | 子图自动标注。|
| `generate_synthetic_data` | 生成带趋势/季节的合成 ET 数据。|
| `plot_extreme_events` 等绘图函数 | 快速绘制极端掩膜、贡献玫瑰图、季节条形图、自相关等。|
| `calculate_event_metrics` | 根据事件列表计算峰值、持续、累计量。|
| `summary_statistics` | 多变量统计汇总表。|

---

## 与四个任务的直接对应关系总结

1. **产品 & 定义**：`penman_monteith` + `extreme_detection` + `nonstationary_threshold`，提供从驱动场到阈值的完整链路。
2. **China vs Global**：`data_processing` + `event_evolution` + `spatial_analysis` + `water_cycle_analysis` 形成可套用到任意区域的流程。
3. **归因**：`contribution_analysis`、`event_evolution`、`water_cycle_analysis`、`multivariate_extremes`、`nonstationary_threshold` 构成“驱动 → 能量 → 水循环 → 复合风险”的闭环。
4. **PET Extreme**：`penman_monteith` 系列函数负责 PET 计算，`contribution_analysis`、`water_cycle_analysis`、`event_evolution` 提供对农业/水资源意义的解读指标。

凭借以上模块化设计，可快速实现“中国极端 ET 事件 vs 全球”以及“PET 极端归因”的论文/报告工作流。
