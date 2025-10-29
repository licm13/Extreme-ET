# Advanced Features Documentation (v1.1.0)

## 🚀 New Advanced Analysis Modules

Based on cutting-edge research from:
- **Markonis, Y. (2025)**. On the Definition of Extreme Evaporation Events. *Geophysical Research Letters*, 52, e2024GL113038.
- **Zhao et al. (2025)**. Regional Variations in Drivers of Extreme Reference Evapotranspiration Across the Contiguous United States. *Water Resources Research*, 61, e2025WR040177.

---

## 📚 Module Overview

### 1. Water Cycle Acceleration Analysis (`water_cycle_analysis.py`)

**Based on Markonis (2025) methodology**

Analyzes how extreme ET events impact regional water availability and water cycle intensification.

#### Key Metrics:
- **P-E (Water Availability)**: Precipitation minus evaporation
  - Positive values = wetter conditions
  - Negative values = drier conditions
- **(P+E)/2 (Water Cycle Intensity)**: Land-atmosphere water exchange rate
  - Higher values = accelerated water cycle
  - Lower values = decelerated water cycle

#### Main Functions:
```python
from water_cycle_analysis import (
    calculate_water_availability,
    calculate_water_cycle_intensity,
    decompose_water_cycle_by_extremes,
    analyze_temporal_changes,
    classify_water_cycle_regime
)

# Calculate basic metrics
water_avail = calculate_water_availability(precipitation, evaporation)
water_intensity = calculate_water_cycle_intensity(precipitation, evaporation)

# Decompose by extreme events
decomposition = decompose_water_cycle_by_extremes(
    precipitation, evaporation, extreme_mask
)

# Analyze temporal changes
changes = analyze_temporal_changes(
    precipitation, evaporation, extreme_mask, split_year_idx
)

# Classify water cycle regime
regime = classify_water_cycle_regime(
    water_availability_change=0.5,
    water_intensity_change=0.3
)
# Returns: "Wetter-Accelerated", "Wetter-Decelerated",
#          "Drier-Accelerated", or "Drier-Decelerated"
```

#### Applications:
- Understanding water cycle acceleration under climate change
- Quantifying extreme event impacts on water resources
- Distinguishing between water availability changes and cycle intensification

---

### 2. Non-Stationary Threshold Analysis (`nonstationary_threshold.py`)

**Adapting to climate change**

Traditional extreme detection assumes constant climatology. This module implements adaptive thresholds that account for long-term trends.

#### Methods:
- **LOESS Smoothing**: Separates forced trends from natural variability
- **Moving Percentile**: Adaptive threshold using rolling windows
- **Trend Detection**: Identifies and removes linear trends
- **Decade-Adaptive**: Thresholds that evolve by decade

#### Main Functions:
```python
from nonstationary_threshold import (
    loess_smoothed_threshold,
    detect_trend_and_detrend,
    compare_stationary_vs_nonstationary
)

# Calculate non-stationary threshold
threshold = loess_smoothed_threshold(
    data, dates, percentile=95.0, smoothing_factor=0.3
)

# Detect and remove trend
slope, p_value, detrended, trend_stats = detect_trend_and_detrend(data, dates)
print(f"Trend: {slope*365:.3f} mm/day per year, p={p_value:.4e}")

# Compare approaches
comparison = compare_stationary_vs_nonstationary(data, dates, percentile=95)
print(f"Stationary: {comparison['n_stationary_extremes']} events")
print(f"Non-stationary: {comparison['n_nonstationary_extremes']} events")
```

#### Applications:
- Climate change adaptation studies
- Separating anthropogenic trends from natural variability
- More accurate extreme event detection in non-stationary climates

---

### 3. Multivariate Extreme Analysis (`multivariate_extremes.py`)

**Copula-based compound event analysis**

Analyzes joint extremes of multiple variables (e.g., high ET0 + low precipitation = flash drought).

#### Copula Types:
- **Gaussian**: Symmetric dependence
- **Clayton**: Lower tail dependence (drought conditions)
- **Gumbel**: Upper tail dependence (concurrent extremes)

#### Main Functions:
```python
from multivariate_extremes import (
    identify_compound_extreme_et_precipitation,
    calculate_joint_return_period,
    calculate_drought_severity_index
)

# Identify compound extremes (high ET + low precip)
compound = identify_compound_extreme_et_precipitation(
    et0, precipitation,
    et0_percentile=95.0,
    precip_percentile=5.0
)

print(f"Compound events: {compound['n_compound_events']}")
print(f"ET-only: {compound['n_et0_only']}")
print(f"Precip-only: {compound['n_precip_only']}")

# Calculate joint return periods
return_periods = calculate_joint_return_period(
    var1, var2, threshold1, threshold2,
    copula_type='clayton'  # or 'gaussian', 'gumbel'
)

print(f"Marginal return period 1: {return_periods['marginal_return_period_1']:.1f}")
print(f"Joint return period (AND): {return_periods['joint_and_return_period']:.1f}")

# Drought severity index
dsi = calculate_drought_severity_index(et0, precipitation, soil_moisture=None)
severe_drought = dsi > np.percentile(dsi, 95)
```

#### Applications:
- Flash drought identification and prediction
- Compound event risk assessment
- Agricultural water stress quantification
- Joint probability analysis for water resources planning

---

### 4. Spatial Analysis (`spatial_analysis.py`)

**Regional patterns and propagation**

Analyzes spatial coherence, propagation, and interpolation of extreme events.

#### Key Capabilities:
- **Spatial Correlation**: How extremes co-occur across space
- **Event Propagation**: Detect if extremes move (e.g., following weather systems)
- **Kriging Interpolation**: Spatial prediction with uncertainty
- **Regional Synchrony**: Within- and between-region coherence

#### Main Functions:
```python
from spatial_analysis import (
    calculate_spatial_correlation,
    detect_event_propagation,
    ordinary_kriging,
    calculate_regional_synchrony
)

# Spatial correlation analysis
distances, correlations, bins = calculate_spatial_correlation(
    data_matrix,  # shape: (n_locations, n_timesteps)
    locations,    # shape: (n_locations, 2) with (lat, lon)
    max_distance=500.0  # km
)

# Event propagation detection
propagation = detect_event_propagation(
    data_matrix, locations, dates,
    max_lag_days=7
)
print(f"Propagation speed: {propagation['propagation_speed']:.1f} km/day")

# Kriging interpolation
interp_vals, interp_var = ordinary_kriging(
    known_points, known_values, target_points,
    variogram_model='exponential'
)

# Regional synchrony
synchrony = calculate_regional_synchrony(data_matrix, region_labels)
for region, metrics in synchrony.items():
    print(f"Region {region}: within={metrics['within_region_synchrony']:.3f}")
```

#### Applications:
- Understanding regional extreme event patterns
- Predicting extremes at ungauged locations
- Identifying synchronized drought/heat regions
- Event propagation and predictability studies

---

### 5. Event Physical Evolution (`event_evolution.py`)

**Onset, peak, and termination analysis**

**Based on Markonis (2025) onset/termination methodology**

Analyzes the physical evolution of extreme events, including energy balance and trigger identification.

#### Key Analyses:
- **Onset Conditions**: Meteorology leading to event initiation
- **Termination Conditions**: What ends the extreme event
- **Energy Partitioning**: Latent vs sensible heat during extremes
- **Event Triggers**: Which variables act as precursors

#### Main Functions:
```python
from event_evolution import (
    analyze_onset_termination_conditions,
    analyze_energy_partitioning,
    identify_event_triggers,
    analyze_event_intensity_evolution
)

# Analyze onset and termination
conditions = analyze_onset_termination_conditions(
    et0, met_data, extreme_mask, window_days=5
)

print("Onset conditions:")
print(f"  Temperature: {conditions['onset']['temperature_mean']:.2f} °C")
print(f"  Radiation: {conditions['onset']['radiation_mean']:.2f} MJ/m²/day")

print("Termination conditions:")
print(f"  Temperature: {conditions['termination']['temperature_mean']:.2f} °C")
print(f"  Radiation: {conditions['termination']['radiation_mean']:.2f} MJ/m²/day")

# Energy partitioning analysis
energy = analyze_energy_partitioning(temperature, radiation, et0, extreme_mask)
print(f"Bowen ratio during extremes: {energy['extreme']['bowen_ratio_mean']:.3f}")
print(f"Latent heat fraction: {energy['extreme']['latent_heat_fraction']:.2%}")

# Identify triggers
triggers = identify_event_triggers(met_data, extreme_mask, lookback_days=7)
for var, info in triggers.items():
    if info['is_positive_trigger']:
        print(f"{var}: anomaly = {info['mean_pre_event_anomaly']:+.2f}σ")

# Event intensity evolution
evolutions = analyze_event_intensity_evolution(et0, extreme_mask)
mean_duration = np.mean([e['duration'] for e in evolutions])
print(f"Mean event duration: {mean_duration:.1f} days")
```

#### Applications:
- Understanding physical mechanisms of extreme ET
- Identifying early warning signals
- Energy budget analysis during extremes
- Predictive model development

---

## 📊 Example Workflow

```python
import numpy as np
import pandas as pd
from water_cycle_analysis import decompose_water_cycle_by_extremes
from nonstationary_threshold import compare_stationary_vs_nonstationary
from multivariate_extremes import identify_compound_extreme_et_precipitation
from spatial_analysis import detect_event_propagation
from event_evolution import analyze_onset_termination_conditions

# 1. Detect extremes with non-stationary threshold
comparison = compare_stationary_vs_nonstationary(et0, dates, percentile=95)
extreme_mask = comparison['nonstationary_extremes']

# 2. Analyze water cycle impacts
water_cycle = decompose_water_cycle_by_extremes(precip, et0, extreme_mask)
print(f"Water availability during extremes: {water_cycle['extreme_days']['water_availability_mean']:.2f} mm/day")

# 3. Identify compound events
compound = identify_compound_extreme_et_precipitation(et0, precip)
print(f"Compound flash drought events: {compound['n_compound_events']}")

# 4. Analyze spatial propagation
propagation = detect_event_propagation(spatial_data, locations, dates)
print(f"Events propagate at {propagation['propagation_speed']:.1f} km/day")

# 5. Study event evolution
conditions = analyze_onset_termination_conditions(et0, met_data, extreme_mask)
print(f"Onset temperature: {conditions['onset']['temperature_mean']:.2f} °C")
print(f"Termination precipitation: {conditions['termination']['precipitation_mean']:.2f} mm/day")
```

---

## 🔬 Scientific Innovations

### From Markonis (2025):

1. **ExEvE Framework**: Event-based analysis with clear onset/termination
2. **Z-score Standardization**: Pentad-based deseasonalization
3. **Hurst Persistence**: Long-term memory and clustering detection
4. **Water Cycle Decomposition**: P-E and (P+E)/2 metrics
5. **Physical Drivers**: Energy balance and moisture dynamics

### From Zhao et al. (2025):

6. **Multiple Detection Methods**: ERT_hist, ERT_clim, OPT comparison
7. **Occurrence-Based Severity**: Standardized extremity levels
8. **Multi-Scale Analysis**: Hourly to monthly aggregation
9. **Spatial Heterogeneity**: Regional driver variations
10. **Uncertainty Quantification**: Data set and method comparisons

---

## 💡 Research Applications

### Climate Change Studies:
- Track water cycle acceleration under warming
- Separate forced trends from natural variability
- Project future extreme event characteristics

### Drought Monitoring:
- Identify flash drought precursors
- Quantify compound event severity
- Early warning system development

### Water Resources Management:
- Assess regional synchrony of extremes
- Estimate extreme event return periods
- Optimize reservoir operations

### Agricultural Risk Assessment:
- Link extreme ET to crop water stress
- Predict irrigation demand anomalies
- Evaluate climate adaptation strategies

---

## 📖 References

1. Markonis, Y. (2025). On the Definition of Extreme Evaporation Events. *Geophysical Research Letters*, 52, e2024GL113038. https://doi.org/10.1029/2024GL113038

2. Zhao, B., Horvat, C., Pearson, C., Shah, D., & Gao, H. (2025). Regional Variations in Drivers of Extreme Reference Evapotranspiration Across the Contiguous United States. *Water Resources Research*, 61, e2025WR040177. https://doi.org/10.1029/2025WR040177

3. Huntington, T. G., et al. (2018). A new indicator framework for quantifying the intensity of the terrestrial water cycle. *Journal of Hydrology*, 559, 361-372.

---

## 🚀 Getting Started

See `examples/example_advanced_analysis.py` for a comprehensive demonstration of all new features.

Quick test:
```bash
python test_new_modules.py
```

---

## 📧 Contact & Support

For questions, bug reports, or feature requests:
- GitHub Issues: https://github.com/your-org/extreme-et/issues
- Documentation: https://extreme-et.readthedocs.io

---

**Version**: 1.1.0
**Release Date**: October 2025
**License**: MIT
