# Extreme Evaporation Events Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python toolkit for analyzing extreme evaporation events, implementing methods from:

- **Markonis (2025)**: "On the Definition of Extreme Evaporation Events" ([Geophysical Research Letters](https://doi.org/10.1029/2024GL113038))
- **Zhao et al. (2025)**: "Regional Variations in Drivers of Extreme Reference Evapotranspiration Across the Contiguous United States" ([Water Resources Research](https://doi.org/10.1029/2025WR040177))

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Methods](#core-methods)
- [Examples](#examples)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## ✨ Features

### Extreme Event Detection
- **ERT_hist**: Historical relative threshold method for identifying large values
- **ERT_clim**: Climatological method for detecting anomalies (similar to heatwaves)
- **OPT Method**: Optimal Path Threshold for severity-based event definition

### Evapotranspiration Calculation
- **ASCE-PM Equation**: Standardized Penman-Monteith for grass reference surface
- Net radiation calculation with extraterrestrial radiation
- Vapor pressure conversions (VPD, relative humidity, actual vapor pressure)
- Wind speed height adjustments

### Contribution Analysis
- Meteorological driver attribution (temperature, radiation, wind, humidity)
- Sensitivity analysis with perturbation methods
- Seasonal contribution patterns
- Regional spatial analysis support

### Statistical Tools
- Z-score standardization (pentad and daily)
- Hurst coefficient for long-term persistence
- Autocorrelation analysis
- Moving average smoothing
- Event clustering identification

## 🚀 Installation

### Option 1: pip install (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/extreme-evaporation-events.git
cd extreme-evaporation-events

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Direct download

```bash
# Download all Python files to your project directory
# Import modules directly
import sys
sys.path.append('path/to/src')
```

### Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🎯 Quick Start

### Example 1: Basic Extreme Detection

```python
import numpy as np
from src.extreme_detection import detect_extreme_events_hist
from src.utils import generate_synthetic_data

# Generate synthetic ET0 data
data = generate_synthetic_data(n_days=365*40)

# Detect extreme events (0.5% severity ≈ 1.8 days/year)
extreme_mask, threshold = detect_extreme_events_hist(
    data['ET0'], severity=0.005
)

print(f"Threshold: {threshold:.2f} mm/day")
print(f"Number of extreme days: {np.sum(extreme_mask)}")
```

### Example 2: Calculate ET0 with Penman-Monteith

```python
from src.penman_monteith import calculate_et0

# Meteorological forcings
T_mean, T_max, T_min = 20, 25, 15  # °C
Rs = 20.0  # Solar radiation (MJ m-2 day-1)
u2 = 2.0   # Wind speed at 2m (m s-1)
ea = 1.5   # Actual vapor pressure (kPa)

# Calculate reference ET0
ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, 
                   z=50, latitude=40)
print(f"ET0 = {ET0:.2f} mm/day")
```

### Example 3: Contribution Analysis

```python
from src.contribution_analysis import calculate_contributions

# Assuming you have meteorological data and extreme_mask
contributions = calculate_contributions(
    T_mean, T_max, T_min, Rs, u2, ea, 
    extreme_mask
)

print("Relative contributions:")
for driver, percent in contributions.items():
    print(f"  {driver.capitalize():12s}: {percent:5.1f}%")
```

## 📚 Core Methods

### 1. Extreme Event Detection

#### ERT_hist Method (Zhao et al. 2025)
```python
from src.extreme_detection import detect_extreme_events_hist

# Detect events based on historical extremes
extreme_mask, threshold, details = detect_extreme_events_hist(
    data, 
    severity=0.005,      # Occurrence rate (0.5%)
    return_details=True  # Include event statistics
)
```

#### ERT_clim Method (Zhao et al. 2025)
```python
from src.extreme_detection import detect_extreme_events_clim

# Detect anomalies relative to climatology
extreme_mask, thresholds, details = detect_extreme_events_clim(
    data,
    severity=0.05,       # 5% occurrence rate
    min_duration=3,      # Minimum consecutive days
    window=7,            # Moving average window
    return_details=True
)
```

### 2. Data Processing

#### Standardization (Markonis 2025)
```python
from src.data_processing import standardize_to_zscore

# Remove seasonality via z-score transformation
z_scores = standardize_to_zscore(data, pentad=True)
```

#### Hurst Coefficient
```python
from src.data_processing import calculate_hurst_exponent

# Estimate long-term persistence
H = calculate_hurst_exponent(data, max_lag=10)
# H ≈ 0.5: no persistence
# H > 0.5: clustering (long-term persistence)
```

### 3. Penman-Monteith Equation

```python
from src.penman_monteith import calculate_et0

# Daily ET0 calculation
ET0 = calculate_et0(
    T_mean, T_max, T_min,  # Temperature (°C)
    Rs,                     # Solar radiation (MJ m-2 day-1)
    u2,                     # Wind speed at 2m (m s-1)
    ea,                     # Vapor pressure (kPa)
    z=50,                   # Elevation (m)
    latitude=40,            # Latitude (degrees)
    doy=None                # Day of year (optional)
)
```

### 4. Contribution Analysis

```python
from src.contribution_analysis import (
    calculate_contributions,
    analyze_seasonal_contributions,
    identify_dominant_driver
)

# Overall contributions
contributions = calculate_contributions(
    T_mean, T_max, T_min, Rs, u2, ea, 
    extreme_mask
)

# Seasonal breakdown
seasonal_contrib = analyze_seasonal_contributions(
    T_mean, T_max, T_min, Rs, u2, ea,
    extreme_mask
)

# Identify dominant driver
dominant, contribution = identify_dominant_driver(contributions)
```

## 📖 Examples

### Complete Workflow Examples

#### Markonis (2025) Method
```bash
cd examples
python example_markonis_2025.py
```

This demonstrates:
- Data standardization and deseasonalization
- Autocorrelation and Hurst coefficient analysis
- ExEvE detection and characterization
- Water cycle decomposition (P-E and (P+E)/2)
- Temporal trend analysis

#### Zhao et al. (2025) Method
```bash
cd examples
python example_zhao_2025.py
```

This demonstrates:
- Penman-Monteith ET0 calculation
- Both ERT_hist and ERT_clim detection
- OPT method for threshold determination
- Contribution analysis of meteorological drivers
- Seasonal pattern analysis
- Sensitivity to temporal scales and severity levels

### Expected Outputs

Both example scripts generate:
- Time series plots with highlighted extreme events
- Contribution pie charts
- Seasonal analysis bar plots
- Trend analysis figures
- Method comparison visualizations

All figures are saved to `/mnt/user-data/outputs/`

## 🧪 Testing

Run comprehensive tests:

```bash
cd tests
python test_methods.py
```

Tests cover:
- ✓ Data processing functions
- ✓ Extreme detection methods
- ✓ Penman-Monteith calculations
- ✓ Contribution analysis
- ✓ Synthetic data generation
- ✓ Complete workflow integration

## 📁 Project Structure

```
extreme-evaporation-events/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation script
│
├── src/                              # Core package
│   ├── __init__.py                   # Package initialization
│   ├── data_processing.py            # Z-scores, Hurst, autocorrelation
│   ├── extreme_detection.py          # ERT_hist, ERT_clim, OPT
│   ├── penman_monteith.py            # ASCE-PM equation
│   ├── contribution_analysis.py      # Driver attribution
│   └── utils.py                      # Plotting and utilities
│
├── examples/                         # Usage examples
│   ├── example_markonis_2025.py      # Markonis (2025) workflow
│   └── example_zhao_2025.py          # Zhao et al. (2025) workflow
│
└── tests/                            # Unit tests
    └── test_methods.py               # Comprehensive test suite
```

## 📊 Key Equations Implemented

### 1. ASCE Standardized Penman-Monteith (Equation 3, Zhao et al. 2025)

```
ET0 = [0.408 Δ (Rn - G) + γ (900/(T+273)) u2 (es - ea)] / [Δ + γ(1 + 0.34 u2)]
```

where:
- Δ = slope of saturation vapor pressure curve (kPa °C⁻¹)
- Rn = net radiation (MJ m⁻² day⁻¹)
- G = soil heat flux (≈ 0 for daily step)
- γ = psychrometric constant (kPa °C⁻¹)
- T = mean air temperature (°C)
- u2 = wind speed at 2m height (m s⁻¹)
- es, ea = saturated and actual vapor pressure (kPa)

### 2. Contribution Analysis (Equations 4-5, Zhao et al. 2025)

```
RCi = (ETorig - ETclim_i) / Σ(ETorig - ETclim_i) × 100%
```

where forcing i is replaced with its climatological value while others remain unchanged.

### 3. Hurst Coefficient (Markonis 2025)

Estimated via Maximum Likelihood from autocorrelation structure:
- H ≈ 0.5: white noise (no clustering)
- H > 0.5: long-term persistence (clustering)
- H ≈ 0.85: typical for evaporation time series

## 🎓 Citation

If you use this toolkit in your research, please cite the original papers:

```bibtex
@article{markonis2025extreme,
  title={On the Definition of Extreme Evaporation Events},
  author={Markonis, Yannis},
  journal={Geophysical Research Letters},
  volume={52},
  pages={e2024GL113038},
  year={2025},
  doi={10.1029/2024GL113038}
}

@article{zhao2025regional,
  title={Regional Variations in Drivers of Extreme Reference Evapotranspiration 
         Across the Contiguous United States},
  author={Zhao, Bingjie and Horvat, Christopher and Pearson, Christopher and 
          Shah, Deep and Gao, Huilin},
  journal={Water Resources Research},
  volume={61},
  pages={e2025WR040177},
  year={2025},
  doi={10.1029/2025WR040177}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original methodologies by Markonis (2025) and Zhao et al. (2025)
- ASCE-PM equation from Allen et al. (1998) FAO-56
- Inspired by the need for standardized extreme evaporation analysis

## 📧 Contact

For questions or support:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This toolkit uses synthetic data for demonstrations. For research applications, use actual meteorological observations or validated reanalysis products (e.g., gridMET, ERA5-Land, Daymet).