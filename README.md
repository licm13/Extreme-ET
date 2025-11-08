# Extreme Evaporation Events Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python toolkit for analyzing extreme evaporation events, implementing methods from:

- Markonis (2025): "On the Definition of Extreme Evaporation Events" (Geophysical Research Letters)
- Zhao et al. (2025): "Regional Variations in Drivers of Extreme Reference Evapotranspiration Across CONUS" (Water Resources Research)
- Egli et al. (2025): "Detecting Anthropogenically Induced Changes in Extreme and Seasonal Evapotranspiration Observations"

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Methods](#core-methods)
- [Examples](#examples)
- [Multi-Resolution Validation & Paper-Style Outputs](#multi-resolution-validation--paper-style-outputs)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key Equations Implemented](#key-equations-implemented)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features

### Extreme Event Detection
- ERT_hist: historical relative threshold
- ERT_clim: climatological, season-aware thresholds (heatwave-like)
- OPT: optimal day-of-year threshold family for target severity
- ETx7d: annual maximum of 7-day rolling sum (Egli et al. 2025)

### Detection and Attribution (D&A) Framework (New in v1.2.0)
- Ridge regression detector for forced response extraction
- Trend analysis for observations, historical simulations, and piControl
- Anthropogenic signal detection and attribution

### Evapotranspiration Calculation
- ASCE-PM reference ET0
- Net radiation and vapor pressure helpers
- Wind speed height adjustments

### Contribution Analysis
- Driver attribution (temperature, radiation, wind, humidity)
- Seasonal contribution patterns and sensitivity analysis

### Statistical Tools
- Z-score standardization (pentad/daily)
- Hurst coefficient and autocorrelation
- Event identification and metrics

## Installation

### Option 1: pip install (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/extreme-et.git
cd extreme-et

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

Optional (for NetCDF IO and interpolation):

```
xarray>=2023.1.0
netCDF4>=1.6.0
```

## Quick Start

### Example 1: Basic Extreme Detection

```python
import numpy as np
from src.extreme_detection import detect_extreme_events_hist
from src.utils import generate_synthetic_data

data = generate_synthetic_data(n_days=365*40)

# 0.5% severity ≈ 1.8 days/year
extreme_mask, threshold, details = detect_extreme_events_hist(
    data['ET0'] if 'ET0' in data else data['Rs'], severity=0.005, return_details=True
)
print(f"Threshold: {threshold:.2f} | Extreme days: {details['n_extreme_days']}")
```

### Example 2: Penman-Monteith ET0

```python
from src.penman_monteith import calculate_et0
T_mean, T_max, T_min = 20, 25, 15  # °C
Rs, u2, ea = 20.0, 2.0, 1.5
ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea, z=50, latitude=40)
print(f"ET0 = {ET0:.2f} mm/day")
```

### Example 3: Real-Data IO (xarray stub)

```python
from src.io_utils import read_netcdf_variable, sample_series_at_point
path = 'path/to/product.nc'  # edit
var  = 'et'                  # edit
lat, lon = 35.0, -120.0

da, lats, lons, times = read_netcdf_variable(path, var)
series_nn = sample_series_at_point(da, lat, lon, method='nearest')
series_bl = sample_series_at_point(da, lat, lon, method='bilinear')
```

## Core Methods

1) ERT_hist: upper-tail percentile across full record — fast/simple
2) ERT_clim: day-of-year thresholds with smoothing and persistence
3) OPT: DOY threshold family optimized to target occurrence rate

## Examples

Complete workflows are under `examples/`:
- example_markonis_2025.py: ExEvE detection, persistence diagnostics, water cycle decomposition, trends
- example_zhao_2025.py: ET0 diagnostics, ERT_hist/ERT_clim/OPT comparisons, contributions, seasonal patterns
- example_egli_2025.py: Detection & attribution (D&A) of anthropogenic forcing in extreme ET (NEW in v1.2.0)
- example_multires_et_extremes.py: Multi-resolution/frequency validation vs. stations

Outputs are saved to `examples/outputs/`.

## Multi-Resolution Validation & Paper-Style Outputs

We added an example and utilities to evaluate how spatial resolution (0.25° vs 0.1°) and temporal sampling (hourly vs daily) affect extreme ET detection against station observations.

- Example: examples/example_multires_et_extremes.py
  - Compares 0.25° and 0.1° products (daily and hourly→daily aggregates)
  - Methods: ERT_hist (0.1–1%), ERT_clim (5%), OPT (DOY thresholds)
  - Metrics: POD, FAR, CSI with ±1-day tolerance; timing-error histograms
  - Outputs: paper-style figures + IMRaD Markdown manuscript
- Utilities: src/evaluation.py (skill, severity sweeps, timing errors)
- Plotting: src/utils.py (set_paper_style, label_subplots)
- Optional IO: src/io_utils.py (read_netcdf_variable, sample_series_at_point)

Run:

```bash
python examples/example_multires_et_extremes.py
```

## Testing

```bash
cd tests
python test_methods.py
```

Tests cover data processing, detection methods, ET0 calculations, contributions, synthetic generation, and workflows.

## Project Structure

```
Extreme-ET/
├── README.md                         # Docs (this file)
├── requirements.txt                  # Base dependencies
├── setup.py                          # Package setup
├── src/                              # Core package
│  ├── __init__.py
│  ├── data_processing.py             # Z-scores, Hurst, autocorrelation
│  ├── extreme_detection.py           # ERT_hist, ERT_clim, OPT
│  ├── penman_monteith.py             # ASCE-PM equation
│  ├── contribution_analysis.py       # Driver attribution
│  ├── evaluation.py                  # Skill metrics, severity sweeps, timing errors
│  ├── io_utils.py                    # xarray/NetCDF IO + interpolation (optional)
│  └── utils.py                       # Plotting utils and paper style presets
├── examples/
│  ├── example_markonis_2025.py       # Markonis (2025) workflow
│  ├── example_zhao_2025.py           # Zhao et al. (2025) workflow
│  ├── example_multires_et_extremes.py# Multi-resolution & frequency validation
│  └── example_realdata_io_stub.py    # Minimal xarray + sampling demo (edit paths)
├── examples/outputs/                 # Generated figures and summaries
└── tests/
   └── test_methods.py                # Comprehensive test suite
```

## Key Equations Implemented

ASCE Standardized Penman-Monteith:

```
ET0 = [0.408 Δ (Rn - G) + γ (900/(T+273)) u2 (es - ea)] / [Δ + γ(1 + 0.34 u2)]
```

Where Δ is slope of saturation vapor pressure curve; Rn net radiation; γ psychrometric constant; T air temperature; u2 2 m wind; es, ea saturated/actual vapor pressure.

## Citation

Please cite the original papers (Markonis 2025; Zhao et al. 2025) when using this toolkit.

## Contributing

Issues and PRs are welcome: bug reports, features, docs.

## License

MIT License (see LICENSE).

## Acknowledgments

- Methodologies by Markonis (2025) and Zhao et al. (2025)
- ASCE-PM from Allen et al. (1998) FAO-56

## Contact

- Open an issue on GitHub
- Email: your-email@example.com

---

Note: Examples use synthetic data for demonstration; use validated observations/reanalysis for research (e.g., gridMET, ERA5-Land, Daymet).
