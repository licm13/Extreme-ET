# Getting Started with Extreme Evaporation Analysis

This guide will help you quickly get started with the Extreme Evaporation Events Analysis Toolkit.

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Organize Files

Create this directory structure:
```
your_project/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── extreme_detection.py
│   ├── penman_monteith.py
│   ├── contribution_analysis.py
│   └── utils.py
├── examples/
│   ├── example_markonis_2025.py
│   └── example_zhao_2025.py
└── tests/
    └── test_methods.py
```

### Step 3: Run Examples

```bash
# Test that everything works
python tests/test_methods.py

# Run Markonis (2025) example
python examples/example_markonis_2025.py

# Run Zhao et al. (2025) example
python examples/example_zhao_2025.py
```

## 📝 Minimal Working Example

Create a file called `my_first_analysis.py`:

```python
import numpy as np
import sys
sys.path.append('.')  # Adjust path as needed

from src.utils import generate_synthetic_data
from src.penman_monteith import calculate_et0
from src.extreme_detection import detect_extreme_events_hist
from src.contribution_analysis import calculate_contributions

# Step 1: Generate data (or load your own)
print("Step 1: Generating synthetic data...")
data = generate_synthetic_data(n_days=365*10, seed=42)

# Step 2: Calculate ET0
print("Step 2: Calculating ET0...")
ET0 = calculate_et0(
    data['T_mean'], data['T_max'], data['T_min'],
    data['Rs'], data['u2'], data['ea']
)
print(f"  Mean ET0: {np.mean(ET0):.2f} mm/day")

# Step 3: Detect extreme events
print("Step 3: Detecting extreme events...")
extreme_mask, threshold = detect_extreme_events_hist(
    ET0, severity=0.01  # 1% extremes
)
n_extreme = np.sum(extreme_mask)
print(f"  Threshold: {threshold:.2f} mm/day")
print(f"  Extreme days: {n_extreme}")

# Step 4: Analyze contributions
print("Step 4: Analyzing meteorological contributions...")
contributions = calculate_contributions(
    data['T_mean'], data['T_max'], data['T_min'],
    data['Rs'], data['u2'], data['ea'],
    extreme_mask
)

print("  Relative contributions:")
for driver, percent in contributions.items():
    print(f"    {driver.capitalize():12s}: {percent:5.1f}%")

print("\nAnalysis complete!")
```

Run it:
```bash
python my_first_analysis.py
```

Expected output:
```
Step 1: Generating synthetic data...
Step 2: Calculating ET0...
  Mean ET0: 2.87 mm/day
Step 3: Detecting extreme events...
  Threshold: 5.23 mm/day
  Extreme days: 36
Step 4: Analyzing meteorological contributions...
  Relative contributions:
    Temperature :  42.3%
    Radiation   :  28.7%
    Wind        :  18.9%
    Humidity    :  10.1%

Analysis complete!
```

## 🎓 Understanding the Output

### 1. ET0 Values
- Typical range: 1-8 mm/day
- Higher in summer, lower in winter
- Affected by all four meteorological forcings

### 2. Extreme Thresholds
- **ERT_hist**: Single threshold for entire period
- **ERT_clim**: Different threshold for each calendar day
- Severity controls occurrence frequency

### 3. Contributions
- Sum to 100%
- Temperature usually dominant (30-60%)
- Regional variations significant
- Seasonal patterns important

## 📊 Visualizing Results

Add these lines to create plots:

```python
import matplotlib.pyplot as plt
from src.utils import plot_extreme_events, plot_contribution_pie

# Plot time series
fig1, ax1 = plot_extreme_events(
    ET0, extreme_mask,
    title='Extreme ET0 Events',
    ylabel='ET0 (mm/day)',
    window=(0, 365*2)  # First 2 years
)
plt.savefig('extreme_events.png', dpi=150)

# Plot contributions
fig2, ax2 = plot_contribution_pie(
    contributions,
    title='Meteorological Driver Contributions'
)
plt.savefig('contributions.png', dpi=150)

plt.show()
```

## 🔧 Working with Your Own Data

Replace the synthetic data generation with your own data:

```python
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Prepare meteorological forcings
T_mean = df['temperature_mean'].values  # °C
T_max = df['temperature_max'].values    # °C
T_min = df['temperature_min'].values    # °C
Rs = df['solar_radiation'].values       # MJ m-2 day-1
u2 = df['wind_speed_2m'].values        # m s-1
ea = df['vapor_pressure'].values        # kPa

# Calculate ET0
from src.penman_monteith import calculate_et0

ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea,
                   z=50,           # Your elevation (m)
                   latitude=40)    # Your latitude (degrees)

# Continue with extreme detection and analysis...
```

## 🎯 Common Use Cases

### Use Case 1: Identify Extreme Days
```python
from src.extreme_detection import detect_extreme_events_hist

# Find top 0.5% of days
extreme_mask, threshold = detect_extreme_events_hist(
    ET0, severity=0.005
)

# Get dates of extreme events
extreme_dates = dates[extreme_mask]
print(f"Extreme dates: {extreme_dates}")
```

### Use Case 2: Compare Two Periods
```python
# Split data into two periods
period1_mask = (years >= 1981) & (years < 2000)
period2_mask = (years >= 2000) & (years <= 2020)

# Detect extremes in each period
extreme1, _ = detect_extreme_events_hist(
    ET0[period1_mask], severity=0.01
)
extreme2, _ = detect_extreme_events_hist(
    ET0[period2_mask], severity=0.01
)

print(f"Period 1: {np.sum(extreme1)} extreme days")
print(f"Period 2: {np.sum(extreme2)} extreme days")
```

### Use Case 3: Seasonal Analysis
```python
from src.contribution_analysis import analyze_seasonal_contributions

# Calculate contributions by season
seasonal_contrib = analyze_seasonal_contributions(
    T_mean, T_max, T_min, Rs, u2, ea,
    extreme_mask
)

# Print results
for season, contrib in seasonal_contrib.items():
    dominant = max(contrib, key=contrib.get)
    print(f"{season}: {dominant.capitalize()} dominates "
          f"({contrib[dominant]:.1f}%)")
```

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution**: Check that `src/` is in your Python path:
```python
import sys
sys.path.append('path/to/src')
```

### Problem: "ET0 values are unrealistic"
**Solution**: Check your input units:
- Temperature: °C
- Solar radiation: MJ m⁻² day⁻¹ (not W m⁻²)
- Wind speed: m s⁻¹ at 2m height
- Vapor pressure: kPa

### Problem: "No extreme events detected"
**Solution**: 
- Lower the severity threshold (e.g., 0.05 instead of 0.005)
- Check that your data has enough variability
- Verify data quality (no missing values)

### Problem: "Contributions don't sum to 100%"
**Solution**: This is usually due to numerical precision. It's normal if they sum to 99-101%.

## 📚 Next Steps

1. **Read the full README.md** for detailed method descriptions
2. **Run the example scripts** to see complete workflows
3. **Explore seasonal patterns** in your region
4. **Compare different severity levels** (0.005, 0.01, 0.05)
5. **Validate with known extreme events** in your study area

## 💡 Tips for Research

1. **Use multiple severity levels**: Compare 0.5%, 1%, 2.5%, 5% to understand sensitivity
2. **Check both methods**: ERT_hist for magnitude, ERT_clim for anomalies
3. **Validate with observations**: Compare with known drought/heat events
4. **Consider spatial patterns**: Apply to multiple sites/regions
5. **Analyze trends**: Look at changes over time periods

## 📖 Additional Resources

- **FAO-56**: Allen et al. (1998) for Penman-Monteith details
- **Markonis (2025)**: Original ExEvE framework paper
- **Zhao et al. (2025)**: Regional driver analysis methodology
- **Test suite**: Run `test_methods.py` for validation examples

## 🆘 Getting Help

1. Check error messages carefully
2. Review the example scripts
3. Run the test suite to verify installation
4. Open an issue on GitHub with:
   - Your Python version
   - Error message (full traceback)
   - Minimal code to reproduce the problem

---

Happy analyzing! 🎉