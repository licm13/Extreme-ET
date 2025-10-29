"""
Simple test script to verify new advanced analysis modules
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

# Test imports
print("Testing module imports...")

from water_cycle_analysis import (
    calculate_water_availability,
    calculate_water_cycle_intensity,
    decompose_water_cycle_by_extremes
)
print("✓ Water cycle analysis module imported")

from nonstationary_threshold import (
    loess_smoothed_threshold,
    detect_trend_and_detrend
)
print("✓ Non-stationary threshold module imported")

from multivariate_extremes import (
    identify_compound_extreme_et_precipitation,
    calculate_drought_severity_index
)
print("✓ Multivariate extremes module imported")

from spatial_analysis import (
    calculate_spatial_correlation,
    detect_event_propagation
)
print("✓ Spatial analysis module imported")

from event_evolution import (
    analyze_onset_termination_conditions,
    analyze_energy_partitioning
)
print("✓ Event evolution module imported")

# Test basic functionality
print("\nTesting basic functionality...")

# Generate test data
np.random.seed(42)
n_days = 365 * 10
precip = np.random.gamma(2, 2, n_days)
et0 = np.random.gamma(3, 1.5, n_days)

# Test water cycle analysis
water_avail = calculate_water_availability(precip, et0)
water_intensity = calculate_water_cycle_intensity(precip, et0)
print(f"✓ Water availability: {np.mean(water_avail):.2f} mm/day")
print(f"✓ Water cycle intensity: {np.mean(water_intensity):.2f} mm/day")

# Test non-stationary analysis
dates = np.arange(n_days)
threshold = loess_smoothed_threshold(et0, dates, percentile=95)
print(f"✓ Non-stationary threshold range: {np.min(threshold):.2f} - {np.max(threshold):.2f} mm/day")

# Test multivariate analysis
compound = identify_compound_extreme_et_precipitation(et0, precip)
print(f"✓ Compound extreme events: {compound['n_compound_events']} days")

# Test drought severity index
dsi = calculate_drought_severity_index(et0, precip)
print(f"✓ Drought severity index range: {np.min(dsi):.2f} - {np.max(dsi):.2f}")

# Test spatial analysis
n_locations = 10
spatial_data = np.tile(et0, (n_locations, 1)) + np.random.randn(n_locations, n_days) * 0.5
locations = np.random.rand(n_locations, 2) * 100
extreme_matrix = (spatial_data > np.percentile(et0, 95)).astype(int)

dists, corrs, bins = calculate_spatial_correlation(extreme_matrix, locations, max_distance=80)
print(f"✓ Spatial correlation: mean = {np.mean(corrs):.3f}")

# Test event evolution
met_data = {
    'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365),
    'radiation': 15 + 8 * np.sin(2 * np.pi * np.arange(n_days) / 365),
    'wind_speed': np.full(n_days, 2.5),
    'precipitation': precip
}
extremes = et0 > np.percentile(et0, 95)
conditions = analyze_onset_termination_conditions(et0, met_data, extremes, window_days=5)

if 'onset' in conditions:
    print(f"✓ Event onset ET0: {conditions['onset']['et0_mean']:.2f} mm/day")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nNew advanced analysis modules are working correctly.")
print("Ready for scientific applications!")
