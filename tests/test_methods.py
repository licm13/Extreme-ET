"""
Comprehensive tests for extreme evaporation analysis methods
"""

import numpy as np
import sys
sys.path.append('..')

from src.data_processing import (
    standardize_to_zscore,
    calculate_hurst_exponent,
    moving_average,
    calculate_autocorrelation
)
from src.extreme_detection import (
    detect_extreme_events_hist,
    detect_extreme_events_clim,
    optimal_path_threshold,
    identify_events_from_mask
)
from src.penman_monteith import (
    calculate_et0,
    calculate_vapor_pressure_from_vpd,
    adjust_wind_speed
)
from src.contribution_analysis import (
    calculate_contributions,
    sensitivity_analysis
)
from src.utils import generate_synthetic_data


def test_data_processing():
    """Test data processing functions."""
    print("Testing data processing functions...")
    
    # Test standardization
    data = np.random.randn(3650)  # 10 years
    z_scores = standardize_to_zscore(data, pentad=True)
    
    assert len(z_scores) == len(data), "Length mismatch"
    assert np.abs(np.mean(z_scores)) < 0.1, "Mean should be near zero"
    print("  ✓ standardize_to_zscore")
    
    # Test Hurst coefficient
    H = calculate_hurst_exponent(data)
    assert 0 <= H <= 1, "Hurst coefficient should be in [0, 1]"
    print(f"  ✓ calculate_hurst_exponent (H = {H:.3f})")
    
    # Test moving average
    smoothed = moving_average(data, window=7)
    assert len(smoothed) == len(data), "Length mismatch"
    assert np.std(smoothed) < np.std(data), "Smoothed data should have lower variance"
    print("  ✓ moving_average")
    
    # Test autocorrelation
    lags, acf = calculate_autocorrelation(data, max_lag=10)
    assert len(lags) == 11, "Should have 11 lags (0-10)"
    assert acf[0] == 1.0, "Zero-lag autocorrelation should be 1"
    print("  ✓ calculate_autocorrelation")
    
    print()


def test_extreme_detection():
    """Test extreme event detection methods."""
    print("Testing extreme event detection...")
    
    # Generate test data with known extremes
    np.random.seed(42)
    data = np.random.gamma(2, 2, 3650)  # 10 years
    
    # Test ERT_hist
    extreme_mask, threshold = detect_extreme_events_hist(data, severity=0.01)
    n_extreme = np.sum(extreme_mask)
    expected = int(len(data) * 0.01)
    
    assert np.abs(n_extreme - expected) <= expected * 0.2, "Occurrence rate mismatch"
    print(f"  ✓ detect_extreme_events_hist ({n_extreme} extreme days)")
    
    # Test with details
    extreme_mask, threshold, details = detect_extreme_events_hist(
        data, severity=0.01, return_details=True
    )
    assert 'n_events' in details, "Details should contain n_events"
    print(f"  ✓ detect_extreme_events_hist with details ({details['n_events']} events)")
    
    # Test ERT_clim
    extreme_mask2, thresholds = detect_extreme_events_clim(
        data, severity=0.05, min_duration=3, window=7
    )
    assert len(thresholds) == 366, "Should have 366 daily thresholds"
    print(f"  ✓ detect_extreme_events_clim ({np.sum(extreme_mask2)} extreme days)")
    
    # Test OPT method
    thresholds_opt = optimal_path_threshold(
        data, target_occurrence_rate=0.05, min_duration=3
    )
    assert len(thresholds_opt) == 366, "Should have 366 thresholds"
    print("  ✓ optimal_path_threshold")
    
    # Test event identification
    test_mask = np.array([False, True, True, True, False, True, True, False])
    events = identify_events_from_mask(test_mask)
    assert len(events) == 2, "Should identify 2 events"
    assert events[0]['duration'] == 3, "First event should last 3 days"
    print("  ✓ identify_events_from_mask")
    
    print()


def test_penman_monteith():
    """Test Penman-Monteith calculations."""
    print("Testing Penman-Monteith equation...")
    
    # Test with typical values
    T_mean, T_max, T_min = 20.0, 25.0, 15.0
    Rs = 20.0  # MJ/m2/day
    u2 = 2.0   # m/s
    ea = 1.5   # kPa
    
    ET0 = calculate_et0(T_mean, T_max, T_min, Rs, u2, ea)
    
    assert isinstance(ET0, (float, np.ndarray)), "ET0 should be numeric"
    assert ET0 > 0, "ET0 should be positive"
    assert ET0 < 20, "ET0 should be reasonable (< 20 mm/day)"
    print(f"  ✓ calculate_et0 (ET0 = {ET0:.2f} mm/day)")
    
    # Test with array inputs
    n = 365
    T_mean_arr = np.full(n, 20.0)
    T_max_arr = np.full(n, 25.0)
    T_min_arr = np.full(n, 15.0)
    Rs_arr = np.full(n, 20.0)
    u2_arr = np.full(n, 2.0)
    ea_arr = np.full(n, 1.5)
    doy_arr = np.arange(1, 366)
    
    ET0_arr = calculate_et0(
        T_mean_arr, T_max_arr, T_min_arr, Rs_arr, u2_arr, ea_arr,
        doy=doy_arr
    )
    
    assert len(ET0_arr) == n, "Array length mismatch"
    assert np.all(ET0_arr > 0), "All ET0 values should be positive"
    print(f"  ✓ calculate_et0 with arrays (mean = {np.mean(ET0_arr):.2f} mm/day)")
    
    # Test vapor pressure calculation
    vpd = 1.0  # kPa
    ea_calc = calculate_vapor_pressure_from_vpd(vpd, 20.0)
    assert ea_calc > 0, "Vapor pressure should be positive"
    print(f"  ✓ calculate_vapor_pressure_from_vpd (ea = {ea_calc:.2f} kPa)")
    
    # Test wind speed adjustment
    u10 = 3.0  # m/s at 10m
    u2_adj = adjust_wind_speed(u10, z=10.0)
    assert u2_adj < u10, "Wind speed at 2m should be less than at 10m"
    print(f"  ✓ adjust_wind_speed ({u10:.1f} m/s → {u2_adj:.2f} m/s)")
    
    print()


def test_contribution_analysis():
    """Test contribution analysis."""
    print("Testing contribution analysis...")
    
    # Generate test data
    data = generate_synthetic_data(n_days=3650, seed=42)
    
    # Calculate ET0
    ET0 = calculate_et0(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea']
    )
    
    # Create extreme mask (top 1%)
    threshold = np.percentile(ET0, 99)
    extreme_mask = ET0 > threshold
    
    # Calculate contributions
    contributions = calculate_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        extreme_mask
    )
    
    # Check results
    assert isinstance(contributions, dict), "Should return dictionary"
    assert len(contributions) == 4, "Should have 4 forcings"
    
    total = sum(contributions.values())
    assert np.abs(total - 100) < 1, f"Contributions should sum to 100% (got {total:.1f}%)"
    
    print("  ✓ calculate_contributions")
    print(f"    Temperature: {contributions['temperature']:.1f}%")
    print(f"    Radiation: {contributions['radiation']:.1f}%")
    print(f"    Wind: {contributions['wind']:.1f}%")
    print(f"    Humidity: {contributions['humidity']:.1f}%")
    
    # Test sensitivity analysis
    sensitivity = sensitivity_analysis(
        data['T_mean'][0], data['T_max'][0], data['T_min'][0],
        data['Rs'][0], data['u2'][0], data['ea'][0],
        perturbation=0.1
    )
    
    assert isinstance(sensitivity, dict), "Should return dictionary"
    assert len(sensitivity) == 4, "Should have 4 sensitivities"
    print("  ✓ sensitivity_analysis")
    
    print()


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")
    
    # Generate data
    data = generate_synthetic_data(n_days=3650, seed=42)
    
    # Check all expected keys are present
    expected_keys = ['T_mean', 'T_max', 'T_min', 'Rs', 'u2', 'ea', 'doy', 'dates']
    for key in expected_keys:
        assert key in data, f"Missing key: {key}"
    
    # Check data properties
    assert len(data['T_mean']) == 3650, "Wrong length"
    assert data['T_max'][0] > data['T_mean'][0], "T_max should be > T_mean"
    assert data['T_mean'][0] > data['T_min'][0], "T_mean should be > T_min"
    assert np.all(data['Rs'] >= 0), "Solar radiation should be non-negative"
    assert np.all(data['u2'] > 0), "Wind speed should be positive"
    
    print("  ✓ generate_synthetic_data")
    print(f"    Temperature: {data['T_mean'].mean():.1f} ± {data['T_mean'].std():.1f}°C")
    print(f"    Solar radiation: {data['Rs'].mean():.1f} ± {data['Rs'].std():.1f} MJ/m²/day")
    print(f"    Wind speed: {data['u2'].mean():.2f} ± {data['u2'].std():.2f} m/s")
    
    print()


def test_full_workflow():
    """Test complete workflow."""
    print("Testing complete workflow...")
    
    # Generate data
    data = generate_synthetic_data(n_days=3650, seed=42)
    
    # Calculate ET0
    ET0 = calculate_et0(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea']
    )
    
    # Detect extremes
    extreme_mask, _, details = detect_extreme_events_hist(
        ET0, severity=0.01, return_details=True
    )
    
    # Calculate contributions
    contributions = calculate_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        extreme_mask
    )
    
    print("  ✓ Complete workflow successful")
    print(f"    Mean ET0: {np.mean(ET0):.2f} mm/day")
    print(f"    Extreme days: {details['n_extreme_days']}")
    print(f"    Dominant driver: {max(contributions, key=contributions.get).capitalize()}")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running Comprehensive Tests")
    print("="*70)
    print()
    
    try:
        test_data_processing()
        test_extreme_detection()
        test_penman_monteith()
        test_contribution_analysis()
        test_synthetic_data_generation()
        test_full_workflow()
        
        print("="*70)
        print("✓ All tests passed successfully!")
        print("="*70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)