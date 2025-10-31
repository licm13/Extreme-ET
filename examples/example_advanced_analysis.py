"""
Advanced Extreme ET Analysis Example

This example demonstrates the new advanced analysis capabilities added to
the Extreme-ET framework, based on insights from:
- Markonis (2025): On the Definition of Extreme Evaporation Events
- Zhao et al. (2025): Regional Variations in Drivers of Extreme ET

Features demonstrated:
1. Water cycle acceleration analysis (P-E and (P+E)/2)
2. Non-stationary threshold evolution
3. Multivariate extreme analysis (Copula-based compound events)
4. Spatial analysis (coherence, propagation, Kriging)
5. Event physical evolution (onset/termination analysis)
"""

import os
import sys
# Ensure project root (parent of 'examples' and 'src') is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from src.extreme_detection import detect_extreme_events_hist
from src.penman_monteith import (
    calculate_et0,
    calculate_vapor_pressure_from_vpd,
)
from src.data_processing import calculate_hurst_exponent
from src.water_cycle_analysis import (
    decompose_water_cycle_by_extremes,
    analyze_temporal_changes,
    classify_water_cycle_regime,
    analyze_seasonal_water_cycle
)
from src.nonstationary_threshold import (
    detect_trend_and_detrend,
    compare_stationary_vs_nonstationary
)
from src.multivariate_extremes import (
    identify_compound_extreme_et_precipitation,
    calculate_drought_severity_index
)
from src.spatial_analysis import (
    calculate_spatial_correlation,
    detect_event_propagation,
    ordinary_kriging
)
from src.event_evolution import (
    analyze_onset_termination_conditions,
    analyze_energy_partitioning,
    identify_event_triggers,
    analyze_event_intensity_evolution
)

def generate_synthetic_data_with_trends(n_years=50, n_locations=20):
    """
    Generate realistic synthetic data with climate trends for demonstration.
    """
    n_days = n_years * 365
    dates = pd.date_range('1980-01-01', periods=n_days, freq='D')

    # Add climate change trend
    time_trend = np.arange(n_days) / 365  # Years
    trend_factor = 1 + 0.015 * time_trend  # 1.5% per decade increase

    # Seasonal component
    day_of_year = dates.dayofyear
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    seasonal_rad = 15 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Generate meteorological data with trend
    temperature = seasonal_temp * trend_factor + np.random.randn(n_days) * 2
    radiation = seasonal_rad * np.sqrt(trend_factor) + np.random.randn(n_days) * 3
    wind_speed = 2.5 + np.random.gamma(2, 0.5, n_days)

    # VPD increases with temperature
    vpd = 0.5 + 0.1 * temperature + np.random.randn(n_days) * 0.2
    vpd = np.clip(vpd, 0.1, 5.0)

    # Convert VPD to actual vapor pressure and calculate ET0
    ea = calculate_vapor_pressure_from_vpd(vpd, temperature)
    et0 = calculate_et0(
        temperature,
        temperature + 3,
        temperature - 3,
        radiation,
        wind_speed,
        ea,
        z=100,
        latitude=40,
    )

    # Generate precipitation (decreasing trend to simulate aridification)
    precip_trend = 1 - 0.005 * time_trend  # Slight decrease
    seasonal_precip = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    precipitation = seasonal_precip * precip_trend * np.random.gamma(2, 1, n_days)
    precipitation = np.clip(precipitation, 0, None)

    # Generate spatial data (multiple locations)
    locations = np.random.rand(n_locations, 2) * 100  # Random locations
    spatial_et0 = np.tile(et0, (n_locations, 1)) + np.random.randn(n_locations, n_days) * 0.5

    return {
        'dates': dates,
        'temperature': temperature,
        'radiation': radiation,
        'wind_speed': wind_speed,
        'vpd': vpd,
        'et0': et0,
        'precipitation': precipitation,
        'locations': locations,
        'spatial_et0': spatial_et0,
        'n_years': n_years
    }


def demo_water_cycle_analysis(data):
    """
    Demonstrate water cycle acceleration analysis.
    """
    print("\n" + "="*80)
    print("1. WATER CYCLE ACCELERATION ANALYSIS (Markonis 2025)")
    print("="*80)

    # Detect extremes
    extreme_mask_hist, threshold_hist, details_hist = detect_extreme_events_hist(
        data['et0'], severity=0.005, return_details=True
    )

    # Decompose water cycle
    decomposition = decompose_water_cycle_by_extremes(
        data['precipitation'],
        data['et0'],
        extreme_mask_hist
    )

    print(f"\nThreshold (0.5% severity): {threshold_hist:.2f} mm/day")
    print(f"Extreme days detected: {details_hist['n_extreme_days']} "
          f"({details_hist['occurrence_rate']*100:.2f}% of days)")
    print("\nWater Cycle Metrics:")
    print("-" * 40)
    for condition in ['all_days', 'extreme_days', 'normal_days']:
        if decomposition[condition] is not None:
            print(f"\n{condition.upper()}:")
            print(f"  Water Availability (P-E): {decomposition[condition]['water_availability_mean']:.2f} mm/day")
            print(f"  Water Cycle Intensity ((P+E)/2): {decomposition[condition]['water_cycle_intensity_mean']:.2f} mm/day")
            print(f"  Mean Precipitation: {decomposition[condition]['p_mean']:.2f} mm/day")
            print(f"  Mean ET0: {decomposition[condition]['e_mean']:.2f} mm/day")

    # Temporal changes
    split_idx = len(data['et0']) // 2  # Split at midpoint
    temporal_changes = analyze_temporal_changes(
        data['precipitation'],
        data['et0'],
        extreme_mask_hist,
        split_idx
    )

    print("\n\nTemporal Changes (Period 2 vs Period 1):")
    print("-" * 40)
    if temporal_changes['changes']['extreme_days'] is not None:
        changes = temporal_changes['changes']['extreme_days']
        print(f"Water Availability Change: {(changes['water_availability_ratio']-1)*100:+.1f}%")
        print(f"Water Cycle Intensity Change: {(changes['water_cycle_intensity_ratio']-1)*100:+.1f}%")
        print(f"Extreme Event Frequency Change: {(changes['extreme_days_ratio']-1)*100:+.1f}%")

        # Classify regime
        wa_change = changes['water_availability_ratio'] - 1
        wci_change = changes['water_cycle_intensity_ratio'] - 1
        regime = classify_water_cycle_regime(wa_change, wci_change)
        print(f"\nWater Cycle Regime Classification: {regime}")

    # Seasonal analysis
    seasonal_stats = analyze_seasonal_water_cycle(
        data['dates'],
        data['precipitation'],
        data['et0'],
        extreme_mask_hist
    )

    print("\n\nSeasonal Water Cycle Analysis:")
    print("-" * 40)
    print(seasonal_stats[['season_name', 'water_availability_mean',
                          'water_intensity_mean', 'extreme_frequency']])

    return {
        'mask': extreme_mask_hist,
        'threshold': threshold_hist,
        'details': details_hist,
        'decomposition': decomposition,
        'seasonal_stats': seasonal_stats,
    }


def demo_nonstationary_analysis(data):
    """
    Demonstrate non-stationary threshold analysis.
    """
    print("\n" + "="*80)
    print("2. NON-STATIONARY THRESHOLD ANALYSIS (Zhao et al. 2025)")
    print("="*80)

    dates_numeric = np.arange(len(data['et0']))

    # Detect trend
    slope, p_value, detrended, trend_stats = detect_trend_and_detrend(
        data['et0'], dates_numeric
    )

    print(f"\nTrend Analysis:")
    print("-" * 40)
    print(f"Trend Slope: {slope:.6f} mm/day per day")
    print(f"  (= {slope*365:.3f} mm/day per year)")
    print(f"P-value: {p_value:.4e}")
    print(f"Trend Direction: {trend_stats['trend_direction']}")
    print(f"Trend Significance: {trend_stats['trend_significance']}")
    print(f"R-squared: {trend_stats['r_squared']:.4f}")

    # Compare stationary vs non-stationary
    comparison = compare_stationary_vs_nonstationary(
        data['et0'], dates_numeric, percentile=95.0
    )

    print(f"\n\nStationary vs Non-Stationary Comparison:")
    print("-" * 40)
    print(f"Stationary threshold: {comparison['stationary_threshold']:.2f} mm/day")
    print(f"Non-stationary threshold range: {np.min(comparison['nonstationary_threshold']):.2f} - "
          f"{np.max(comparison['nonstationary_threshold']):.2f} mm/day")
    print(f"\nExtreme events detected:")
    print(f"  Stationary method: {comparison['n_stationary_extremes']} events")
    print(f"  Non-stationary method: {comparison['n_nonstationary_extremes']} events")
    print(f"  Difference: {comparison['n_extremes_difference']:+d} events")

    return comparison


def demo_multivariate_extremes(data):
    """
    Demonstrate multivariate extreme analysis with copulas.
    """
    print("\n" + "="*80)
    print("3. MULTIVARIATE EXTREME ANALYSIS (Copula-based)")
    print("="*80)

    # Identify compound extremes
    compound_results = identify_compound_extreme_et_precipitation(
        data['et0'],
        data['precipitation'],
        et0_percentile=95.0,
        precip_percentile=5.0
    )

    print(f"\nCompound Extreme Events (High ET0 + Low Precipitation):")
    print("-" * 40)
    print(f"Total compound extreme days: {compound_results['n_compound_events']}")
    print(f"ET0-only extreme days: {compound_results['n_et0_only']}")
    print(f"Precipitation-only extreme days: {compound_results['n_precip_only']}")
    print(f"Compound event frequency: {compound_results['compound_frequency']:.2%}")

    print(f"\nThresholds:")
    print(f"  ET0 threshold: {compound_results['et0_threshold']:.2f} mm/day")
    print(f"  Precipitation threshold: {compound_results['precip_threshold']:.2f} mm/day")

    # Copula analysis
    copula = compound_results['copula_analysis']
    print(f"\nCopula Analysis:")
    print(f"  Copula type: {copula['copula_type']}")
    print(f"  Copula parameter: {copula['copula_parameter']:.3f}")
    print(f"\nReturn Periods:")
    print(f"  ET0 alone: {copula['marginal_return_period_1']:.1f} events")
    print(f"  Precip alone: {copula['marginal_return_period_2']:.1f} events")
    print(f"  Joint (AND): {copula['joint_and_return_period']:.1f} events")
    print(f"  Joint (OR): {copula['joint_or_return_period']:.1f} events")

    # Drought severity index
    dsi = calculate_drought_severity_index(
        data['et0'],
        data['precipitation']
    )

    print(f"\nDrought Severity Index:")
    print(f"  Mean DSI: {np.mean(dsi):.2f}")
    print(f"  Severe drought days (DSI > 95th percentile): {np.sum(dsi > np.percentile(dsi, 95))}")

    return compound_results, dsi


def demo_spatial_analysis(data):
    """
    Demonstrate spatial analysis capabilities.
    """
    print("\n" + "="*80)
    print("4. SPATIAL ANALYSIS")
    print("="*80)

    # Spatial correlation
    extreme_matrix = (data['spatial_et0'] > np.percentile(data['et0'], 95)).astype(int)

    distances, correlations, dist_bins = calculate_spatial_correlation(
        extreme_matrix,
        data['locations'],
        max_distance=80.0
    )

    print(f"\nSpatial Correlation Analysis:")
    print("-" * 40)
    print(f"Number of location pairs analyzed: {len(distances)}")
    print(f"Mean correlation: {np.mean(correlations):.3f}")
    print(f"Correlation range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")

    # Event propagation
    propagation = detect_event_propagation(
        extreme_matrix,
        data['locations'],
        np.arange(len(data['et0'])),
        max_lag_days=5
    )

    print(f"\nEvent Propagation Analysis:")
    print("-" * 40)
    print(f"Optimal lag: {propagation['optimal_lag']} days")
    print(f"Propagation speed: {propagation['propagation_speed']:.1f} km/day")
    print(f"Maximum lag correlation: {propagation['max_correlation']:.3f}")

    # Kriging interpolation example
    n_known = 15
    known_idx = np.random.choice(len(data['locations']), n_known, replace=False)
    known_points = data['locations'][known_idx]
    known_values = data['et0'][-1] * np.ones(n_known)  # Last day's ET0

    target_points = np.random.rand(30, 2) * 100
    interp_vals, interp_var = ordinary_kriging(
        known_points, known_values, target_points
    )

    print(f"\nKriging Interpolation:")
    print("-" * 40)
    print(f"Interpolated {len(target_points)} points from {n_known} known locations")
    print(f"Interpolated values range: [{np.min(interp_vals):.2f}, {np.max(interp_vals):.2f}] mm/day")
    print(f"Mean interpolation variance: {np.mean(interp_var):.3f}")

    return propagation


def demo_event_evolution(data):
    """
    Demonstrate event physical evolution analysis.
    """
    print("\n" + "="*80)
    print("5. EVENT PHYSICAL EVOLUTION ANALYSIS (Markonis 2025)")
    print("="*80)

    # Detect extremes
    extreme_mask, _, details = detect_extreme_events_hist(
        data['et0'], severity=0.01, return_details=True
    )

    # Meteorological data for analysis
    met_data = {
        'temperature': data['temperature'],
        'radiation': data['radiation'],
        'wind_speed': data['wind_speed'],
        'precipitation': data['precipitation']
    }

    # Onset and termination conditions
    conditions = analyze_onset_termination_conditions(
        data['et0'],
        met_data,
        extreme_mask,
        window_days=5
    )

    if 'error' not in conditions:
        print(f"\nOnset vs Termination Conditions:")
        print("-" * 40)

        for phase in ['onset', 'termination']:
            print(f"\n{phase.upper()}:")
            print(f"  ET0: {conditions[phase]['et0_mean']:.2f} ± {conditions[phase]['et0_std']:.2f} mm/day")
            print(f"  Temperature: {conditions[phase]['temperature_mean']:.2f} ± {conditions[phase]['temperature_std']:.2f} °C")
            print(f"  Radiation: {conditions[phase]['radiation_mean']:.2f} ± {conditions[phase]['radiation_std']:.2f} MJ/m²/day")
            print(f"  Precipitation: {conditions[phase]['precipitation_mean']:.2f} ± {conditions[phase]['precipitation_std']:.2f} mm/day")

        # Energy partitioning
        energy = analyze_energy_partitioning(
            data['temperature'],
            data['radiation'],
            data['et0'],
            extreme_mask
        )

        print(f"\n\nEnergy Partitioning:")
        print("-" * 40)
        for condition in ['extreme', 'normal']:
            print(f"\n{condition.upper()} conditions:")
            print(f"  Latent heat flux: {energy[condition]['latent_heat_flux_mean']:.2f} MJ/m²/day")
            print(f"  Sensible heat: {energy[condition]['sensible_heat_mean']:.2f} MJ/m²/day")
            print(f"  Bowen ratio: {energy[condition]['bowen_ratio_mean']:.3f}")
            print(f"  Latent heat fraction: {energy[condition]['latent_heat_fraction']:.2%}")

        # Event triggers
        triggers = identify_event_triggers(met_data, extreme_mask, lookback_days=7)

        if 'error' not in triggers:
            print(f"\n\nEvent Triggers (7-day lookback):")
            print("-" * 40)
            for var, info in triggers.items():
                if info['is_positive_trigger']:
                    print(f"{var}: Mean anomaly = {info['mean_pre_event_anomaly']:+.2f} σ "
                          f"(consistency: {info['consistency']:.0%})")

        # Event intensity evolution
        evolutions = analyze_event_intensity_evolution(data['et0'], extreme_mask)

        if len(evolutions) > 0:
            print(f"\n\nEvent Intensity Evolution:")
            print("-" * 40)
            mean_duration = np.mean([e['duration'] for e in evolutions])
            mean_peak = np.mean([e['peak_intensity'] for e in evolutions])
            mean_cumulative = np.mean([e['cumulative_et'] for e in evolutions])

            print(f"Number of events: {len(evolutions)}")
            print(f"Mean duration: {mean_duration:.1f} days")
            print(f"Mean peak intensity: {mean_peak:.2f} mm/day")
            print(f"Mean cumulative ET per event: {mean_cumulative:.1f} mm")

    summary = {
        'mask': extreme_mask,
        'details': details,
        'conditions': conditions,
        'evolutions': evolutions if 'error' not in conditions else [],
    }
    return summary


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "="*80)
    print("ADVANCED EXTREME ET ANALYSIS DEMONSTRATION")
    print("Based on Markonis (2025) and Zhao et al. (2025)")
    print("="*80)

    # Generate data
    print("\nGenerating synthetic data with climate trends...")
    data = generate_synthetic_data_with_trends(n_years=50, n_locations=20)
    print(f"Generated {data['n_years']} years of daily data ({len(data['et0'])} days)")
    print(f"Spatial data: {len(data['locations'])} locations")

    # Calculate Hurst exponent to verify persistence
    hurst = calculate_hurst_exponent(data['et0'])
    print(f"\nHurst Exponent (persistence): {hurst:.3f}")
    if hurst > 0.5:
        print("  → Long-term persistence detected (clustering behavior)")

    # Run demonstrations
    try:
        demo_water_cycle_analysis(data)
        demo_nonstationary_analysis(data)
        demo_multivariate_extremes(data)
        demo_spatial_analysis(data)
        demo_event_evolution(data)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nAll advanced analysis modules have been successfully demonstrated.")
        print("These new capabilities significantly extend the Extreme-ET framework")
        print("with cutting-edge methods from recent literature.")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
