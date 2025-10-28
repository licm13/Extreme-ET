"""
Example replicating Markonis (2025) methodology

This script demonstrates:
1. Data standardization to z-scores
2. Autocorrelation analysis and Hurst coefficient calculation
3. Extreme Evaporation Event (ExEvE) detection
4. Analysis of physical drivers (radiation, precipitation)
5. Water cycle decomposition (P-E and (P+E)/2)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.data_processing import (
    standardize_to_zscore,
    calculate_hurst_exponent,
    calculate_autocorrelation
)
from src.extreme_detection import (
    detect_extreme_events_hist,
    identify_events_from_mask,
    calculate_event_statistics
)
from src.utils import (
    generate_synthetic_data,
    plot_extreme_events,
    plot_autocorrelation,
    calculate_event_metrics
)


def main():
    """
    Main function demonstrating Markonis (2025) methodology.
    """
    print("="*70)
    print("Replicating Markonis (2025): Extreme Evaporation Events")
    print("="*70)
    
    # Step 1: Generate synthetic evaporation data
    print("\n[Step 1] Generating synthetic evaporation data (40 years)...")
    n_days = 365 * 40
    np.random.seed(42)
    
    # Generate seasonal evaporation with trends and random variability
    doy = np.arange(n_days) % 365
    years = np.arange(n_days) / 365
    
    # Base evaporation with seasonality (mm/day)
    evap = 1.3 + 1.5 * np.sin(2 * np.pi * doy / 365 - np.pi/2)  # Seasonal cycle
    evap += 0.02 * years  # Long-term trend
    evap += np.random.randn(n_days) * 0.3  # Random noise
    
    # Add occasional extreme events
    extreme_idx = np.random.choice(n_days, size=int(n_days * 0.015), replace=False)
    evap[extreme_idx] += np.random.gamma(2, 0.5, len(extreme_idx))
    
    # Ensure non-negative
    evap = np.maximum(evap, 0.1)
    
    print(f"  Generated {n_days} days of evaporation data")
    print(f"  Mean: {np.mean(evap):.2f} mm/day")
    print(f"  Range: {np.min(evap):.2f} to {np.max(evap):.2f} mm/day")
    
    # Step 2: Standardize to z-scores (remove seasonality)
    print("\n[Step 2] Standardizing data to z-scores over pentads...")
    z_scores = standardize_to_zscore(evap, pentad=True)
    print(f"  Standardized data: mean = {np.mean(z_scores):.3f}, std = {np.std(z_scores):.3f}")
    
    # Step 3: Calculate autocorrelation structure
    print("\n[Step 3] Analyzing autocorrelation structure...")
    lags, autocorr = calculate_autocorrelation(z_scores, max_lag=10)
    hurst = calculate_hurst_exponent(z_scores, max_lag=10)
    
    print(f"  Hurst coefficient: {hurst:.3f}")
    if hurst > 0.5:
        print(f"  → Long-term persistence detected (clustering behavior)")
    else:
        print(f"  → No significant persistence")
    
    # Step 4: Define Extreme Evaporation Events (ExEvEs)
    print("\n[Step 4] Detecting Extreme Evaporation Events (ExEvEs)...")
    print("  Using definition: consecutive days with z-score > 0.8 quantile,")
    print("  with at least one day > 0.95 quantile")
    
    # For simplicity, use ERT_hist method with 0.5% severity
    extreme_mask, threshold, details = detect_extreme_events_hist(
        evap, severity=0.005, return_details=True
    )
    
    print(f"  Threshold: {threshold:.2f} mm/day")
    print(f"  Number of extreme days: {details['n_extreme_days']}")
    print(f"  Number of events: {details['n_events']}")
    print(f"  Occurrence rate: {details['occurrence_rate']*100:.2f}%")
    
    # Calculate event metrics
    events = details['events']
    metrics = calculate_event_metrics(events, evap)
    print(f"  Mean event duration: {metrics['mean_duration']:.1f} days")
    print(f"  Max event duration: {metrics['max_duration']:.0f} days")
    print(f"  Mean intensity during events: {metrics['mean_intensity']:.2f} mm/day")
    
    # Step 5: Analyze temporal trends
    print("\n[Step 5] Analyzing temporal trends...")
    
    # Calculate annual extreme event frequency
    years_unique = np.unique(years.astype(int))
    annual_extreme_days = []
    
    for year in years_unique:
        year_mask = (years.astype(int) == year)
        annual_extreme_days.append(np.sum(extreme_mask[year_mask]))
    
    # Fit linear trend
    trend_coef = np.polyfit(years_unique, annual_extreme_days, 1)[0]
    print(f"  Trend in extreme days: {trend_coef*10:.2f} days/decade")
    
    # Step 6: Water cycle analysis
    print("\n[Step 6] Water cycle decomposition...")
    
    # Generate synthetic precipitation
    precip = 2.0 + 1.0 * np.sin(2 * np.pi * doy / 365)
    precip += np.random.gamma(2, 0.5, n_days)
    precip = np.maximum(precip, 0)
    
    # Calculate P-E (water availability) and (P+E)/2 (water cycle intensity)
    P_minus_E = precip - evap
    P_plus_E_half = (precip + evap) / 2
    
    # Compare during extreme vs non-extreme days
    P_minus_E_extreme = np.mean(P_minus_E[extreme_mask])
    P_minus_E_normal = np.mean(P_minus_E[~extreme_mask])
    
    P_plus_E_extreme = np.mean(P_plus_E_half[extreme_mask])
    P_plus_E_normal = np.mean(P_plus_E_half[~extreme_mask])
    
    print(f"  During ExEvEs:")
    print(f"    P-E (water availability): {P_minus_E_extreme:.2f} mm/day")
    print(f"    (P+E)/2 (water cycle intensity): {P_plus_E_extreme:.2f} mm/day")
    print(f"  During normal days:")
    print(f"    P-E (water availability): {P_minus_E_normal:.2f} mm/day")
    print(f"    (P+E)/2 (water cycle intensity): {P_plus_E_normal:.2f} mm/day")
    
    # Step 7: Visualization
    print("\n[Step 7] Creating visualizations...")
    
    # Plot 1: Autocorrelation
    fig1, ax1 = plot_autocorrelation(lags, autocorr, 
                                      title=f'Autocorrelation Structure (H={hurst:.3f})')
    plt.savefig('/mnt/user-data/outputs/markonis_autocorrelation.png', dpi=150, bbox_inches='tight')
    print("  Saved: markonis_autocorrelation.png")
    
    # Plot 2: Time series with extreme events (zoom in on 3 years)
    start_idx = 365 * 20  # Year 21
    window = (start_idx, start_idx + 365 * 3)
    
    fig2, ax2 = plot_extreme_events(
        evap, extreme_mask,
        title='Evaporation Time Series with Extreme Events',
        ylabel='Evaporation (mm/day)',
        window=window
    )
    plt.savefig('/mnt/user-data/outputs/markonis_timeseries.png', dpi=150, bbox_inches='tight')
    print("  Saved: markonis_timeseries.png")
    
    # Plot 3: Annual trend in extreme events
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(years_unique, annual_extreme_days, color='coral', alpha=0.7, edgecolor='black')
    ax3.plot(years_unique, np.polyval(np.polyfit(years_unique, annual_extreme_days, 1), years_unique),
             'r--', linewidth=2, label=f'Trend: {trend_coef*10:.2f} days/decade')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Number of Extreme Days', fontsize=12)
    ax3.set_title('Temporal Trend in Extreme Evaporation Days', fontsize=14, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/markonis_trend.png', dpi=150, bbox_inches='tight')
    print("  Saved: markonis_trend.png")
    
    # Plot 4: Water cycle comparison
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    
    categories = ['ExEvEs', 'Normal']
    P_E_values = [P_minus_E_extreme, P_minus_E_normal]
    cycle_values = [P_plus_E_extreme, P_plus_E_normal]
    
    ax4a.bar(categories, P_E_values, color=['coral', 'skyblue'], edgecolor='black')
    ax4a.set_ylabel('P - E (mm/day)', fontsize=12)
    ax4a.set_title('Water Availability', fontsize=12, weight='bold')
    ax4a.grid(True, alpha=0.3, axis='y')
    
    ax4b.bar(categories, cycle_values, color=['coral', 'skyblue'], edgecolor='black')
    ax4b.set_ylabel('(P + E) / 2 (mm/day)', fontsize=12)
    ax4b.set_title('Water Cycle Intensity', fontsize=12, weight='bold')
    ax4b.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/markonis_water_cycle.png', dpi=150, bbox_inches='tight')
    print("  Saved: markonis_water_cycle.png")
    
    print("\n" + "="*70)
    print("Analysis complete! Check the outputs directory for figures.")
    print("="*70)


if __name__ == "__main__":
    main()