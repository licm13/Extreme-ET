"""
Example replicating Zhao et al. (2025) methodology

This script demonstrates:
1. Penman-Monteith ETo calculation
2. ERT_hist and ERT_clim extreme event detection
3. OPT method for severity-based thresholds
4. Contribution analysis of meteorological drivers
5. Sensitivity to temporal scale and severity level
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the absolute path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.penman_monteith import calculate_et0
from src.extreme_detection import (
    detect_extreme_events_hist,
    detect_extreme_events_clim,
    optimal_path_threshold
)
from src.contribution_analysis import (
    calculate_contributions,
    analyze_seasonal_contributions,
    identify_dominant_driver
)
from src.utils import (
    generate_synthetic_data,
    plot_extreme_events,
    plot_contribution_pie,
    plot_seasonal_contributions
)


def main():
    """
    Main function demonstrating Zhao et al. (2025) methodology.
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*70)
    print("Replicating Zhao et al. (2025): Drivers of Extreme ET0")
    print("="*70)
    
    # Step 1: Generate synthetic meteorological data
    print("\n[Step 1] Generating synthetic meteorological data (40 years)...")
    data = generate_synthetic_data(n_days=365*40, seed=42)
    
    print(f"  Temperature: {data['T_mean'].mean():.1f}°C "
          f"(range: {data['T_mean'].min():.1f} to {data['T_mean'].max():.1f})")
    print(f"  Solar Radiation: {data['Rs'].mean():.1f} MJ/m²/day")
    print(f"  Wind Speed: {data['u2'].mean():.2f} m/s")
    print(f"  Vapor Pressure: {data['ea'].mean():.2f} kPa")
    
    # Step 2: Calculate ET0 using Penman-Monteith
    print("\n[Step 2] Calculating reference ET0 using ASCE-PM equation...")
    ET0 = calculate_et0(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        z=50.0, latitude=40.0, doy=data['doy']
    )
    
    print(f"  Mean ET0: {np.mean(ET0):.2f} mm/day")
    print(f"  Range: {np.min(ET0):.2f} to {np.max(ET0):.2f} mm/day")
    
    # Step 3: Detect extreme events using ERT_hist
    print("\n[Step 3] Detecting extreme events using ERT_hist method...")
    print("  Severity level: 0.5% (approximately 1.8 days/year)")
    
    extreme_hist, threshold_hist, details_hist = detect_extreme_events_hist(
        ET0, severity=0.005, return_details=True
    )
    
    print(f"  Threshold: {threshold_hist:.2f} mm/day")
    print(f"  Extreme days: {details_hist['n_extreme_days']}")
    print(f"  Number of events: {details_hist['n_events']}")
    
    # Step 4: Detect extreme events using ERT_clim
    print("\n[Step 4] Detecting extreme events using ERT_clim method...")
    print("  Severity level: 5% (approximately 18 days/year)")
    print("  Minimum duration: 3 consecutive days")
    print("  Moving average window: 7 days")
    
    extreme_clim, thresholds_clim, details_clim = detect_extreme_events_clim(
        ET0, severity=0.05, min_duration=3, window=7, return_details=True
    )
    
    print(f"  Mean threshold: {np.mean(thresholds_clim):.2f} mm/day")
    print(f"  Extreme days: {details_clim['n_extreme_days']}")
    print(f"  Number of events: {details_clim['n_events']}")
    
    # Step 5: Contribution analysis for ERT_hist extremes
    print("\n[Step 5] Analyzing meteorological contributions (ERT_hist)...")
    
    contributions_hist = calculate_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        extreme_hist
    )
    
    print("  Relative contributions:")
    for driver, contribution in contributions_hist.items():
        print(f"    {driver.capitalize():12s}: {contribution:5.1f}%")
    
    dominant_hist, primary_contrib = identify_dominant_driver(contributions_hist)
    print(f"  Dominant driver: {dominant_hist.capitalize()} ({primary_contrib:.1f}%)")
    
    # Step 6: Contribution analysis for ERT_clim extremes
    print("\n[Step 6] Analyzing meteorological contributions (ERT_clim)...")
    
    contributions_clim = calculate_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        extreme_clim
    )
    
    print("  Relative contributions:")
    for driver, contribution in contributions_clim.items():
        print(f"    {driver.capitalize():12s}: {contribution:5.1f}%")
    
    dominant_clim, _ = identify_dominant_driver(contributions_clim)
    print(f"  Dominant driver: {dominant_clim.capitalize()}")
    
    # Step 7: Seasonal contribution analysis
    print("\n[Step 7] Seasonal contribution analysis...")
    
    seasonal_contrib = analyze_seasonal_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        extreme_hist
    )
    
    for season, contrib in seasonal_contrib.items():
        dominant_season, primary_season = identify_dominant_driver(contrib)
        print(f"  {season.capitalize():6s}: {dominant_season.capitalize()} dominates ({primary_season:.1f}%)")
    
    # Step 8: Compare different severity levels
    print("\n[Step 8] Sensitivity to severity levels...")
    
    severities = [0.001, 0.005, 0.01, 0.025]
    print("  Testing ERT_hist at different severity levels:")
    
    for severity in severities:
        extreme_temp, threshold_temp = detect_extreme_events_hist(ET0, severity=severity)
        n_extreme = np.sum(extreme_temp)
        print(f"    {severity*100:4.1f}%: {n_extreme:4d} extreme days, threshold = {threshold_temp:.2f} mm/day")
    
    # Step 9: Visualizations
    print("\n[Step 9] Creating visualizations...")
    
    # Plot 1: ERT_hist time series (zoom on 2 years)
    start_idx = 365 * 20
    window = (start_idx, start_idx + 365 * 2)
    
    fig1, ax1 = plot_extreme_events(
        ET0, extreme_hist,
        title='ERT_hist Extreme ET0 Events (0.5% severity)',
        ylabel='ET0 (mm/day)',
        window=window
    )
    plt.savefig(os.path.join(output_dir, 'zhao_ert_hist_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_ert_hist_timeseries.png")
    
    # Plot 2: ERT_clim time series
    fig2, ax2 = plot_extreme_events(
        ET0, extreme_clim,
        title='ERT_clim Extreme ET0 Events (5% severity)',
        ylabel='ET0 (mm/day)',
        window=window
    )
    plt.savefig(os.path.join(output_dir, 'zhao_ert_clim_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_ert_clim_timeseries.png")
    
    # Plot 3: Contribution pie chart (ERT_hist)
    fig3, ax3 = plot_contribution_pie(
        contributions_hist,
        title='Meteorological Drivers of Extreme ET0 (ERT_hist)'
    )
    plt.savefig(os.path.join(output_dir, 'zhao_contributions_hist.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_contributions_hist.png")
    
    # Plot 4: Contribution pie chart (ERT_clim)
    fig4, ax4 = plot_contribution_pie(
        contributions_clim,
        title='Meteorological Drivers of Extreme ET0 (ERT_clim)'
    )
    plt.savefig(os.path.join(output_dir, 'zhao_contributions_clim.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_contributions_clim.png")
    
    # Plot 5: Seasonal contributions
    if len(seasonal_contrib) > 0:
        fig5, ax5 = plot_seasonal_contributions(seasonal_contrib)
        plt.savefig(os.path.join(output_dir, 'zhao_seasonal_contributions.png'), dpi=150, bbox_inches='tight')
        print("  Saved: zhao_seasonal_contributions.png")
    
    # Plot 6: Daily threshold variation (ERT_clim)
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    doy_array = np.arange(1, 366)
    ax6.plot(doy_array, thresholds_clim[:365], 'b-', linewidth=2)
    ax6.fill_between(doy_array, 0, thresholds_clim[:365], alpha=0.3)
    ax6.set_xlabel('Day of Year', fontsize=12)
    ax6.set_ylabel('Threshold (mm/day)', fontsize=12)
    ax6.set_title('Daily ET0 Thresholds for Extreme Events (ERT_clim)', 
                  fontsize=14, weight='bold')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zhao_daily_thresholds.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_daily_thresholds.png")
    
    # Plot 7: Comparison of methods
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compare number of extreme days per year
    years = np.arange(len(ET0)) // 365
    years_unique = np.unique(years)
    
    annual_hist = [np.sum(extreme_hist[years == y]) for y in years_unique]
    annual_clim = [np.sum(extreme_clim[years == y]) for y in years_unique]
    
    ax7a.plot(years_unique, annual_hist, 'o-', label='ERT_hist (0.5%)', linewidth=2)
    ax7a.plot(years_unique, annual_clim, 's-', label='ERT_clim (5%)', linewidth=2)
    ax7a.set_xlabel('Year', fontsize=12)
    ax7a.set_ylabel('Number of Extreme Days', fontsize=12)
    ax7a.set_title('Annual Extreme Days', fontsize=12, weight='bold')
    ax7a.legend()
    ax7a.grid(True, alpha=0.3)
    
    # Compare contributions
    drivers = list(contributions_hist.keys())
    x = np.arange(len(drivers))
    width = 0.35
    
    hist_values = [contributions_hist[d] for d in drivers]
    clim_values = [contributions_clim[d] for d in drivers]
    
    ax7b.bar(x - width/2, hist_values, width, label='ERT_hist', alpha=0.8)
    ax7b.bar(x + width/2, clim_values, width, label='ERT_clim', alpha=0.8)
    ax7b.set_xlabel('Driver', fontsize=12)
    ax7b.set_ylabel('Contribution (%)', fontsize=12)
    ax7b.set_title('Driver Contributions', fontsize=12, weight='bold')
    ax7b.set_xticks(x)
    ax7b.set_xticklabels([d.capitalize() for d in drivers], rotation=45, ha='right')
    ax7b.legend()
    ax7b.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zhao_method_comparison.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_method_comparison.png")
    
    print("\n" + "="*70)
    print("Analysis complete! Check the outputs directory for figures.")
    print("\nKey findings:")
    print(f"  - ERT_hist dominant driver: {dominant_hist.capitalize()}")
    print(f"  - ERT_clim dominant driver: {dominant_clim.capitalize()}")
    print(f"  - Temperature contribution: {contributions_hist['temperature']:.1f}% (hist), "
          f"{contributions_clim['temperature']:.1f}% (clim)")
    print("="*70)


if __name__ == "__main__":
    main()