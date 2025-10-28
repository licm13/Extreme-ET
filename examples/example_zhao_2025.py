"""Advanced replication of Zhao et al. (2025) methodology.

This expanded example emphasises complex, multi-scenario workflows:

1. Multi-station synthetic meteorology with contrasting climate regimes
2. Data quality control (bias injection + gap infilling) prior to ET0 modelling
3. Penman-Monteith ET0 diagnostics for altitude/latitude sensitivity
4. Severity-varying extreme detection using ERT_hist, ERT_clim, and OPT methods
5. Driver contribution analysis for baseline and perturbed climates
6. Compound-event triage for top-percentile, multi-day extremes
7. Comprehensive visual dashboards comparing methods, severities, and stations
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.penman_monteith import calculate_et0
from src.extreme_detection import (
    detect_extreme_events_hist,
    detect_extreme_events_clim,
    optimal_path_threshold,
)
from src.contribution_analysis import (
    calculate_contributions,
    analyze_seasonal_contributions,
    identify_dominant_driver,
)
from src.utils import (
    generate_synthetic_data,
    plot_extreme_events,
    plot_contribution_pie,
    plot_seasonal_contributions,
)


def _fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs while preserving edge values."""
    arr = arr.astype(float)
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr

    x = np.arange(arr.size)
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _inject_bias(data: Dict[str, np.ndarray], bias_dict: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Apply additive biases to meteorological variables (in-place)."""
    for key, bias in bias_dict.items():
        if key in data:
            data[key] = data[key] + bias
    return data


def _mask_and_fill(data: Dict[str, np.ndarray], mask_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    """Inject random gaps into each series and infill them."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(data.values())))
    mask = rng.random(n) < mask_fraction
    for key in ['T_mean', 'T_max', 'T_min', 'Rs', 'u2', 'ea']:
        series = data[key].astype(float).copy()
        series[mask] = np.nan
        data[key] = _fill_nan_linear(series)
    return data


def main():
    """Run an enhanced Zhao et al. (2025) style driver attribution analysis."""

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 70)
    print("Replicating Zhao et al. (2025) with multi-scenario driver analysis")
    print("=" * 70)

    # Step 1: Generate synthetic meteorological data for contrasting regimes
    print("\n[Step 1] Generating synthetic meteorological datasets (45 years)...")

    n_days = 365 * 45
    scenario_specs = {
        'baseline_valley': {
            'seed': 42,
            'altitude': 80.0,
            'latitude': 40.0,
            'bias': {'T_mean': 0.0, 'Rs': 0.0, 'u2': 0.0},
            'mask_fraction': 0.02,
        },
        'warming_plateau': {
            'seed': 2025,
            'altitude': 1450.0,
            'latitude': 32.0,
            'bias': {'T_mean': 1.8, 'Rs': -1.0, 'u2': 0.4},
            'mask_fraction': 0.035,
        },
    }

    scenarios: Dict[str, Dict[str, np.ndarray]] = {}
    for name, spec in scenario_specs.items():
        print(f"  • {name.replace('_', ' ').title()} (seed={spec['seed']})")
        data = generate_synthetic_data(n_days=n_days, seed=spec['seed'])
        data = _inject_bias(data, spec['bias'])
        data = _mask_and_fill(data, spec['mask_fraction'], seed=spec['seed'] + 99)
        scenarios[name] = data
        print(
            f"    Mean temperature: {np.mean(data['T_mean']):6.2f}°C | "
            f"Solar radiation: {np.mean(data['Rs']):6.2f} MJ/m²/day | "
            f"Wind: {np.mean(data['u2']):4.2f} m/s"
        )

    # Step 2: Calculate ET0 with Penman-Monteith across scenarios
    print("\n[Step 2] Calculating reference ET0 (ASCE-PM) for each regime...")

    scenario_results: Dict[str, Dict[str, np.ndarray]] = {}
    for name, spec in scenario_specs.items():
        data = scenarios[name]
        ET0 = calculate_et0(
            data['T_mean'], data['T_max'], data['T_min'],
            data['Rs'], data['u2'], data['ea'],
            z=spec['altitude'], latitude=spec['latitude'], doy=data['doy']
        )
        scenario_results[name] = {'ET0': ET0, 'data': data, 'spec': spec}
        print(
            f"  {name.replace('_', ' ').title():<20s}: "
            f"Mean={np.mean(ET0):5.2f} mm/day, Range=({np.min(ET0):.2f}, {np.max(ET0):.2f})"
        )

    # Choose baseline scenario for detailed analysis
    baseline_key = 'baseline_valley'
    baseline = scenario_results[baseline_key]
    ET0 = baseline['ET0']
    data = baseline['data']

    # Step 3: Multi-method extreme detection (ERT_hist and ERT_clim)
    print("\n[Step 3] Detecting extremes using ERT_hist and ERT_clim...")

    hist_severity_levels = [0.001, 0.005, 0.01]
    hist_results: Dict[float, Dict[str, np.ndarray]] = {}
    for severity in hist_severity_levels:
        extreme_mask, threshold, details = detect_extreme_events_hist(
            ET0, severity=severity, return_details=True
        )
        hist_results[severity] = {
            'mask': extreme_mask,
            'threshold': threshold,
            'details': details,
        }
        print(
            f"  ERT_hist severity {severity * 100:4.1f}% → "
            f"threshold {threshold:.2f} mm/day, events={details['n_events']}, "
            f"extreme days={details['n_extreme_days']}"
        )

    # ERT_clim with longer persistence constraint
    extreme_clim, thresholds_clim, details_clim = detect_extreme_events_clim(
        ET0, severity=0.025, min_duration=4, window=11, return_details=True
    )
    print(
        f"  ERT_clim severity 2.5% → mean threshold {np.mean(thresholds_clim):.2f} mm/day, "
        f"events={details_clim['n_events']}, extreme days={details_clim['n_extreme_days']}"
    )

    # Step 4: OPT thresholds across severities
    print("\n[Step 4] Deriving OPT daily thresholds across severities...")
    opt_severities = np.logspace(-3, -1, 5)
    opt_threshold_collection: Dict[float, Tuple[np.ndarray, float]] = {}
    for severity in opt_severities:
        thresholds_opt = optimal_path_threshold(
            ET0, target_occurrence_rate=severity, min_duration=3
        )
        day_thresholds = thresholds_opt[data['doy'] - 1]
        mask = ET0 >= day_thresholds
        actual_rate = mask.mean()
        opt_threshold_collection[severity] = (thresholds_opt, actual_rate)
        print(
            f"  OPT target {severity * 100:5.2f}% → realised {actual_rate * 100:5.2f}% occurrence"
        )

    # Step 5: Contribution analysis for multiple scenarios
    print("\n[Step 5] Meteorological driver contributions...")
    contributions_hist = calculate_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        hist_results[0.005]['mask']
    )
    dominant_hist, primary_hist = identify_dominant_driver(contributions_hist)
    print(f"  Baseline (ERT_hist 0.5%) dominant driver: {dominant_hist} ({primary_hist:.1f}%)")

    seasonal_hist = analyze_seasonal_contributions(
        data['T_mean'], data['T_max'], data['T_min'],
        data['Rs'], data['u2'], data['ea'],
        hist_results[0.005]['mask']
    )

    # Compare warming scenario contributions using same mask logic
    warming = scenario_results['warming_plateau']
    contrib_warming = calculate_contributions(
        warming['data']['T_mean'], warming['data']['T_max'], warming['data']['T_min'],
        warming['data']['Rs'], warming['data']['u2'], warming['data']['ea'],
        warming['ET0'] >= hist_results[0.005]['threshold']
    )
    dominant_warm, primary_warm = identify_dominant_driver(contrib_warming)
    print(f"  Warming plateau dominant driver proxy: {dominant_warm} ({primary_warm:.1f}%)")

    # Step 6: Compound-event diagnostics for high-impact sequences
    print("\n[Step 6] High-impact compound extreme diagnostics...")
    reference_mask = hist_results[0.001]['mask']
    rolling_window = 5
    rolling_mean = np.convolve(ET0, np.ones(rolling_window) / rolling_window, mode='same')
    compound_mask = reference_mask & (rolling_mean >= np.quantile(ET0, 0.995))
    print(f"  Ultra-severe days captured: {compound_mask.sum()} (of {reference_mask.sum()} top-tier days)")

    # Step 7: Visualisations
    print("\n[Step 7] Generating visual dashboards...")

    years = np.arange(len(ET0)) // 365
    years_unique = np.unique(years)

    # Plot 1: ERT_hist time series for baseline
    start_idx = 365 * 20
    window = (start_idx, start_idx + 365 * 3)
    fig1, ax1 = plot_extreme_events(
        ET0, hist_results[0.005]['mask'],
        title='ERT_hist Extreme ET0 Events (0.5% severity)',
        ylabel='ET0 (mm/day)',
        window=window,
    )
    ax1.axhline(hist_results[0.005]['threshold'], color='r', linestyle='--', label='0.5% threshold')
    ax1.legend()
    plt.savefig(os.path.join(output_dir, 'zhao_ert_hist_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_ert_hist_timeseries.png")

    # Plot 2: ERT_clim time series
    fig2, ax2 = plot_extreme_events(
        ET0, extreme_clim,
        title='ERT_clim Extreme ET0 Events (2.5% severity)',
        ylabel='ET0 (mm/day)',
        window=window,
    )
    plt.savefig(os.path.join(output_dir, 'zhao_ert_clim_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_ert_clim_timeseries.png")

    # Plot 3: Contribution pie chart (baseline)
    fig3, ax3 = plot_contribution_pie(
        contributions_hist,
        title='Drivers of Extreme ET0 (Baseline, 0.5%)'
    )
    plt.savefig(os.path.join(output_dir, 'zhao_contributions_hist.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_contributions_hist.png")

    # Plot 4: Contribution pie chart (warming scenario)
    fig4, ax4 = plot_contribution_pie(
        contrib_warming,
        title='Drivers of Extreme ET0 (Warming Plateau)'
    )
    plt.savefig(os.path.join(output_dir, 'zhao_contributions_warming.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_contributions_warming.png")

    # Plot 5: Seasonal contributions heatmap (if data available)
    if seasonal_hist:
        fig5, ax5 = plot_seasonal_contributions(seasonal_hist)
        plt.savefig(os.path.join(output_dir, 'zhao_seasonal_contributions.png'), dpi=150, bbox_inches='tight')
        print("  Saved: zhao_seasonal_contributions.png")

    # Plot 6: OPT threshold family
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    doy_array = np.arange(1, 367)
    for severity, (thresholds_opt, actual_rate) in opt_threshold_collection.items():
        ax6.plot(
            doy_array,
            thresholds_opt[:366],
            label=f"Target {severity * 100:4.2f}% → actual {actual_rate * 100:4.2f}%"
        )
    ax6.set_xlabel('Day of Year')
    ax6.set_ylabel('Threshold (mm/day)')
    ax6.set_title('OPT Daily Threshold Families')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zhao_opt_thresholds.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_opt_thresholds.png")

    # Plot 7: Scenario comparison of ET0 distributions
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(np.min(res['ET0']) for res in scenario_results.values()),
                       max(np.max(res['ET0']) for res in scenario_results.values()), 50)
    for name, result in scenario_results.items():
        ax7.hist(result['ET0'], bins=bins, alpha=0.5, label=name.replace('_', ' '), density=True)
    ax7.set_xlabel('ET0 (mm/day)')
    ax7.set_ylabel('Probability Density')
    ax7.set_title('Scenario ET0 Distribution Comparison')
    ax7.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zhao_scenario_distribution.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_scenario_distribution.png")

    # Plot 8: Annual extreme day counts per method
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    annual_hist = {
        sev: [np.sum(mask[years == y]) for y in years_unique]
        for sev, mask in {sev: res['mask'] for sev, res in hist_results.items()}.items()
    }
    annual_clim = [np.sum(extreme_clim[years == y]) for y in years_unique]
    for sev, counts in annual_hist.items():
        ax8.plot(years_unique, counts, label=f"ERT_hist {sev * 100:.1f}%")
    ax8.plot(years_unique, annual_clim, label='ERT_clim 2.5%', linewidth=2, linestyle='--')
    ax8.set_xlabel('Year')
    ax8.set_ylabel('Extreme Days per Year')
    ax8.set_title('Annual Extreme Day Totals by Method')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zhao_method_comparison.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_method_comparison.png")

    # Plot 9: Compound event overlay (baseline)
    fig9, ax9 = plot_extreme_events(
        ET0, compound_mask,
        title='Compound Ultra-Severe ET0 Days',
        ylabel='ET0 (mm/day)',
        window=window,
    )
    plt.savefig(os.path.join(output_dir, 'zhao_compound_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: zhao_compound_timeseries.png")

    print("\n" + "=" * 70)
    print("Analysis complete! Explore the outputs directory for figures.")
    print("Key comparisons:")
    print(
        f"  - Baseline dominant driver: {dominant_hist} ({primary_hist:.1f}%) | "
        f"Warming dominant driver: {dominant_warm} ({primary_warm:.1f}%)"
    )
    print(
        f"  - OPT thresholds span {opt_severities.min() * 100:.2f}% to "
        f"{opt_severities.max() * 100:.2f}% severity targets"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
