"""Example replicating (and extending) the Markonis (2025) methodology

This script now showcases a substantially more complex workflow:

1. Synthetic multi-regime evaporation generation with drought/heatwave blocks
2. Data quality control (missing data injection + interpolation recovery)
3. Multi-timescale standardization (pentad & monthly) and persistence diagnostics
4. Severity-stratified Extreme Evaporation Event (ExEvE) detection & comparison
5. Compound-event analysis with precipitation deficits and soil moisture proxies
6. Water cycle decomposition (P-E and (P+E)/2) for extreme vs. normal regimes
7. Advanced visualisations summarising variability across severities & regimes
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the absolute path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.data_processing import (
    standardize_to_zscore,
    calculate_hurst_exponent,
    calculate_autocorrelation
)
from src.extreme_detection import (
    detect_extreme_events_hist,
)
from src.utils import (
    plot_extreme_events,
    calculate_event_metrics
)


def main():
    """Run an extended Markonis (2025) style Extreme Evaporation analysis."""
    print("=" * 70)
    print("Replicating Markonis (2025) with extended ExEvE diagnostics")
    print("=" * 70)

    # Step 1: Generate synthetic evaporation data with compound regimes
    print("\n[Step 1] Generating synthetic evaporation data (45 years, 2 regimes)...")
    n_days = 365 * 45
    np.random.seed(42)

    doy = np.arange(n_days) % 365
    years = np.arange(n_days) / 365

    # Construct a baseline seasonal cycle with multi-decadal oscillation
    base_cycle = 1.4 + 1.6 * np.sin(2 * np.pi * doy / 365 - np.pi / 2)
    oscillation = 0.4 * np.sin(2 * np.pi * years / 7.5)  # quasi-decadal mode

    # Introduce two climate regimes (e.g., maritime vs continental influence)
    regime_switch = (years % 6 >= 3).astype(float)
    continental_boost = regime_switch * (0.4 + 0.3 * np.sin(2 * np.pi * doy / 365))

    evap = base_cycle + oscillation + continental_boost
    evap += 0.018 * years  # long-term warming trend
    evap += np.random.normal(0, 0.35, n_days)

    # Embed multi-week drought/heatwave blocks for realism
    block_centers = np.random.choice(np.arange(60, n_days - 60, 120), size=18, replace=False)
    for center in block_centers:
        block = slice(center - 15, center + 20)
        evap[block] += np.linspace(0.0, 1.4, block.stop - block.start)

    # Add occasional sharp spikes to mimic sudden advection events
    spike_idx = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    evap[spike_idx] += np.random.gamma(3, 0.6, len(spike_idx))

    # Ensure physical bounds
    evap = np.clip(evap, 0.05, None)

    print(f"  Generated {n_days} days of evaporation data")
    print(f"  Mean 卤 std: {np.mean(evap):.2f} 卤 {np.std(evap):.2f} mm/day")
    print(f"  Range: {np.min(evap):.2f} to {np.max(evap):.2f} mm/day")

    # Step 1b: Inject and repair missing observations
    print("\n[Step 1b] Injecting 3% missing values and applying linear infill...")
    missing_mask = np.random.choice([False, True], size=n_days, p=[0.97, 0.03])
    evap_with_gaps = evap.copy()
    evap_with_gaps[missing_mask] = np.nan

    gap_count = np.sum(missing_mask)
    print(f"  Injected {gap_count} missing observations")

    # Linear interpolation while preserving NaNs at edges
    valid_idx = np.where(~np.isnan(evap_with_gaps))[0]
    interp_evap = np.interp(
        np.arange(n_days),
        valid_idx,
        evap_with_gaps[valid_idx]
    )

    # Restore original missing edges if any
    if np.isnan(evap_with_gaps[0]):
        first_valid = valid_idx[0]
        interp_evap[:first_valid] = evap_with_gaps[first_valid]
    if np.isnan(evap_with_gaps[-1]):
        last_valid = valid_idx[-1]
        interp_evap[last_valid + 1:] = evap_with_gaps[last_valid]

    evap = interp_evap
    print("  Missing data successfully infilled.")

    # Step 2: Standardize at multiple temporal scales
    print("\n[Step 2] Standardizing data to z-scores over pentads and months...")
    z_scores_pentad = standardize_to_zscore(evap, pentad=True)
    z_scores_monthly = standardize_to_zscore(evap, pentad=False)

    print(f"  Pentad mean 卤 std: {np.mean(z_scores_pentad):.3f} 卤 {np.std(z_scores_pentad):.3f}")
    print(f"  Monthly mean 卤 std: {np.mean(z_scores_monthly):.3f} 卤 {np.std(z_scores_monthly):.3f}")

    # Step 3: Persistence diagnostics for both scales
    print("\n[Step 3] Assessing persistence across temporal scales...")
    lags_pentad, autocorr_pentad = calculate_autocorrelation(z_scores_pentad, max_lag=12)
    lags_monthly, autocorr_monthly = calculate_autocorrelation(z_scores_monthly, max_lag=12)

    hurst_pentad = calculate_hurst_exponent(z_scores_pentad, max_lag=12)
    hurst_monthly = calculate_hurst_exponent(z_scores_monthly, max_lag=12)

    print(f"  Hurst (pentad):  {hurst_pentad:.3f}")
    print(f"  Hurst (monthly): {hurst_monthly:.3f}")
    print(
        "  Persistence is stronger at coarser (monthly) scale"
        if hurst_monthly > hurst_pentad
        else "  Persistence comparable across scales"
    )
    severities = [0.001, 0.005, 0.01]
    severity_results = {}

    for sev in severities:
        extreme_mask, threshold, details = detect_extreme_events_hist(
            evap, severity=sev, return_details=True
        )
        metrics = calculate_event_metrics(details['events'], evap)
        severity_results[sev] = {
            'mask': extreme_mask,
            'threshold': threshold,
            'details': details,
            'metrics': metrics
        }
        print(
            f"  Severity {sev * 100:4.1f}% - threshold {threshold:.2f} mm/day, "
            f"{details['n_events']} events, mean duration {metrics['mean_duration']:.1f} days"
        )
    # Choose a reference severity (0.5%) for downstream composites
    ref_severity = 0.005
    extreme_mask = severity_results[ref_severity]['mask']
    details = severity_results[ref_severity]['details']
    threshold = severity_results[ref_severity]['threshold']
    metrics = severity_results[ref_severity]['metrics']

    print("\n  Reference severity (0.5%) diagnostics:")
    print(f"    Extreme days: {details['n_extreme_days']}")
    print(f"    Occurrence rate: {details['occurrence_rate'] * 100:.2f}%")
    print(f"    Mean intensity during events: {metrics['mean_intensity']:.2f} mm/day")
    print(f"    Longest duration: {metrics['max_duration']:.0f} days")

    # Step 5: Trend diagnostics across regimes
    print("\n[Step 5] Regime-aware temporal trend analysis...")
    years_int = years.astype(int)
    years_unique = np.unique(years_int)
    annual_extreme_days = []
    regime_extreme_days = {0: [], 1: []}

    for year in years_unique:
        year_mask = years_int == year
        annual_count = np.sum(extreme_mask[year_mask])
        annual_extreme_days.append(annual_count)
        regime_state = int(regime_switch[np.where(year_mask)[0][0]])
        regime_extreme_days[regime_state].append(annual_count)

    trend_coef = np.polyfit(years_unique, annual_extreme_days, 1)[0]
    print(f"  Overall trend in extreme days: {trend_coef * 10:.2f} days/decade")
    if regime_extreme_days[0] and regime_extreme_days[1]:
        print(
            f"  Regime-0 mean: {np.mean(regime_extreme_days[0]):.1f} days/year | "
            f"Regime-1 mean: {np.mean(regime_extreme_days[1]):.1f} days/year"
        )

    # Step 6: Hydro-meteorological drivers & compound deficit analysis
    print("\n[Step 6] Evaluating compound water cycle anomalies...")

    precip = 2.1 + 1.2 * np.sin(2 * np.pi * doy / 365 + 0.3)
    precip += np.random.gamma(2.2, 0.45, n_days)
    precip = np.clip(precip - 0.003 * years * 365, 0, None)

    soil_moisture_proxy = 0.3 + 0.15 * np.cos(2 * np.pi * doy / 365)
    soil_moisture_proxy -= 0.05 * regime_switch
    soil_moisture_proxy += np.random.normal(0, 0.02, n_days)

    P_minus_E = precip - evap
    P_plus_E_half = (precip + evap) / 2

    compound_mask = extreme_mask & (P_minus_E < np.percentile(P_minus_E, 20))

    def summarize(label, mask):
        return (
            np.mean(P_minus_E[mask]),
            np.mean(P_plus_E_half[mask]),
            np.mean(soil_moisture_proxy[mask])
        )

    P_E_extreme, cycle_extreme, soil_extreme = summarize("Extremes", extreme_mask)
    P_E_compound, cycle_compound, soil_compound = summarize("Compound", compound_mask)
    P_E_normal, cycle_normal, soil_normal = summarize("Normal", ~extreme_mask)

    print("  Composite diagnostics (mean values):")
    print(f"    Extreme only  : P-E={P_E_extreme:.2f}, (P+E)/2={cycle_extreme:.2f}, SM={soil_extreme:.3f}")
    print(f"    Compound (dry): P-E={P_E_compound:.2f}, (P+E)/2={cycle_compound:.2f}, SM={soil_compound:.3f}")
    print(f"    Non-extreme   : P-E={P_E_normal:.2f}, (P+E)/2={cycle_normal:.2f}, SM={soil_normal:.3f}")

    # Step 7: Visualization suite
    print("\n[Step 7] Creating advanced visualizations...")

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot 1: Autocorrelation comparison
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(lags_pentad, autocorr_pentad, 'o-', label=f'Pentad (H={hurst_pentad:.2f})')
    ax1.plot(lags_monthly, autocorr_monthly, 's-', label=f'Monthly (H={hurst_monthly:.2f})')
    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Autocorrelation Structure across Scales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'markonis_autocorrelation.png'), dpi=150, bbox_inches='tight')
    print("  Saved: markonis_autocorrelation.png")

    # Plot 2: Time series with severity overlays (zoom 4 years)
    start_idx = 365 * 22
    window = (start_idx, start_idx + 365 * 4)

    fig2, ax2 = plot_extreme_events(
        evap, extreme_mask,
        title='Evaporation Time Series with Extreme Events (0.5% severity)',
        ylabel='Evaporation (mm/day)',
        window=window
    )
    ax2.fill_between(
        np.arange(len(evap))[window[0]:window[1]],
        threshold,
        np.max(evap[window[0]:window[1]]) + 0.5,
        color='coral', alpha=0.1, label='0.5% threshold'
    )
    plt.savefig(os.path.join(output_dir, 'markonis_timeseries.png'), dpi=150, bbox_inches='tight')
    print("  Saved: markonis_timeseries.png")

    # Plot 3: Annual trend with regime colouring
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    colors = ['steelblue' if regime_switch[np.where(years_int == y)[0][0]] == 0 else 'indianred'
              for y in years_unique]
    ax3.bar(years_unique, annual_extreme_days, color=colors, edgecolor='black')
    ax3.plot(
        years_unique,
        np.polyval(np.polyfit(years_unique, annual_extreme_days, 1), years_unique),
        'k--', linewidth=2, label=f'Trend: {trend_coef * 10:.2f} days/decade'
    )
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Extreme Days')
    ax3.set_title('Temporal Trend in Extreme Evaporation Days (Regime-coloured)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'markonis_trend.png'), dpi=150, bbox_inches='tight')
    print("  Saved: markonis_trend.png")

    # Plot 4: Water cycle + soil moisture comparison
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    categories = ['Normal', 'Extreme', 'Compound']
    P_E_values = [P_E_normal, P_E_extreme, P_E_compound]
    cycle_values = [cycle_normal, cycle_extreme, cycle_compound]
    soil_values = [soil_normal, soil_extreme, soil_compound]

    width = 0.25
    x = np.arange(len(categories))
    ax4.bar(x - width, P_E_values, width, label='P - E (mm/day)')
    ax4.bar(x, cycle_values, width, label='(P + E) / 2 (mm/day)')
    ax4.bar(x + width, soil_values, width, label='Soil Moisture Proxy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_title('Hydro-meteorological States by Regime')
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'markonis_water_cycle.png'), dpi=150, bbox_inches='tight')
    print("  Saved: markonis_water_cycle.png")

    # Plot 5: Severity comparison dashboard
    fig5, axes5 = plt.subplots(1, len(severities), figsize=(14, 4), sharey=True)
    for ax, sev in zip(axes5, severities):
        metrics_sev = severity_results[sev]['metrics']
        details_sev = severity_results[sev]['details']
        labels = ['Events', 'Mean Dur. (d)', 'Max Dur. (d)', 'Mean Intensity']
        values = [
            details_sev['n_events'],
            metrics_sev['mean_duration'],
            metrics_sev['max_duration'],
            metrics_sev['mean_intensity']
        ]
        ax.bar(labels, values, color='coral', alpha=0.7)
        ax.set_title(f"Severity {sev * 100:.1f}%")
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'markonis_severity_comparison.png'), dpi=150, bbox_inches='tight')
    print("  Saved: markonis_severity_comparison.png")

    print("\n" + "=" * 70)
    print("Analysis complete! Check the outputs directory for figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
