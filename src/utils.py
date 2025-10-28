"""
Utility functions for extreme evaporation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def set_paper_style(context: str = "nature") -> None:
    """Configure Matplotlib/Seaborn for a Nature/Science-like style.

    Parameters
    ----------
    context : {"nature", "science", "talk", "poster"}
        Slightly adjusts font sizes and line widths.
    """
    import matplotlib as mpl

    # Base palette and style
    sns.set_theme(style="whitegrid")
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]
    sns.set_palette(palette)

    # Typography and layout tuned for clarity
    base = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }

    if context in {"talk", "poster"}:
        base.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
        })

    mpl.rcParams.update(base)


def label_subplots(axes: List, xpos: float = -0.08, ypos: float = 1.02, start: str = 'a') -> None:
    """Add (a), (b), ... labels to a list/array of axes in-place.

    Parameters
    ----------
    axes : list of Axes
        Axes to annotate in reading order.
    xpos, ypos : float
        Relative axes coordinates for label placement.
    start : str
        Starting letter, default 'a'.
    """
    letters = [chr(ord(start) + i) for i in range(len(axes))]
    for ax, letter in zip(np.ravel(axes), letters):
        ax.text(xpos, ypos, f"({letter})", transform=ax.transAxes, fontsize=12, weight="bold", va="bottom")


def generate_synthetic_data(n_days=365*40, add_trend=True, add_seasonality=True, 
                           seed=42):
    """
    Generate synthetic meteorological data for testing.
    
    Parameters
    ----------
    n_days : int, default=14600 (40 years)
        Number of days to generate
    add_trend : bool, default=True
        Add long-term trend
    add_seasonality : bool, default=True
        Add seasonal cycle
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    data : dict
        Dictionary containing synthetic meteorological forcings
    
    Examples
    --------
    >>> data = generate_synthetic_data(n_days=365*10)
    >>> print(f"Temperature range: {data['T_mean'].min():.1f} to {data['T_mean'].max():.1f}°C")
    """
    np.random.seed(seed)
    
    # Time array
    days = np.arange(n_days)
    years = days / 365.25
    doy = days % 365
    
    # Base values
    T_base = 15.0  # °C
    Rs_base = 15.0  # MJ m-2 day-1
    u2_base = 2.0  # m s-1
    RH_base = 0.6  # Relative humidity
    
    # Seasonal component
    if add_seasonality:
        T_seasonal = 10 * np.sin(2 * np.pi * doy / 365 - np.pi/2)
        Rs_seasonal = 10 * np.sin(2 * np.pi * doy / 365 - np.pi/2)
        RH_seasonal = 0.2 * np.sin(2 * np.pi * doy / 365 + np.pi)
    else:
        T_seasonal = 0
        Rs_seasonal = 0
        RH_seasonal = 0
    
    # Trend component
    if add_trend:
        T_trend = 0.02 * years  # Warming trend
        Rs_trend = 0.01 * years
    else:
        T_trend = 0
        Rs_trend = 0
    
    # Random variability
    T_noise = np.random.randn(n_days) * 2
    Rs_noise = np.random.randn(n_days) * 3
    u2_noise = np.random.randn(n_days) * 0.5
    RH_noise = np.random.randn(n_days) * 0.05
    
    # Generate forcings
    T_mean = T_base + T_seasonal + T_trend + T_noise
    T_max = T_mean + 5 + np.random.randn(n_days)
    T_min = T_mean - 5 + np.random.randn(n_days)
    
    Rs = np.maximum(Rs_base + Rs_seasonal + Rs_trend + Rs_noise, 0)
    u2 = np.maximum(u2_base + u2_noise, 0.5)
    
    # Generate humidity (via relative humidity)
    RH = np.clip(RH_base + RH_seasonal + RH_noise, 0.2, 0.95)
    
    # Calculate vapor pressure from RH and temperature
    es = 0.6108 * np.exp(17.27 * T_mean / (T_mean + 237.3))
    ea = RH * es
    
    # Add occasional extreme events
    extreme_idx = np.random.choice(n_days, size=int(n_days * 0.01), replace=False)
    T_mean[extreme_idx] += np.random.randn(len(extreme_idx)) * 5 + 5
    Rs[extreme_idx] += np.random.randn(len(extreme_idx)) * 5 + 5
    u2[extreme_idx] += np.random.randn(len(extreme_idx)) * 2 + 2

    # Ensure physical bounds after perturbations
    Rs = np.maximum(Rs, 0)
    u2 = np.maximum(u2, 0.5)

    data = {
        'T_mean': T_mean,
        'T_max': T_max,
        'T_min': T_min,
        'Rs': Rs,
        'u2': u2,
        'ea': ea,
        'doy': doy + 1,
        'dates': pd.date_range('1981-01-01', periods=n_days, freq='D'),
    }
    
    return data


def plot_extreme_events(data_series, extreme_mask, title='Extreme Events',
                       ylabel='Value', window=None):
    """
    Visualize extreme events in time series.
    
    Parameters
    ----------
    data_series : array-like
        Time series data
    extreme_mask : array-like (bool)
        Boolean mask for extreme events
    title : str
        Plot title
    ylabel : str
        Y-axis label
    window : tuple (int, int), optional
        (start, end) indices to zoom in
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    data_series = np.asarray(data_series)
    extreme_mask = np.asarray(extreme_mask, dtype=bool)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Determine plot window
    if window is not None:
        start, end = window
    else:
        start, end = 0, len(data_series)
    
    x = np.arange(start, end)
    
    # Plot data
    ax.plot(x, data_series[start:end], 'k-', alpha=0.5, linewidth=0.5, label='Data')
    
    # Highlight extremes
    ax.scatter(x[extreme_mask[start:end]], 
              data_series[start:end][extreme_mask[start:end]],
              c='red', s=20, zorder=5, label='Extreme Events')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def plot_contribution_pie(contributions, title='Contribution Analysis'):
    """
    Create pie chart of forcing contributions.
    
    Parameters
    ----------
    contributions : dict
        Contribution percentages for each forcing
    title : str
        Plot title
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    labels = list(contributions.keys())
    # Take absolute values to handle negative contributions
    sizes = [abs(val) for val in contributions.values()]
    
    # Normalize to ensure sum is 100%
    total = sum(sizes)
    sizes = [100 * s/total for s in sizes]
    
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    
    # Beautify text
    for text in texts:
        text.set_fontsize(12)
        text.set_weight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    ax.set_title(title + '\n(Absolute Values)', fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    return fig, ax


def plot_seasonal_contributions(seasonal_contributions):
    """
    Create bar plot of seasonal contributions.
    
    Parameters
    ----------
    seasonal_contributions : dict
        Nested dict: season -> forcing -> contribution
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    seasons = list(seasonal_contributions.keys())
    forcings = ['temperature', 'radiation', 'wind', 'humidity']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(seasons))
    width = 0.2
    
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    
    for i, forcing in enumerate(forcings):
        values = [seasonal_contributions[s].get(forcing, 0) for s in seasons]
        ax.bar(x + i*width, values, width, label=forcing.capitalize(),
               color=colors[i])
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('Seasonal Contribution of Meteorological Forcings', 
                fontsize=14, weight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig, ax


def plot_autocorrelation(lags, autocorr, title='Autocorrelation'):
    """
    Plot autocorrelation function.
    
    Parameters
    ----------
    lags : array-like
        Lag values
    autocorr : array-like
        Autocorrelation coefficients
    title : str
        Plot title
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.stem(lags, autocorr, basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Add confidence interval
    n = 1000  # Approximate sample size
    conf_level = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_level, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf_level, color='b', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def calculate_event_metrics(events, data):
    """
    Calculate various metrics for events.
    
    Parameters
    ----------
    events : list of dict
        Events from identify_events_from_mask
    data : array-like
        Time series data
    
    Returns
    -------
    metrics : dict
        Event statistics
    """
    if len(events) == 0:
        return {
            'n_events': 0,
            'mean_duration': 0,
            'max_duration': 0,
            'mean_intensity': 0,
            'max_intensity': 0,
        }
    
    durations = [e['duration'] for e in events]
    
    # Calculate intensities
    intensities = []
    for event in events:
        start, end = event['start'], event['end']
        event_mean = np.mean(data[start:end+1])
        intensities.append(event_mean)
    
    metrics = {
        'n_events': len(events),
        'mean_duration': np.mean(durations),
        'max_duration': np.max(durations),
        'mean_intensity': np.mean(intensities),
        'max_intensity': np.max(intensities),
    }
    
    return metrics


def summary_statistics(data_dict):
    """
    Calculate summary statistics for all forcings.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of forcings
    
    Returns
    -------
    summary : pd.DataFrame
        Summary statistics table
    """
    stats = {}
    
    for key, values in data_dict.items():
        if isinstance(values, np.ndarray) or isinstance(values, list):
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
            }
    
    summary = pd.DataFrame(stats).T
    
    return summary
