"""
Evaluation utilities for comparing extreme ET detection across datasets.

This module provides helpers to:
- Aggregate hourly to daily series
- Map station locations to grid cells (nearest-neighbor)
- Compute skill metrics between station and product extreme masks
- Sweep severity levels and summarize skill
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .extreme_detection import (
    detect_extreme_events_hist,
    identify_events_from_mask,
)


def aggregate_hourly_to_daily(hourly: np.ndarray, method: str = "sum") -> np.ndarray:
    """Aggregate hourly values into daily values.

    Assumes the input length is divisible by 24. If not, drops trailing hours.

    - For ET in mm/h, use method='sum'.
    - For temperature-like variables in degC, use method='mean'.
    """
    arr = np.asarray(hourly, dtype=float)
    n_hours = arr.size - (arr.size % 24)
    if n_hours != arr.size:
        arr = arr[:n_hours]
    reshaped = arr.reshape(-1, 24)
    if method.lower() == "sum":
        return reshaped.sum(axis=1)
    elif method.lower() == "mean":
        return reshaped.mean(axis=1)
    else:
        raise ValueError("method must be 'sum' or 'mean'")


def nearest_grid_point(
    lat: float, lon: float, grid_lats: np.ndarray, grid_lons: np.ndarray
) -> Tuple[int, int]:
    """Return indices of the nearest grid point to the given lat/lon.

    Expects 1D grid_lats and grid_lons (regular lat/lon grid).
    """
    grid_lats = np.asarray(grid_lats)
    grid_lons = np.asarray(grid_lons)
    i = int(np.argmin(np.abs(grid_lats - lat)))
    j = int(np.argmin(np.abs(grid_lons - lon)))
    return i, j


def compute_daywise_skill(
    station_mask: np.ndarray,
    product_mask: np.ndarray,
    max_lag_days: int = 1,
) -> Dict[str, float]:
    """Compute day-wise detection skill with temporal tolerance.

    - A hit occurs if a station extreme day has a product extreme within ±max_lag_days.
    - A false alarm occurs if a product extreme has no station extreme within ±max_lag_days.
    - Miss = station extreme with no matched product extreme.
    - Also reports POD, FAR, CSI, precision, recall, F1.
    """
    s = np.asarray(station_mask, dtype=bool)
    p = np.asarray(product_mask, dtype=bool)
    n = min(s.size, p.size)
    s = s[:n]
    p = p[:n]

    station_days = np.where(s)[0]
    product_days = np.where(p)[0]

    # Match station extremes to product extremes within tolerance
    hits = 0
    matched_product_idx: List[int] = []
    for d in station_days:
        # Find the closest product extreme within the window
        lo = d - max_lag_days
        hi = d + max_lag_days
        candidates = product_days[(product_days >= lo) & (product_days <= hi)]
        if candidates.size > 0:
            hits += 1
            # mark the first candidate as matched to avoid double counting
            matched_product_idx.append(int(candidates[0]))

    misses = int(station_days.size - hits)
    false_alarms = int(product_days.size - len(set(matched_product_idx)))

    pod = hits / station_days.size if station_days.size else 0.0  # probability of detection
    far = false_alarms / product_days.size if product_days.size else 0.0  # false alarm ratio
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else 0.0

    precision = hits / (hits + false_alarms) if (hits + false_alarms) else 0.0
    recall = pod
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "hits": float(hits),
        "misses": float(misses),
        "false_alarms": float(false_alarms),
        "pod": float(pod),
        "far": float(far),
        "csi": float(csi),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_station_extremes": float(station_days.size),
        "n_product_extremes": float(product_days.size),
    }


def event_timing_errors(
    station_mask: np.ndarray,
    product_mask: np.ndarray,
    max_lag_days: int = 5,
) -> np.ndarray:
    """Compute signed timing errors (product day - station day) for matched extremes.

    Returns an array of timing errors in days for all matched events.
    """
    s = np.asarray(station_mask, dtype=bool)
    p = np.asarray(product_mask, dtype=bool)
    n = min(s.size, p.size)
    s = s[:n]
    p = p[:n]

    station_days = np.where(s)[0]
    product_days = np.where(p)[0]
    errors: List[int] = []
    for d in station_days:
        lo = d - max_lag_days
        hi = d + max_lag_days
        candidates = product_days[(product_days >= lo) & (product_days <= hi)]
        if candidates.size:
            # choose the closest product day (smallest absolute difference)
            idx = int(candidates[np.argmin(np.abs(candidates - d))])
            errors.append(idx - d)
    return np.asarray(errors, dtype=float)


def severity_sweep_skill(
    station_series: np.ndarray,
    product_series: np.ndarray,
    severities: Iterable[float] = (0.001, 0.0025, 0.005, 0.01, 0.02),
    max_lag_days: int = 1,
) -> Dict[float, Dict[str, float]]:
    """Sweep severity levels and compute skill metrics vs. station.

    Uses the historical ERT method for both series at each severity.
    """
    station_series = np.asarray(station_series, dtype=float)
    product_series = np.asarray(product_series, dtype=float)
    out: Dict[float, Dict[str, float]] = {}
    for sev in severities:
        s_mask, _ = detect_extreme_events_hist(station_series, severity=sev)
        p_mask, _ = detect_extreme_events_hist(product_series, severity=sev)
        out[float(sev)] = compute_daywise_skill(s_mask, p_mask, max_lag_days=max_lag_days)
    return out


def matched_event_intensity_bias(
    station_series: np.ndarray,
    product_series: np.ndarray,
    station_mask: np.ndarray,
    product_mask: np.ndarray,
    max_lag_days: int = 1,
) -> float:
    """Compute mean intensity bias (product - station) over matched extreme days.

    Intensity here is simply the series value on the matched day.
    """
    s = np.asarray(station_mask, dtype=bool)
    p = np.asarray(product_mask, dtype=bool)
    n = min(s.size, p.size)
    s = s[:n]
    p = p[:n]
    station_series = np.asarray(station_series, dtype=float)[:n]
    product_series = np.asarray(product_series, dtype=float)[:n]

    station_days = np.where(s)[0]
    product_days = np.where(p)[0]
    diffs: List[float] = []
    for d in station_days:
        lo = d - max_lag_days
        hi = d + max_lag_days
        candidates = product_days[(product_days >= lo) & (product_days <= hi)]
        if candidates.size:
            idx = int(candidates[np.argmin(np.abs(candidates - d))])
            diffs.append(product_series[idx] - station_series[d])
    return float(np.mean(diffs)) if diffs else 0.0


def serialize_skill_summary(skill_by_sev: Dict[float, Dict[str, float]]) -> str:
    """Format a concise skill summary across severities for manuscript insertion."""
    lines: List[str] = []
    for sev in sorted(skill_by_sev.keys()):
        k = skill_by_sev[sev]
        lines.append(
            f"- Severity {sev*100:.2f}%: POD {k['pod']:.2f}, FAR {k['far']:.2f}, CSI {k['csi']:.2f}"
        )
    return "\n".join(lines)

