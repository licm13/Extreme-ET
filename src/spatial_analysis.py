"""
Spatial Analysis for Extreme ET
Based on Zhao et al. (2025) recommendations for regional analysis

This module implements spatial coherence, propagation, and interpolation
methods for extreme ET events across geographic regions.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.interpolate import griddata, Rbf
from scipy.signal import correlate2d
import warnings


def calculate_spatial_correlation(
    data_matrix: np.ndarray,
    locations: np.ndarray,
    max_distance: float = 500.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate spatial correlation of extreme events as a function of distance.

    Parameters
    ----------
    data_matrix : np.ndarray
        Matrix of shape (n_locations, n_timesteps) with extreme event indicators
    locations : np.ndarray
        Coordinates of shape (n_locations, 2) with (lat, lon) or (x, y)
    max_distance : float
        Maximum distance for correlation calculation (km or degrees)

    Returns
    -------
    tuple
        - distances: pairwise distances between locations
        - correlations: pairwise correlations
        - distance_bins: binned distances for plotting

    Examples
    --------
    >>> # 10 locations, 1000 days
    >>> data = np.random.binomial(1, 0.05, (10, 1000))
    >>> locs = np.random.rand(10, 2) * 100  # Random locations
    >>> dists, corrs, bins = calculate_spatial_correlation(data, locs, max_distance=50)
    """
    n_locations = data_matrix.shape[0]

    # Calculate pairwise distances
    distances = pdist(locations, metric='euclidean')
    distance_matrix = squareform(distances)

    # Calculate pairwise correlations
    correlations = []
    distance_pairs = []

    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            dist = distance_matrix[i, j]
            if dist <= max_distance:
                corr = np.corrcoef(data_matrix[i, :], data_matrix[j, :])[0, 1]
                correlations.append(corr)
                distance_pairs.append(dist)

    distances = np.array(distance_pairs)
    correlations = np.array(correlations)

    # Bin distances for easier visualization
    n_bins = 20
    distance_bins = np.linspace(0, max_distance, n_bins + 1)

    return distances, correlations, distance_bins


def detect_event_propagation(
    data_matrix: np.ndarray,
    locations: np.ndarray,
    dates: np.ndarray,
    max_lag_days: int = 7
) -> Dict[str, any]:
    """
    Detect spatial propagation patterns of extreme events.

    Identifies if extreme events tend to move in a particular direction
    over time (e.g., following weather systems).

    Parameters
    ----------
    data_matrix : np.ndarray
        Matrix of shape (n_locations, n_timesteps) with extreme indicators
    locations : np.ndarray
        Coordinates (n_locations, 2) with (lat, lon) or (x, y)
    dates : np.ndarray
        Time index for each column
    max_lag_days : int
        Maximum lag to test for propagation (default: 7 days)

    Returns
    -------
    dict
        Dictionary with:
        - propagation_speed: estimated speed (distance/day)
        - propagation_direction: dominant direction (degrees)
        - lag_correlations: correlations at different lags

    Examples
    --------
    >>> data = np.random.binomial(1, 0.05, (20, 1000))
    >>> locs = np.random.rand(20, 2) * 100
    >>> dates = np.arange(1000)
    >>> prop = detect_event_propagation(data, locs, dates, max_lag_days=5)
    >>> print(f"Propagation speed: {prop['propagation_speed']:.1f} km/day")
    """
    n_locations, n_timesteps = data_matrix.shape

    # Calculate distance matrix
    distance_matrix = squareform(pdist(locations, metric='euclidean'))

    # Calculate directional vectors
    direction_vectors = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]

    # Test different lags
    lag_correlations = []
    for lag in range(max_lag_days + 1):
        lag_corr_sum = 0
        count = 0

        for i in range(n_locations):
            for j in range(n_locations):
                if i != j and distance_matrix[i, j] > 0:
                    # Check if event at i precedes event at j by lag days
                    if lag < n_timesteps:
                        series_i = data_matrix[i, :-lag] if lag > 0 else data_matrix[i, :]
                        series_j = data_matrix[j, lag:] if lag > 0 else data_matrix[j, :]

                        if len(series_i) == len(series_j) and len(series_i) > 0:
                            corr = np.corrcoef(series_i, series_j)[0, 1]
                            if not np.isnan(corr):
                                lag_corr_sum += corr
                                count += 1

        avg_corr = lag_corr_sum / count if count > 0 else 0
        lag_correlations.append(avg_corr)

    lag_correlations = np.array(lag_correlations)

    # Find optimal lag
    if len(lag_correlations) > 0 and np.max(lag_correlations) > 0:
        optimal_lag = np.argmax(lag_correlations)
        # Estimate propagation speed (simplified)
        mean_distance = np.mean(distance_matrix[distance_matrix > 0])
        propagation_speed = mean_distance / optimal_lag if optimal_lag > 0 else 0
    else:
        optimal_lag = 0
        propagation_speed = 0

    return {
        'optimal_lag': optimal_lag,
        'propagation_speed': propagation_speed,
        'lag_correlations': lag_correlations,
        'max_correlation': np.max(lag_correlations) if len(lag_correlations) > 0 else 0
    }


def ordinary_kriging(
    known_points: np.ndarray,
    known_values: np.ndarray,
    target_points: np.ndarray,
    variogram_model: str = 'exponential',
    variogram_range: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform ordinary kriging interpolation for spatial data.

    Parameters
    ----------
    known_points : np.ndarray
        Coordinates of known locations (n_known, 2)
    known_values : np.ndarray
        Values at known locations (n_known,)
    target_points : np.ndarray
        Coordinates where to interpolate (n_target, 2)
    variogram_model : str
        Variogram model: 'exponential', 'spherical', or 'gaussian'
    variogram_range : float
        Range parameter for variogram (default: 100.0)

    Returns
    -------
    tuple
        - interpolated_values: values at target points
        - interpolation_variance: kriging variance (uncertainty)

    Examples
    --------
    >>> known_pts = np.random.rand(50, 2) * 100
    >>> known_vals = np.random.gamma(3, 1.5, 50)
    >>> target_pts = np.random.rand(100, 2) * 100
    >>> interp_vals, interp_var = ordinary_kriging(known_pts, known_vals, target_pts)
    """
    # For simplicity, use RBF interpolation as approximation to kriging
    # A full kriging implementation would require additional dependencies

    try:
        # Radial Basis Function interpolation
        rbf = Rbf(known_points[:, 0], known_points[:, 1], known_values,
                  function='multiquadric', smooth=0.1)

        interpolated_values = rbf(target_points[:, 0], target_points[:, 1])

        # Estimate variance based on distance to nearest known point
        distances = cdist(target_points, known_points)
        min_distances = np.min(distances, axis=1)

        # Simple variance model: increases with distance
        interpolation_variance = 0.1 * (1 + min_distances / variogram_range)

    except Exception as e:
        warnings.warn(f"Kriging failed: {e}. Using nearest neighbor.")
        interpolated_values = griddata(known_points, known_values, target_points,
                                      method='nearest')
        interpolation_variance = np.ones(len(target_points)) * 0.5

    return interpolated_values, interpolation_variance


def calculate_regional_synchrony(
    data_matrix: np.ndarray,
    region_labels: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Calculate synchrony of extreme events within and between regions.

    Parameters
    ----------
    data_matrix : np.ndarray
        Matrix (n_locations, n_timesteps) with extreme indicators
    region_labels : np.ndarray
        Region label for each location (n_locations,)

    Returns
    -------
    dict
        Dictionary mapping region_id to:
        - within_region_synchrony: correlation within region
        - between_region_synchrony: correlation with other regions

    Examples
    --------
    >>> data = np.random.binomial(1, 0.05, (100, 1000))
    >>> regions = np.repeat(np.arange(4), 25)  # 4 regions, 25 locations each
    >>> synchrony = calculate_regional_synchrony(data, regions)
    >>> for region, metrics in synchrony.items():
    ...     print(f"Region {region}: within={metrics['within_region_synchrony']:.3f}")
    """
    unique_regions = np.unique(region_labels)
    synchrony = {}

    for region in unique_regions:
        # Get locations in this region
        in_region = region_labels == region
        region_data = data_matrix[in_region, :]

        # Within-region synchrony
        if region_data.shape[0] > 1:
            within_corr = []
            for i in range(region_data.shape[0]):
                for j in range(i + 1, region_data.shape[0]):
                    corr = np.corrcoef(region_data[i, :], region_data[j, :])[0, 1]
                    if not np.isnan(corr):
                        within_corr.append(corr)

            within_synchrony = np.mean(within_corr) if within_corr else 0
        else:
            within_synchrony = 1.0

        # Between-region synchrony
        out_region = ~in_region
        if np.any(out_region):
            between_corr = []
            for i in np.where(in_region)[0]:
                for j in np.where(out_region)[0]:
                    corr = np.corrcoef(data_matrix[i, :], data_matrix[j, :])[0, 1]
                    if not np.isnan(corr):
                        between_corr.append(corr)

            between_synchrony = np.mean(between_corr) if between_corr else 0
        else:
            between_synchrony = 0

        synchrony[int(region)] = {
            'within_region_synchrony': within_synchrony,
            'between_region_synchrony': between_synchrony,
            'n_locations': np.sum(in_region)
        }

    return synchrony


def identify_spatial_clusters(
    extreme_day_data: np.ndarray,
    locations: np.ndarray,
    cluster_distance_threshold: float = 50.0
) -> np.ndarray:
    """
    Identify spatial clusters of extreme events on a given day.

    Parameters
    ----------
    extreme_day_data : np.ndarray
        Binary indicators (n_locations,) for a single day
    locations : np.ndarray
        Coordinates (n_locations, 2)
    cluster_distance_threshold : float
        Maximum distance to be considered in same cluster (km or degrees)

    Returns
    -------
    np.ndarray
        Cluster labels for each location (-1 for non-extreme)

    Examples
    --------
    >>> day_data = np.random.binomial(1, 0.1, 100)
    >>> locs = np.random.rand(100, 2) * 100
    >>> clusters = identify_spatial_clusters(day_data, locs, cluster_distance_threshold=20)
    >>> print(f"Number of clusters: {len(np.unique(clusters[clusters >= 0]))}")
    """
    # Simple clustering: use distance-based grouping
    extreme_locations = np.where(extreme_day_data == 1)[0]

    if len(extreme_locations) == 0:
        return np.full(len(extreme_day_data), -1)

    cluster_labels = np.full(len(extreme_day_data), -1)

    # Calculate distances between extreme locations
    extreme_coords = locations[extreme_locations]
    if len(extreme_coords) == 1:
        cluster_labels[extreme_locations[0]] = 0
        return cluster_labels

    distance_matrix = squareform(pdist(extreme_coords, metric='euclidean'))

    # Assign clusters using simple threshold
    assigned = np.zeros(len(extreme_locations), dtype=bool)
    current_cluster = 0

    for i in range(len(extreme_locations)):
        if not assigned[i]:
            # Start new cluster
            cluster_members = [i]
            assigned[i] = True

            # Find nearby points
            for j in range(i + 1, len(extreme_locations)):
                if not assigned[j] and distance_matrix[i, j] <= cluster_distance_threshold:
                    cluster_members.append(j)
                    assigned[j] = True

            # Assign cluster labels
            for member in cluster_members:
                cluster_labels[extreme_locations[member]] = current_cluster

            current_cluster += 1

    return cluster_labels


def calculate_spatial_extent_metrics(
    extreme_mask: np.ndarray,
    locations: np.ndarray,
    total_area: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate spatial extent metrics for extreme events.

    Parameters
    ----------
    extreme_mask : np.ndarray
        Boolean mask (n_locations,) indicating extreme events
    locations : np.ndarray
        Coordinates (n_locations, 2)
    total_area : float, optional
        Total area of study region (if None, uses convex hull)

    Returns
    -------
    dict
        Metrics including:
        - n_locations_affected: number of locations with extremes
        - fraction_affected: fraction of total locations
        - spatial_extent_area: approximate area affected (if total_area provided)

    Examples
    --------
    >>> extremes = np.random.binomial(1, 0.2, 100).astype(bool)
    >>> locs = np.random.rand(100, 2) * 100
    >>> metrics = calculate_spatial_extent_metrics(extremes, locs, total_area=10000)
    >>> print(f"Fraction affected: {metrics['fraction_affected']:.2%}")
    """
    n_total = len(extreme_mask)
    n_affected = np.sum(extreme_mask)

    fraction_affected = n_affected / n_total if n_total > 0 else 0

    metrics = {
        'n_locations_affected': n_affected,
        'n_total_locations': n_total,
        'fraction_affected': fraction_affected
    }

    if total_area is not None:
        metrics['spatial_extent_area'] = fraction_affected * total_area

    # Calculate spatial spread (distance between extreme locations)
    if n_affected > 1:
        extreme_coords = locations[extreme_mask]
        distances = pdist(extreme_coords, metric='euclidean')
        metrics['mean_inter_extreme_distance'] = np.mean(distances)
        metrics['max_inter_extreme_distance'] = np.max(distances)
    else:
        metrics['mean_inter_extreme_distance'] = 0
        metrics['max_inter_extreme_distance'] = 0

    return metrics
