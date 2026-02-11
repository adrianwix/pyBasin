from typing import Any, cast

import numpy as np
from scipy.signal import find_peaks  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore[import-untyped]
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import euclidean_distances  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors


class DBSCANClusterer(BaseEstimator, ClusterMixin):  # type: ignore[misc]
    """DBSCAN clustering for basin stability analysis with optional epsilon auto-tuning (unsupervised learning).

    When ``auto_tune=True``, replicates the epsilon search from the MATLAB
    bSTAB ``classify_solution.m`` unsupervised branch:

    1. Precompute the pairwise Euclidean distance matrix.
    2. Build an epsilon grid from the feature ranges.
    3. For each candidate epsilon, run DBSCAN and record the minimum
       per-sample silhouette score (worst-case cluster quality).
    4. Find the most prominent peak in the silhouette curve above a
       height threshold.
    5. Fall back to the global maximum if no peak is found.
    """

    display_name: str = "DBSCAN Clustering"

    dbscan: Any
    auto_tune: bool
    n_eps_grid: int
    tune_sample_size: int
    min_peak_height: float
    assign_noise: bool
    nearest_neighbors: Any
    optimal_eps_: float | None

    def __init__(
        self,
        dbscan: DBSCAN | None = None,
        auto_tune: bool = False,
        n_eps_grid: int = 200,
        tune_sample_size: int = 2000,
        min_peak_height: float = 0.9,
        assign_noise: bool = False,
        nearest_neighbors: NearestNeighbors | None = None,
    ):
        """
        Initialize DBSCAN clusterer.

        :param dbscan: A configured ``sklearn.cluster.DBSCAN`` instance, or
            ``None`` to create a default one (``eps=0.5``, ``min_samples=10``).
            When ``auto_tune=True``, the tuned epsilon overrides ``dbscan.eps``.
        :param auto_tune: Whether to automatically find the optimal epsilon
            using silhouette-based peak analysis (MATLAB bSTAB algorithm).
        :param n_eps_grid: Number of epsilon candidates to evaluate during
            auto-tuning.
        :param tune_sample_size: Maximum number of samples to use during
            the epsilon search. If the dataset is larger, a random subsample
            is drawn to keep the search fast.
        :param min_peak_height: Minimum silhouette peak height for the peak
            finder during auto-tuning.
        :param assign_noise: Whether to assign noise points (-1) to the
            nearest cluster using KNN.
        :param nearest_neighbors: A configured ``sklearn.neighbors.NearestNeighbors``
            instance for noise assignment, or ``None`` to create a default
            one (``n_neighbors=5``). Only used when ``assign_noise=True``.
        """
        if dbscan is None:
            dbscan = DBSCAN(eps=0.5, min_samples=10)

        self.dbscan = dbscan
        self.auto_tune = auto_tune
        self.n_eps_grid = n_eps_grid
        self.tune_sample_size = tune_sample_size
        self.min_peak_height = min_peak_height
        self.assign_noise = assign_noise
        self.nearest_neighbors = nearest_neighbors
        self.optimal_eps_ = None

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:  # type: ignore[override]
        """
        Fit and predict labels using DBSCAN clustering.

        :param X: Feature array of shape ``(n_samples, n_features)``.
        :param y: Ignored (present for sklearn API compatibility).
        :return: Cluster labels (``-1`` for noise unless ``assign_noise=True``).
        """
        if self.auto_tune:
            eps = self._find_optimal_eps(X)
            self.optimal_eps_ = eps
            self.dbscan.eps = eps

        labels: np.ndarray = self.dbscan.fit_predict(X)

        if self.assign_noise:
            labels = self._assign_noise_to_clusters(X, labels)

        return labels

    def _find_optimal_eps(self, features: np.ndarray) -> float:
        """Find optimal epsilon using silhouette-based peak analysis.

        Replicates the MATLAB bSTAB ``classify_solution.m`` unsupervised
        epsilon search: precomputes pairwise distances, sweeps an epsilon
        grid derived from the feature ranges, records the minimum per-sample
        silhouette value for each epsilon, and selects the most prominent
        peak above ``min_peak_height``.

        When the dataset exceeds ``tune_sample_size``, a random subsample is
        used for the search to keep runtime manageable.

        :param features: Feature matrix ``(n_samples, n_features)``.
        :return: Optimal epsilon value.
        """
        min_samples: int = self.dbscan.min_samples

        n_samples: int = len(features)
        if n_samples > self.tune_sample_size:
            rng: np.random.Generator = np.random.default_rng(seed=42)
            idx: np.ndarray = rng.choice(n_samples, size=self.tune_sample_size, replace=False)
            features = features[idx]

        distance_matrix: np.ndarray = euclidean_distances(features)

        feat_ranges: np.ndarray = np.max(features, axis=0) - np.min(features, axis=0)
        min_range: float = float(np.min(feat_ranges))

        if min_range <= 0:
            return float(self.dbscan.eps)

        eps_grid: np.ndarray = np.linspace(min_range / self.n_eps_grid, min_range, self.n_eps_grid)

        s_grid: np.ndarray = np.zeros(self.n_eps_grid)

        for i, eps_candidate in enumerate(eps_grid):
            labels: np.ndarray = DBSCAN(
                eps=eps_candidate,
                min_samples=min_samples,
                metric="precomputed",
            ).fit_predict(distance_matrix)

            valid_mask: np.ndarray = labels != -1
            n_valid: int = int(np.sum(valid_mask))
            n_clusters: int = len(np.unique(labels[valid_mask])) if n_valid > 0 else 0

            if n_clusters >= 2 and n_valid > n_clusters:
                s: np.ndarray = cast(
                    np.ndarray,
                    silhouette_samples(
                        distance_matrix[np.ix_(valid_mask, valid_mask)],
                        labels[valid_mask],
                        metric="precomputed",
                    ),
                )
                s_grid[i] = float(np.min(s))

        s_grid[0] = 0.0
        s_grid[-1] = 0.0

        peaks, properties = find_peaks(s_grid, height=self.min_peak_height, prominence=0)

        if len(peaks) > 0:
            prominences: np.ndarray = properties["prominences"]
            best_peak_idx: int = int(peaks[np.argmax(prominences)])
            return float(eps_grid[best_peak_idx])

        best_idx: int = int(np.argmax(s_grid))
        return float(eps_grid[best_idx])

    def _assign_noise_to_clusters(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Assign noise points (-1 label) to nearest clusters using KNN.

        :param features: Feature matrix ``(n_samples, n_features)``.
        :param labels: Cluster labels with -1 for noise.
        :return: Updated labels with noise points assigned to clusters.
        """
        labels_updated: np.ndarray = labels.copy()
        noise_mask: np.ndarray = labels == -1

        if not noise_mask.any():
            return labels_updated

        labeled_mask: np.ndarray = ~noise_mask
        labeled_features: np.ndarray = features[labeled_mask]
        labeled_labels: np.ndarray = labels[labeled_mask]

        if len(labeled_features) == 0:
            return labels_updated

        noise_features: np.ndarray = features[noise_mask]
        nn: Any = (
            self.nearest_neighbors
            if self.nearest_neighbors is not None
            else NearestNeighbors(n_neighbors=5)
        )
        n_neighbors: int = nn.get_params()["n_neighbors"]
        k_actual: int = min(n_neighbors, len(labeled_features))
        nn.set_params(n_neighbors=k_actual)
        nn.fit(labeled_features)
        _, indices = nn.kneighbors(noise_features)

        noise_indices: np.ndarray = np.where(noise_mask)[0]
        for i, neighbor_indices in enumerate(indices):
            neighbor_labels: np.ndarray = labeled_labels[neighbor_indices]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            labels_updated[noise_indices[i]] = most_common_label

        return labels_updated
