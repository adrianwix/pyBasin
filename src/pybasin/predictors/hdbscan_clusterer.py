from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors


class HDBSCANClusterer(BaseEstimator, ClusterMixin):  # type: ignore[misc]
    """HDBSCAN clustering for basin stability analysis with optional auto-tuning and noise assignment (unsupervised learning)."""

    display_name: str = "HDBSCAN Clustering"

    hdbscan: Any
    assign_noise: bool
    nearest_neighbors: Any
    auto_tune: bool

    def __init__(
        self,
        hdbscan: Any = None,
        assign_noise: bool = False,
        nearest_neighbors: NearestNeighbors | None = None,
        auto_tune: bool = False,
    ):
        """
        Initialize HDBSCAN clusterer.

        :param hdbscan: A configured ``sklearn.cluster.HDBSCAN`` instance, or
            ``None`` to create a default one (``min_cluster_size=50``,
            ``min_samples=10``).
        :param assign_noise: Whether to assign noise points to nearest
            clusters using KNN.
        :param nearest_neighbors: A configured ``sklearn.neighbors.NearestNeighbors``
            instance for noise assignment, or ``None`` to create a default
            one (``n_neighbors=5``). Only used when ``assign_noise=True``.
        :param auto_tune: Whether to automatically tune ``min_cluster_size``
            using silhouette score. The tuned value overrides the one on the
            ``hdbscan`` instance.
        """
        if hdbscan is None:
            hdbscan = HDBSCAN(min_cluster_size=50, min_samples=10, copy=True)  # type: ignore[call-arg]

        self.hdbscan = hdbscan
        self.assign_noise = assign_noise
        self.nearest_neighbors = nearest_neighbors
        self.auto_tune = auto_tune

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:  # type: ignore[override]
        """
        Fit and predict labels using HDBSCAN clustering with optional noise assignment.

        :param X: Feature array to cluster.
        :param y: Ignored (present for sklearn API compatibility).
        :return: Cluster labels.
        """
        features = X
        if self.auto_tune:
            optimal_size = self._find_optimal_min_cluster_size(features)
            self.hdbscan.min_cluster_size = optimal_size
            self.hdbscan.min_samples = min(10, optimal_size // 5)

        labels = cast(np.ndarray, self.hdbscan.fit_predict(features))

        if self.assign_noise:
            labels = self._assign_noise_to_clusters(features, labels)

        return labels

    def _assign_noise_to_clusters(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Assign noise points (-1 label) to nearest clusters using KNN.

        :param features: Feature matrix (n_samples, n_features)
        :param labels: Cluster labels with -1 for noise
        :return: Updated labels with noise assigned to clusters
        """
        labels_updated = labels.copy()
        noise_mask = labels == -1

        if not noise_mask.any():
            return labels_updated

        labeled_mask = ~noise_mask
        labeled_features = features[labeled_mask]
        labeled_labels = labels[labeled_mask]

        if len(labeled_features) == 0:
            return labels_updated

        noise_features = features[noise_mask]
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

        noise_indices = np.where(noise_mask)[0]
        for i, neighbor_indices in enumerate(indices):
            neighbor_labels = labeled_labels[neighbor_indices]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            labels_updated[noise_indices[i]] = most_common_label

        return labels_updated

    def _find_optimal_min_cluster_size(self, features: np.ndarray) -> int:
        """Find optimal min_cluster_size using silhouette score.

        :param features: Feature matrix (n_samples, n_features)
        :return: Best min_cluster_size value
        """
        n_samples = len(features)

        min_sizes = [
            max(10, int(0.005 * n_samples)),
            max(25, int(0.01 * n_samples)),
            max(50, int(0.02 * n_samples)),
            max(100, int(0.03 * n_samples)),
            max(150, int(0.05 * n_samples)),
        ]

        scores: dict[int, float] = {}
        best_score = -1.0
        best_min_size = min_sizes[0]

        for min_size in min_sizes:
            self.hdbscan.min_cluster_size = min_size
            self.hdbscan.min_samples = min(10, min_size // 5)
            labels = cast(np.ndarray, self.hdbscan.fit_predict(features))  # type: ignore[attr-defined]

            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) >= 2:
                mask = cast(np.ndarray, labels != -1)
                if np.sum(mask) > 1:
                    score = cast(float, silhouette_score(features[mask], labels[mask]))  # type: ignore[arg-type]
                    scores[min_size] = score

                    if score > best_score:
                        best_score = score
                        best_min_size = min_size

        return best_min_size
