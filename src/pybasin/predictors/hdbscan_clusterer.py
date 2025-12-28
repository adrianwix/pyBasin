from typing import Any, cast

import numpy as np
from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors

from pybasin.predictors.base import ClustererPredictor


class HDBSCANClusterer(ClustererPredictor):
    """HDBSCAN clustering for basin stability analysis with optional auto-tuning and noise assignment (unsupervised learning)."""

    display_name: str = "HDBSCAN Clustering"

    clusterer: Any
    assign_noise: bool
    k_neighbors: int
    auto_tune: bool

    def __init__(
        self,
        clusterer: Any = None,
        assign_noise: bool = False,
        k_neighbors: int = 5,
        auto_tune: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize HDBSCAN clusterer.

        :param clusterer: HDBSCAN instance, or None to create default.
        :param assign_noise: Whether to assign noise points to nearest clusters using KNN.
        :param k_neighbors: Number of neighbors for KNN noise assignment.
        :param auto_tune: Whether to automatically tune min_cluster_size using silhouette score.
        :param kwargs: Additional arguments for HDBSCAN if clusterer is None.
                       Common: min_cluster_size=50, min_samples=10
        """
        if clusterer is None:
            if "min_cluster_size" not in kwargs:
                kwargs["min_cluster_size"] = 50
            if "min_samples" not in kwargs:
                kwargs["min_samples"] = min(10, kwargs.get("min_cluster_size", 50) // 5)
            clusterer = HDBSCAN(**kwargs)  # type: ignore[call-arg]

        self.clusterer = clusterer
        self.assign_noise = assign_noise
        self.k_neighbors = k_neighbors
        self.auto_tune = auto_tune

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using HDBSCAN clustering with optional noise assignment.

        :param features: Feature array to cluster.
        :return: Cluster labels.
        """
        if self.auto_tune:
            optimal_size = self._find_optimal_min_cluster_size(features)
            self.clusterer.min_cluster_size = optimal_size
            self.clusterer.min_samples = min(10, optimal_size // 5)

        labels = cast(np.ndarray, self.clusterer.fit_predict(features))

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
        k_actual = min(self.k_neighbors, len(labeled_features))
        nbrs = NearestNeighbors(n_neighbors=k_actual).fit(labeled_features)
        _, indices = nbrs.kneighbors(noise_features)

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
            clusterer = HDBSCAN(  # type: ignore[call-arg]
                min_cluster_size=min_size,
                min_samples=min(10, min_size // 5),
            )
            labels = cast(np.ndarray, clusterer.fit_predict(features))  # type: ignore[attr-defined]

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
