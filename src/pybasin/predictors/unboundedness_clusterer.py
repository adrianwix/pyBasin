from collections.abc import Callable
from typing import Any

import numpy as np

from pybasin.predictors.base import ClustererPredictor


def default_unbounded_detector(x: np.ndarray) -> np.ndarray:
    """
    Default unbounded trajectory detector.

    Detects unbounded trajectories based on:
    - Inf or -Inf values (from JAX solver)
    - Values at extreme bounds: 1e10 or -1e10 (from torch feature extractor with imputation)

    :param x: Feature array of shape (n_samples, n_features).
    :return: Boolean array of shape (n_samples,) where True indicates unbounded.
    """
    has_inf = np.isinf(x).any(axis=1)
    has_extreme = (np.abs(x) >= 1e10).any(axis=1)
    return has_inf | has_extreme  # type: ignore[return-value]


class UnboundednessClusterer(ClustererPredictor):
    """
    Meta-clusterer for separately labeling unbounded trajectories.

    This meta-clusterer wraps another ClustererPredictor and handles unbounded trajectories
    separately. Unbounded trajectories are identified using a detector function and assigned
    a special label, while bounded trajectories are processed using the wrapped clusterer.

    This is particularly useful in basin stability calculations where some trajectories
    may diverge to infinity (e.g., in the Lorenz system). By excluding unbounded trajectories
    from clustering, the wrapped clusterer can focus on discovering patterns in bounded basins
    without contamination from divergent trajectories.

    Parameters
    ----------
    clusterer : ClustererPredictor
        The base clusterer to use for bounded trajectories. Must implement the
        ClustererPredictor interface (predict_labels method).

    unbounded_detector : callable, default=None
        Function to detect unbounded trajectories. Should take a feature array
        of shape (n_samples, n_features) and return a boolean array of shape
        (n_samples,) where True indicates unbounded. If None, uses the default
        detector which identifies:
        - Trajectories with Inf/-Inf values (from JAX solver)
        - Trajectories with values at ±1e10 (from torch feature extractor)

    unbounded_label : int or str, default="unbounded"
        Label to assign to unbounded trajectories.

    Attributes
    ----------
    clusterer : ClustererPredictor
        The wrapped clusterer instance.

    unbounded_detector : callable
        Function used to detect unbounded trajectories.

    unbounded_label : int or str
        Label assigned to unbounded trajectories.

    Examples
    --------
    >>> from pybasin.predictors.unboundedness_clusterer import UnboundednessClusterer
    >>> from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
    >>> import numpy as np
    >>> # Create features with some unbounded samples
    >>> features = np.random.randn(100, 10)
    >>> features[0, :] = np.inf  # Unbounded sample
    >>> features[1, :] = 1e10    # Unbounded sample
    >>> # Wrap HDBSCAN with unboundedness handling
    >>> base_clusterer = HDBSCANClusterer(min_cluster_size=5)
    >>> clusterer = UnboundednessClusterer(base_clusterer)
    >>> labels = clusterer.predict_labels(features)
    >>> print(f"Unbounded samples: {np.sum(labels == 'unbounded')}")

    Notes
    -----
    - Only bounded samples are passed to the wrapped clusterer for clustering
    - The unbounded label is automatically tracked and returned for unbounded samples
    - If all samples are unbounded, all labels will be the unbounded label
    - This prevents unbounded trajectories from distorting cluster centroids and boundaries
    """

    display_name: str = "Unboundedness Meta-Clusterer"

    def __init__(
        self,
        clusterer: ClustererPredictor,
        unbounded_detector: Callable[[np.ndarray], np.ndarray] | None = None,
        unbounded_label: int | str = "unbounded",
        **kwargs: Any,
    ):
        """
        Initialize the unboundedness meta-clusterer.

        :param clusterer: Base clusterer to use for bounded trajectories.
        :param unbounded_detector: Function to detect unbounded trajectories.
        :param unbounded_label: Label to assign to unbounded trajectories.
        :param kwargs: Additional arguments passed to ClustererPredictor base (unused).
        """
        self.clusterer = clusterer
        self.unbounded_detector = unbounded_detector
        self.unbounded_label = unbounded_label

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for features, separating unbounded trajectories.

        Unbounded trajectories are detected and labeled separately, while bounded
        trajectories are clustered using the wrapped clusterer.

        :param features: Feature array of shape (n_samples, n_features).
        :return: Array of predicted labels with unbounded trajectories labeled separately.
        """
        detector = (
            self.unbounded_detector
            if self.unbounded_detector is not None
            else default_unbounded_detector
        )
        unbounded_mask = detector(features)

        labels = np.empty(features.shape[0], dtype=object)
        labels[unbounded_mask] = self.unbounded_label

        if np.any(~unbounded_mask):
            bounded_features = features[~unbounded_mask]
            bounded_labels = self.clusterer.predict_labels(bounded_features)
            labels[~unbounded_mask] = bounded_labels

        return labels
