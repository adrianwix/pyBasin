"""Default feature selector using variance threshold and correlation filtering.

This module provides a Pipeline that combines VarianceThreshold and
CorrelationSelector for feature filtering.
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from pybasin.feature_selector.correlation_selector import CorrelationSelector


class DefaultFeatureSelector(Pipeline):
    """Feature selector combining variance threshold and correlation filtering.

    This class extends sklearn's Pipeline with two steps:

    1. VarianceThreshold: Removes features with variance below threshold
    2. CorrelationSelector: Removes highly correlated features (|corr| > threshold)

    The correlation threshold uses absolute correlation values, meaning both
    positive and negative correlations above the threshold will trigger removal.

    As a Pipeline subclass, this implements the full sklearn transformer API:
    fit(), transform(), fit_transform(), get_params(), set_params(), etc.

    ```python
    selector = DefaultFeatureSelector(variance_threshold=0.01, correlation_threshold=0.95)
    features_filtered = selector.fit_transform(features)
    ```

    :ivar variance_threshold: Minimum variance required to keep a feature.
    :ivar correlation_threshold: Maximum absolute correlation allowed between features. Features with |correlation| > threshold will be removed.
    :ivar min_features: Minimum number of features to keep.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        min_features: int = 2,
    ):
        self.variance_threshold: float = variance_threshold
        self.correlation_threshold: float = correlation_threshold
        self.min_features: int = min_features

        super().__init__(  # type: ignore[misc]
            [
                ("variance", VarianceThreshold(threshold=variance_threshold)),
                (
                    "correlation",
                    CorrelationSelector(threshold=correlation_threshold, min_features=min_features),
                ),
            ]
        )

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of features that passed the filter.

        :param indices: If True, returns indices. If False, returns boolean mask.
        :return: Boolean mask or integer indices of selected features.
        """
        # Get support from each step and combine
        variance_support: np.ndarray = self.named_steps["variance"].get_support(indices=False)  # type: ignore[assignment]
        correlation_support: np.ndarray = self.named_steps["correlation"].get_support(indices=False)  # type: ignore[assignment]

        # Start with all features from variance step
        combined_support: np.ndarray = variance_support.copy()  # type: ignore[assignment]

        # Apply correlation mask to the variance-filtered features
        # The correlation step only sees variance-filtered features
        variance_indices: np.ndarray = np.where(variance_support)[0]  # type: ignore[arg-type]
        kept_after_correlation: np.ndarray = variance_indices[correlation_support]

        # Create final mask
        combined_support[:] = False
        combined_support[kept_after_correlation] = True

        if indices:
            return np.where(combined_support)[0]  # type: ignore[arg-type]
        return combined_support  # type: ignore[return-value]
