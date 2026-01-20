"""Scikit-learn transformer for removing highly correlated features."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to remove highly correlated features.

    This transformer removes features with high pairwise correlations,
    keeping only one feature from each correlated group.

    :ivar threshold: Correlation threshold. Features with absolute correlation above this value will be considered redundant.
    :ivar min_features: Minimum number of features to keep. If removing correlated features would result in fewer than this many, some correlated features are retained.
    :ivar support_: Boolean mask of selected features.
    :ivar n_features_in_: Number of input features.
    """

    def __init__(self, threshold: float = 0.95, min_features: int = 2):
        self.threshold: float = threshold
        self.min_features: int = min_features

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Compute which features to keep.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :return: Fitted transformer.
        """
        self.n_features_in_ = X.shape[1]

        if self.n_features_in_ <= self.min_features:
            self.support_ = np.ones(self.n_features_in_, dtype=bool)
            return self

        corr_matrix = np.corrcoef(X.T)

        to_drop: set[int] = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if (
                    abs(corr_matrix[i, j]) > self.threshold
                    and self.n_features_in_ - len(to_drop) > self.min_features
                ):
                    to_drop.add(j)

        self.support_ = np.array([i not in to_drop for i in range(self.n_features_in_)])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove correlated features.

        :param X: Input data of shape (n_samples, n_features).
        :return: Data with correlated features removed, shape (n_samples, n_features_out).
        """
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        """Get a mask or indices of selected features.

        :param indices: If True, return feature indices. Otherwise, return boolean mask.
        :return: Boolean mask or integer indices of selected features.
        """
        if indices:
            return np.where(self.support_)[0]
        return self.support_
