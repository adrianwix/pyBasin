"""Scikit-learn transformer for removing highly correlated features.

Uses the mean absolute correlation ranking from Kuhn & Johnson (2013),
"Applied Predictive Modeling", Chapter 3. When two features exceed the
correlation threshold, the one with higher mean absolute correlation
across all remaining features is dropped (more globally redundant).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to remove highly correlated features.

    When a pair of features has absolute correlation above ``threshold``,
    the feature with the higher mean absolute correlation against all
    remaining features is removed. This follows the algorithm described
    by Kuhn & Johnson (2013) and implemented in R's
    ``caret::findCorrelation()``.

    The mean absolute correlation is recomputed after each removal so
    that subsequent decisions reflect the current feature set.

    :ivar threshold: Correlation threshold. Feature pairs with absolute
        correlation above this value trigger a removal decision.
    :ivar min_features: Minimum number of features to keep.
    :ivar support_: Boolean mask of selected features (set after ``fit``).
    :ivar n_features_in_: Number of input features (set after ``fit``).
    """

    def __init__(self, threshold: float = 0.9, min_features: int = 3):
        self.threshold: float = threshold
        self.min_features: int = min_features

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Compute which features to keep using mean absolute correlation ranking.

        For each pair exceeding the threshold, the feature with the higher
        mean absolute correlation across all remaining features is dropped.
        The correlation statistics are recomputed after each removal.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :return: Fitted transformer.
        """
        self.n_features_in_ = X.shape[1]

        if self.n_features_in_ <= self.min_features:
            self.support_ = np.ones(self.n_features_in_, dtype=bool)
            return self

        corr_matrix: np.ndarray = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(corr_matrix, 0.0)

        remaining: list[int] = list(range(self.n_features_in_))

        while len(remaining) > self.min_features:
            sub_corr: np.ndarray = corr_matrix[np.ix_(remaining, remaining)]
            max_corr: float = float(np.max(sub_corr))

            if max_corr <= self.threshold:
                break

            i_local, j_local = divmod(int(np.argmax(sub_corr)), len(remaining))
            mean_corr_i: float = float(np.mean(sub_corr[i_local]))  # type: ignore[arg-type]
            mean_corr_j: float = float(np.mean(sub_corr[j_local]))  # type: ignore[arg-type]

            drop_local: int = i_local if mean_corr_i >= mean_corr_j else j_local
            remaining.pop(drop_local)

        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[remaining] = True

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
