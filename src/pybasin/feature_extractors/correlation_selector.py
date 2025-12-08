"""Scikit-learn transformer for removing highly correlated features."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to remove highly correlated features.

    This transformer removes features with high pairwise correlations,
    keeping only one feature from each correlated group.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. Features with absolute correlation above
        this value will be considered redundant.

    Attributes
    ----------
    support_ : ndarray of shape (n_features,)
        Boolean mask of selected features.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: np.ndarray, y=None):
        """Compute which features to keep.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.n_features_in_ = X.shape[1]

        corr_matrix = np.corrcoef(X.T)

        to_drop = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.threshold:
                    to_drop.add(j)

        self.support_ = np.array([i not in to_drop for i in range(self.n_features_in_)])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove correlated features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features_out)
            Data with correlated features removed.
        """
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        """Get a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return feature indices. Otherwise, return boolean mask.

        Returns
        -------
        support : ndarray
            Boolean mask or integer indices of selected features.
        """
        if indices:
            return np.where(self.support_)[0]
        return self.support_
