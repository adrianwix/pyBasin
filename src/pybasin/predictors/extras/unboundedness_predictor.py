# TODO: Delete if not needed

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_clusterer,  # type: ignore
)
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)


def default_unbounded_detector(x: np.ndarray) -> np.ndarray:
    """
    Default unbounded trajectory detector.

    Detects unbounded trajectories based on:
    - NaN values (invalid/undefined trajectories)
    - Inf or -Inf values (from JAX solver)
    - Values at extreme bounds: 1e10 or -1e10 (from torch feature extractor with imputation)

    :param x: Feature array of shape (n_samples, n_features).
    :return: Boolean array of shape (n_samples,) where True indicates unbounded.
    """
    has_nan = np.isnan(x).any(axis=1)
    has_inf = np.isinf(x).any(axis=1)
    has_extreme = (np.abs(x) >= 1e10).any(axis=1)
    return has_nan | has_inf | has_extreme  # type: ignore[return-value]


class UnboundednessPredictor(MetaEstimatorMixin, BaseEstimator):
    """
    Meta-estimator for separately labeling unbounded trajectories.

    This meta-estimator wraps another estimator (classifier or clusterer) and handles
    unbounded trajectories separately. Unbounded trajectories are identified using a
    detector function and assigned a special label, while bounded trajectories are
    processed using the wrapped estimator.

    The API adapts to the wrapped estimator type (similar to sklearn.pipeline.Pipeline):
    - If estimator is a clusterer: provides fit(), fit_predict(), predict()
    - If estimator is a classifier: provides fit(), predict(), and potentially predict_proba()

    This is particularly useful in basin stability calculations where some trajectories
    may diverge to infinity (e.g., in the Lorenz system).

    Parameters
    ----------
    estimator : estimator object
        The base estimator to use for bounded trajectories. Must be a classifier or
        clusterer implementing `fit` and `predict` methods (or `fit_predict` for clustering).

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
    estimator_ : estimator object
        The fitted base estimator (only fitted on bounded samples).

    classes_ : ndarray of shape (n_classes,)
        The classes labels (only for classifiers), including the unbounded label.

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample from the last fit operation (only for clusterers).

    n_features_in_ : int
        Number of features seen during fit.

    bounded_mask_ : ndarray of shape (n_samples,)
        Boolean mask indicating which training samples were bounded.

    Examples
    --------
    >>> from pybasin.predictors.extras import UnboundednessPredictor
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> import numpy as np
    >>> X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    >>> # Add some "unbounded" samples with extreme values
    >>> X[0, :] = 1e10
    >>> X[1, :] = -1e10
    >>> clf = UnboundednessPredictor(KMeans(n_clusters=3, random_state=42))
    >>> clf.fit(X)
    >>> labels = clf.predict(X)
    >>> print(f"Unbounded samples: {np.sum(labels == 'unbounded')}")

    Notes
    -----
    - Only bounded samples are used to fit the base estimator
    - The unbounded label is automatically tracked
    - If all samples are unbounded, the estimator will only predict the unbounded label
    - The estimator type validation ensures only classifiers or clusterers are accepted
    """

    def __init__(
        self,
        estimator: Any,
        unbounded_detector: Callable[[np.ndarray], np.ndarray] | None = None,
        unbounded_label: int | str = "unbounded",
    ):
        self.estimator = estimator
        self.unbounded_detector = unbounded_detector
        self.unbounded_label = unbounded_label

    def _validate_estimator(self) -> None:
        """Validate that the estimator is a classifier or clusterer."""
        if not (is_classifier(self.estimator) or is_clusterer(self.estimator)):
            raise ValueError(
                f"estimator must be a classifier or clusterer, got {type(self.estimator).__name__}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "UnboundednessPredictor":
        """
        Fit the meta-estimator.

        Detects unbounded samples, then fits the base estimator only on bounded samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target values. Only used if the base estimator is a classifier.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_estimator()

        X = cast(np.ndarray, check_array(X, ensure_all_finite=False))  # type: ignore[call-arg]
        self.n_features_in_: int = X.shape[1]

        detector = (
            self.unbounded_detector
            if self.unbounded_detector is not None
            else default_unbounded_detector
        )
        unbounded_mask = detector(X)
        self.bounded_mask_ = ~unbounded_mask

        if np.all(unbounded_mask):
            self.estimator_ = None
            if is_classifier(self.estimator):
                self.classes_ = np.array([self.unbounded_label])
            else:
                self.labels_ = np.full(X.shape[0], self.unbounded_label)
                self.classes_ = np.array([self.unbounded_label])
        else:
            X_bounded = X[self.bounded_mask_]
            self.estimator_ = clone(self.estimator)

            # Handle classifiers (need y) vs clusterers (don't need y)
            if is_classifier(self.estimator):
                if y is None:
                    raise ValueError("y is required when using a classifier estimator")
                y_bounded = y[self.bounded_mask_]  # type: ignore[index]
                self.estimator_.fit(X_bounded, y_bounded)

                # Get classes from fitted classifier
                if hasattr(self.estimator_, "classes_"):
                    base_classes = self.estimator_.classes_
                else:
                    labels_bounded = self.estimator_.predict(X_bounded)
                    base_classes = np.unique(labels_bounded)

                if unbounded_mask.any():
                    self.classes_ = np.append(base_classes, self.unbounded_label)
                else:
                    self.classes_ = base_classes
            else:
                # Clusterer case
                if hasattr(self.estimator_, "fit_predict"):
                    labels_bounded = self.estimator_.fit_predict(X_bounded)
                else:
                    self.estimator_.fit(X_bounded)
                    labels_bounded = self.estimator_.predict(X_bounded)

                # Store labels for all samples
                self.labels_ = np.empty(X.shape[0], dtype=object)
                self.labels_[unbounded_mask] = self.unbounded_label
                self.labels_[self.bounded_mask_] = labels_bounded

                # Also provide classes_ for consistency with classifiers
                base_classes = np.unique(labels_bounded)
                if unbounded_mask.any():
                    self.classes_ = np.append(base_classes, self.unbounded_label)
                else:
                    self.classes_ = base_classes

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = cast(np.ndarray, check_array(X, ensure_all_finite=False))  # type: ignore[call-arg]

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but UnboundednessPredictor "
                f"is expecting {self.n_features_in_} features as input."
            )

        detector = (
            self.unbounded_detector
            if self.unbounded_detector is not None
            else default_unbounded_detector
        )
        unbounded_mask = detector(X)

        labels = np.empty(X.shape[0], dtype=object)
        labels[unbounded_mask] = self.unbounded_label

        if self.estimator_ is not None and np.any(~unbounded_mask):
            X_bounded = X[~unbounded_mask]
            bounded_labels = self.estimator_.predict(X_bounded)
            labels[~unbounded_mask] = bounded_labels

        # Convert to int if all labels are integers (for sklearn compatibility)
        try:
            labels_int = labels.astype(int)
            if np.array_equal(labels, labels_int):
                return labels_int
        except (ValueError, TypeError):
            pass

        return labels

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit the meta-estimator and predict labels (for clusterers).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target values (ignored for clusterers, required for classifiers).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        self.fit(X, y)
        if is_clusterer(self.estimator):
            return self.labels_
        else:
            return self.predict(X)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given features (compatibility with LabelPredictor interface).

        :param features: Feature array to predict labels for.
        :return: Array of predicted labels.
        """
        return self.predict(features)

    def __sklearn_tags__(self) -> Any:
        """
        Provide sklearn tags based on the wrapped estimator type.

        The meta-estimator adapts its behavior based on the wrapped estimator,
        similar to Pipeline.
        """
        tags = super().__sklearn_tags__()  # type: ignore[misc]

        # Set estimator type based on wrapped estimator
        if is_classifier(self.estimator):
            tags.estimator_type = "classifier"  # type: ignore[attr-defined]
        elif is_clusterer(self.estimator):
            tags.estimator_type = "clusterer"  # type: ignore[attr-defined]

        tags.input_tags.allow_nan = True  # type: ignore[attr-defined]
        return tags  # type: ignore[return-value]
