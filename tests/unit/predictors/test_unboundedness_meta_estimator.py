import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils.estimator_checks import (
    check_estimator,  # pyright: ignore[reportUnknownVariableType]
)

from pybasin.predictors.unboundedness_meta_estimator import (
    UnboundednessMetaEstimator,
    default_unbounded_detector,
)


class TestDefaultUnboundedDetector:
    """Test the default unbounded trajectory detector."""

    def test_detects_inf_values(self):
        X = np.array([[1.0, 2.0], [np.inf, 1.0], [1.0, -np.inf], [0.5, 0.5]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(unbounded, expected)

    def test_detects_extreme_values(self):
        X = np.array([[1.0, 2.0], [1e10, 1.0], [1.0, -1e10], [0.5, 0.5]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(unbounded, expected)

    def test_detects_nan_values(self):
        X = np.array([[1.0, 2.0], [np.nan, 1.0], [1.0, np.nan], [0.5, 0.5]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(unbounded, expected)

    def test_detects_mixed_conditions(self):
        X = np.array([[1.0, 2.0], [np.inf, 1.0], [1.0, -1e10], [0.5, 0.5], [1e10, np.inf]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([False, True, True, False, True])
        np.testing.assert_array_equal(unbounded, expected)

    def test_all_bounded(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(unbounded, expected)

    def test_all_unbounded(self):
        X = np.array([[np.inf, 2.0], [1e10, 4.0], [-1e10, 6.0]])
        unbounded = default_unbounded_detector(X)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(unbounded, expected)


class TestUnboundednessMetaEstimator:
    """Test the UnboundednessMetaEstimator meta-estimator."""

    def test_initialization(self):
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)
        assert clf.estimator is estimator
        assert clf.unbounded_detector is None
        assert clf.unbounded_label == "unbounded"

    def test_custom_unbounded_label(self):
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator, unbounded_label=-1)
        assert clf.unbounded_label == -1

    def test_custom_detector(self):
        def custom_detector(X: np.ndarray) -> np.ndarray:
            return X[:, 0] > 100

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator, unbounded_detector=custom_detector)
        assert clf.unbounded_detector is custom_detector

    def test_fit_with_bounded_data_only(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)

        assert hasattr(clf, "estimator_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_features_in_")
        assert hasattr(clf, "bounded_mask_")
        assert clf.n_features_in_ == 5  # pyright: ignore[reportUnknownMemberType]
        assert np.all(clf.bounded_mask_)
        assert "unbounded" not in clf.classes_

    def test_fit_with_unbounded_data(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf
        X[1, :] = -1e10
        X[2, 0] = 1e10

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)

        assert hasattr(clf, "bounded_mask_")
        assert np.sum(~clf.bounded_mask_) == 3
        assert "unbounded" in clf.classes_

    def test_fit_all_unbounded(self):
        X = np.array([[np.inf, 1.0], [1e10, 2.0], [-1e10, 3.0]])
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)

        assert clf.estimator_ is None
        assert len(clf.classes_) == 1
        assert clf.classes_[0] == "unbounded"

    def test_predict_bounded_only(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)
        labels = clf.predict(X)

        assert labels.shape == (100,)
        assert not np.any(labels == "unbounded")

    def test_predict_with_unbounded(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf
        X[1, :] = -1e10

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)
        labels = clf.predict(X)

        assert labels.shape == (100,)
        assert labels[0] == "unbounded"
        assert labels[1] == "unbounded"
        assert not np.all(labels == "unbounded")

    def test_predict_new_unbounded_samples(self):
        X_train, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X_test = X_train[:10].copy()
        X_test[5, :] = np.inf

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X_train)
        labels = clf.predict(X_test)

        assert labels.shape == (10,)
        assert labels[5] == "unbounded"
        assert not np.all(labels == "unbounded")

    def test_predict_all_unbounded(self):
        X_train = np.random.randn(100, 5)
        X_test = np.array([[np.inf, 1.0, 2.0, 3.0, 4.0], [1e10, 2.0, 3.0, 4.0, 5.0]])

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X_train)
        labels = clf.predict(X_test)

        assert labels.shape == (2,)
        assert np.all(labels == "unbounded")

    def test_custom_detector_function(self):
        def custom_detector(X: np.ndarray) -> np.ndarray:
            return X[:, 0] > 50

        X = np.random.randn(100, 5) * 10
        X[:5, 0] = 100

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator, unbounded_detector=custom_detector)

        clf.fit(X)
        labels = clf.predict(X)

        assert np.sum(labels == "unbounded") == 5

    def test_custom_unbounded_label_int(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator, unbounded_label=-999)

        clf.fit(X)
        labels = clf.predict(X)

        assert labels[0] == -999
        assert -999 in clf.classes_

    def test_feature_dimension_validation(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(10, 3)

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X_train)

        with pytest.raises(ValueError, match="expecting 5 features"):
            clf.predict(X_test)

    def test_sklearn_compatibility(self):
        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator, unbounded_label=-1)

        check_estimator(clf)

    def test_classes_attribute_includes_base_classes(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf

        estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
        clf = UnboundednessMetaEstimator(estimator)

        clf.fit(X)

        assert "unbounded" in clf.classes_
        assert len(clf.classes_) > 1

    def test_example_usage_pattern(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=10, centers=3, random_state=42
        )
        X[0, :] = 1e10
        X[1, :] = -1e10

        clf = UnboundednessMetaEstimator(KMeans(n_clusters=3, random_state=42, n_init=10))
        clf.fit(X)
        labels = clf.predict(X)

        assert np.sum(labels == "unbounded") == 2
        assert labels.shape == (100,)
