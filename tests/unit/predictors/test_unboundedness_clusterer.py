import numpy as np
from sklearn.datasets import make_blobs

from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.predictors.unboundedness_clusterer import (
    UnboundednessClusterer,
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


class TestUnboundednessClusterer:
    """Test the UnboundednessClusterer meta-clusterer."""

    def test_initialization(self):
        base_clusterer = HDBSCANClusterer()

        clusterer = UnboundednessClusterer(base_clusterer)

        assert clusterer.clusterer is base_clusterer
        assert clusterer.unbounded_detector is None
        assert clusterer.unbounded_label == "unbounded"

    def test_custom_unbounded_label(self):
        base_clusterer = HDBSCANClusterer()

        clusterer = UnboundednessClusterer(base_clusterer, unbounded_label=-1)

        assert clusterer.unbounded_label == -1

    def test_custom_detector(self):
        def custom_detector(X: np.ndarray) -> np.ndarray:
            return X[:, 0] > 100

        base_clusterer = HDBSCANClusterer()

        clusterer = UnboundednessClusterer(base_clusterer, unbounded_detector=custom_detector)

        assert clusterer.unbounded_detector is custom_detector

    def test_predict_labels_with_bounded_data_only(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert labels.shape == (100,)
        assert not np.any(labels == "unbounded")

    def test_predict_labels_with_unbounded_data(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf
        X[1, :] = -1e10
        X[2, 0] = 1e10

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert labels.shape == (100,)
        assert labels[0] == "unbounded"
        assert labels[1] == "unbounded"
        assert labels[2] == "unbounded"
        assert not np.all(labels == "unbounded")

    def test_predict_labels_all_unbounded(self):
        X = np.array(
            [[np.inf, 1.0, 2.0, 3.0, 4.0], [1e10, 2.0, 3.0, 4.0, 5.0], [-1e10, 1.0, 2.0, 3.0, 4.0]]
        )

        base_clusterer = HDBSCANClusterer(min_cluster_size=2)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert labels.shape == (3,)
        assert np.all(labels == "unbounded")

    def test_unbounded_trajectories_excluded_from_clustering(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf
        X[1, :] = -1e10

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        bounded_labels = labels[2:]
        unique_bounded = np.unique(bounded_labels)

        assert "unbounded" not in unique_bounded
        assert len(unique_bounded) >= 1

    def test_custom_detector_function(self):
        def custom_detector(X: np.ndarray) -> np.ndarray:
            return X[:, 0] > 50

        X = np.random.randn(100, 5) * 10
        X[:5, 0] = 100

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer, unbounded_detector=custom_detector)
        labels = clusterer.predict_labels(X)

        assert np.sum(labels == "unbounded") == 5

    def test_custom_unbounded_label_int(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42
        )
        X[0, :] = np.inf

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer, unbounded_label=-999)
        labels = clusterer.predict_labels(X)

        assert labels[0] == -999

    def test_display_name_attribute(self):
        base_clusterer = HDBSCANClusterer()

        clusterer = UnboundednessClusterer(base_clusterer)

        assert hasattr(clusterer, "display_name")
        assert isinstance(clusterer.display_name, str)
        assert clusterer.display_name == "Unboundedness Meta-Clusterer"

    def test_example_usage_pattern(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=10, centers=3, random_state=42
        )
        X[0, :] = 1e10
        X[1, :] = -1e10

        base_clusterer = HDBSCANClusterer(min_cluster_size=5)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert np.sum(labels == "unbounded") == 2
        assert labels.shape == (100,)

    def test_mixed_unbounded_with_multiple_clusters(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=200, n_features=10, centers=4, random_state=42, cluster_std=1.5
        )

        X[0, :] = np.inf
        X[50, :] = -1e10
        X[100, 0] = 1e10
        X[150, :] = -np.inf

        base_clusterer = HDBSCANClusterer(min_cluster_size=10)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert np.sum(labels == "unbounded") == 4
        assert labels.shape == (200,)

        bounded_labels = labels[labels != "unbounded"]
        assert len(bounded_labels) == 196
        assert len(np.unique(bounded_labels)) >= 1

    def test_no_contamination_from_unbounded(self):
        X, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
            n_samples=100, n_features=5, centers=3, random_state=42, cluster_std=0.5
        )

        X_with_unbounded = X.copy()
        X_with_unbounded[:5, :] = np.inf

        base_clusterer1 = HDBSCANClusterer(min_cluster_size=5)
        base_clusterer2 = HDBSCANClusterer(min_cluster_size=5)

        clusterer_with_protection = UnboundednessClusterer(base_clusterer1)
        labels_protected = clusterer_with_protection.predict_labels(X_with_unbounded)

        labels_clean = base_clusterer2.predict_labels(X)

        # Verify unbounded samples are correctly labeled
        assert np.sum(labels_protected == "unbounded") == 5

        # Verify bounded samples have same number of clusters
        bounded_mask = labels_protected != "unbounded"
        bounded_labels_protected = labels_protected[bounded_mask]
        unique_protected = np.unique(bounded_labels_protected)
        unique_clean = np.unique(labels_clean[bounded_mask])

        # Should have same number of clusters (labels may differ but structure should be same)
        assert len(unique_protected) == len(unique_clean)

    def test_empty_bounded_set(self):
        X = np.array([[np.inf, np.inf], [1e10, 1e10], [-1e10, -1e10]])

        base_clusterer = HDBSCANClusterer(min_cluster_size=2)

        clusterer = UnboundednessClusterer(base_clusterer)
        labels = clusterer.predict_labels(X)

        assert np.all(labels == "unbounded")
