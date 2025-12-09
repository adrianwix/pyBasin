from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar, cast

import numpy as np
import torch
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore[attr-defined]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solution import Solution

# TypeVar for ODE parameters
P = TypeVar("P")


class ClusterClassifier(ABC):
    """Abstract base class for clustering/classification algorithms."""

    display_name: str = "Classifier"

    @abstractmethod
    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given features.

        :param features: Feature array to classify.
        :return: Array of predicted labels.
        """
        pass


class SupervisedClassifier[P](ClusterClassifier):
    """Base class for supervised classifiers that require template data."""

    labels: list[str]
    template_y0: list[list[float]]  # Stored as list, converted to tensor during integration
    classifier: Any  # Type depends on subclass

    def __init__(
        self,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: P,
        solver: SolverProtocol | None = None,
    ):
        """
        Initialize the supervised classifier.

        :param template_y0: Template initial conditions as a list of lists (e.g., [[0.5, 0.0], [2.7, 0.0]]).
                           Will be converted to tensor with appropriate device during integration.
        :param labels: Ground truth labels for template conditions.
        :param ode_params: ODE parameters to use for generating training data.
        :param solver: Optional solver for template integration. If provided, this solver
                       will be used instead of the main solver (useful for CPU-based
                       template integration when templates are few).
        """
        self.labels = labels
        self.template_y0 = template_y0
        self.ode_params = deepcopy(ode_params)
        self.solver = solver
        self.solution: Solution | None = None  # Populated by integrate_templates

    @property
    def has_dedicated_solver(self) -> bool:
        """Check if the classifier has its own dedicated solver for template integration."""
        return self.solver is not None

    def integrate_templates(
        self,
        solver: SolverProtocol | None,
        ode_system: ODESystemProtocol,
    ) -> None:
        """
        Integrate ODE for template initial conditions (without feature extraction).

        This method should be called before fit_with_features() to allow the main
        feature extraction to fit the scaler first.

        By default, if no dedicated solver was provided at init, this method will
        automatically create a CPU variant of the passed solver. This is because
        CPU is typically faster than GPU for small batch sizes (like templates).

        :param solver: Fallback solver if no solver was provided at init. Can be None
                       if a solver was provided during classifier initialization.
        :param ode_system: ODE system to integrate (ODESystem or JaxODESystem).
        """
        classifier_ode_system = deepcopy(ode_system)
        classifier_ode_system.params = self.ode_params

        # Determine which solver to use
        if self.solver is not None:
            # User provided a dedicated solver - use it as-is
            effective_solver = self.solver
            solver_source = "dedicated"
        elif solver is not None:
            # No dedicated solver - auto-create CPU variant for better performance
            # GPU has overhead that hurts small batch sizes (templates are typically 2-5 samples)
            if hasattr(solver, "with_device") and str(solver.device) != "cpu":
                effective_solver = solver.with_device("cpu")
                solver_source = "auto-cpu"
                print(
                    "    [SupervisedClassifier] Auto-created CPU solver for templates "
                    "(faster for small batch sizes)"
                )
            else:
                effective_solver = solver
                solver_source = "fallback"
        else:
            raise ValueError(
                "No solver available. Either pass a solver to integrate_templates() "
                "or provide one during classifier initialization."
            )

        print(f"    [SupervisedClassifier] ODE params: {classifier_ode_system.params}")
        print(f"    [SupervisedClassifier] Template ICs: {len(self.template_y0)} templates")
        print(f"    [SupervisedClassifier] Labels: {self.labels}")
        print(
            f"    [SupervisedClassifier] Using solver: {type(effective_solver).__name__} ({solver_source})"
        )

        # Convert template_y0 to tensor on the solver's device
        template_tensor = torch.tensor(
            self.template_y0, dtype=torch.float32, device=effective_solver.device
        )

        t, y = effective_solver.integrate(classifier_ode_system, template_tensor)
        self.solution = Solution(initial_condition=template_tensor, time=t, y=y)

    def fit_with_features(
        self,
        feature_extractor: FeatureExtractor,
        feature_selector: Any | None = None,
    ) -> None:
        """
        Fit the classifier using pre-integrated template solutions.

        Must call integrate_templates() first to populate self.solution.

        :param feature_extractor: Feature extractor to transform trajectories.
        :param feature_selector: Optional feature selector (already fitted on main data).
                                If provided, applies the same filtering to template features.
        :raises ValueError: If filtering removes all template features.
        """
        if self.solution is None:
            raise RuntimeError("Must call integrate_templates() before fit_with_features()")

        # Extract features from pre-integrated solution
        features = feature_extractor.extract_features(self.solution)

        # Apply feature filtering if selector provided
        if feature_selector is not None:
            features_np = features.detach().cpu().numpy()
            features_filtered_np = feature_selector.transform(features_np)

            if features_filtered_np.shape[1] == 0:
                raise ValueError(
                    f"Feature filtering removed all {features_np.shape[1]} template features. "
                    "This should not happen if the selector was fitted on main data."
                )

            features = torch.from_numpy(features_filtered_np).to(  # type: ignore[misc]
                dtype=features.dtype, device=features.device
            )

        train_x = features.detach().cpu().numpy()
        train_y = self.labels

        print(
            f"    Training classifier with {train_x.shape[0]} samples, {train_x.shape[1]} features"
        )

        self.classifier.fit(train_x, train_y)

    def fit(
        self,
        solver: SolverProtocol,
        ode_system: ODESystemProtocol,
        feature_extractor: FeatureExtractor,
    ) -> None:
        """
        Fit the classifier using template initial conditions.

        WARNING: This method extracts features from templates FIRST, which means
        the scaler will be fitted on template data (often just 2 samples). For
        better normalization, use integrate_templates() + fit_with_features()
        to allow the main data to fit the scaler first.

        :param solver: Solver to integrate the ODE system (Solver or JaxSolver).
        :param ode_system: ODE system to integrate (ODESystem or JaxODESystem).
        :param feature_extractor: Feature extractor to transform trajectories.
        """
        # Use the new two-step methods for consistency
        self.integrate_templates(solver, ode_system)
        self.fit_with_features(feature_extractor)


class KNNCluster[P](SupervisedClassifier[P]):
    """K-Nearest Neighbors classifier for basin stability analysis."""

    display_name: str = "KNN Classifier"

    def __init__(
        self,
        classifier: KNeighborsClassifier | None,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: P,
        solver: SolverProtocol | None = None,
        **kwargs: Any,
    ):
        """
        Initialize KNN classifier.

        :param classifier: KNeighborsClassifier instance, or None to create default.
        :param template_y0: Template initial conditions as a list of lists.
        :param labels: Ground truth labels.
        :param ode_params: ODE parameters.
        :param solver: Optional solver for template integration.
        :param kwargs: Additional arguments for KNeighborsClassifier if classifier is None.
        """
        if classifier is None:
            classifier = KNeighborsClassifier(**kwargs)
        self.classifier = classifier
        super().__init__(template_y0, labels, ode_params, solver)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using the fitted KNN classifier.

        :param features: Feature array to classify.
        :return: Predicted labels.
        """
        return self.classifier.predict(features)


class UnsupervisedClassifier[P](ClusterClassifier):
    """Base class for unsupervised clustering algorithms."""

    def __init__(self, template_y0: torch.Tensor, ode_params: P):
        """
        Initialize the unsupervised classifier.

        :param template_y0: Template initial conditions to cluster.
        :param ode_params: ODE parameters.
        """
        self.template_y0 = template_y0
        self.ode_params = ode_params


class DBSCANCluster(UnsupervisedClassifier[Any]):
    """DBSCAN clustering for basin stability analysis."""

    display_name: str = "DBSCAN Clustering"

    classifier: DBSCAN

    def __init__(self, classifier: DBSCAN | None = None, **kwargs: Any):
        """
        Initialize DBSCAN classifier.

        :param classifier: DBSCAN instance, or None to create default.
        :param kwargs: Additional arguments for DBSCAN if classifier is None.
        """
        if classifier is None:
            classifier = DBSCAN(**kwargs)
        self.classifier = classifier

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using DBSCAN clustering.

        :param features: Feature array to cluster.
        :return: Cluster labels.
        """
        return self.classifier.fit_predict(features)


class HDBSCANCluster(UnsupervisedClassifier[Any]):
    """HDBSCAN clustering for basin stability analysis with optional auto-tuning and noise assignment."""

    display_name: str = "HDBSCAN Clustering"

    classifier: Any
    assign_noise: bool
    k_neighbors: int
    auto_tune: bool

    def __init__(
        self,
        classifier: Any = None,
        assign_noise: bool = False,
        k_neighbors: int = 5,
        auto_tune: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize HDBSCAN classifier.

        :param classifier: HDBSCAN instance, or None to create default.
        :param assign_noise: Whether to assign noise points to nearest clusters using KNN.
        :param k_neighbors: Number of neighbors for KNN noise assignment.
        :param auto_tune: Whether to automatically tune min_cluster_size using silhouette score.
        :param kwargs: Additional arguments for HDBSCAN if classifier is None.
                       Common: min_cluster_size=50, min_samples=10
        """
        if classifier is None:
            if "min_cluster_size" not in kwargs:
                kwargs["min_cluster_size"] = 50
            if "min_samples" not in kwargs:
                kwargs["min_samples"] = min(10, kwargs.get("min_cluster_size", 50) // 5)
            classifier = HDBSCAN(**kwargs)  # type: ignore[call-arg]

        self.classifier = classifier
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
            self.classifier.min_cluster_size = optimal_size
            self.classifier.min_samples = min(10, optimal_size // 5)

        labels = cast(np.ndarray, self.classifier.fit_predict(features))

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
