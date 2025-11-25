from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

from pybasin.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solution import Solution

# TypeVar for ODE parameters
P = TypeVar("P")


class ClusterClassifier(ABC):
    """Abstract base class for clustering/classification algorithms."""

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
    ) -> None:
        """
        Fit the classifier using pre-integrated template solutions.

        Must call integrate_templates() first to populate self.solution.

        :param feature_extractor: Feature extractor to transform trajectories.
        """
        if self.solution is None:
            raise RuntimeError("Must call integrate_templates() before fit_with_features()")

        # Extract features from pre-integrated solution
        features = feature_extractor.extract_features(self.solution)

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
