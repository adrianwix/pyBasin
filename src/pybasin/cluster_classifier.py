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

# TODO: Fix namings

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
    initial_conditions: torch.Tensor
    classifier: Any  # Type depends on subclass

    def __init__(
        self,
        initial_conditions: torch.Tensor,
        labels: list[str],
        ode_params: P,
    ):
        """
        Initialize the supervised classifier.

        :param initial_conditions: Template initial conditions for training.
        :param labels: Ground truth labels for template conditions.
        :param ode_params: ODE parameters to use for generating training data.
        """
        self.labels = labels
        self.initial_conditions = initial_conditions
        self.ode_params = deepcopy(ode_params)
        self.solution: Solution | None = None  # Populated by integrate_templates

    def integrate_templates(
        self,
        solver: SolverProtocol,
        ode_system: ODESystemProtocol,
    ) -> None:
        """
        Integrate ODE for template initial conditions (without feature extraction).

        This method should be called before fit_with_features() to allow the main
        feature extraction to fit the scaler first.

        :param solver: Solver to integrate the ODE system (Solver or JaxSolver).
        :param ode_system: ODE system to integrate (ODESystem or JaxODESystem).
        """
        classifier_ode_system = deepcopy(ode_system)
        classifier_ode_system.params = self.ode_params

        print(f"    [SupervisedClassifier] ODE params: {classifier_ode_system.params}")
        print(f"    [SupervisedClassifier] Template ICs: {self.initial_conditions.shape}")
        print(f"    [SupervisedClassifier] Labels: {self.labels}")

        t, y = solver.integrate(classifier_ode_system, self.initial_conditions)  # type: ignore[reportUnknownArgumentType]
        self.solution = Solution(initial_condition=self.initial_conditions, time=t, y=y)

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

    # TODO: Group initial_conditions, labels, and ode_params into a single argument
    def __init__(
        self,
        classifier: KNeighborsClassifier | None,
        initial_conditions: torch.Tensor,
        labels: list[str],
        ode_params: P,
        **kwargs: Any,
    ):
        """
        Initialize KNN classifier.

        :param classifier: KNeighborsClassifier instance, or None to create default.
        :param initial_conditions: Template initial conditions.
        :param labels: Ground truth labels.
        :param ode_params: ODE parameters.
        :param kwargs: Additional arguments for KNeighborsClassifier if classifier is None.
        """
        if classifier is None:
            classifier = KNeighborsClassifier(**kwargs)
        self.classifier = classifier
        super().__init__(initial_conditions, labels, ode_params)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using the fitted KNN classifier.

        :param features: Feature array to classify.
        :return: Predicted labels.
        """
        return self.classifier.predict(features)


class UnsupervisedClassifier[P](ClusterClassifier):
    """Base class for unsupervised clustering algorithms."""

    def __init__(self, initial_conditions: torch.Tensor, ode_params: P):
        """
        Initialize the unsupervised classifier.

        :param initial_conditions: Initial conditions to cluster.
        :param ode_params: ODE parameters.
        """
        self.initial_conditions = initial_conditions
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
