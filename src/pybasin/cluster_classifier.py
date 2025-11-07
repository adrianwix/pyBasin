from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from torch import tensor

from pybasin.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.solution import Solution
from pybasin.solver import Solver

# TODO: Fix namings


class ClusterClassifier(ABC):
    @abstractmethod
    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        pass


class SupervisedClassifier(ClusterClassifier):
    labels: np.ndarray
    initial_conditions: tensor

    def __init__(self, initial_conditions: tensor, labels: np.ndarray, ode_params: dict):
        self.labels = labels
        self.initial_conditions = initial_conditions
        self.ode_params = deepcopy(ode_params)

    def fit(
        self, solver: Solver, ode_system: ODESystem, feature_extractor: FeatureExtractor
    ) -> None:
        # Generate features for each template initial condition
        classifier_ode_system = deepcopy(ode_system)
        classifier_ode_system.params = self.ode_params

        print("classifier.fit - ode_system.params: ", classifier_ode_system.params)
        print("classifier.fit - self.initial_conditions: ", self.initial_conditions)
        print("classifier.fit - self.labels: ", self.labels)

        t, y = solver.integrate(classifier_ode_system, self.initial_conditions)

        self.solution = Solution(initial_condition=self.initial_conditions, time=t, y=y)

        # Build the features array from the Solution instances.
        features = feature_extractor.extract_features(self.solution)

        trainX = features.detach().cpu().numpy()
        trainY = self.labels

        print(trainX, trainY)

        self.classifier.fit(trainX, trainY)


class KNNCluster(SupervisedClassifier):
    # TODO: Group initiial_conditions, labels, and ode_params into a single argument
    def __init__(
        self,
        classifier: KNeighborsClassifier,
        initial_conditions: tensor,
        labels: np.ndarray,
        ode_params: dict,
    ):
        if classifier is None:
            classifier = KNeighborsClassifier(**kwargs)
        self.classifier = classifier
        super().__init__(initial_conditions, labels, ode_params)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)


class UnsupervisedClassifier(ClusterClassifier):
    def __init__(self, initial_conditions: tensor, ode_params: dict):
        super().__init__(initial_conditions, ode_params)


class DBSCANCluster(UnsupervisedClassifier):
    def __init__(self, classifier: DBSCAN = None, **kwargs):
        if classifier is None:
            classifier = DBSCAN(**kwargs)
        self.classifier = classifier

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.fit_predict(features)
