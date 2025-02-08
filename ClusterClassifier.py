from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.base import ClusterMixin
from torch import tensor
from FeatureExtractor import FeatureExtractor
from ODESystem import ODESystem
from Solution import Solution
from Solver import Solver


class ClusterClassifier(ABC):
    """
    Abstract base class for clustering/classification.

    Subclasses must implement the get_labels(features) method, which takes a
    feature array (shape: (N, num_features)) and returns an array of labels.
    """

    @abstractmethod
    def get_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Compute labels for each feature vector.

        Parameters
        ----------
        features : np.ndarray
            Array of features (shape: (N, num_features)).

        Returns
        -------
        np.ndarray
            Array of labels.
        """
        pass

    def fit(self, solver: Solver, ode_system: ODESystem, feature_extractor: FeatureExtractor) -> None:
        # Generate features for each template initial condition
        template_features = []

        t, y = solver.integrate(
            ode_system, self.initial_conditions)

        # Iterate over each initial condition (batch)
        for i in range(self.initial_conditions.shape[0]):
            solution = Solution(
                initial_condition=self.initial_conditions[i],
                time=t,
                y=y[:, i, :]
            )
            features = feature_extractor.extract_features(solution)
            template_features.append(features)

        trainX = np.array(template_features)
        trainY = np.array(self.labels)

        self.classifier.fit(trainX, trainY)


class KNNCluster(ClusterClassifier):
    """
    A supervised clustering/classification class using k-Nearest Neighbors.

    The user must pass an instance of KNeighborsClassifier along with training data:
    - trainX: A NumPy array of shape (n_samples, n_features) containing the training features.
    - trainY: A NumPy array of shape (n_samples,) containing the corresponding string labels.

    In our example, trainY should contain the strings "Fixed Point" and "Limit Cycle".
    """

    def __init__(self, classifier: KNeighborsClassifier, initial_conditions: tensor, labels: np.ndarray):
        """
        Initialize the KNNCluster.

        Parameters
        ----------
        classifier : KNeighborsClassifier
            A scikit-learn k-NN classifier (e.g., with n_neighbors=1).
        trainX : np.ndarray
            Training features (shape: (n_samples, n_features)).
        trainY : np.ndarray
            Training labels (shape: (n_samples,)). Expected labels are strings.
        """
        self.classifier = classifier
        self.initial_conditions = initial_conditions
        self.labels = labels

    def get_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given features using the trained k-NN classifier.

        Parameters
        ----------
        features : np.ndarray
            Array of features (shape: (N, num_features)).

        Returns
        -------
        np.ndarray
            Array of string labels.
        """
        return self.classifier.predict(features)


class DBSCANCluster(ClusterClassifier):
    """
    Unsupervised clustering using DBSCAN (or any other clustering classifier).

    When instantiated, the user can either provide an already-configured classifier
    (which must implement the fit_predict method) or pass desired options via keyword
    arguments to create a DBSCAN instance.
    """

    def __init__(self, classifier: Optional[ClusterMixin] = None, **kwargs):
        """
        Initialize the DBSCANCluster.

        Parameters
        ----------
        classifier : Optional[ClusterMixin]
            An instance of a clustering classifier that implements fit_predict.
            If None, a new DBSCAN instance is created using the provided kwargs.
        **kwargs :
            Keyword arguments to be passed to DBSCAN if classifier is None.
        """
        if classifier is None:
            classifier = DBSCAN(**kwargs)
        self.classifier = classifier
        self.classifier.fit()

    def get_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using the classifier's fit_predict method.

        Parameters
        ----------
        features : np.ndarray
            Array of features (shape: (N, num_features)).

        Returns
        -------
        np.ndarray
            Array of integer labels. Note that DBSCAN may label noise points as -1.
        """
        return self.classifier.predict(features)
