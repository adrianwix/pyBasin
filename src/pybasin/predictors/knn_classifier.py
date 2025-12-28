from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from pybasin.predictors.base import ClassifierPredictor
from pybasin.protocols import SolverProtocol


class KNNClassifier(ClassifierPredictor):
    """K-Nearest Neighbors classifier for basin stability analysis (supervised learning)."""

    display_name: str = "KNN Classifier"

    def __init__(
        self,
        classifier: KNeighborsClassifier | None,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: Mapping[str, Any],
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
