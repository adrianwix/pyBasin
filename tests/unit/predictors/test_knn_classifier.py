from typing import TypedDict

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.predictors.knn_classifier import KNNClassifier
from pybasin.solution import Solution
from pybasin.solver import TorchOdeSolver


class SimpleParams(TypedDict):
    a: float


class SimpleODE(ODESystem[SimpleParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.params["a"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['a']} * y"


class SimpleFeatureExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        return solution.y[-1, :, :]

    @property
    def feature_names(self) -> list[str]:
        return [f"final_state_{i}" for i in range(2)]


def test_knn_classifier_fit_predict():
    params: SimpleParams = {"a": -1.0}
    ode_system = SimpleODE(params)
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)
    feature_extractor = SimpleFeatureExtractor(time_steady=0)

    template_y0 = [[1.0], [2.0]]
    labels = ["A", "B"]

    knn = KNNClassifier(
        classifier=KNeighborsClassifier(n_neighbors=1),
        template_y0=template_y0,
        labels=labels,
        ode_params=params,
    )

    knn.fit(solver, ode_system, feature_extractor)

    test_features = np.array([[0.3], [0.7]])
    predicted = knn.predict_labels(test_features)

    # Two predictions (one per test sample)
    assert len(predicted) == 2
    # Prediction is one of the trained labels
    assert predicted[0] in ["A", "B"]
