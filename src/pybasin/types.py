from typing import Any, TypedDict

from pybasin.cluster_classifier import ClusterClassifier
from pybasin.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.sampler import Sampler
from pybasin.solver import Solver


class SetupProperties(TypedDict):
    """
    Standard properties returned by setup functions for case studies.

    Note: This is a flexible type definition. Actual implementations
    may use more specific types (e.g., GridSampler instead of Sampler).
    """

    n: int  # Number of samples
    ode_system: ODESystem[Any]
    sampler: Sampler
    solver: Solver
    feature_extractor: FeatureExtractor
    cluster_classifier: ClusterClassifier
