from typing import TypedDict

from pybasin.cluster_classifier import ClusterClassifier
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler


class SetupProperties(TypedDict):
    """
    Standard properties returned by setup functions for case studies.

    Note: This is a flexible type definition. Actual implementations
    may use more specific types (e.g., GridSampler instead of Sampler).
    """

    n: int  # Number of samples
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: SolverProtocol
    feature_extractor: FeatureExtractor
    cluster_classifier: ClusterClassifier
