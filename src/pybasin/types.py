from typing import NotRequired, TypedDict

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.predictors.base import LabelPredictor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler


class ErrorInfo(TypedDict):
    e_abs: float
    e_rel: float


class SetupProperties(TypedDict):
    """
    Standard properties returned by setup functions for case studies.

    Note: This is a flexible type definition. Actual implementations
    may use more specific types (e.g., GridSampler instead of Sampler).
    """

    n: int
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: NotRequired[SolverProtocol]
    feature_extractor: NotRequired[FeatureExtractor]
    cluster_classifier: NotRequired[LabelPredictor]
