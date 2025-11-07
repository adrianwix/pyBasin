from typing import TypedDict

from pybasin.cluster_classifier import KNNCluster
from pybasin.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.sampler import UniformRandomSampler
from pybasin.solver import TorchDiffEqSolver


class SetupProperties(TypedDict):
    N: int
    ode_system: ODESystem
    sampler: UniformRandomSampler
    solver: TorchDiffEqSolver
    feature_extractor: FeatureExtractor
    cluster_classifier: KNNCluster
