from typing import TypedDict
from pybasin.ode_system import ODESystem
from pybasin.ClusterClassifier import KNNCluster
from pybasin.FeatureExtractor import FeatureExtractor
from pybasin.Sampler import UniformRandomSampler
from pybasin.Solver import TorchDiffEqSolver


class SetupProperties(TypedDict):
    N: int
    ode_system: ODESystem
    sampler: UniformRandomSampler
    solver: TorchDiffEqSolver
    feature_extractor: FeatureExtractor
    cluster_classifier: KNNCluster
