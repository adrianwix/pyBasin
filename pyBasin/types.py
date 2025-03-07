from typing import TypedDict
from pybasin.ODESystem import LorenzODE
from pybasin.ClusterClassifier import KNNCluster
from pybasin.FeatureExtractor import LorenzFeatureExtractor
from pybasin.Sampler import UniformRandomSampler
from pybasin.Solver import TorchDiffEqSolver


class SetupProperties(TypedDict):
    N: int
    ode_system: LorenzODE
    sampler: UniformRandomSampler
    solver: TorchDiffEqSolver
    feature_extractor: LorenzFeatureExtractor
    cluster_classifier: KNNCluster
