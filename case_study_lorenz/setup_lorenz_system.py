from sklearn.neighbors import KNeighborsClassifier
from pybasin.types import SetupProperties
from pybasin.ClusterClassifier import KNNCluster
from pybasin.ODESystem import LorenzODE, LorenzParams
from pybasin.Sampler import UniformRandomSampler
from pybasin.Solver import TorchDiffEqSolver
from pybasin.FeatureExtractor import LorenzFeatureExtractor
import torch


def setup_lorenz_system() -> SetupProperties:
    N = 1 * 10**4

    # Parameters for broken butterfly system
    params: LorenzParams = {
        "sigma": 0.12,
        "r": 0.0,
        "b": -0.6
    }

    ode_system = LorenzODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0],
        max_limits=[10.0, 20.0, 0.0]
    )

    solver = TorchDiffEqSolver(
        time_span=(0, 1000),
        fs=25
    )

    feature_extractor = LorenzFeatureExtractor(time_steady=900)

    classifier_initial_conditions = torch.tensor([
        [0.8, -3.0, 0.0],
        [-0.8, 3.0, 0.0],
        [10.0, 50.0, 0.0],
    ], dtype=torch.float32)

    classifier_labels = ['butterfly1', 'butterfly2', 'unbounded']

    knn = KNeighborsClassifier(n_neighbors=1)

    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params
    )

    return {
        "N": N,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster
    }
