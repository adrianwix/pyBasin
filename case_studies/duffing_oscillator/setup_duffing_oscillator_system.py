import torch
from case_study_duffing_oscillator.DuffingFeatureExtractor import DuffingFeatureExtractor

# Local imports
from case_study_duffing_oscillator.DuffingODE import DuffingODE, DuffingParams

# External imports
from sklearn.neighbors import KNeighborsClassifier

from pybasin.cluster_classifier import KNNCluster
from pybasin.sampler import GridSampler
from pybasin.solver import TorchDiffEqSolver
from pybasin.types import SetupProperties


def setup_duffing_oscillator_system() -> SetupProperties:
    N = 10000

    # Create ODE system instance
    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    ode_system = DuffingODE(params)

    # Instantiate sampler, solver, feature extractor, and cluster classifier
    sampler = GridSampler(min_limits=(-1, -0.5), max_limits=(1, 1))
    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25)
    feature_extractor = DuffingFeatureExtractor(time_steady=950)

    classifier_initial_conditions = torch.tensor(
        [
            [-0.21, 0.02],
            [1.05, 0.77],
            [-0.67, 0.02],
            [-0.46, 0.30],
            [-0.43, 0.12],
        ],
        dtype=torch.float32,
    )

    classifier_labels = [
        "y1: small n=1 cycle",
        "y2: large n=1 cycle",
        "y3: first n=2 cycle",
        "y4: second n=2 cycle",
        "y5: n=3 cycle",
    ]

    knn = KNeighborsClassifier(n_neighbors=1)

    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params,
    )

    return {
        "N": N,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster,
    }
