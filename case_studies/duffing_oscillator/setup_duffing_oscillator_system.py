import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.duffing_oscillator.duffing_feature_extractor import DuffingFeatureExtractor
from case_studies.duffing_oscillator.duffing_ode import DuffingODE, DuffingParams
from pybasin.cluster_classifier import KNNCluster
from pybasin.sampler import GridSampler
from pybasin.solver import TorchDiffEqSolver
from pybasin.types import SetupProperties


def setup_duffing_oscillator_system() -> SetupProperties:
    n = 10000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Duffing oscillator system on device: {device}")

    # Create ODE system instance
    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    ode_system = DuffingODE(params)

    # Instantiate sampler, solver, feature extractor, and cluster classifier
    sampler = GridSampler(min_limits=[-1, -0.5], max_limits=[1, 1], device=device)
    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25, device=device)
    feature_extractor = DuffingFeatureExtractor(time_steady=950)

    classifier_initial_conditions = [
        [-0.21, 0.02],
        [1.05, 0.77],
        [-0.67, 0.02],
        [-0.46, 0.30],
        [-0.43, 0.12],
    ]

    classifier_labels = [
        "y1",
        "y2",
        "y3",
        "y4",
        "y5",
    ]

    knn = KNeighborsClassifier(n_neighbors=1)

    knn_cluster = KNNCluster(
        classifier=knn,
        template_y0=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params,
    )

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster,
    }
