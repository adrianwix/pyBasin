import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.lorenz.lorenz_feature_extractor import LorenzFeatureExtractor
from case_studies.lorenz.lorenz_ode import LorenzODE, LorenzParams
from pybasin.cluster_classifier import KNNCluster
from pybasin.sampler import UniformRandomSampler
from pybasin.solver import TorchOdeSolver
from pybasin.types import SetupProperties


def setup_lorenz_system() -> SetupProperties:
    n = 1 * 10**4

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    # Parameters for broken butterfly system
    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}

    ode_system = LorenzODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0], max_limits=[10.0, 20.0, 0.0], device=device
    )

    solver = TorchOdeSolver(time_span=(0, 1000), n_steps=1000, device=device)

    feature_extractor = LorenzFeatureExtractor(time_steady=900)

    classifier_initial_conditions = torch.tensor(
        [
            [0.8, -3.0, 0.0],
            [-0.8, 3.0, 0.0],
            [10.0, 50.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    classifier_labels = ["butterfly1", "butterfly2", "unbounded"]

    knn = KNeighborsClassifier(n_neighbors=1)

    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
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
