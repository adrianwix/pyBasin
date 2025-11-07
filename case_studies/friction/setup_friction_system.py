import torch
from case_friction.FrictionFeatureExtractor import FrictionFeatureExtractor
from case_friction.FrictionODE import FrictionODE, FrictionParams
from sklearn.neighbors import KNeighborsClassifier

from pybasin.cluster_classifier import KNNCluster
from pybasin.sampler import UniformRandomSampler
from pybasin.solver import TorchDiffEqSolver
from pybasin.types import SetupProperties


def setup_friction_system() -> SetupProperties:
    N = 1 * 10**3  # Number of samples as in setup_friction.m

    # Parameters from setup_friction.m
    params: FrictionParams = {
        "v_d": 1.5,  # Driving velocity
        "xi": 0.05,  # Damping ratio
        "musd": 2.0,  # Ratio static to dynamic friction coefficient
        "mud": 0.5,  # Dynamic coefficient of friction
        "muv": 0.0,  # Linear strengthening parameter
        "v0": 0.5,  # Reference velocity for exponential decay
    }

    ode_system = FrictionODE(params)

    # Sampling limits from setup_friction.m
    # sampler = UniformRandomSampler(
    #     min_limits=[-2.0, 0.0],    # [disp, vel]
    #     max_limits=[2.0, 2.0]      # [disp, vel]
    # )

    sampler = UniformRandomSampler(
        min_limits=[0.5, -2.0],  # [disp, vel]
        max_limits=[2.5, 0.0],  # [disp, vel]
    )

    # Time integration parameters from setup_friction.m
    solver = TorchDiffEqSolver(
        time_span=(0, 500),  # tSpan
        fs=100,  # Sampling frequency
    )

    # Feature extraction (using last 100 time units as in setup_friction.m)
    feature_extractor = FrictionFeatureExtractor(time_steady=400)

    # Template initial conditions for classification from setup_friction.m
    classifier_initial_conditions = torch.tensor(
        [
            [0.1, 0.1],  # Fixed point (FP)
            [2.0, 2.0],  # Limit cycle (LC)
        ],
        dtype=torch.float64,
    )

    classifier_labels = ["FP", "LC"]

    # KNN classifier as specified in setup_friction.m
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
