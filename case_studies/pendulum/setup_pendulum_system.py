import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# from tsfresh.feature_extraction import (  # pyright: ignore[reportMissingTypeStubs]
#     MinimalFCParameters,
# )
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.cluster_classifier import KNNCluster
from pybasin.jax_feature_extractor import JaxFeatureExtractor
from pybasin.sampler import GridSampler
from pybasin.solvers import JaxSolver

# from pybasin.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.types import SetupProperties


def setup_pendulum_system() -> SetupProperties:
    n = 10000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up pendulum system on device: {device}")

    # Define the parameters of the pendulum
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    # Instantiate JAX ODE system for the pendulum (high-performance).
    ode_system = PendulumJaxODE(params)

    # Define sampling limits based on the pendulum parameters.
    # Here the angular limits for theta are adjusted using arcsin(T/K).
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )

    solver = JaxSolver(
        time_span=(0, 1000),
        n_steps=1000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        use_cache=False,
    )

    # Instantiate the feature extractor with a steady state time.
    # feature_extractor = TsfreshFeatureExtractor(
    #     time_steady=950.0,  # Same as PendulumFeatureExtractor
    #     default_fc_parameters=MinimalFCParameters(),
    #     n_jobs=1,  # Use n_jobs=1 for deterministic results (n_jobs>1 causes non-determinism)
    #     normalize=False,  # Don't normalize - causes issues with KNN when templates differ from main data
    # )
    feature_extractor = JaxFeatureExtractor(time_steady=950.0, normalize=False)

    # Define template initial conditions and labels (e.g., for Fixed Point and Limit Cycle).
    classifier_initial_conditions = torch.tensor(
        [
            [0.5, 0.0],  # FP: fixed point
            [2.7, 0.0],  # LC: limit cycle
        ],
        dtype=torch.float32,
        device=device,
    )
    classifier_labels = ["FP", "LC"]

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
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
