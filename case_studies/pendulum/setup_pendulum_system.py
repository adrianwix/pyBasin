import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# from tsfresh.feature_extraction import (  # pyright: ignore[reportMissingTypeStubs]
#     MinimalFCParameters,
# )
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.feature_extractors.jax_feature_extractor import JaxFeatureExtractor
from pybasin.predictors.knn_classifier import KNNClassifier
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
        use_cache=True,
    )

    # Instantiate the feature extractor with a steady state time.
    # For the pendulum, we use the "log_delta" feature on velocity (state 1).
    # Log transformation of delta makes the exponential range more linear:
    #   FP: log(delta) ~ -11 to -4 (delta ~ 1e-5 to 1e-2)
    #   LC: log(delta) ~ -4 to 2+  (delta ~ 1e-2 to 10+)
    # This provides better KNN separation across different T parameters.
    feature_extractor = JaxFeatureExtractor(
        time_steady=950.0,
        features=None,
        features_per_state={
            1: {"log_delta": None},
        },
        normalize=False,
    )

    # Define template initial conditions and labels (e.g., for Fixed Point and Limit Cycle).
    # Templates are defined as plain Python lists - the classifier will convert them
    # to tensors on the appropriate device automatically.
    template_y0 = [
        [0.5, 0.0],  # FP: fixed point
        [2.7, 0.0],  # LC: limit cycle
    ]
    classifier_labels = ["FP", "LC"]

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNClassifier with the training data.
    knn_cluster = KNNClassifier(
        classifier=knn,
        template_y0=template_y0,
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
