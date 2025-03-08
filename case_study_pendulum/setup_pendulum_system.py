from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from pybasin.ClusterClassifier import KNNCluster
from pybasin.Plotter import Plotter
from pybasin.Sampler import UniformRandomSampler
from pybasin.Solver import TorchDiffEqSolver
from pybasin.types import SetupProperties
from pybasin.utils import time_execution

# Case Study classes
from PendulumODE import PendulumODE, PendulumParams
from PendulumFeatureExtractor import PendulumFeatureExtractor

# Third Parties
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch


def setup_pendulum_system() -> SetupProperties:
    N = 1000

    # Define the parameters of the pendulum
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    # Instantiate ODE system for the pendulum.
    ode_system = PendulumODE(params)

    # Define sampling limits based on the pendulum parameters.
    # Here the angular limits for theta are adjusted using arcsin(T/K).
    sampler = UniformRandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]), 10.0)
    )

    # Create the solver with specified integration time and frequency.
    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25)

    # Instantiate the feature extractor with a steady state time.
    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Define template initial conditions and labels (e.g., for Fixed Point and Limit Cycle).
    classifier_initial_conditions = torch.tensor([
        [0.5, 0.0],    # FP: fixed point
        [2.7, 0.0]     # LC: limit cycle
    ], dtype=torch.float32)
    classifier_labels = ['FP', 'LC']

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
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
