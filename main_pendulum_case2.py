from copy import deepcopy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams
from ClusterClassifier import KNNCluster
from ODESystem import PendulumODE, PendulumParams
from Sampler import UniformRandomSampler
from Solver import TorchDiffEqSolver
from FeatureExtractor import PendulumFeatureExtractor
from utils import time_execution  # Import the utility function


def main():
    N = 10000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = UniformRandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    # 25 Hz sampling frequency
    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Template initial conditions from MATLAB code
    classifier_initial_conditions = torch.tensor([
        [0.5, 0.0],    # FP: stable fixed point
        [2.7, 0.0],    # LC: limit cycle
    ], dtype=torch.float32)
    # Also define the params for this conditions since they vary during AP study => Labels are not the same
    classifier_labels = ['FP', 'LC']  # Original MATLAB labels

    classifier_ode_params = params

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=classifier_ode_params
    )

    as_params = AdaptiveStudyParams(
        # adaptative_parameter_values=np.arange(0.01, 1.05, 0.05),
        adaptative_parameter_values=np.arange(0.21, 0.55, 0.05),
        adaptative_parameter_name='ode_system.params["T"]')

    bse = ASBasinStabilityEstimator(
        name="pendulum_case2",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster,
        as_params=as_params
    )

    param_values, basin_stabilities = bse.estimate_as_bs()

    # Improved readability for the print statement
    for i in range(len(param_values)):
        print(
            f"Parameter value: {param_values[i]}, Basin Stability: {basin_stabilities[i]}")

    bse.plot_basin_stability_variation()

    # Disabled since result file is too big
    # bse.save("basin_stability_results.json")


if __name__ == "__main__":
    time_execution("main_pendulum_case2.py", main)
