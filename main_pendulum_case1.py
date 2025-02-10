import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from BasinStabilityEstimator import BasinStabilityEstimator
from ClusterClassifier import KNNCluster
from ODESystem import PendulumODE, PendulumParams
from Sampler import RandomSampler
from Solver import TorchDiffEqSolver
from FeatureExtractor import PendulumFeatureExtractor
from utils import time_execution
import torch


def preview_plot_templates():
    N = 10000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.2, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Template initial conditions from MATLAB code
    classifier_initial_conditions = torch.tensor([
        [0.5, 0.0],    # FP: stable fixed point
        [0, 6],        # LC: limit cycle
    ], dtype=torch.float32)
    classifier_labels = ['FP', 'LC']  # Original MATLAB labels

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels)

    bse = BasinStabilityEstimator(
        name="pendulum_case1_templates",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    bse.plot_templates(
        plotted_var=1,
        time_span=(0, 200))


def main():
    # We can test and visualize the templates before running the Basin Stability Estimator
    preview_plot_templates()
    exit(1)

    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 10000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.2, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = TorchDiffEqSolver(time_span=(0, 1000), fs=25)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Template initial conditions from MATLAB code
    classifier_initial_conditions = torch.tensor([
        [0.5, 0.0],    # FP: stable fixed point
        [2.7, 0.0],    # LC: limit cycle
    ], dtype=torch.float32)
    classifier_labels = ['FP', 'LC']  # Original MATLAB labels

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels)

    bse = BasinStabilityEstimator(
        name="pendulum_case1",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    # bse.plot_templates(
    #     plotted_var=1,
    #     time_span=(0, 50 * np.pi))

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    bse.plots()

    # Disabled since result file is too big
    # bse.save("basin_stability_results.json")


if __name__ == "__main__":
    time_execution("main_pendulum_case1.py", main)
