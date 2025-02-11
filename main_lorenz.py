import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from BasinStabilityEstimator import BasinStabilityEstimator
from ClusterClassifier import KNNCluster
from ODESystem import LorenzODE, LorenzParams
from Sampler import UniformRandomSampler
from Solver import TorchDiffEqSolver
from FeatureExtractor import LorenzFeatureExtractor
from utils import time_execution
import torch


def preview_plot_templates():
    """Preview the template trajectories before running the full analysis"""
    N = 1000  # Following MATLAB example

    # Parameters for broken butterfly system
    params: LorenzParams = {
        "sigma": 0.12,  # Prandtl number
        "r": 0.0,       # Rayleigh number
        "b": -0.6       # Physical dimension parameter
    }

    ode_system = LorenzODE(params)

    # Region of Interest settings from MATLAB
    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0],  # Following MATLAB ROI
        max_limits=[10.0, 20.0, 0.0]
    )

    # Time integration settings from MATLAB
    solver = TorchDiffEqSolver(
        time_span=(0, 1000),  # tSpan from MATLAB
        fs=25  # Sampling frequency from MATLAB
    )

    # Steady state time from MATLAB (tStar = tSpan(end)-100)
    feature_extractor = LorenzFeatureExtractor(time_steady=900)

    # Template initial conditions from MATLAB
    classifier_initial_conditions = torch.tensor([
        [0.8, -3.0, 0.0],    # butterfly1
        [-0.8, 3.0, 0.0],    # butterfly2
        [10.0, 50.0, 0.0],   # unbounded
    ], dtype=torch.float32)

    classifier_labels = ['butterfly1', 'butterfly2', 'unbounded']

    # Create a KNeighborsClassifier with k=1
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params
    )

    bse = BasinStabilityEstimator(
        name="lorenz_templates",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    print("Generating template trajectories...")

    # Plot the first state variable (x) trajectories
    bse.plot_templates(
        plotted_var=1,
        time_span=(0, 1000)
    )


# def lorenz_termination_event(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """Stops integration if any state exceeds magnitude of 200."""
#     return 200.0 - torch.max(torch.abs(y))


def main():
    # Preview templates first
    # preview_plot_templates()
    # exit(1)

    N = 10000  # Following MATLAB example

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

    bse = BasinStabilityEstimator(
        name="lorenz_case1",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    bse.plots()


if __name__ == "__main__":
    time_execution("main_lorenz.py", main)
