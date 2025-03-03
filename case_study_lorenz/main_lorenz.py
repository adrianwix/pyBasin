import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from ..pybasin.ClusterClassifier import KNNCluster
from ..pybasin.ODESystem import LorenzODE, LorenzParams
from ..pybasin.Sampler import UniformRandomSampler
from ..pybasin.Solver import TorchDiffEqSolver
from ..pybasin.FeatureExtractor import LorenzFeatureExtractor
from ..pybasin.utils import time_execution
import torch


def setup_lorenz_system():
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

    return {
        "N": N,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster
    }


def preview_plot_templates():
    """Preview the template trajectories before running the full analysis"""
    props = setup_lorenz_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    params = props["cluster_classifier"].ode_params

    # Only plot the 2 bounded solutions

    # Template initial conditions from MATLAB
    classifier_initial_conditions = torch.tensor([
        [0.8, -3.0, 0.0],    # butterfly1
        [-0.8, 3.0, 0.0],    # butterfly2
        # [10.0, 50.0, 0.0],   # unbounded
    ], dtype=torch.float32)

    classifier_labels = ['butterfly1', 'butterfly2']

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
        time_span=(0, 200)
    )

    bse.plot_phase(x_var=1, y_var=2)


def main():
    # Preview templates first
    preview_plot_templates()
    exit(1)

    props = setup_lorenz_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    knn_cluster = props["cluster_classifier"]

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
