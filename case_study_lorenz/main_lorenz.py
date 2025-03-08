import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from case_study_lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from pybasin.ClusterClassifier import KNNCluster
from pybasin.Plotter import Plotter
from pybasin.utils import time_execution
import torch


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

    plt = Plotter(bse=bse)

    plt.plot_bse_results()


def preview_plot_templates():
    """Preview the template trajectories before running the full analysis"""
    props = setup_lorenz_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    params = props["cluster_classifier"].ode_params

    # Here we override the setup_lorenz because we only want to plot the 2 bounded solutions

    classifier_initial_conditions = torch.tensor([
        [0.8, -3.0, 0.0],    # butterfly1
        [-0.8, 3.0, 0.0],    # butterfly2
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
        cluster_classifier=knn_cluster,
        save_to="results"
    )

    plt = Plotter(bse=bse)

    print("Generating template trajectories...")

    # Plot the first state variable (x) trajectories
    plt.plot_templates(
        plotted_var=1,
        time_span=(0, 200)
    )

    plt.plot_phase(x_var=1, y_var=2)


if __name__ == "__main__":
    time_execution("main_lorenz.py", main)
