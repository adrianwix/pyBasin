import torch
from setup_lorenz_system import setup_lorenz_system
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import KNNCluster
from pybasin.plotter import Plotter
from pybasin.utils import time_execution


def main():
    # Preview templates first
    # preview_plot_templates()

    props = setup_lorenz_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_case1",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    plotter = Plotter(bse=bse)

    plotter.plot_bse_results()
    plotter.plot_phase(x_var=1, y_var=2)

    bse.save()


# TODO: Having example trajectories and previewing them before running the full analysis
# seem like a useful feature to have in other case studies as well, and part of the library
def preview_plot_templates():
    """Preview the template trajectories before running the full analysis"""
    props = setup_lorenz_system()
    n = props["n"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    params = props["cluster_classifier"].ode_params # pyright: ignore[reportAttributeAccessIssue]

    # Here we override the setup_lorenz because we only want to plot the 2 bounded solutions

    classifier_initial_conditions = torch.tensor(
        [
            [0.8, -3.0, 0.0],  # butterfly1
            [-0.8, 3.0, 0.0],  # butterfly2
        ],
        dtype=torch.float32,
    )
    classifier_labels = ["butterfly1", "butterfly2"]

    # Create a KNeighborsClassifier with k=1
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params,
    )

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster,
        save_to="results",
    )

    plotter = Plotter(bse=bse)

    print("Generating template trajectories...")

    # Plot the first state variable (x) trajectories
    plotter.plot_templates(plotted_var=1, time_span=(0, 200))

    plotter.plot_phase(x_var=1, y_var=2)


if __name__ == "__main__":
    time_execution("main_lorenz.py", main)
