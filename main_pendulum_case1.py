import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from BasinStabilityEstimator import BasinStabilityEstimator
from ClusterClassifier import KNNCluster
from ODESystem import PendulumODE, PendulumParams
from Sampler import RandomSampler
from Solver import SciPySolver
from FeatureExtractor import PendulumFeatureExtractor


def preview_plot_templates():
    N = 1000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Template initial conditions from MATLAB code
    golden_ratio = (1 + np.sqrt(5)) / 2
    epsilom = 0.1
    classifier_initial_conditions = [
        [-epsilom, epsilom],    # FP: stable fixed point
        [golden_ratio - np.pi + epsilom, epsilom],    # LC: limit cycle
    ]
    classifier_labels = ['FP', 'LC']  # Original MATLAB labels

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(
        classifier=knn,
        initial_conditions=classifier_initial_conditions,
        labels=classifier_labels)

    bse = BasinStabilityEstimator(
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    bse.plot_templates(
        plotted_var=1,
        time_span=(0, 50 * np.pi),
        y_lims=[(-0.6, 0.6), (-1, params["T"] / params["alpha"] + 1)])


if __name__ == "__main__":
    # We can test and visualize the templates before running the Basin Stability Estimator
    # preview_plot_templates()

    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 1000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits=(-np.pi + np.arcsin(params["T"] / params["K"]), -10.0),
        max_limits=(np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Template initial conditions from MATLAB code
    classifier_initial_conditions = [
        [0.5, 0.0],    # FP: stable fixed point
        [2.7, 0.0],    # LC: limit cycle
    ]
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
