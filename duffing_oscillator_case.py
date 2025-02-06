import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from BasinStabilityEstimator import BasinStabilityEstimator
from ClusterClassifier import KNNCluster
from ODESystem import DuffingODE, DuffingParams, PendulumODE, PendulumParams
from Sampler import RandomSampler, GridSampler
from Solution import Solution
from Solver import SciPySolver

from FeatureExtractor import DuffingFeatureExtractor, PendulumFeatureExtractor, PendulumOHE

if __name__ == "__main__":
    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 5000

    # Instantiate your ODE system, sampler, and solver: delta = 0.08, k3 = 1, A = 0.2
    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}

    ode_system = DuffingODE(params)  # for example

    sampler = GridSampler(
        min_limits= (-1, -0.5), 
        max_limits= (1, 1))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    feature_extractor = DuffingFeatureExtractor(time_steady=950)

    # Template initial conditions
    classifier_initial_conditions = [
        [-0.21, 0.02],   # y1: small n=1 cycle
        [1.05, 0.77],    # y2: large n=1 cycle
        [-0.67, 0.02],   # y3: first n=2 cycle
        [-0.46, 0.30],   # y4: second n=2 cycle
        [-0.43, 0.12],   # y5: n=3 cycle
    ]
    classifier_labels = [
        'y1: small n=1 cycle', 
        'y2: large n=1 cycle', 
        'y3: first n=2 cycle', 
        'y4: second n=2 cycle', 
        'y5: n=3 cycle']

    # Create a KNeighborsClassifier with k=1
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(
        classifier=knn, 
        initial_conditions=classifier_initial_conditions, 
        labels=classifier_labels)
    
    bse = BasinStabilityEstimator(
        name="duffing_oscillator",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster
    )

    bse.plot_templates(plotted_var=1, time_span=(0, 50), y_lims=(-1, 1))

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    bse.plots()

    # Disabled since result file is too big
    # bse.save("duffing_oscillator_case_results.json")
