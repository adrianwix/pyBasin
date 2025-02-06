import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from BasinStabilityEstimator import BasinStabilityEstimator
from ClusterClassifier import KNNCluster
from ODESystem import PendulumODE, PendulumParams
from Sampler import RandomSampler
from Solver import SciPySolver

from FeatureExtractor import PendulumFeatureExtractor, PendulumOHE

if __name__ == "__main__":
    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 1000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits= (-np.pi + np.arcsin(params["T"] / params["K"]), -10.0), 
        max_limits= (np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Define training data.
    trainX = np.array([PendulumOHE["FP"], PendulumOHE["LC"]])
    trainY = np.array(["Fixed Point", "Limit Cycle"])

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Instantiate the KNNCluster with the training data.
    knn_cluster = KNNCluster(classifier=knn, trainX=trainX, trainY=trainY)

    bse = BasinStabilityEstimator(
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

    # Disabled since result file is too big
    # bse.save("basin_stability_results.json")
