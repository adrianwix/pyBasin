import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams
from ClusterClassifier import KNNCluster
from ODESystem import PendulumODE, PendulumParams
from Sampler import RandomSampler
from Solver import SciPySolver
from FeatureExtractor import PendulumFeatureExtractor
from utils import time_execution  # Import the utility function

def main():
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

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=np.arange(0.01, 1.05, 0.3),
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

    print({param_values[i]: basin_stabilities[i]
          for i in range(len(param_values))})

    bse.plot_basin_stability_variation()

    # Disabled since result file is too big
    # bse.save("basin_stability_results.json")

if __name__ == "__main__":
    time_execution("main_pendulum_case2.py", main)
