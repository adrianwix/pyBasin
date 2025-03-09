import numpy as np
from case_study_pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams
from pybasin.utils import time_execution  # Import the utility function


def main():
    props = setup_pendulum_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    knn_cluster = props["cluster_classifier"]

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=np.arange(0.01, 1.05, 0.05),
        adaptative_parameter_name='ode_system.params["T"]')

    bse = ASBasinStabilityEstimator(
        name="pendulum_case2",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster,
        as_params=as_params,
        save_to="results_case2"
    )

    param_values, basin_stabilities = bse.estimate_as_bs()

    # Improved readability for the print statement
    for i in range(len(param_values)):
        print(
            f"Parameter value: {param_values[i]}, Basin Stability: {basin_stabilities[i]}")

    bse.plot_basin_stability_variation()

    bse.save()


if __name__ == "__main__":
    time_execution("main_pendulum_case2.py", main)
