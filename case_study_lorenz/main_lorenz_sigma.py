import numpy as np
from case_study_lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams


def main():
    props = setup_lorenz_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    knn_cluster = props["cluster_classifier"]

    as_params = AdaptiveStudyParams(
        # adaptative_parameter_values=np.arange(0.01, 1.05, 0.05),
        adaptative_parameter_values=np.arange(0.12, 0.1825, 0.0025),
        adaptative_parameter_name='ode_system.params["sigma"]')

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

    print("Estimating Basin Stability...")
    param_values, basin_stabilities = bse.estimate_as_bs()

    # Improved readability for the print statement
    for i in range(len(param_values)):
        print(
            f"Parameter value: {param_values[i]}, Basin Stability: {basin_stabilities[i]}")

    bse.plot_basin_stability_variation()

    bse.plots()


if __name__ == "__main__":
    # time_execution("main_lorenz.py", main)
    main()
