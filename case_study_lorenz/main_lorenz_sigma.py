import numpy as np
from case_study_lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams
from pybasin.ASPlotter import ASPlotter


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
        as_params=as_params,
        save_to='results_sigma'
    )

    print("Estimating Basin Stability...")
    bse.estimate_as_bs()

    plotter = ASPlotter(bse)

    # plotter.plot_basin_stability_variation()

    plotter.plot_bifurcation_diagram(dof=[0, 1, 2])

    bse.save()


if __name__ == "__main__":
    # time_execution("main_lorenz.py", main)
    main()
