import numpy as np

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.as_plotter import ASPlotter


def main():
    props = setup_lorenz_system()

    as_params = AdaptiveStudyParams(
        # adaptative_parameter_values=np.arange(0.01, 1.05, 0.05),
        adaptative_parameter_values=np.arange(0.12, 0.1825, 0.0025),
        adaptative_parameter_name='ode_system.params["sigma"]',
    )

    bse = ASBasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        as_params=as_params,
        save_to="results_sigma",
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
