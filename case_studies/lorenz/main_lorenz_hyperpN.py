import numpy as np

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.as_plotter import ASPlotter


def main():
    props = setup_lorenz_system()

    as_params = AdaptiveStudyParams(
        # adaptative_parameter_values=np.arange(0.01, 1.05, 0.05),
        adaptative_parameter_values=2 * np.logspace(2, 4, 50, dtype=np.int64),
        adaptative_parameter_name="N",
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None, "solver is required for ASBasinStabilityEstimator"
    assert feature_extractor is not None, (
        "feature_extractor is required for ASBasinStabilityEstimator"
    )
    assert cluster_classifier is not None, (
        "cluster_classifier is required for ASBasinStabilityEstimator"
    )

    bse = ASBasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        as_params=as_params,
        save_to="results_hyperpN",
    )

    print("Estimating Basin Stability...")
    bse.estimate_as_bs()

    plotter = ASPlotter(bse)

    plotter.plot_basin_stability_variation(interval="log")

    bse.save()


if __name__ == "__main__":
    # time_execution("main_lorenz.py", main)
    main()
