import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.matplotlib_as_plotter import ASPlotter
from pybasin.utils import time_execution  # Import the utility function


def main():
    props = setup_pendulum_system()

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=np.arange(0.01, 0.97, 0.05),
        # adaptative_parameter_values=np.arange(0.1, 1.00, 0.2),
        adaptative_parameter_name='ode_system.params["T"]',
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
        save_to="results_case2",
    )

    bse.estimate_as_bs()

    plotter = ASPlotter(bse)

    plotter.plot_basin_stability_variation()
    plotter.plot_bifurcation_diagram(dof=[1])

    bse.save()


if __name__ == "__main__":
    time_execution("main_pendulum_case2.py", main)
