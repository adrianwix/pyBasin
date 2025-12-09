"""Hyperparameter sensitivity study for the pendulum case.

This script varies the number of sampling points (N) to study the
sensitivity of basin stability values against this hyperparameter.
Based on the MATLAB bSTAB implementation.
"""

import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.matplotlib_as_plotter import ASPlotter
from pybasin.utils import time_execution


def main():
    """Run hyperparameter sensitivity study for pendulum system."""
    props = setup_pendulum_system()

    # Hyperparameter study: vary N (number of sampling points)
    # Using log-spaced values from 50 to 5000 (20 points)
    # Equivalent to MATLAB: 5*logspace(1, 3, 20)
    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=5 * np.logspace(1, 3, 20),
        adaptative_parameter_name="n",  # Varying the number of samples
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
        n=props["n"],  # Initial value, will be overridden by adaptive study
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        as_params=as_params,
        save_to="results_hyperparameters",
    )

    bse.estimate_as_bs()

    plotter = ASPlotter(bse)

    # Plot basin stability variation against hyperparameter
    plotter.plot_basin_stability_variation()

    bse.save()


if __name__ == "__main__":
    time_execution("main_pendulum_hyperparameters.py", main)
