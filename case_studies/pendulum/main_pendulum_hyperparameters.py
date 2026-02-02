"""Hyperparameter sensitivity study for the pendulum case.

This script varies the number of sampling points (N) to study the
sensitivity of basin stability values against this hyperparameter.
Based on the MATLAB bSTAB implementation.
"""

import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_as_plotter import ASPlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution


def main():
    """Run hyperparameter sensitivity study for pendulum system."""
    props = setup_pendulum_system()

    study_params = SweepStudyParams(
        name="n",
        values=list(5 * np.logspace(1, 3, 20)),
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert cluster_classifier is not None, "cluster_classifier is required for BasinStabilityStudy"

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        study_params=study_params,
        save_to="results_hyperparameters",
    )

    bse.estimate_as_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_hyperparameters.py", main)

    plotter = ASPlotter(bse)

    plotter.plot_basin_stability_variation()

    bse.save()
