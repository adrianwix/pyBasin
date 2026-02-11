from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DBSCANClusterer
from pybasin.utils import time_execution


def main():
    setup = setup_duffing_oscillator_system()

    estimator = DBSCANClusterer(auto_tune=True, assign_noise=True)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=estimator,
        save_to="results_unsupervised",
        feature_selector=None,
    )

    bse.estimate_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_duffing_oscillator_unsupervised.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "duffing"
        / "main_duffing_unsupervised.json"
    )

    if bse.bs_vals is not None:
        errors = bse.get_errors()
        compare_with_expected_by_size(bse.bs_vals, expected_file, errors)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
