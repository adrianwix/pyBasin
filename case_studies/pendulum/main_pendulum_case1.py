from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("cluster_classifier"),
        save_to="results_case1",
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()
    # bse.save_to_excel()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case1.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "pendulum"
        / "main_pendulum_case1.json"
    )

    if bse.bs_vals is not None:
        errors = bse.get_errors()
        compare_with_expected_by_size(bse.bs_vals, expected_file, errors)

    plotter = InteractivePlotter(bse, state_labels={0: "θ", 1: "ω"})
    # plotter.run(port=8050)
