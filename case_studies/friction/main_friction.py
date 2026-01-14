from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.friction.setup_friction_system import setup_friction_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_friction_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        cluster_classifier=props.get("cluster_classifier"),
        save_to="results_friction",
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_friction.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "friction"
        / "main_friction_case1.json"
    )

    if bse.bs_vals is not None:
        errors = bse.get_errors()
        compare_with_expected_by_size(bse.bs_vals, expected_file, errors)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
