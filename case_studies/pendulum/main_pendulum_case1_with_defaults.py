from pathlib import Path

from case_studies.comparison_utils import compare_with_expected
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        ode_system=props["ode_system"],
        sampler=props["sampler"],
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", {k: float(v) for k, v in basin_stability.items()})

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case_1_with_defaults.py", main)

    label_mapping = {"0": "LC", "1": "FP"}
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "pendulum"
        / "main_pendulum_case1.json"
    )

    if bse.bs_vals is not None:
        compare_with_expected(bse.bs_vals, label_mapping, expected_file)

    plotter = InteractivePlotter(bse, state_labels={0: "θ", 1: "ω"})
    # plotter.run(port=8050)
