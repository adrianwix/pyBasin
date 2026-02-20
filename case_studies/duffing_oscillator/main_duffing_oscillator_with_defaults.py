from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.sampler import Sampler
from pybasin.utils import time_execution


def main(sampler_override: Sampler | None = None) -> BasinStabilityEstimator:
    setup = setup_duffing_oscillator_system()
    sampler = sampler_override if sampler_override is not None else setup["sampler"]

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=sampler,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", {k: float(v) for k, v in basin_stability.items()})

    return bse


if __name__ == "__main__":
    bse = time_execution("main_duffing_oscillator_with_defaults.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "duffing"
        / "main_duffing_supervised.json"
    )

    if bse.bs_vals is not None:
        compare_with_expected_by_size(bse.bs_vals, expected_file)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
