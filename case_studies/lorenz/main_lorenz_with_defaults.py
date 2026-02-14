from pathlib import Path

import torch

from case_studies.comparison_utils import compare_with_expected
from case_studies.lorenz.setup_lorenz_system import lorenz_stop_event, setup_lorenz_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.types import InteractivePlotterOptions
from pybasin.solvers.jax_solver import JaxSolver
from pybasin.utils import time_execution


def main():
    props = setup_lorenz_system()

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The default solver does not work here because it needs the event_fn to stop
    solver = JaxSolver(
        device=device,
        cache_dir=".pybasin_cache/lorenz",
        event_fn=lorenz_stop_event,
    )

    bse = BasinStabilityEstimator(
        n=props.get("n"),
        ode_system=props.get("ode_system"),
        sampler=props.get("sampler"),
        solver=solver,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz.py", main)

    label_mapping = {"0": "butterfly1", "1": "butterfly2", "unbounded": "unbounded"}
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "lorenz"
        / "main_lorenz.json"
    )
    if bse.bs_vals is not None:
        compare_with_expected(bse.bs_vals, label_mapping, expected_file)

    options: InteractivePlotterOptions = {
        "templates_phase_space": {"x_axis": 1, "y_axis": 2, "exclude_templates": ["unbounded"]},
        "feature_space": {"exclude_labels": ["unbounded"]},
    }
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
    plotter.run(port=8050)
