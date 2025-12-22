import torch

from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE, LorenzParams
from case_studies.lorenz.setup_lorenz_system import lorenz_stop_event
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.types import (
    FeatureSpaceOptions,
    InteractivePlotterOptions,
    PhasePlotOptions,
)
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers.jax_solver import JaxSolver
from pybasin.utils import time_execution


def main():
    n = 1 * 10**4

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    # Parameters for broken butterfly system
    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}

    ode_system = LorenzJaxODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0], max_limits=[10.0, 20.0, 0.0], device=device
    )

    # The default solver does not work here because it needs the event_fn to stop
    solver = JaxSolver(
        device=device,
        event_fn=lorenz_stop_event,
    )

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        save_to="results_case1_with_defaults",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz.py", main)
    options = InteractivePlotterOptions(
        phase_plot=PhasePlotOptions(x_var=1, y_var=2, exclude_templates=["unbounded"]),
        feature_space=FeatureSpaceOptions(exclude_labels=["unbounded"]),
    )
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
    # plotter.run(port=8050)
