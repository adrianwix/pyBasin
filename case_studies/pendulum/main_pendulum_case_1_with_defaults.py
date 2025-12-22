from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        save_to="results_case_1_with_defaults",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", {k: float(v) for k, v in basin_stability.items()})

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case_1_with_defaults.py", main)
    plotter = InteractivePlotter(bse, state_labels={0: "θ", 1: "ω"})
    plotter.run(port=8050)
