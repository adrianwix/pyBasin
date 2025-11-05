# Pybasin Imports
from setup_pendulum_system import setup_pendulum_system
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from pybasin.Plotter import Plotter
from pybasin.utils import time_execution

# Third Parties
import numpy as np


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        N=props["N"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_case1"
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    plotter = Plotter(bse=bse)
    plotter.plot_templates(
        plotted_var=1,
        time_span=(0, 100))
    plotter.plot_bse_results()

    bse.save()
    bse.save_to_excel()


if __name__ == "__main__":
    time_execution("main_pendulum_case1.py", main)
