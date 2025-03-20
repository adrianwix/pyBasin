# Pybasin Imports
from case_study_pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator

from pybasin.Sampler import GridSampler
from pybasin.utils import time_execution

# Third Parties
import numpy as np


def main():
    props = setup_pendulum_system()

    props["ode_system"].params["T"] = 0.01
    props["N"] = 10000
    props["sampler"] = GridSampler(
        min_limits=props["sampler"].min_limits,
        max_limits=props["sampler"].max_limits)

    bse = BasinStabilityEstimator(
        name="main_pendulum_T_0_01",
        N=props["N"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="main_pendulum_T_0_01"
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # plotter = Plotter(bse=bse)
    # plotter.plot_templates(
    #     plotted_var=1,
    #     time_span=(0, 200))
    # plotter.plot_bse_results()

    bse.save()


if __name__ == "__main__":
    time_execution("main_pendulum_case1.py", main)
