# Pybasin Imports
from case_study_pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator

from pybasin.utils import time_execution

# Third Parties
import numpy as np


def main():
    props = setup_pendulum_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    knn_cluster = props["cluster_classifier"]

    bse = BasinStabilityEstimator(
        name="pendulum_case1",
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster,
        save_to="results_case1"
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
