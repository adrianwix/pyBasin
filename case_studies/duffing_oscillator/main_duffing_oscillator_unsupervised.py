from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DBSCANClusterer
from pybasin.utils import time_execution


def main():
    setup = setup_duffing_oscillator_system()

    cluster_classifier = DBSCANClusterer(eps=0.08)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup["solver"],
        feature_extractor=setup["feature_extractor"],
        cluster_classifier=cluster_classifier,
        save_to="results_unsupervised",
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_duffing_oscillator_unsupervised.py", main)
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    plotter.run(port=8050)
