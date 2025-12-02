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
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_friction",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_friction.py", main)
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    plotter.run(port=8050)
