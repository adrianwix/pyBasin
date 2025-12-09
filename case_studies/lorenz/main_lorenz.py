from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.types import (
    FeatureSpaceOptions,
    InteractivePlotterOptions,
    PhasePlotOptions,
)
from pybasin.utils import time_execution


def main():
    props = setup_lorenz_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        cluster_classifier=props.get("cluster_classifier"),
        save_to="results_case1",
        feature_selector=None,
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
    plotter.run(port=8050)
