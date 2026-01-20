from case_studies.pendulum.setup_pendulum_system_sklearn import setup_pendulum_system_sklearn
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system_sklearn()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("cluster_classifier"),
        save_to="results_case1_sklearn",
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    # bse.save()
    # bse.save_to_excel()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case1_sklearn.py", main)
    plotter = InteractivePlotter(bse, state_labels={0: "θ", 1: "ω"})
    plotter.run(port=8050)
