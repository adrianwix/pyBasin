from case_friction.setup_friction_system import setup_friction_system
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from pybasin.Plotter import Plotter
from pybasin.utils import time_execution


def main():
    props = setup_friction_system()

    bse = BasinStabilityEstimator(
        name="friction",
        N=props["N"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_friction"
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    plotter = Plotter(bse=bse)

    plotter.plot_bse_results()
    plotter.plot_templates(
        plotted_var=1,
        time_span=(0, 200)
    )
    plotter.plot_phase(x_var=0, y_var=1)

    bse.save()


if __name__ == "__main__":
    time_execution("main_friction.py", main)
