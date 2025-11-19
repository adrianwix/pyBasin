# Pybasin Imports
# Third Parties
from case_studies.pendulum.setup_pendulum_system_torchode import (
    setup_pendulum_system_torchode,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.utils import time_execution


def main():
    """
    Test case study for pendulum using TorchOdeSolver.

    This demonstrates the usage of torchode as an alternative ODE solver
    to torchdiffeq. The results should be similar to the standard case study.
    """
    print("=" * 80)
    print("Pendulum Case Study - TorchOdeSolver")
    print("=" * 80)

    props = setup_pendulum_system_torchode()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_case1_torchode",
    )

    basin_stability = bse.estimate_bs()
    print("\n" + "=" * 80)
    print("Basin Stability Results:")
    print("=" * 80)
    for label, value in basin_stability.items():
        print(f"  {label}: {value:.4f}")
    print("=" * 80)

    # Uncomment to generate plots
    # plotter = Plotter(bse=bse)
    # plotter.plot_templates(plotted_var=1, time_span=(0, 100))
    # plotter.plot_bse_results()

    # Uncomment to save results
    # bse.save()
    # bse.save_to_excel()


if __name__ == "__main__":
    time_execution("main_pendulum_case1_torchode.py", main)
