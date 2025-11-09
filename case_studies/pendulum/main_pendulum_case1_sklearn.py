# Pybasin Imports
# Third Parties
from case_studies.pendulum.setup_pendulum_system_sklearn import setup_pendulum_system_sklearn
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.utils import time_execution


def main():
    """
    Pendulum case study using SklearnParallelSolver.

    This case study demonstrates the use of the sklearn-based parallel solver
    that leverages Python 3.14's free-threading capabilities (PEP 703) to
    achieve true parallel execution without the Global Interpreter Lock (GIL).

    The sklearn parallel solver uses:
    - RK45 adaptive step size method for accurate ODE integration
    - sklearn's joblib backend for efficient parallel processing
    - Python 3.14 free-threading for true multi-core parallelism
    - CPU-based execution with automatic core utilization

    Performance characteristics:
    - Excellent scalability on multi-core CPUs
    - No CUDA required - pure CPU parallelism
    - Efficient memory usage with adaptive step size
    - Ideal for systems without GPU acceleration
    """
    print("=" * 70)
    print("Pendulum Basin Stability Analysis - SklearnParallelSolver")
    print("Python 3.14 Free-Threading Parallel Execution")
    print("=" * 70)

    props = setup_pendulum_system_sklearn()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
        save_to="results_case1_sklearn",
    )

    basin_stability = bse.estimate_bs()
    print("=" * 70)
    print(f"Basin Stability: {basin_stability}")
    print("=" * 70)

    # Optional: Uncomment to visualize results
    # plotter = Plotter(bse=bse)
    # plotter.plot_templates(plotted_var=1, time_span=(0, 100))
    # plotter.plot_bse_results()

    # Optional: Uncomment to save results
    # bse.save()
    # bse.save_to_excel()


if __name__ == "__main__":
    time_execution("main_pendulum_case1_sklearn.py", main)
