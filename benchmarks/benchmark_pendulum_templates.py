"""
Benchmark template solving performance for the pendulum system.

This script measures how long it takes to solve the template initial conditions.
Cache is always cleared before benchmarking to get accurate timing measurements.
"""

import shutil
import time
from pathlib import Path
from typing import cast

import torch

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.predictors import ClassifierPredictor


def clear_cache():
    """Clear the solver cache directory."""
    # The cache is created relative to where setup_pendulum_system is defined
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "case_studies" / "pendulum" / "cache"

    if cache_dir.exists():
        print(f"Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Cache cleared.")
    else:
        print(f"Cache directory does not exist: {cache_dir}")
        print("Will be created on first integration.")


def benchmark_template_solving():
    """
    Benchmark the template solving process.
    Cache is cleared before benchmarking to measure true computation time.
    """
    clear_cache()

    print("\n" + "=" * 80)
    print("PENDULUM SYSTEM TEMPLATE SOLVING BENCHMARK")
    print("=" * 80)

    # Setup the pendulum system
    print("\nSetting up pendulum system...")
    props = setup_pendulum_system()

    ode_system = props["ode_system"]
    solver = props.get("solver")
    cluster_classifier = cast(ClassifierPredictor, props.get("cluster_classifier"))

    print(f"\nDevice: {solver.device}")
    print(f"Number of templates: {len(cluster_classifier.initial_conditions)}")
    print(f"Template initial conditions:\n{cluster_classifier.initial_conditions}")
    print(f"Template labels: {cluster_classifier.labels}")
    print(f"Integration time span: {solver.time_span}")
    print(f"Sampling frequency: {solver.fs} Hz")
    print(f"Number of time steps: {solver.n_steps}")

    print("\n" + "-" * 80)
    print("BENCHMARKING TEMPLATE INTEGRATION (Cache Disabled)")
    print("-" * 80)

    # Measure time for integration
    print("\nIntegrating template initial conditions...")
    start_time = time.perf_counter()

    # This is what happens during classifier.fit()
    # It integrates the template initial conditions
    # Use no_grad() to disable gradient computation for faster inference
    with torch.no_grad():
        t, y = solver.integrate(ode_system, cluster_classifier.initial_conditions)

    integration_time = time.perf_counter() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nIntegration time: {integration_time:.3f} seconds")
    print(f"Solution shape:   {y.shape}")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    benchmark_template_solving()


if __name__ == "__main__":
    main()
