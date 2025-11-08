"""Test GPU performance improvement."""

import time

import numpy as np
import torch

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator


def test_gpu_performance():
    """Compare GPU performance."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = setup_pendulum_system()

    # Run a smaller test
    print("\n" + "=" * 60)
    print("Running Basin Stability Estimation on GPU")
    print("=" * 60)

    start = time.time()
    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props["solver"],
        feature_extractor=props["feature_extractor"],
        cluster_classifier=props["cluster_classifier"],
    )
    result = bse.estimate_bs()
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Result: {result}")


if __name__ == "__main__":
    test_gpu_performance()
