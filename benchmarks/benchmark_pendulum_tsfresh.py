# pyright: basic
"""
Benchmark pendulum basin stability estimation with tsfresh comprehensive features.

This benchmark compares tsfresh ComprehensiveFCParameters vs JAX feature extraction
in the context of a full basin stability estimation workflow.

Usage:
    uv run python benchmarks/benchmark_pendulum_tsfresh.py
    uv run python benchmarks/benchmark_pendulum_tsfresh.py --n=1000
    uv run python benchmarks/benchmark_pendulum_tsfresh.py --jax-only
    uv run python benchmarks/benchmark_pendulum_tsfresh.py --tsfresh-only
"""

import sys
import time
import warnings

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
)

# Suppress tsfresh warnings about NaN/inf values
warnings.filterwarnings("ignore", message=".*did not have any finite values.*")
warnings.filterwarnings("ignore", message=".*Filling with zeros.*")
warnings.filterwarnings("ignore", message=".*The columns.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tsfresh")
warnings.filterwarnings("ignore", category=UserWarning, module="tsfresh")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tsfresh")

from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import KNNCluster
from pybasin.feature_extractors.jax_feature_extractor import JaxFeatureExtractor
from pybasin.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.sampler import GridSampler
from pybasin.solvers import JaxSolver
from pybasin.types import SetupProperties


def create_pendulum_setup(
    n: int, feature_extractor_type: str = "jax", comprehensive: bool = True
) -> SetupProperties:
    """Create pendulum setup with specified feature extractor.

    Args:
        n: Number of samples
        feature_extractor_type: "jax" or "tsfresh"
        comprehensive: If True, use comprehensive features; if False, use minimal
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumJaxODE(params)

    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )

    solver = JaxSolver(
        time_span=(0, 1000),
        n_steps=1000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        use_cache=True,
    )

    if feature_extractor_type == "jax":
        feature_extractor = JaxFeatureExtractor(
            time_steady=950.0,
            normalize=False,
        )
    else:
        fc_params = ComprehensiveFCParameters() if comprehensive else MinimalFCParameters()

        feature_extractor = TsfreshFeatureExtractor(
            time_steady=950.0,
            default_fc_parameters=fc_params,
            n_jobs=-1,
            normalize=False,
        )

    template_y0 = [
        [0.5, 0.0],
        [2.7, 0.0],
    ]
    classifier_labels = ["FP", "LC"]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn_cluster = KNNCluster(
        classifier=knn,
        template_y0=template_y0,
        labels=classifier_labels,
        ode_params=params,
    )

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster,
    }


def run_benchmark(n: int, extractor_type: str, comprehensive: bool = True) -> dict:
    """Run a single benchmark and return timing results."""
    print(f"\n{'=' * 80}")
    print(f"Running {extractor_type.upper()} benchmark with n={n}")
    if extractor_type == "tsfresh":
        print(f"Feature set: {'Comprehensive' if comprehensive else 'Minimal'}")
    print("=" * 80)

    setup = create_pendulum_setup(n, extractor_type, comprehensive)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup["solver"],
        feature_extractor=setup["feature_extractor"],
        cluster_classifier=setup["cluster_classifier"],
        feature_selector=None,
    )

    t0 = time.perf_counter()
    bs_vals = bse.estimate_bs(parallel_integration=True)
    total_time = time.perf_counter() - t0

    n_features = 0
    if bse.solution is not None and bse.solution.features is not None:
        n_features = bse.solution.features.shape[1]
    print(f"Number of features extracted: {n_features}")

    return {
        "extractor": extractor_type,
        "n": n,
        "comprehensive": comprehensive,
        "total_time": total_time,
        "bs_vals": bs_vals,
        "n_features": n_features,
    }


def main():
    print("=" * 80)
    print("PENDULUM BASIN STABILITY BENCHMARK: JAX vs TSFRESH")
    print("=" * 80)

    n = 1000
    run_jax = True
    run_tsfresh = True

    for arg in sys.argv:
        if arg.startswith("--n="):
            n = int(arg.split("=")[1])
        if arg == "--jax-only":
            run_tsfresh = False
        if arg == "--tsfresh-only":
            run_jax = False

    print(f"\nConfiguration:")
    print(f"  Samples (n): {n}")
    print(f"  Run JAX: {run_jax}")
    print(f"  Run tsfresh: {run_tsfresh}")

    results = []

    if run_jax:
        jax_result = run_benchmark(n, "jax")
        results.append(jax_result)

    if run_tsfresh:
        tsfresh_result = run_benchmark(n, "tsfresh", comprehensive=False)
        results.append(tsfresh_result)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n{r['extractor'].upper()}:")
        print(f"  Number of features: {r['n_features']}")
        print(f"  Total time: {r['total_time']:.2f}s")
        print(f"  Basin stability: {r['bs_vals']}")

    if run_jax and run_tsfresh:
        jax_time = results[0]["total_time"]
        tsfresh_time = results[1]["total_time"]
        speedup = tsfresh_time / jax_time if jax_time > 0 else 0
        print(f"\nJAX speedup over tsfresh: {speedup:.1f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
