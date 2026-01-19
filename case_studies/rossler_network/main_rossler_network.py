"""Main entry point for the Rössler network basin stability case study.

This replicates the basin stability analysis of synchronization in coupled
Rössler oscillator networks from the original paper.

Expected result for K=0.218: S_B ≈ 0.496 (±0.05 due to sampling variance)
"""

from case_studies.rossler_network.setup_rossler_network_system import (
    setup_rossler_network_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.utils import time_execution


def main(k: float = 0.218):
    props = setup_rossler_network_system(k=k)

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        cluster_classifier=props.get("cluster_classifier"),
        save_to="results",
        feature_selector=None,
        detect_unbounded=False,
    )

    basin_stability = bse.estimate_bs()
    print(f"\nBasin Stability for k={k}:")
    print(basin_stability)

    expected_sb = {
        0.119: 0.226,
        0.139: 0.274,
        0.159: 0.330,
        0.179: 0.346,
        0.198: 0.472,
        0.218: 0.496,
        0.238: 0.594,
        0.258: 0.628,
        0.278: 0.656,
        0.297: 0.694,
        0.317: 0.690,
    }
    if k in expected_sb:
        print(f"\nExpected S_B (synchronized) from paper: {expected_sb[k]}")

    return bse


if __name__ == "__main__":
    bse = time_execution("main_rossler_network.py", main)
