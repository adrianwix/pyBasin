"""Adaptive parameter study for Rössler network basin stability across K values.

This replicates the full K-sweep from the original paper, computing basin stability
at 11 equally spaced values of K in the stability interval.

Expected results from paper:
    K=0.119: S_B=0.226    K=0.238: S_B=0.594
    K=0.139: S_B=0.274    K=0.258: S_B=0.628
    K=0.159: S_B=0.330    K=0.278: S_B=0.656
    K=0.179: S_B=0.346    K=0.297: S_B=0.694
    K=0.198: S_B=0.472    K=0.317: S_B=0.690
    K=0.218: S_B=0.496

Mean basin stability: S̄_B ≈ 0.49
"""

import numpy as np

from case_studies.rossler_network.setup_rossler_network_system import (
    setup_rossler_network_system,
)
from pybasin.as_basin_stability_estimator import (
    AdaptiveStudyParams,
    ASBasinStabilityEstimator,
)
from pybasin.utils import time_execution


def main():
    props = setup_rossler_network_system(k=0.119)

    k_values = np.array(
        [0.119, 0.139, 0.159, 0.179, 0.198, 0.218, 0.238, 0.258, 0.278, 0.297, 0.317]
    )

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=k_values,
        adaptative_parameter_name='ode_system.params["K"]',
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None
    assert feature_extractor is not None
    assert cluster_classifier is not None

    bse = ASBasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        as_params=as_params,
        save_to="results_k_study",
    )

    bse.estimate_as_bs()

    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 60)

    expected = {
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

    n_samples = props["n"]

    print(f"{'K':>8} | {'Expected':>8} | {'Computed':>8} | {'Diff':>7} | {'e_abs':>6} | {'2σ':>6}")
    print("-" * 60)

    computed_sync: list[float] = []
    within_2sigma = 0
    for param_val, bs_dict in zip(bse.parameter_values, bse.basin_stabilities, strict=True):
        sync_val = bs_dict.get("synchronized", 0.0)
        computed_sync.append(sync_val)
        diff = sync_val - expected[param_val]

        e_abs = np.sqrt(sync_val * (1 - sync_val) / n_samples)
        e_combined = e_abs * np.sqrt(2)
        within = abs(diff) <= 2 * e_combined
        within_2sigma += int(within)
        status = "✓" if within else "✗"

        print(
            f"{param_val:>8.3f} | {expected[param_val]:>8.3f} | {sync_val:>8.3f} | {diff:>+7.3f} | {e_abs:>6.3f} | {status:>6}"
        )

    if computed_sync:
        mean_sb = np.mean(computed_sync)
        e_mean = np.std(computed_sync, ddof=1) / np.sqrt(len(computed_sync))
        mean_diff = mean_sb - 0.490
        mean_within = abs(mean_diff) <= 2 * e_mean
        mean_status = "✓" if mean_within else "✗"
        print("-" * 60)
        print(
            f"{'Mean S_B':>8} | {'0.490':>8} | {mean_sb:>8.3f} | {mean_diff:>+7.3f} | {e_mean:>6.3f} | {mean_status:>6}"
        )
        print(
            f"\nWithin 2σ: {within_2sigma}/{len(expected)} ({within_2sigma / len(expected) * 100:.0f}%)"
        )
        print(f"Note: e_abs = sqrt(S_B*(1-S_B)/N), N={n_samples}; e_mean = std(S_B)/sqrt(K)")

    bse.save()

    return bse


if __name__ == "__main__":
    time_execution("main_rossler_network_k_study.py", main)
