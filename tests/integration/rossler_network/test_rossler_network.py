"""Integration tests for the Rössler network basin stability case study.

These tests validate the basin stability estimation for coupled Rössler oscillator
networks against expected values from the reference paper (Menck et al., 2013).

Note: Unlike other case studies, this does not have MATLAB bSTAB reference implementation,
so we validate against the published paper results using statistical error bounds.
"""

import numpy as np
import pytest

from case_studies.rossler_network.setup_rossler_network_system import (
    setup_rossler_network_system,
)
from pybasin.as_basin_stability_estimator import (
    AdaptiveStudyParams,
    ASBasinStabilityEstimator,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    AttractorComparison,
    ComparisonResult,
    compute_statistical_comparison,
)

EXPECTED_VALUES_FROM_PAPER: dict[float, float] = {
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

EXPECTED_MEAN_SB = 0.490


class TestRosslerNetwork:
    """Integration tests for Rössler network basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Rössler network baseline parameters (K=0.218).

        Expected from paper:
        - K=0.218: S_B ≈ 0.496

        Verifies:
        1. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        2. Results are documented for artifact generation
        """
        k_val = 0.218
        expected_sb = EXPECTED_VALUES_FROM_PAPER[k_val]

        props = setup_rossler_network_system(k=k_val)

        bse = BasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props.get("solver"),
            feature_extractor=props.get("feature_extractor"),
            predictor=props.get("cluster_classifier"),
            feature_selector=None,
        )

        basin_stability = bse.estimate_bs()
        computed_sb = basin_stability.get("synchronized", 0.0) + basin_stability.get(
            "desynchronized", 0.0
        )

        n_samples = props["n"]
        e_abs_computed = np.sqrt(computed_sb * (1 - computed_sb) / n_samples)
        e_abs_paper = np.sqrt(expected_sb * (1 - expected_sb) / n_samples)

        e_combined = np.sqrt(e_abs_computed**2 + e_abs_paper**2)
        diff = abs(computed_sb - expected_sb)
        z_score = diff / e_combined if e_combined > 0 else 0.0

        stats_comp = compute_statistical_comparison(
            computed_sb, e_abs_computed, expected_sb, e_abs_paper
        )

        print(f"\nRössler Network K={k_val}:")
        print(f"  Expected S_B: {expected_sb:.3f} ± {e_abs_paper:.3f}")
        print(f"  Computed S_B: {computed_sb:.3f} ± {e_abs_computed:.3f}")
        print(f"  Difference:   {diff:+.3f}")
        print(f"  Z-score:      {stats_comp.z_score:.2f}")
        print(f"  P-value:      {stats_comp.p_value:.4f}")
        print("  Threshold:    3.00 (99.7% CI)")

        comparison = AttractorComparison(
            label="synchronized",
            python_bs=computed_sb,
            python_se=e_abs_computed,
            matlab_bs=expected_sb,
            matlab_se=e_abs_paper,
            z_score=stats_comp.z_score,
            p_value=stats_comp.p_value,
            ci_lower=stats_comp.ci_lower,
            ci_upper=stats_comp.ci_upper,
            confidence=stats_comp.confidence,
        )

        comparison_result = ComparisonResult(
            system_name="rossler_network",
            case_name="baseline",
            attractors=[comparison],
            z_threshold=3.0,
        )

        assert z_score < 3.0, (
            f"Basin stability for K={k_val}: expected {expected_sb:.3f} ± {e_abs_paper:.3f}, "
            f"got {computed_sb:.3f} ± {e_abs_computed:.3f}, "
            f"z-score {z_score:.2f} exceeds threshold 3.0"
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(bse, comparison_result)

    @pytest.mark.integration
    def test_parameter_k(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Rössler network coupling strength (K) parameter sweep.

        Studies the effect of varying K from 0.119 to 0.317 (11 values) on basin stability.
        This replicates the K-sweep from the paper.

        Expected from paper: Mean S̄_B ≈ 0.49, monotonically increasing trend

        Verifies:
        1. Parameter sweep over K (coupling strength) executes successfully
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        3. Results are documented for artifact generation
        """
        props = setup_rossler_network_system(k=0.119)

        k_values = np.array(list(EXPECTED_VALUES_FROM_PAPER.keys()))

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

        as_bse = ASBasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=solver,
            feature_extractor=feature_extractor,
            cluster_classifier=cluster_classifier,
            as_params=as_params,
        )

        as_bse.estimate_as_bs()

        comparison_results: list[ComparisonResult] = []
        computed_sync: list[float] = []
        n_samples = props["n"]

        print("\n" + "=" * 80)
        print("RÖSSLER NETWORK K-SWEEP: COMPARISON WITH PAPER")
        print("=" * 80)
        print(
            f"{'K':>8} | {'Expected':>8} | {'Computed':>8} | {'Diff':>7} | {'e_abs':>6} | {'Z-score':>8} | {'Status':>6}"
        )
        print("-" * 80)

        within_2sigma = 0
        for param_val, bs_dict in zip(
            as_bse.parameter_values, as_bse.basin_stabilities, strict=True
        ):
            sync_val = bs_dict.get("synchronized", 0.0) + bs_dict.get("desynchronized", 0.0)
            computed_sync.append(sync_val)
            expected_val = EXPECTED_VALUES_FROM_PAPER[param_val]

            e_abs_computed = np.sqrt(sync_val * (1 - sync_val) / n_samples)
            e_abs_paper = np.sqrt(expected_val * (1 - expected_val) / n_samples)

            stats_comp = compute_statistical_comparison(
                sync_val, e_abs_computed, expected_val, e_abs_paper
            )

            diff = sync_val - expected_val

            within = stats_comp.z_score < 3.0
            within_2sigma += int(within)
            status = "✓" if within else "✗"

            print(
                f"{param_val:>8.3f} | {expected_val:>8.3f} | {sync_val:>8.3f} | "
                f"{diff:>+7.3f} | {e_abs_computed:>6.3f} | {stats_comp.z_score:>8.2f} | {status:>6}"
            )

            comparison = AttractorComparison(
                label="synchronized",
                python_bs=sync_val,
                python_se=e_abs_computed,
                matlab_bs=expected_val,
                matlab_se=e_abs_paper,
                z_score=stats_comp.z_score,
                p_value=stats_comp.p_value,
                ci_lower=stats_comp.ci_lower,
                ci_upper=stats_comp.ci_upper,
                confidence=stats_comp.confidence,
            )

            comparison_result = ComparisonResult(
                system_name="rossler_network",
                case_name="k_sweep",
                attractors=[comparison],
                parameter_value=param_val,
                z_threshold=3.0,
            )
            comparison_results.append(comparison_result)

            assert stats_comp.z_score < 3.0, (
                f"Basin stability for K={param_val}: expected {expected_val:.3f} ± {e_abs_paper:.3f}, "
                f"got {sync_val:.3f} ± {e_abs_computed:.3f}, "
                f"z-score {stats_comp.z_score:.2f} exceeds threshold 3.0"
            )

        mean_sb = np.mean(computed_sync)
        print("-" * 80)
        print(f"Mean S_B: {mean_sb:.3f} (expected ~{EXPECTED_MEAN_SB:.3f} from paper)")
        print(
            f"\nWithin 2σ: {within_2sigma}/{len(EXPECTED_VALUES_FROM_PAPER)} "
            f"({within_2sigma / len(EXPECTED_VALUES_FROM_PAPER) * 100:.0f}%)"
        )
        print(f"Note: e_abs = sqrt(S_B*(1-S_B)/N), N={n_samples}")

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparison_results)
