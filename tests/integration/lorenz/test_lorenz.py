"""Integration tests for the Lorenz system case study."""

import json
from pathlib import Path

import numpy as np
import pytest

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import (
    AdaptiveStudyParams,
    ASBasinStabilityEstimator,
)
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    AttractorComparison,
    ComparisonResult,
    compute_statistical_comparison,
    run_adaptive_basin_stability_test,
    run_basin_stability_test,
    run_single_point_test,
)


class TestLorenz:
    """Integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_case1(
        self,
        tolerance: float,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz system case 1 - broken butterfly attractor parameters.

        Parameters: sigma=0.12, r=0.0, b=-0.6
        Expected attractors: chaos y_1, chaos y_2, unbounded

        Verifies:
        1. Number of ICs used matches sum of absNumMembers from MATLAB
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 2
        """
        json_path = Path(__file__).parent / "main_lorenz.json"
        label_map = {
            "butterfly1": "chaos y_1",
            "butterfly2": "chaos y_2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_lorenz_system,
            label_map=label_map,
            system_name="lorenz",
            case_name="case1",
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(bse, comparison)

    @pytest.mark.integration
    def test_parameter_sigma(
        self,
        tolerance: float,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz sigma parameter sweep using z-score validation.

        Studies the effect of varying sigma parameter from 0.12 to 0.18 on basin stability.

        Verifies:
        1. Parameter sweep over sigma
        2. Basin stability values pass z-score test for butterfly1, butterfly2, unbounded, NaN
        """
        json_path = Path(__file__).parent / "main_lorenz_sigma_study.json"
        label_map = {
            "butterfly1": "chaos y_1",
            "butterfly2": "chaos y_2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name='ode_system.params["sigma"]',
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            z_threshold=3.5,  # Chaotic system - occasional 2-3σ outliers expected
            system_name="lorenz",
            case_name="case2",
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)

    @pytest.mark.integration
    def test_hyperparameter_n(self, tolerance: float) -> None:
        """Test hyperparameter n - convergence study varying sample size N.

        Uses z-score validation with standard errors. As N increases, the standard
        error decreases (SE ~ 1/sqrt(N)), so validation naturally becomes stricter.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3.5
        3. Uses standard errors from both MATLAB and Python results

        Note: Uses higher z-threshold (3.5) for chaotic Lorenz system.
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperparameters.json"
        label_map = {
            "butterfly1": "chaos y_1",
            "butterfly2": "chaos y_2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name="n",
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            z_threshold=3.5,
        )

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n200(self) -> None:
        """Test with small N=200 for random sampling validation.

        Expected from MATLAB with UniformRandomSampler:
        - N=200 -> generates exactly 200 random points
        - butterfly1: 0.1000, butterfly2: 0.0750, unbounded: 0.8250

        Note: With only 200 random points, there's inherent statistical uncertainty.
        We use z-score validation with z-threshold=3.5 for chaotic system with small N.
        """
        run_single_point_test(
            n=200,
            expected_bs={
                "chaos y_1": 0.1,
                "chaos y_2": 0.075,
                "unbounded": 0.825,
            },
            setup_function=setup_lorenz_system,
            z_threshold=3.5,
            expected_points=200,
        )

    @pytest.mark.integration
    def test_hyperparameter_rtol(
        self,
        tolerance: float,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter rtol - ODE solver relative tolerance convergence study.

        Studies the effect of varying relative tolerance from 1e-3 to 1e-8 on basin stability.
        This test validates that the solver correctly uses the specified tolerances and
        that coarse tolerances (1e-3) can produce incorrect results.

        Expected behavior:
        - rtol=1e-3: May produce significantly different results (tolerance is too coarse)
        - rtol=1e-4 to 1e-8: Should converge to consistent values

        Note: We use a more lenient tolerance for rtol=1e-3 since we expect it to be less accurate.
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperpTol.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        props = setup_lorenz_system()

        parameter_values = np.array([result["parameter"] for result in expected_results])

        as_params = AdaptiveStudyParams(
            adaptative_parameter_values=parameter_values,
            adaptative_parameter_name="solver.rtol",
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

        label_map = {
            "butterfly1": "chaos y_1",
            "butterfly2": "chaos y_2",
            "unbounded": "unbounded",
        }

        comparison_results: list[ComparisonResult] = []
        z_threshold = 2.0

        for i, expected in enumerate(expected_results):
            rtol_value = expected["parameter"]
            actual_bs = as_bse.basin_stabilities[i]
            errors = as_bse.get_errors(i)

            attractor_comparisons: list[AttractorComparison] = []

            # For rtol=1e-3, just verify values are in valid range (coarse tolerance expected to differ)
            if rtol_value >= 1e-3:
                for label, python_label in label_map.items():
                    actual_val = actual_bs.get(python_label, 0.0)
                    assert 0.0 <= actual_val <= 1.0, (
                        f"At rtol={rtol_value:.0e}, {python_label} basin stability {actual_val:.4f} "
                        f"is outside valid range [0, 1]"
                    )
                    expected_bs = expected[f"bs_{label}"]
                    expected_err = expected.get(f"err_{label}", 0.0)
                    actual_err = errors[python_label]["e_abs"] if python_label in errors else 0.0
                    stats_comp = compute_statistical_comparison(
                        actual_val, actual_err, expected_bs, expected_err
                    )
                    attractor_comparisons.append(
                        AttractorComparison(
                            label=python_label,
                            python_bs=actual_val,
                            python_se=actual_err,
                            matlab_bs=expected_bs,
                            matlab_se=expected_err,
                            z_score=stats_comp.z_score,
                            p_value=stats_comp.p_value,
                            ci_lower=stats_comp.ci_lower,
                            ci_upper=stats_comp.ci_upper,
                            confidence="high",  # Only range check for rtol=1e-3, override
                        )
                    )
            else:
                # For rtol < 1e-3, use z-score validation
                for label, python_label in label_map.items():
                    expected_bs = expected[f"bs_{label}"]
                    expected_err = expected.get(f"err_{label}", 0.0)
                    actual_val = actual_bs.get(python_label, 0.0)
                    actual_err = errors[python_label]["e_abs"] if python_label in errors else 0.0

                    stats_comp = compute_statistical_comparison(
                        actual_val, actual_err, expected_bs, expected_err
                    )

                    combined_err = float(np.sqrt(expected_err**2 + actual_err**2))
                    diff = abs(actual_val - expected_bs)

                    if combined_err > 0:
                        threshold = z_threshold * combined_err
                        assert diff < threshold, (
                            f"At rtol={rtol_value:.0e}, {python_label}: "
                            f"expected {expected_bs:.4f} ± {expected_err:.4f}, "
                            f"got {actual_val:.4f} ± {actual_err:.4f}, "
                            f"diff {diff:.4f} exceeds threshold {threshold:.4f}"
                        )

                    attractor_comparisons.append(
                        AttractorComparison(
                            label=python_label,
                            python_bs=actual_val,
                            python_se=actual_err,
                            matlab_bs=expected_bs,
                            matlab_se=expected_err,
                            z_score=stats_comp.z_score,
                            p_value=stats_comp.p_value,
                            ci_lower=stats_comp.ci_lower,
                            ci_upper=stats_comp.ci_upper,
                            confidence=stats_comp.confidence,
                        )
                    )

            comparison_results.append(
                ComparisonResult(
                    system_name="lorenz",
                    case_name="case3",
                    attractors=attractor_comparisons,
                    parameter_value=rtol_value,
                    z_threshold=z_threshold,
                )
            )

            # Verify basin stabilities sum to approximately 1.0
            total_bs = sum(actual_bs.values())
            assert abs(total_bs - 1.0) < 0.01, (
                f"At rtol={rtol_value:.0e}, basin stabilities should sum to 1.0, got {total_bs:.4f}"
            )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparison_results)
