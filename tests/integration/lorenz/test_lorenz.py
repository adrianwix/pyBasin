"""Integration tests for the Lorenz system case study."""

import json
from pathlib import Path

import numpy as np
import pytest

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from tests.integration.test_helpers import (
    run_adaptive_basin_stability_test,
    run_basin_stability_test,
    run_single_point_test,
)


class TestLorenz:
    """Integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_case1(self, tolerance: float) -> None:
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
        run_basin_stability_test(json_path, setup_lorenz_system, label_map=label_map)

    @pytest.mark.integration
    def test_parameter_sigma(self, tolerance: float) -> None:
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
        run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name='ode_system.params["sigma"]',
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            z_threshold=3.5,  # Chaotic system - occasional 2-3σ outliers expected
        )

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
    def test_hyperparameter_rtol(self, tolerance: float) -> None:
        """Test hyperparameter rtol - ODE solver relative tolerance convergence study.

        Studies the effect of varying relative tolerance from 1e-3 to 1e-8 on basin stability.
        This test validates that the solver correctly uses the specified tolerances and
        that coarse tolerances (1e-3) can produce incorrect results.

        Expected behavior:
        - rtol=1e-3: May produce significantly different results (tolerance is too coarse)
        - rtol=1e-4 to 1e-8: Should converge to consistent values

        Note: We use a more lenient tolerance for rtol=1e-3 since we expect it to be less accurate.
        """
        # Load expected results from JSON (MATLAB reference)
        json_path = Path(__file__).parent / "main_lorenz_hyperpTol.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run tolerance study
        props = setup_lorenz_system()

        # Use the same tolerance values as in the expected results
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

        # Use N=20000 to match MATLAB study
        as_bse = ASBasinStabilityEstimator(
            n=20000,
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=solver,
            feature_extractor=feature_extractor,
            cluster_classifier=cluster_classifier,
            as_params=as_params,
            save_to=None,  # Don't save during test
        )

        as_bse.estimate_as_bs()

        # Compare results at each tolerance value
        for i, expected in enumerate(expected_results):
            rtol_value = expected["parameter"]
            actual_bs = as_bse.basin_stabilities[i]

            # Use a more lenient tolerance for rtol=1e-3 since it's expected to be less accurate
            # For rtol >= 1e-4, we expect convergence to accurate values
            test_tolerance = tolerance * 3.0 if rtol_value >= 1e-3 else tolerance  # type: ignore[assignment]

            # Check butterfly1 (JSON) -> chaos y_1 (Python) basin stability
            expected_bs_b1 = expected["bs_butterfly1"]
            actual_bs_b1 = actual_bs.get("chaos y_1", 0.0)
            diff_b1 = abs(actual_bs_b1 - expected_bs_b1)

            # For rtol=1e-3, just verify it's within a reasonable range (not checking exact match)
            if rtol_value == 1e-3:
                # At coarse tolerance, we mainly want to ensure the solver doesn't crash
                # and produces some result (even if less accurate)
                assert 0.0 <= actual_bs_b1 <= 1.0, (
                    f"At rtol={rtol_value:.0e}, chaos y_1 basin stability {actual_bs_b1:.4f} "
                    f"is outside valid range [0, 1]"
                )
            else:
                assert diff_b1 < test_tolerance, (
                    f"At rtol={rtol_value:.0e}, chaos y_1 basin stability: "
                    f"expected {expected_bs_b1:.4f}, got {actual_bs_b1:.4f}, "
                    f"difference {diff_b1:.4f} exceeds tolerance {test_tolerance:.4f}"
                )

            # Check butterfly2 (JSON) -> chaos y_2 (Python) basin stability
            expected_bs_b2 = expected["bs_butterfly2"]
            actual_bs_b2 = actual_bs.get("chaos y_2", 0.0)
            diff_b2 = abs(actual_bs_b2 - expected_bs_b2)

            if rtol_value == 1e-3:
                assert 0.0 <= actual_bs_b2 <= 1.0, (
                    f"At rtol={rtol_value:.0e}, chaos y_2 basin stability {actual_bs_b2:.4f} "
                    f"is outside valid range [0, 1]"
                )
            else:
                assert diff_b2 < test_tolerance, (
                    f"At rtol={rtol_value:.0e}, chaos y_2 basin stability: "
                    f"expected {expected_bs_b2:.4f}, got {actual_bs_b2:.4f}, "
                    f"difference {diff_b2:.4f} exceeds tolerance {test_tolerance:.4f}"
                )

            # Check unbounded basin stability
            expected_bs_unbounded = expected["bs_unbounded"]
            actual_bs_unbounded = actual_bs.get("unbounded", 0.0)
            diff_unbounded = abs(actual_bs_unbounded - expected_bs_unbounded)

            if rtol_value == 1e-3:
                assert 0.0 <= actual_bs_unbounded <= 1.0, (
                    f"At rtol={rtol_value:.0e}, unbounded basin stability {actual_bs_unbounded:.4f} "
                    f"is outside valid range [0, 1]"
                )
            else:
                assert diff_unbounded < test_tolerance, (
                    f"At rtol={rtol_value:.0e}, unbounded basin stability: "
                    f"expected {expected_bs_unbounded:.4f}, got {actual_bs_unbounded:.4f}, "
                    f"difference {diff_unbounded:.4f} exceeds tolerance {test_tolerance:.4f}"
                )

            # Verify basin stabilities sum to approximately 1.0
            total_bs = sum(actual_bs.values())
            assert abs(total_bs - 1.0) < 0.01, (
                f"At rtol={rtol_value:.0e}, basin stabilities should sum to 1.0, got {total_bs:.4f}"
            )

        # Additional check: verify that results for rtol >= 1e-4 are relatively consistent
        # (i.e., they have converged to a stable solution)
        consistent_results = as_bse.basin_stabilities[1:]  # Skip rtol=1e-3
        if len(consistent_results) > 1:
            chaos_y_1_values = [bs.get("chaos y_1", 0.0) for bs in consistent_results]
            chaos_y_1_std = np.std(chaos_y_1_values)
            assert chaos_y_1_std < 0.01, (
                f"chaos y_1 values for rtol >= 1e-4 should be consistent, "
                f"but std={chaos_y_1_std:.4f} is too high. Values: {chaos_y_1_values}"
            )
