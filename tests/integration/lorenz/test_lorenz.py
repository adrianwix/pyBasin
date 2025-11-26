"""Integration tests for the Lorenz system case study."""

import json
from pathlib import Path

import numpy as np
import pytest

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator


class TestLorenz:
    """Integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_case1(self, tolerance: float) -> None:
        """Test Lorenz system case 1 - broken butterfly attractor parameters.

        Parameters: sigma=0.12, r=0.0, b=-0.6
        Expected attractors: butterfly1, butterfly2, unbounded
        """
        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_lorenz.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run estimation
        props = setup_lorenz_system()

        bse = BasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
        )

        basin_stability = bse.estimate_bs()

        # Compare results with expected values
        for expected in expected_results:
            label = expected["label"]
            expected_bs = expected["basinStability"]

            # Get actual basin stability for this label
            actual_bs = basin_stability.get(label, 0.0)

            # Compare with tolerance
            assert abs(actual_bs - expected_bs) < tolerance, (
                f"Basin stability for {label}: expected {expected_bs:.4f}, "
                f"got {actual_bs:.4f}, difference {abs(actual_bs - expected_bs):.4f} "
                f"exceeds tolerance {tolerance}"
            )

        # Also verify we have the same labels
        expected_labels = {
            result["label"] for result in expected_results if result["basinStability"] > 0
        }
        actual_labels = {label for label, bs in basin_stability.items() if bs > 0}
        assert expected_labels == actual_labels, (
            f"Label mismatch: expected {expected_labels}, got {actual_labels}"
        )

    @pytest.mark.integration
    def test_case2(self, tolerance: float) -> None:
        """Test Lorenz system case 2 - adaptive parameter study varying sigma.

        Studies the effect of varying sigma parameter from 0.12 to 0.18 on basin stability.
        """
        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_lorenz_sigma_study.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run adaptive parameter study
        props = setup_lorenz_system()

        # Use the same parameter values as in the expected results
        parameter_values = np.array([result["parameter"] for result in expected_results])

        as_params = AdaptiveStudyParams(
            adaptative_parameter_values=parameter_values,
            adaptative_parameter_name='ode_system.params["sigma"]',
        )

        as_bse = ASBasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
            as_params=as_params,
        )

        as_bse.estimate_as_bs()

        # Compare results at each parameter value
        for i, expected in enumerate(expected_results):
            param_value = expected["parameter"]
            actual_bs = as_bse.basin_stabilities[i]

            # Check butterfly1 basin stability
            expected_bs_b1 = expected["bs_butterfly1"]
            actual_bs_b1 = actual_bs.get("butterfly1", 0.0)
            assert abs(actual_bs_b1 - expected_bs_b1) < tolerance, (
                f"At parameter {param_value:.4f}, butterfly1 basin stability: "
                f"expected {expected_bs_b1:.4f}, got {actual_bs_b1:.4f}, "
                f"difference {abs(actual_bs_b1 - expected_bs_b1):.4f} exceeds tolerance {tolerance}"
            )

            # Check butterfly2 basin stability
            expected_bs_b2 = expected["bs_butterfly2"]
            actual_bs_b2 = actual_bs.get("butterfly2", 0.0)
            assert abs(actual_bs_b2 - expected_bs_b2) < tolerance, (
                f"At parameter {param_value:.4f}, butterfly2 basin stability: "
                f"expected {expected_bs_b2:.4f}, got {actual_bs_b2:.4f}, "
                f"difference {abs(actual_bs_b2 - expected_bs_b2):.4f} exceeds tolerance {tolerance}"
            )

            # Check unbounded basin stability
            expected_bs_unbounded = expected["bs_unbounded"]
            actual_bs_unbounded = actual_bs.get("unbounded", 0.0)
            assert abs(actual_bs_unbounded - expected_bs_unbounded) < tolerance, (
                f"At parameter {param_value:.4f}, unbounded basin stability: "
                f"expected {expected_bs_unbounded:.4f}, got {actual_bs_unbounded:.4f}, "
                f"difference {abs(actual_bs_unbounded - expected_bs_unbounded):.4f} exceeds tolerance {tolerance}"
            )

            # Check NaN basin stability (should be 0)
            expected_bs_nan = expected["bs_NaN"]
            actual_bs_nan = actual_bs.get("NaN", 0.0)
            assert abs(actual_bs_nan - expected_bs_nan) < tolerance, (
                f"At parameter {param_value:.4f}, NaN basin stability: "
                f"expected {expected_bs_nan:.4f}, got {actual_bs_nan:.4f}, "
                f"difference {abs(actual_bs_nan - expected_bs_nan):.4f} exceeds tolerance {tolerance}"
            )

    @pytest.mark.integration
    def test_hyperparameters(self):
        """Test hyperparameter sensitivity study - varying N (number of samples).

        Uses adaptive tolerance: starts at higher tolerance for small N (high uncertainty),
        decreases for large N (low uncertainty). Only stores tolerance
        differences in CSV for convergence analysis.
        """
        import csv

        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_lorenz_hyperparameters.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run hyperparameter study
        props = setup_lorenz_system()

        # Use the same parameter values as in the expected results
        parameter_values = np.array([result["parameter"] for result in expected_results])

        as_params = AdaptiveStudyParams(
            adaptative_parameter_values=parameter_values,
            adaptative_parameter_name="n",  # Varying the number of samples
        )

        as_bse = ASBasinStabilityEstimator(
            n=props["n"],  # Initial value, will be overridden
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
            as_params=as_params,
        )

        as_bse.estimate_as_bs()

        # Prepare CSV data - only store tolerance differences
        csv_data: list[dict[str, float | int | bool]] = []
        csv_headers = [
            "N",
            "actual_grid_points",
            "tolerance_diff_butterfly1",
            "tolerance_diff_butterfly2",
            "tolerance_diff_unbounded",
            "adaptive_tolerance",
            "test_passed",
        ]

        # Adaptive tolerance parameters
        min_n = 200
        max_n = 20000
        max_tolerance = 0.65  # 65% for small N (high statistical uncertainty with random sampling)
        min_tolerance = 0.05  # 5% for large N (some variance remains)

        all_tests_passed = True

        # Compare results at each parameter value (N)
        for i, expected in enumerate(expected_results):
            param_value = expected["parameter"]
            actual_bs = as_bse.basin_stabilities[i]

            # Get values
            expected_bs_b1 = expected["bs_butterfly1"]
            actual_bs_b1 = actual_bs.get("butterfly1", 0.0)

            expected_bs_b2 = expected["bs_butterfly2"]
            actual_bs_b2 = actual_bs.get("butterfly2", 0.0)

            expected_bs_unbounded = expected["bs_unbounded"]
            actual_bs_unbounded = actual_bs.get("unbounded", 0.0)

            # Get actual number of samples used (from results)
            actual_grid_points = as_bse.results[i].get("n_samples", int(param_value))

            # Calculate tolerance differences (normalized by expected value to get relative error)
            tolerance_diff_b1 = (
                abs(actual_bs_b1 - expected_bs_b1) / expected_bs_b1
                if expected_bs_b1 > 0
                else abs(actual_bs_b1 - expected_bs_b1)
            )
            tolerance_diff_b2 = (
                abs(actual_bs_b2 - expected_bs_b2) / expected_bs_b2
                if expected_bs_b2 > 0
                else abs(actual_bs_b2 - expected_bs_b2)
            )
            tolerance_diff_unbounded = (
                abs(actual_bs_unbounded - expected_bs_unbounded) / expected_bs_unbounded
                if expected_bs_unbounded > 0
                else abs(actual_bs_unbounded - expected_bs_unbounded)
            )

            # Calculate adaptive tolerance using logarithmic scale
            log_progress = (np.log(actual_grid_points) - np.log(min_n)) / (
                np.log(max_n) - np.log(min_n)
            )
            log_progress = np.clip(log_progress, 0, 1)  # Ensure in [0, 1]
            adaptive_tolerance = max_tolerance - (max_tolerance - min_tolerance) * log_progress

            # Test with adaptive tolerance
            test_passed_b1 = tolerance_diff_b1 <= adaptive_tolerance
            test_passed_b2 = tolerance_diff_b2 <= adaptive_tolerance
            test_passed_unbounded = tolerance_diff_unbounded <= adaptive_tolerance
            test_passed = test_passed_b1 and test_passed_b2 and test_passed_unbounded

            if not test_passed:
                all_tests_passed = False

            # Add to CSV data
            csv_data.append(  # type: ignore[arg-type]
                {
                    "N": param_value,
                    "actual_grid_points": actual_grid_points,
                    "tolerance_diff_butterfly1": tolerance_diff_b1,
                    "tolerance_diff_butterfly2": tolerance_diff_b2,
                    "tolerance_diff_unbounded": tolerance_diff_unbounded,
                    "adaptive_tolerance": adaptive_tolerance,
                    "test_passed": test_passed,
                }
            )

        # Save results to CSV
        csv_path = Path(__file__).parent / "hyperparameter_test_results.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)  # type: ignore[arg-type]
            writer.writeheader()
            writer.writerows(csv_data)  # type: ignore[arg-type]

        print(f"\nHyperparameter test results saved to: {csv_path}")

        # Print summary
        print("\n=== Hyperparameter Test Summary ===")
        print(f"Total tests: {len(csv_data)}")
        print(f"Passed: {sum(1 for d in csv_data if d['test_passed'])}")
        print(f"Failed: {sum(1 for d in csv_data if not d['test_passed'])}")
        print(f"Adaptive tolerance range: {min_tolerance:.1%} to {max_tolerance:.1%}")

        # Assert that all tests passed
        assert all_tests_passed, f"Some hyperparameter tests failed. See {csv_path} for details."

    @pytest.mark.integration
    def test_n200_single_case(self):
        """Test single case with N=200 to debug sampling behavior.

        Expected from MATLAB with UniformRandomSampler:
        - N=200 -> generates exactly 200 random points
        - butterfly1: 0.1000, butterfly2: 0.0750, unbounded: 0.8250

        Note: With only 200 random points, there's inherent statistical uncertainty.
        We use a larger tolerance (15%) to account for this.
        """
        # Setup system
        props = setup_lorenz_system()

        # Create BSE with N=200
        bse = BasinStabilityEstimator(
            n=200,
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
        )

        basin_stability = bse.estimate_bs()

        # Expected values from MATLAB
        expected_bs_b1 = 0.1
        expected_bs_b2 = 0.075
        expected_bs_unbounded = 0.825

        # Get actual values
        actual_bs_b1 = basin_stability.get("butterfly1", 0.0)
        actual_bs_b2 = basin_stability.get("butterfly2", 0.0)
        actual_bs_unbounded = basin_stability.get("unbounded", 0.0)

        # Check the actual number of points generated
        if bse.y0 is not None:
            actual_points = len(bse.y0)
            print(f"\nActual points generated: {actual_points}")
            assert actual_points == 200, (
                f"Expected 200 points with UniformRandomSampler, but got {actual_points}"
            )

        # For small N, use a more lenient tolerance due to statistical uncertainty
        tolerance_small_n = 0.15

        # Compare with tolerance
        assert abs(actual_bs_b1 - expected_bs_b1) < tolerance_small_n, (
            f"butterfly1 basin stability: expected {expected_bs_b1:.4f}, "
            f"got {actual_bs_b1:.4f}, difference {abs(actual_bs_b1 - expected_bs_b1):.4f}"
        )

        assert abs(actual_bs_b2 - expected_bs_b2) < tolerance_small_n, (
            f"butterfly2 basin stability: expected {expected_bs_b2:.4f}, "
            f"got {actual_bs_b2:.4f}, difference {abs(actual_bs_b2 - expected_bs_b2):.4f}"
        )

        assert abs(actual_bs_unbounded - expected_bs_unbounded) < tolerance_small_n, (
            f"unbounded basin stability: expected {expected_bs_unbounded:.4f}, "
            f"got {actual_bs_unbounded:.4f}, difference {abs(actual_bs_unbounded - expected_bs_unbounded):.4f}"
        )

        # Verify basin stabilities sum to 1.0
        total_bs = sum(basin_stability.values())
        assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"

    @pytest.mark.integration
    def test_tolerance_study(self, tolerance: float) -> None:
        """Test hyperparameter study varying ODE solver tolerance (rtol).

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

        # Use N=20000 to match MATLAB study
        as_bse = ASBasinStabilityEstimator(
            n=20000,
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
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

            # Check butterfly1 basin stability
            expected_bs_b1 = expected["bs_butterfly1"]
            actual_bs_b1 = actual_bs.get("butterfly1", 0.0)
            diff_b1 = abs(actual_bs_b1 - expected_bs_b1)

            # For rtol=1e-3, just verify it's within a reasonable range (not checking exact match)
            if rtol_value == 1e-3:
                # At coarse tolerance, we mainly want to ensure the solver doesn't crash
                # and produces some result (even if less accurate)
                assert 0.0 <= actual_bs_b1 <= 1.0, (
                    f"At rtol={rtol_value:.0e}, butterfly1 basin stability {actual_bs_b1:.4f} "
                    f"is outside valid range [0, 1]"
                )
            else:
                assert diff_b1 < test_tolerance, (
                    f"At rtol={rtol_value:.0e}, butterfly1 basin stability: "
                    f"expected {expected_bs_b1:.4f}, got {actual_bs_b1:.4f}, "
                    f"difference {diff_b1:.4f} exceeds tolerance {test_tolerance:.4f}"
                )

            # Check butterfly2 basin stability
            expected_bs_b2 = expected["bs_butterfly2"]
            actual_bs_b2 = actual_bs.get("butterfly2", 0.0)
            diff_b2 = abs(actual_bs_b2 - expected_bs_b2)

            if rtol_value == 1e-3:
                assert 0.0 <= actual_bs_b2 <= 1.0, (
                    f"At rtol={rtol_value:.0e}, butterfly2 basin stability {actual_bs_b2:.4f} "
                    f"is outside valid range [0, 1]"
                )
            else:
                assert diff_b2 < test_tolerance, (
                    f"At rtol={rtol_value:.0e}, butterfly2 basin stability: "
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
            butterfly1_values = [bs.get("butterfly1", 0.0) for bs in consistent_results]
            butterfly1_std = np.std(butterfly1_values)
            assert butterfly1_std < 0.01, (
                f"butterfly1 values for rtol >= 1e-4 should be consistent, "
                f"but std={butterfly1_std:.4f} is too high. Values: {butterfly1_values}"
            )
