"""Integration tests for the pendulum case study."""

import json
from pathlib import Path

import numpy as np
import pytest

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator


class TestPendulum:
    """Integration tests for pendulum basin stability estimation."""

    @pytest.mark.integration
    def test_case1(self, tolerance):
        """Test pendulum case 1 parameters."""
        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_pendulum_case1.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run estimation
        props = setup_pendulum_system()

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
            # Use a slightly larger tolerance for numerical variations
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
    def test_case2(self, tolerance):
        """Test pendulum case 2 parameters - adaptive parameter study."""
        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_pendulum_case2.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run adaptive parameter study
        props = setup_pendulum_system()

        # Use the same parameter values as in the expected results
        parameter_values = np.array([result["parameter"] for result in expected_results])

        as_params = AdaptiveStudyParams(
            adaptative_parameter_values=parameter_values,
            adaptative_parameter_name='ode_system.params["T"]',
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

            # Check FP basin stability
            expected_bs_fp = expected["bs_FP"]
            actual_bs_fp = actual_bs.get("FP", 0.0)
            assert abs(actual_bs_fp - expected_bs_fp) < tolerance, (
                f"At parameter {param_value:.3f}, FP basin stability: "
                f"expected {expected_bs_fp:.4f}, got {actual_bs_fp:.4f}, "
                f"difference {abs(actual_bs_fp - expected_bs_fp):.4f} exceeds tolerance {tolerance}"
            )

            # Check LC basin stability
            expected_bs_lc = expected["bs_LC"]
            actual_bs_lc = actual_bs.get("LC", 0.0)
            assert abs(actual_bs_lc - expected_bs_lc) < tolerance, (
                f"At parameter {param_value:.3f}, LC basin stability: "
                f"expected {expected_bs_lc:.4f}, got {actual_bs_lc:.4f}, "
                f"difference {abs(actual_bs_lc - expected_bs_lc):.4f} exceeds tolerance {tolerance}"
            )

            # Check NaN basin stability (should be 0)
            expected_bs_nan = expected["bs_NaN"]
            actual_bs_nan = actual_bs.get("NaN", 0.0)
            assert abs(actual_bs_nan - expected_bs_nan) < tolerance, (
                f"At parameter {param_value:.3f}, NaN basin stability: "
                f"expected {expected_bs_nan:.4f}, got {actual_bs_nan:.4f}, "
                f"difference {abs(actual_bs_nan - expected_bs_nan):.4f} exceeds tolerance {tolerance}"
            )

    @pytest.mark.integration
    def test_grid_sampling(self, tolerance):
        """Test grid-based sampling approach."""
        pytest.skip("To be implemented after case study refactoring")
