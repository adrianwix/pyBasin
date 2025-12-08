"""Integration tests for the friction oscillator case study."""

import json
from pathlib import Path

import numpy as np
import pytest

from case_studies.friction.setup_friction_system import setup_friction_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator


class TestFriction:
    """Integration tests for friction oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_case1(self, tolerance: float) -> None:
        """Test friction oscillator case 1 parameters.

        Parameters: v_d=1.5, ξ=0.05, μsd=2.0, μd=0.5, μv=0.0, v0=0.5
        Expected attractors: FP (fixed point), LC (limit cycle)

        Note: Uses larger tolerance due to small sample size (N=1000)
        and sensitivity of friction system.
        """
        # Use larger tolerance for friction system due to small N and sensitivity
        tolerance = 0.04

        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_friction_case1.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run estimation
        props = setup_friction_system()

        bse = BasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
            feature_selector=None,
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

        # Verify basin stabilities sum to 1.0
        total_bs = sum(basin_stability.values())
        assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"

    @pytest.mark.integration
    def test_case_v_study(self, tolerance: float) -> None:
        """Test friction oscillator case 2 - adaptive parameter study varying v_d.

        Studies the effect of varying driving velocity v_d from 1.85 to 2.0.
        """
        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_friction_v_study.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run adaptive parameter study
        props = setup_friction_system()

        # Use the same parameter values as in the expected results
        parameter_values = np.array([result["parameter"] for result in expected_results])

        as_params = AdaptiveStudyParams(
            adaptative_parameter_values=parameter_values,
            adaptative_parameter_name='ode_system.params["v_d"]',
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
