"""Integration tests for the Duffing oscillator case study."""

import json
from pathlib import Path

import pytest

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import DBSCANCluster


class TestDuffing:
    """Integration tests for Duffing oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_supervised(self, tolerance: float) -> None:
        """Test Duffing oscillator supervised classification approach.

        Parameters: δ=0.08, k3=1, A=0.2
        Expected attractors: y1-y5 (various n-cycles)
        Uses supervised KNN classification with known attractor templates.
        """
        # Use a slightly higher tolerance for Duffing due to its chaotic nature
        tolerance = 0.015

        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_duffing_supervised.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run estimation
        props = setup_duffing_oscillator_system()

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
    def test_unsupervised(self, tolerance: float) -> None:
        """Test Duffing oscillator unsupervised clustering approach.

        Parameters: δ=0.08, k3=1, A=0.2
        Uses unsupervised clustering (DBSCAN) to discover attractors.

        Note: Unsupervised clustering assigns arbitrary numeric labels to clusters,
        so we compare sorted basin stability values rather than label-specific values.
        """
        # Use a slightly higher tolerance for unsupervised due to clustering variations
        tolerance = 0.015

        # Load expected results from JSON
        json_path = Path(__file__).parent / "main_duffing_unsupervised.json"
        with open(json_path) as f:
            expected_results = json.load(f)

        # Setup system and run estimation with DBSCAN clustering
        props = setup_duffing_oscillator_system()

        # Use DBSCAN clustering for unsupervised approach
        cluster_classifier = DBSCANCluster(eps=0.08)

        bse = BasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=cluster_classifier,
            feature_selector=None,
        )

        basin_stability = bse.estimate_bs()

        # Since unsupervised clustering produces arbitrary labels,
        # compare sorted basin stability values
        expected_bs_values = sorted(
            [r["basinStability"] for r in expected_results if r["label"] != "NaN"], reverse=True
        )
        actual_bs_values = sorted(
            [bs for label, bs in basin_stability.items() if label != "NaN"], reverse=True
        )

        # Verify we found the same number of clusters
        assert len(actual_bs_values) == len(expected_bs_values), (
            f"Number of clusters mismatch: expected {len(expected_bs_values)}, "
            f"got {len(actual_bs_values)}"
        )

        # Compare sorted basin stability values
        for i, (expected_bs, actual_bs) in enumerate(
            zip(expected_bs_values, actual_bs_values, strict=True)
        ):
            assert abs(actual_bs - expected_bs) < tolerance, (
                f"Cluster {i} basin stability: expected {expected_bs:.4f}, "
                f"got {actual_bs:.4f}, difference {abs(actual_bs - expected_bs):.4f} "
                f"exceeds tolerance {tolerance}"
            )

        # Verify basin stabilities sum to 1.0
        total_bs = sum(basin_stability.values())
        assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"
