"""Integration tests for the Duffing oscillator case study."""

import json
from pathlib import Path

import pytest

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.predictors.dbscan_clusterer import DBSCANClusterer
from tests.integration.test_helpers import run_basin_stability_test


class TestDuffing:
    """Integration tests for Duffing oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline_supervised(self, tolerance: float) -> None:
        """Test Duffing oscillator baseline with supervised classification approach.

        Parameters: δ=0.08, k3=1, A=0.2
        Expected attractors: y1-y5 (various n-cycles)
        Uses supervised KNN classification with known attractor templates.

        Verifies:
        1. Number of ICs used matches sum of absNumMembers from MATLAB
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 2

        Note: Uses larger z-threshold due to chaotic nature of Duffing system.
        """
        json_path = Path(__file__).parent / "main_duffing_supervised.json"
        run_basin_stability_test(json_path, setup_duffing_oscillator_system, z_threshold=2.5)

    @pytest.mark.integration
    def test_baseline_unsupervised(self, tolerance: float) -> None:
        """Test Duffing oscillator baseline with unsupervised clustering approach.

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
        cluster_classifier = DBSCANClusterer(eps=0.08)

        bse = BasinStabilityEstimator(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props.get("solver"),
            feature_extractor=props.get("feature_extractor"),
            predictor=cluster_classifier,
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
