"""Integration tests for the friction oscillator case study."""

from pathlib import Path

import pytest

from case_studies.friction.setup_friction_system import setup_friction_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_adaptive_basin_stability_test,
    run_basin_stability_test,
)


class TestFriction:
    """Integration tests for friction oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        tolerance: float,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator baseline parameters.

        Parameters: v_d=1.5, ξ=0.05, μsd=2.0, μd=0.5, μv=0.0, v0=0.5
        Expected attractors: FP (fixed point), LC (limit cycle)

        Verifies:
        1. Number of ICs used matches sum of absNumMembers from MATLAB
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 2

        Note: Uses larger z-threshold due to small sample size (N=1000)
        and sensitivity of friction system.
        """
        json_path = Path(__file__).parent / "main_friction_case1.json"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_friction_system,
            z_threshold=2.5,
            system_name="friction",
            case_name="case1",
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(bse, comparison)

    @pytest.mark.integration
    def test_parameter_v_d(
        self,
        tolerance: float,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator v_d parameter sweep - adaptive parameter study varying v_d.

        Studies the effect of varying driving velocity v_d from 1.85 to 2.0.

        Verifies:
        1. Parameter sweep over v_d (driving velocity)
        2. Basin stability values pass z-score test for FP, LC, NaN
        """
        json_path = Path(__file__).parent / "main_friction_v_study.json"
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_friction_system,
            adaptative_parameter_name='ode_system.params["v_d"]',
            label_keys=["FP", "LC", "NaN"],
            system_name="friction",
            case_name="case2",
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)
