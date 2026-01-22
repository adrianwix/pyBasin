"""Integration tests for the pendulum case study."""

from pathlib import Path

import pytest

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_adaptive_basin_stability_test,
    run_basin_stability_test,
    run_single_point_test,
)


class TestPendulum:
    """Integration tests for pendulum basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum baseline using exact MATLAB initial conditions.

        Uses CsvSampler to load the exact ICs from MATLAB bSTAB, eliminating
        sampling variance. Any differences are due to numerical integration
        or feature extraction only.

        Verifies:
        1. Basin stability values match MATLAB within tight tolerance
        2. Z-score validation: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 0.5
        """
        json_path = Path(__file__).parent / "main_pendulum_case1.json"
        ground_truth_csv = (
            Path(__file__).parent / "ground_truths" / "case1" / "main_pendulum_case1.csv"
        )

        bse, comparison = run_basin_stability_test(
            json_path,
            setup_pendulum_system,
            z_threshold=0.5,
            system_name="pendulum",
            case_name="case1",
            ground_truth_csv=ground_truth_csv,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(bse, comparison)

    @pytest.mark.integration
    def test_parameter_t(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum period (T) parameter sweep using z-score validation.

        Verifies:
        1. Parameter sweep over T (driving period)
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 2
        3. Uses standard errors from both MATLAB (err_FP, err_LC) and Python results
        """
        json_path = Path(__file__).parent / "main_pendulum_case2.json"
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_pendulum_system,
            adaptative_parameter_name='ode_system.params["T"]',
            label_keys=["FP", "LC", "NaN"],
            system_name="pendulum",
            case_name="case2",
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)

    @pytest.mark.integration
    def test_hyperparameter_n(self) -> None:
        """Test hyperparameter n - convergence study varying sample size N.

        Uses z-score validation with standard errors. As N increases, the standard
        error decreases (SE ~ 1/sqrt(N)), so validation naturally becomes stricter.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 2.5
        3. Uses standard errors from both MATLAB (err_FP, err_LC) and Python results
        """
        json_path = Path(__file__).parent / "main_pendulum_hyperparameters.json"
        _as_bse, _comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_pendulum_system,
            adaptative_parameter_name="n",
            label_keys=["FP", "LC", "NaN"],
            z_threshold=2.5,
        )

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n50(self) -> None:
        """Test with small N=50 for grid sampling validation.

        Expected from MATLAB:
        - N=50 -> ceil(50^0.5) = 8 -> 8x8 = 64 grid points
        - FP: 0.1000, LC: 0.9000

        Note: With only 64 grid points, there's inherent statistical uncertainty.
        We use z-score validation with z-threshold=3.0 for small sample size.
        """
        run_single_point_test(
            n=50,
            expected_bs={"FP": 0.1, "LC": 0.9},
            setup_function=setup_pendulum_system,
            z_threshold=3.0,
            expected_points=64,
        )
