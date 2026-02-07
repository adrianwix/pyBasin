"""Integration tests for the Lorenz system case study."""

from pathlib import Path

import pytest

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_adaptive_basin_stability_test,
    run_basin_stability_test,
    run_single_point_test,
)


class TestLorenz:
    """Integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_case1(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz system case 1 - broken butterfly attractor parameters.

        Parameters: sigma=0.12, r=0.0, b=-0.6
        Expected attractors: chaotic attractor 1, chaotic attractor 2, unbounded

        Verifies:
        1. Number of ICs used matches sum of absNumMembers from MATLAB
        2. Classification metrics: MCC >= 0.95
        """
        json_path = Path(__file__).parent / "main_lorenz.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_lorenz.csv"
        label_map = {
            "butterfly1": "chaotic attractor 1",
            "butterfly2": "chaotic attractor 2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_lorenz_system,
            label_map=label_map,
            system_name="lorenz",
            case_name="case1",
            ground_truth_csv=ground_truth_csv,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                phase_space_axes=(0, 2),
            )

    @pytest.mark.integration
    def test_parameter_sigma(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz sigma parameter sweep using classification metrics.

        Studies the effect of varying sigma parameter from 0.12 to 0.18 on basin stability.

        Verifies:
        1. Parameter sweep over sigma
        2. Classification metrics: MCC >= 0.95 for each parameter point
        """
        json_path = Path(__file__).parent / "main_lorenz_sigma_study.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "sigmaStudy"
        label_map = {
            "butterfly1": "chaotic attractor 1",
            "butterfly2": "chaotic attractor 2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name='ode_system.params["sigma"]',
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            system_name="lorenz",
            case_name="case2",
            ground_truths_dir=ground_truths_dir,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)

    @pytest.mark.integration
    def test_hyperparameter_n(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter n - convergence study varying sample size N.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Classification metrics: MCC >= 0.95 for each parameter point
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpN"
        label_map = {
            "butterfly1": "chaotic attractor 1",
            "butterfly2": "chaotic attractor 2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name="n",
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            system_name="lorenz",
            case_name="case4",
            ground_truths_dir=ground_truths_dir,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n200(self) -> None:
        """Test with small N=200 for random sampling validation.

        Expected from MATLAB with UniformRandomSampler:
        - N=200 -> generates exactly 200 random points
        - butterfly1: 0.1000, butterfly2: 0.0750, unbounded: 0.8250

        Note: With only 200 random points, there's inherent statistical uncertainty.
        """
        run_single_point_test(
            n=200,
            expected_bs={
                "chaotic attractor 1": 0.1,
                "chaotic attractor 2": 0.075,
                "unbounded": 0.825,
            },
            setup_function=setup_lorenz_system,
            expected_points=200,
        )

    @pytest.mark.integration
    def test_hyperparameter_rtol(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter rtol - ODE solver relative tolerance convergence study.

        Studies the effect of varying relative tolerance from 1e-3 to 1e-8 on basin stability.
        This test validates that the solver correctly uses the specified tolerances using
        classification metrics to compare against ground truth labels.

        Expected behavior:
        - rtol=1e-3: May produce different results (tolerance is too coarse)
        - rtol=1e-4 to 1e-8: Should converge to consistent classification results
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperpTol.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpTol"
        label_map = {
            "butterfly1": "chaotic attractor 1",
            "butterfly2": "chaotic attractor 2",
            "unbounded": "unbounded",
            "NaN": "NaN",
        }
        as_bse, comparisons = run_adaptive_basin_stability_test(
            json_path,
            setup_lorenz_system,
            adaptative_parameter_name="solver.rtol",
            label_keys=["butterfly1", "butterfly2", "unbounded", "NaN"],
            label_map=label_map,
            system_name="lorenz",
            case_name="case3",
            ground_truths_dir=ground_truths_dir,
            mcc_threshold=0.4,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(as_bse, comparisons)
