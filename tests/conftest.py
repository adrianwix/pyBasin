"""Test configuration and fixtures for pybasin tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.integration.artifact_generator import (
    generate_parameter_sweep_artifacts,
    generate_single_point_artifacts,
    generate_unsupervised_artifacts,
)

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator
    from pybasin.basin_stability_study import BasinStabilityStudy
    from tests.integration.test_helpers import ComparisonResult, UnsupervisedComparisonResult


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "no_artifacts: mark test to skip artifact generation (validation tests with small N)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --generate-artifacts command line option."""
    parser.addoption(
        "--generate-artifacts",
        action="store_true",
        default=False,
        help="Generate documentation artifacts (JSON and plots) after tests pass",
    )


@dataclass
class ArtifactEntry:
    """Entry for artifact collection.

    :ivar bse: BasinStabilityEstimator instance (for single-point tests).
    :ivar bs_study: BasinStabilityStudy instance (for parameter sweep tests).
    :ivar comparison: Single ComparisonResult (for single-point tests).
    :ivar comparisons: List of ComparisonResult (for parameter sweep tests).
    :ivar unsupervised_comparison: UnsupervisedComparisonResult (for unsupervised tests).
    :ivar trajectory_state: State variable index for trajectory plot.
    :ivar trajectory_x_limits: X-axis limits (x_min, x_max) for trajectory plot.
    :ivar trajectory_y_limits: Y-axis limits (y_min, y_max) for trajectory plot.
    :ivar phase_space_axes: Tuple (x_var, y_var) for 2D phase space plot.
    """

    bse: BasinStabilityEstimator | None = None
    bs_study: BasinStabilityStudy | None = None
    comparison: ComparisonResult | None = None
    comparisons: list[ComparisonResult] | None = None
    unsupervised_comparison: UnsupervisedComparisonResult | None = None
    trajectory_state: int = 0
    trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None
    trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None
    phase_space_axes: tuple[int, int] | None = None


class ArtifactCollector:
    """Collects artifacts from tests for later generation."""

    def __init__(self) -> None:
        self.entries: list[ArtifactEntry] = []

    def add_single_point(
        self,
        bse: BasinStabilityEstimator,
        comparison: ComparisonResult,
        trajectory_state: int = 0,
        trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        phase_space_axes: tuple[int, int] | None = None,
    ) -> None:
        """Add a single-point test result.

        :param bse: The BasinStabilityEstimator instance with results.
        :param comparison: The comparison result with validation metrics.
        :param trajectory_state: State variable index for trajectory plot.
        :param trajectory_x_limits: X-axis limits for trajectory plot.
        :param trajectory_y_limits: Y-axis limits for trajectory plot.
        :param phase_space_axes: Tuple (x_var, y_var) for 2D phase space plot.
        """
        self.entries.append(
            ArtifactEntry(
                bse=bse,
                comparison=comparison,
                trajectory_state=trajectory_state,
                trajectory_x_limits=trajectory_x_limits,
                trajectory_y_limits=trajectory_y_limits,
                phase_space_axes=phase_space_axes,
            )
        )

    def add_parameter_sweep(
        self,
        bs_study: BasinStabilityStudy,
        comparisons: list[ComparisonResult],
    ) -> None:
        """Add a parameter sweep test result."""
        self.entries.append(ArtifactEntry(bs_study=bs_study, comparisons=comparisons))

    def add_unsupervised(
        self,
        bse: BasinStabilityEstimator,
        comparison: UnsupervisedComparisonResult,
        trajectory_state: int = 0,
        trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        phase_space_axes: tuple[int, int] | None = None,
    ) -> None:
        """Add an unsupervised clustering test result.

        :param bse: The BasinStabilityEstimator instance with results.
        :param comparison: The unsupervised comparison result with cluster metrics.
        :param trajectory_state: State variable index for trajectory plot.
        :param trajectory_x_limits: X-axis limits for trajectory plot.
        :param trajectory_y_limits: Y-axis limits for trajectory plot.
        :param phase_space_axes: Tuple (x_var, y_var) for 2D phase space plot.
        """
        self.entries.append(
            ArtifactEntry(
                bse=bse,
                unsupervised_comparison=comparison,
                trajectory_state=trajectory_state,
                trajectory_x_limits=trajectory_x_limits,
                trajectory_y_limits=trajectory_y_limits,
                phase_space_axes=phase_space_axes,
            )
        )


_artifact_collector: ArtifactCollector | None = None


@pytest.fixture(scope="session")
def artifact_collector(request: pytest.FixtureRequest) -> ArtifactCollector | None:
    """Session-scoped fixture for collecting artifacts.

    Returns None if --generate-artifacts is not set.
    """
    global _artifact_collector
    if request.config.getoption("--generate-artifacts"):
        _artifact_collector = ArtifactCollector()
        return _artifact_collector
    return None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Generate artifacts after all tests complete successfully."""
    global _artifact_collector

    if _artifact_collector is None:
        return

    if exitstatus != 0:
        print("\n⚠️  Tests failed, skipping artifact generation")
        return

    if not _artifact_collector.entries:
        print("\n⚠️  No artifacts collected")
        return

    print(f"\n{'=' * 60}")
    print("Generating documentation artifacts...")
    print(f"{'=' * 60}")

    for entry in _artifact_collector.entries:
        if entry.bse is not None and entry.comparison is not None:
            print(f"\n📊 {entry.comparison.system_name} - {entry.comparison.case_name}")
            generate_single_point_artifacts(
                entry.bse,
                entry.comparison,
                trajectory_state=entry.trajectory_state,
                trajectory_x_limits=entry.trajectory_x_limits,
                trajectory_y_limits=entry.trajectory_y_limits,
                phase_space_axes=entry.phase_space_axes,
            )
        elif entry.bse is not None and entry.unsupervised_comparison is not None:
            print(
                f"\n📊 {entry.unsupervised_comparison.system_name} - "
                f"{entry.unsupervised_comparison.case_name} (unsupervised)"
            )
            generate_unsupervised_artifacts(
                entry.bse,
                entry.unsupervised_comparison,
                trajectory_state=entry.trajectory_state,
                trajectory_x_limits=entry.trajectory_x_limits,
                trajectory_y_limits=entry.trajectory_y_limits,
                phase_space_axes=entry.phase_space_axes,
            )
        elif entry.bs_study is not None and entry.comparisons is not None:
            system_name = entry.comparisons[0].system_name
            case_name = entry.comparisons[0].case_name
            print(f"\n📊 {system_name} - {case_name} (parameter sweep)")
            generate_parameter_sweep_artifacts(entry.bs_study, entry.comparisons)

    print(f"\n{'=' * 60}")
    print("✅ Artifact generation complete")
    print(f"{'=' * 60}\n")
