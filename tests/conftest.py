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
    :ivar as_bse: BasinStabilityStudy instance (for parameter sweep tests).
    :ivar comparison: Single ComparisonResult (for single-point tests).
    :ivar comparisons: List of ComparisonResult (for parameter sweep tests).
    :ivar unsupervised_comparison: UnsupervisedComparisonResult (for unsupervised tests).
    """

    bse: BasinStabilityEstimator | None = None
    as_bse: BasinStabilityStudy | None = None
    comparison: ComparisonResult | None = None
    comparisons: list[ComparisonResult] | None = None
    unsupervised_comparison: UnsupervisedComparisonResult | None = None


class ArtifactCollector:
    """Collects artifacts from tests for later generation."""

    def __init__(self) -> None:
        self.entries: list[ArtifactEntry] = []

    def add_single_point(
        self,
        bse: BasinStabilityEstimator,
        comparison: ComparisonResult,
    ) -> None:
        """Add a single-point test result."""
        self.entries.append(ArtifactEntry(bse=bse, comparison=comparison))

    def add_parameter_sweep(
        self,
        as_bse: BasinStabilityStudy,
        comparisons: list[ComparisonResult],
    ) -> None:
        """Add a parameter sweep test result."""
        self.entries.append(ArtifactEntry(as_bse=as_bse, comparisons=comparisons))

    def add_unsupervised(
        self,
        bse: BasinStabilityEstimator,
        comparison: UnsupervisedComparisonResult,
    ) -> None:
        """Add an unsupervised clustering test result."""
        self.entries.append(ArtifactEntry(bse=bse, unsupervised_comparison=comparison))


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
            generate_single_point_artifacts(entry.bse, entry.comparison)
        elif entry.bse is not None and entry.unsupervised_comparison is not None:
            print(
                f"\n📊 {entry.unsupervised_comparison.system_name} - "
                f"{entry.unsupervised_comparison.case_name} (unsupervised)"
            )
            generate_unsupervised_artifacts(entry.bse, entry.unsupervised_comparison)
        elif entry.as_bse is not None and entry.comparisons is not None:
            system_name = entry.comparisons[0].system_name
            case_name = entry.comparisons[0].case_name
            print(f"\n📊 {system_name} - {case_name} (parameter sweep)")
            generate_parameter_sweep_artifacts(entry.as_bse, entry.comparisons)

    print(f"\n{'=' * 60}")
    print("✅ Artifact generation complete")
    print(f"{'=' * 60}\n")
