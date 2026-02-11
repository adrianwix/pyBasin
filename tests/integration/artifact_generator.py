"""Artifact generator for case study documentation.

Generates JSON comparison results and plot images from integration tests.
Uses CPSME color palette and export utilities for thesis-quality plots.
"""

# pyright: basic

import json
from pathlib import Path

import matplotlib.pyplot as plt

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.thesis_utils import (
    LORENZ_PALETTE,
    recolor_axes,
    recolor_figure,
    recolor_stacked_figure,
    thesis_export,
)
from tests.integration.test_helpers import ComparisonResult, UnsupervisedComparisonResult

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DOCS_ASSETS_DIR = Path(__file__).parent.parent.parent / "docs" / "assets" / "case_studies"


def ensure_directories() -> None:
    """Create artifact directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_single_point_artifacts(
    bse: BasinStabilityEstimator,
    comparison: ComparisonResult,
    trajectory_state: int = 0,
    trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    phase_space_axes: tuple[int, int] | None = None,
) -> None:
    """Generate artifacts for a single-point basin stability test.

    Writes comparison JSON to artifacts/results/ and plot images to docs/assets/.

    :param bse: The BasinStabilityEstimator instance with results.
    :param comparison: The comparison result with validation metrics.
    :param trajectory_state: State variable index for trajectory plot.
    :param trajectory_x_limits: X-axis limits. Tuple applies to all, dict maps label to limits.
    :param trajectory_y_limits: Y-axis limits. Tuple applies to all, dict maps label to limits.
    :param phase_space_axes: Tuple (x_axis, y_axis) for 2D phase space plot. If None, skipped.
    """
    ensure_directories()

    system_name = comparison.system_name
    case_name = comparison.case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    # Use Lorenz-specific palette for Lorenz system
    palette = LORENZ_PALETTE if system_name == "lorenz" else None

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)

    print(f"  Written: {json_path}")

    plotter = MatplotlibPlotter(bse)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_basin_stability_bars(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_basin_stability.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_state_space(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_state_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_feature_space(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_feature_space.png", DOCS_ASSETS_DIR)

    # Generate stacked trajectory plot if template_integrator is available
    if bse.template_integrator is not None:
        fig = plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        recolor_stacked_figure(fig, palette)
        thesis_export(fig, f"{prefix}_trajectories.png", DOCS_ASSETS_DIR)

    # Generate 2D phase space plot if axes specified
    if phase_space_axes is not None and bse.template_integrator is not None:
        fig = plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        recolor_figure(fig, palette)
        thesis_export(fig, f"{prefix}_phase_space.png", DOCS_ASSETS_DIR)


def generate_unsupervised_artifacts(
    bse: BasinStabilityEstimator,
    comparison: UnsupervisedComparisonResult,
    trajectory_state: int = 0,
    trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    phase_space_axes: tuple[int, int] | None = None,
) -> None:
    """Generate artifacts for an unsupervised clustering test.

    Writes comparison JSON (including cluster purity metrics) to artifacts/results/
    and plot images to docs/assets/.

    :param bse: The BasinStabilityEstimator instance with results.
    :param comparison: The unsupervised comparison result with cluster metrics.
    :param trajectory_state: State variable index for trajectory plot.
    :param trajectory_x_limits: X-axis limits. Tuple applies to all, dict maps label to limits.
    :param trajectory_y_limits: Y-axis limits. Tuple applies to all, dict maps label to limits.
    :param phase_space_axes: Tuple (x_axis, y_axis) for 2D phase space plot. If None, skipped.
    """
    ensure_directories()

    system_name = comparison.system_name
    case_name = comparison.case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    # Use Lorenz-specific palette for Lorenz system
    palette = LORENZ_PALETTE if system_name == "lorenz" else None

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)

    print(f"  Written: {json_path}")

    plotter = MatplotlibPlotter(bse)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_basin_stability_bars(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_basin_stability.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_state_space(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_state_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotter.plot_feature_space(ax=ax)
    recolor_axes(ax, palette)
    thesis_export(fig, f"{prefix}_feature_space.png", DOCS_ASSETS_DIR)

    # Generate stacked trajectory plot if template_integrator is available
    if bse.template_integrator is not None:
        fig = plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        recolor_stacked_figure(fig, palette)
        thesis_export(fig, f"{prefix}_trajectories.png", DOCS_ASSETS_DIR)

    # Generate 2D phase space plot if axes specified
    if phase_space_axes is not None and bse.template_integrator is not None:
        fig = plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        recolor_figure(fig, palette)
        thesis_export(fig, f"{prefix}_phase_space.png", DOCS_ASSETS_DIR)


def generate_parameter_sweep_artifacts(
    as_bse: BasinStabilityStudy,
    comparisons: list[ComparisonResult],
) -> None:
    """Generate artifacts for a parameter sweep basin stability test.

    Writes comparison JSON to artifacts/results/ and plot images to docs/assets/.

    :param as_bse: The BasinStabilityStudy instance with results.
    :param comparisons: List of comparison results, one per parameter point.
    """
    if not comparisons:
        return

    ensure_directories()

    system_name = comparisons[0].system_name
    case_name = comparisons[0].case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    # Use Lorenz-specific palette for Lorenz system
    palette = LORENZ_PALETTE if system_name == "lorenz" else None

    # Check if this is a paper validation (propagate from first comparison)
    is_paper_validation = comparisons[0].paper_validation if comparisons else False

    data: dict[
        str,
        str
        | float
        | bool
        | list[dict[str, str | float | list[dict[str, str | float | bool]] | None]],
    ] = {
        "system_name": system_name,
        "case_name": case_name,
        "parameter_results": [c.to_dict() for c in comparisons],
    }
    if is_paper_validation:
        data["paper_validation"] = True

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Written: {json_path}")

    plotter = MatplotlibStudyPlotter(as_bse)

    fig = plotter.plot_basin_stability_variation(show=False)
    recolor_figure(fig, palette)
    thesis_export(fig, f"{prefix}_basin_stability_variation.png", DOCS_ASSETS_DIR)

    state_dim = as_bse.sampler.state_dim
    if state_dim > 3:
        print(f"  Skipped bifurcation_diagram plot: state_dim={state_dim} > 3")
        return

    dof = list(range(state_dim))

    try:
        fig = plotter.plot_bifurcation_diagram(dof=dof, show=False)
        recolor_figure(fig, palette)
        thesis_export(fig, f"{prefix}_bifurcation_diagram.png", DOCS_ASSETS_DIR)
    except (ValueError, KeyError) as e:
        print(f"  Skipped bifurcation_diagram plot: {e}")
