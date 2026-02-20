# Plotters

Plotters turn basin stability results into figures -- static matplotlib plots for publication or a Dash web app for interactive exploration. Three plotter classes cover single-run diagnostics, parameter study visualizations, and browser-based drill-down. Each accepts a completed `BasinStabilityEstimator` or `BasinStabilityStudy` and produces ready-to-use output with minimal configuration.

## Available Plotters

| Class                    | Type    | Input                                                    | Best for                              |
| ------------------------ | ------- | -------------------------------------------------------- | ------------------------------------- |
| `MatplotlibPlotter`      | Static  | `BasinStabilityEstimator`                                | Publication figures, quick inspection |
| `InteractivePlotter`     | Web app | `BasinStabilityEstimator` or `BasinStabilityStudy`       | Exploration, presentations            |
| `MatplotlibStudyPlotter` | Static  | `BasinStabilityStudy`                                    | Parameter study bifurcation diagrams  |

---

## MatplotlibPlotter

Generates static matplotlib figures from a single `BasinStabilityEstimator` run.

```python
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter

plotter = MatplotlibPlotter(bse)

# 2x2 diagnostic panel (bar chart, state space, feature space, placeholder)
plotter.plot_bse_results()

# Individual plots
plotter.plot_basin_stability_bars()
plotter.plot_state_space()
plotter.plot_feature_space()

# Template trajectory visualizations
plotter.plot_templates_phase_space(x_var=0, y_var=1, time_range=(700, 1000))
plotter.plot_templates_trajectories(plotted_var=0, y_limits=(-1.4, 1.4))

# Save all pending figures to bse.save_to
plotter.save(dpi=300)
```

### Constructor Parameters

| Parameter | Type                      | Default  | Description                          |
| --------- | ------------------------- | -------- | ------------------------------------ |
| `bse`     | `BasinStabilityEstimator` | Required | BSE instance with computed results.  |

### Methods

| Method                         | Signature                                                                         | Description                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `plot_bse_results`             | `() -> Figure`                                                                    | 2x2 grid combining bar chart, state space, feature space, and a placeholder panel.    |
| `plot_basin_stability_bars`    | `(ax: Axes \| None = None) -> Axes`                                               | Bar chart of basin stability values per attractor.                                    |
| `plot_state_space`             | `(ax: Axes \| None = None) -> Axes`                                               | Scatter plot of initial conditions colored by attractor label.                         |
| `plot_feature_space`           | `(ax: Axes \| None = None) -> Axes`                                               | Feature space scatter with classifier results. Handles 1D (strip plot) and 2D cases.  |
| `plot_templates_phase_space`   | `(x_var=0, y_var=1, z_var=None, time_range=(700, 1000)) -> Figure`                | Template trajectories in 2D or 3D phase space.                                        |
| `plot_templates_trajectories`  | `(plotted_var: int, y_limits=None, x_limits=None) -> Figure`                      | Stacked subplots, one per template trajectory.                                        |
| `save`                         | `(dpi: int = 300) -> None`                                                        | Saves all pending figures as PNG files to `bse.save_to`.                              |
| `show`                         | `() -> None`                                                                      | Calls `plt.show()`.                                                                   |

### Composing into Existing Figures

Methods that accept an optional `ax` parameter (`plot_basin_stability_bars`, `plot_state_space`, `plot_feature_space`) can draw onto an existing matplotlib `Axes`. When `ax=None`, the plotter creates a new figure and tracks it for later `save()`. Passing your own axes lets you compose multiple plots into custom layouts without affecting the pending-figures queue.

### Template Trajectory Plots

Both `plot_templates_phase_space` and `plot_templates_trajectories` re-integrate the template trajectories using a CPU solver clone with 10x the configured `n_steps`, producing smoother curves than the original integration. This happens transparently on each call.

`plot_templates_trajectories` supports per-label axis limits via dictionaries:

```python
plotter.plot_templates_trajectories(
    plotted_var=0,
    y_limits={"attractor_1": (-2.0, 2.0), "attractor_2": (-5.0, 5.0)},
    x_limits=(0, 50),
)
```

---

## InteractivePlotter

Launches a Dash web app with Plotly figures, providing dropdown selectors, axis pickers, and click-to-inspect trajectory modals. It operates in two modes depending on whether it receives a `BasinStabilityEstimator` or a `BasinStabilityStudy`.

```python
from pybasin.plotters.interactive_plotter import InteractivePlotter

plotter = InteractivePlotter(
    bse,
    state_labels={0: "theta", 1: "omega"},
)
plotter.run(port=8050)
```

### Constructor Parameters

| Parameter      | Type                                                    | Default  | Description                                                                |
| -------------- | ------------------------------------------------------- | -------- | -------------------------------------------------------------------------- |
| `bse`          | `BasinStabilityEstimator \| BasinStabilityStudy`        | Required | A completed BSE or study instance. Must have been run before plotting.     |
| `state_labels` | `dict[int, str] \| None`                                | `None`   | Maps state indices to display labels, e.g. `{0: "theta", 1: "omega"}`.    |
| `options`      | `InteractivePlotterOptions \| None`                     | `None`   | Configuration controlling default views and per-page settings.             |

!!! warning "BSE Must Be Run First"
    The constructor raises `ValueError` if the estimator or study has not been executed yet. Call `bse.run()` or `bse.estimate_bs()` before creating the plotter.

### Two Operating Modes

**BSE mode** -- when given a `BasinStabilityEstimator`, the app shows five pages:

- Basin Stability (bar chart)
- State Space (initial conditions colored by label)
- Feature Space (scatter or strip plot of extracted features)
- Templates Phase Space (2D/3D phase portrait)
- Templates Time Series (per-state time series of template trajectories)

**Study mode** -- when given a `BasinStabilityStudy`, the app shows:

- Parameter Overview (basin stability across the parameter sweep)
- Parameter Bifurcation (attractor amplitude evolution)
- Per-parameter-value BSE pages accessible through a dropdown selector

```python
from pybasin.plotters.interactive_plotter import InteractivePlotter

# With a parameter study
plotter = InteractivePlotter(bss, state_labels={0: "theta", 1: "omega"})
plotter.run(port=8050)
```

### InteractivePlotterOptions

Options are passed as a `TypedDict` that deep-merges with sensible defaults -- partial overrides work correctly. Import the type from `pybasin.plotters.types`.

```python
from pybasin.plotters.types import InteractivePlotterOptions

options: InteractivePlotterOptions = {
    "initial_view": "feature_space",
    "templates_phase_space": {"x_axis": 1, "y_axis": 2, "exclude_templates": ["unbounded"]},
    "feature_space": {"exclude_labels": ["unbounded"]},
    "templates_time_series": {"time_range": (0, 0.15)},
}

plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
plotter.run(port=8050, debug=True)
```

**Top-level keys:**

| Key                       | Type                            | Default                | Description                                                                          |
| ------------------------- | ------------------------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `initial_view`            | `ViewType`                      | `"basin_stability"`    | Starting page. Defaults to `"param_overview"` for study mode.                        |
| `state_space`             | `StateSpaceOptions`             | x=0, y=1, time=(0, 1) | State space page defaults.                                                           |
| `feature_space`           | `FeatureSpaceOptions`           | x=0, y=1, filtered     | Feature space page defaults.                                                         |
| `templates_phase_space`   | `TemplatesPhaseSpaceOptions`    | x=0, y=1, z=None       | Phase space page defaults.                                                           |
| `templates_time_series`   | `TemplatesTimeSeriesOptions`    | var=0, time=(0, 1)     | Time series page defaults.                                                           |
| `param_overview`          | `ParamOverviewOptions`          | linear scale            | Parameter overview page defaults (study mode only).                                  |
| `param_bifurcation`       | `ParamBifurcationOptions`       | --                      | Parameter bifurcation page defaults (study mode only).                               |

`ViewType` accepts: `"basin_stability"`, `"state_space"`, `"feature_space"`, `"templates_phase_space"`, `"templates_time_series"`, `"param_overview"`, `"param_bifurcation"`.

### Per-Page Options

**StateSpaceOptions:**

| Key          | Type                    | Description                                              |
| ------------ | ----------------------- | -------------------------------------------------------- |
| `x_axis`     | `int`                   | State index for the x-axis.                              |
| `y_axis`     | `int`                   | State index for the y-axis.                              |
| `time_range` | `tuple[float, float]`   | Normalized time range (0.0 to 1.0) for displayed points. |

**FeatureSpaceOptions:**

| Key              | Type              | Description                                                  |
| ---------------- | ----------------- | ------------------------------------------------------------ |
| `x_axis`         | `int`             | Feature index for the x-axis.                                |
| `y_axis`         | `int \| None`     | Feature index for the y-axis. `None` produces a strip plot.  |
| `use_filtered`   | `bool`            | Whether to use the filtered feature set.                     |
| `include_labels` | `list[str]`       | Show only these attractor labels.                            |
| `exclude_labels` | `list[str]`       | Hide these attractor labels.                                 |

**TemplatesPhaseSpaceOptions:**

| Key                   | Type              | Description                                   |
| --------------------- | ----------------- | --------------------------------------------- |
| `x_axis`              | `int`             | State index for the x-axis.                   |
| `y_axis`              | `int`             | State index for the y-axis.                   |
| `z_axis`              | `int \| None`     | State index for the z-axis (3D plot).         |
| `include_templates`   | `list[str]`       | Show only these template labels.              |
| `exclude_templates`   | `list[str]`       | Hide these template labels.                   |

**TemplatesTimeSeriesOptions:**

| Key                   | Type                                            | Description                                    |
| --------------------- | ----------------------------------------------- | ---------------------------------------------- |
| `state_variable`      | `int`                                           | Which state variable to plot.                  |
| `time_range`          | `tuple[float, float]`                           | Normalized time window.                        |
| `include_templates`   | `list[str]`                                     | Show only these template labels.               |
| `exclude_templates`   | `list[str]`                                     | Hide these template labels.                    |
| `y_limits`            | `tuple[float, float] \| dict[str, tuple[...]]`  | Global or per-label y-axis limits.             |

**ParamOverviewOptions:**

| Key               | Type                       | Description                                |
| ----------------- | -------------------------- | ------------------------------------------ |
| `x_scale`         | `"linear"` or `"log"`     | Scale for the parameter axis.              |
| `selected_labels` | `list[str]`                | Pre-selected attractor labels to display.  |

**ParamBifurcationOptions:**

| Key                | Type         | Description                                  |
| ------------------ | ------------ | -------------------------------------------- |
| `state_dimensions` | `list[int]`  | State dimensions to show in the bifurcation. |

!!! note "Include/exclude are mutually exclusive"
    Providing both `include_*` and `exclude_*` in the same options dict raises `ValueError`. Pick one or the other.

---

## MatplotlibStudyPlotter

Produces static bifurcation-style figures from a `BasinStabilityStudy` parameter sweep.

!!! note "Import path"
    Unlike the other two plotters, `MatplotlibStudyPlotter` lives at `pybasin.matplotlib_study_plotter` -- not inside the `plotters/` subpackage.

```python
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter

plotter = MatplotlibStudyPlotter(bss)
plotter.plot_basin_stability_variation()
plotter.plot_bifurcation_diagram(dof=[1])
plotter.save(dpi=300)
```

### Constructor Parameters

| Parameter  | Type                    | Default  | Description                                 |
| ---------- | ----------------------- | -------- | ------------------------------------------- |
| `bs_study` | `BasinStabilityStudy`   | Required | Study instance with computed results.       |

### Methods

| Method                           | Signature                                                                                    | Description                                                                                                    |
| -------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `plot_basin_stability_variation` | `(interval: "linear" \| "log" = "linear", parameters: list[str] \| None = None) -> list[Figure]` | Basin stability vs. parameter value. One figure per parameter. Supports linear or log x-axis.                  |
| `plot_bifurcation_diagram`       | `(dof: list[int], parameters: list[str] \| None = None) -> list[Figure]`                     | Bifurcation diagram showing attractor amplitude locations. One figure per parameter.                            |
| `save`                           | `(dpi: int = 300) -> None`                                                                   | Saves all pending figures as PNG files to `bs_study.save_to`.                                                  |
| `show`                           | `() -> None`                                                                                 | Calls `plt.show()`.                                                                                            |

The `parameters` argument on both plotting methods filters which study parameters to plot. When `None`, all parameters are included. The `dof` argument in `plot_bifurcation_diagram` selects which state dimensions appear in the diagram -- amplitudes are extracted per dimension using k-means clustering internally.

---

## Saving Figures

Both `MatplotlibPlotter` and `MatplotlibStudyPlotter` track figures internally. Each call to a `plot_*` method adds the resulting figure to a pending queue, and `save()` writes them all as PNG files to the directory specified by `bse.save_to` or `bs_study.save_to`.

!!! warning "Requirements for `save()`"
    `save()` raises `ValueError` if the output directory (`save_to`) has not been set on the estimator or study, or if no figures are pending. Call at least one plot method before saving, and make sure `save_to` points to a valid path.

`show()` is a thin wrapper around `plt.show()` -- useful for quick inspection in notebooks or scripts. It does not clear the pending queue, so you can call both `show()` and `save()` on the same plotter.

For full class signatures and docstrings, see the [API reference](../api/plotters.md).
