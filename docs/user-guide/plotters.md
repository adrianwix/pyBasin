# Plotters

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Plotters visualize basin stability results.

## Available Plotters

| Class                | Type    | Use Case                              |
| -------------------- | ------- | ------------------------------------- |
| `MatplotlibPlotter`  | Static  | Publication figures, quick inspection |
| `InteractivePlotter` | Web app | Exploration, presentations            |
| `ASPlotter`          | Static  | Parameter study bifurcation diagrams  |

## MatplotlibPlotter

```python
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter

plotter = MatplotlibPlotter(bse)
plotter.plot_bse_results()      # 4-panel diagnostic plot
plotter.plot_phase(x_var=0, y_var=1)  # Phase space
plotter.plot_templates(plotted_var=0) # Template time series
```

## InteractivePlotter

```python
from pybasin.plotters.interactive_plotter import InteractivePlotter

plotter = InteractivePlotter(
    bse,
    state_labels={0: "θ", 1: "ω"},
)
plotter.run(port=8050)  # Opens web browser
```

## ASPlotter

Used for visualizing parameter study results from `ASBasinStabilityEstimator`.
