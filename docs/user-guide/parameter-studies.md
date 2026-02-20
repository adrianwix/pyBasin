# Parameter Studies

`BasinStabilityStudy` runs basin stability estimation across multiple parameter configurations. Instead of computing basin stability once, it systematically varies one or more parameters and records how the stability fractions change. This enables bifurcation-style analysis -- tracking how attractor basins grow, shrink, or disappear as system parameters shift.

The class wraps `BasinStabilityEstimator` internally, creating a fresh instance for each parameter combination. Results include basin stability values, error estimates, and steady-state amplitudes needed for bifurcation diagrams.

## When to Use Parameter Studies

Parameter studies answer questions like:

- How does damping strength affect which attractor dominates?
- At what coupling constant does a new attractor basin appear?
- How do basin boundaries shift across a 2D parameter grid?

For single-point basin stability (one parameter configuration), use `BasinStabilityEstimator` directly. When you need to sweep parameters, `BasinStabilityStudy` handles the iteration, logging, memory management, and result aggregation.

## Study Parameter Classes

Four classes define how parameters vary across runs:

| Class               | Pattern            | Use Case                                                 |
| ------------------- | ------------------ | -------------------------------------------------------- |
| `SweepStudyParams`  | 1D sweep           | Single parameter varied across N values                  |
| `GridStudyParams`   | Cartesian product  | All combinations of multiple parameters (grid study)     |
| `ZipStudyParams`    | Parallel iteration | Parameters that must vary together (same-length arrays)  |
| `CustomStudyParams` | User-defined       | Full control over arbitrary parameter combinations       |

## Parameter Paths

Parameters are specified as Python attribute paths relative to the components passed into `BasinStabilityStudy`. The path syntax supports dictionary access for ODE parameters:

```python
# ODE system parameter (dict-style)
'ode_system.params["K"]'
'ode_system.params["sigma"]'

# Direct attribute access
'solver.rtol'
'sampler'  # Replace the entire sampler object

# Sample count
'n'
```

Internally, each assignment executes as `context[path] = value`, so any valid Python attribute chain works.

---

## SweepStudyParams

Sweeps a single parameter through a list of values. Each value produces one run.

```python
import numpy as np
from pybasin.study_params import SweepStudyParams

study_params = SweepStudyParams(
    name='ode_system.params["sigma"]',
    values=list(np.arange(0.12, 0.18, 0.01)),
)
# Runs: sigma=0.12, sigma=0.13, ..., sigma=0.17
```

| Parameter | Type        | Description                                           |
| --------- | ----------- | ----------------------------------------------------- |
| `name`    | `str`       | Parameter path to vary                                |
| `values`  | `list[Any]` | Values to sweep through (converted to list if needed) |

---

## GridStudyParams

Creates all combinations of multiple parameters (Cartesian product). For K values in parameter A and M values in parameter B, this produces K x M runs.

```python
import numpy as np
from pybasin.study_params import GridStudyParams

T_VALUES = list(np.linspace(0.1, 0.9, 5))      # 5 values
ALPHA_VALUES = list(np.linspace(0.05, 0.3, 5)) # 5 values

study_params = GridStudyParams(
    **{
        'ode_system.params["T"]': T_VALUES,
        'ode_system.params["alpha"]': ALPHA_VALUES,
    }
)
# Runs: 5 x 5 = 25 combinations
```

!!! warning "Combinatorial Explosion"
    Grid studies grow exponentially. Three parameters with 10 values each produce 1,000 runs. Start with coarse grids and refine regions of interest.

| Parameter  | Type            | Description                                 |
| ---------- | --------------- | ------------------------------------------- |
| `**params` | `dict[str, list[Any]]` | Mapping of parameter paths to value arrays |

---

## ZipStudyParams

Iterates multiple parameters in parallel (like Python's `zip`). All arrays must have the same length. Use this when parameters must vary together -- for example, loading different CSV samplers alongside corresponding ODE parameters.

```python
import numpy as np
from pybasin.sampler import CsvSampler
from pybasin.study_params import ZipStudyParams

t_values = list(np.arange(0.01, 0.97, 0.05))
samplers = [CsvSampler(f"ground_truth_T_{t:.2f}.csv") for t in t_values]

study_params = ZipStudyParams(
    **{
        'ode_system.params["T"]': t_values,
        'sampler': samplers,
    }
)
# Runs: (T=0.01, sampler_0), (T=0.06, sampler_1), ...
```

| Parameter  | Type            | Description                                          |
| ---------- | --------------- | ---------------------------------------------------- |
| `**params` | `dict[str, list[Any]]` | Mapping of parameter paths to equal-length arrays |

---

## CustomStudyParams

For full control, provide explicit `RunConfig` objects. Each config specifies assignments and a study label for identification.

```python
from pybasin.study_params import CustomStudyParams, RunConfig, ParamAssignment

configs = [
    RunConfig(
        assignments=[
            ParamAssignment('ode_system.params["K"]', 0.1),
            ParamAssignment('n', 500),
        ],
        study_label={"K": 0.1, "n": 500},
    ),
    RunConfig(
        assignments=[
            ParamAssignment('ode_system.params["K"]', 0.2),
            ParamAssignment('n', 1000),
        ],
        study_label={"K": 0.2, "n": 1000},
    ),
]

study_params = CustomStudyParams(configs)
```

A convenience factory builds configs from dictionaries:

```python
study_params = CustomStudyParams.from_dicts([
    {'ode_system.params["K"]': 0.1, 'n': 500},
    {'ode_system.params["K"]': 0.2, 'n': 1000},
])
```

---

## BasinStabilityStudy

The main class that orchestrates parameter sweeps.

### Constructor Parameters

| Parameter             | Type                           | Default     | Description                                                        |
| --------------------- | ------------------------------ | ----------- | ------------------------------------------------------------------ |
| `n`                   | `int`                          | Required    | Number of samples per parameter combination                        |
| `ode_system`          | `ODESystemProtocol`            | Required    | The dynamical system (modified via parameter assignments)          |
| `sampler`             | `Sampler`                      | Required    | Initial condition generator                                        |
| `solver`              | `SolverProtocol`               | Required    | ODE integrator                                                     |
| `feature_extractor`   | `FeatureExtractor`             | Required    | Extracts features from trajectories                                |
| `estimator`           | `BaseEstimator`                | Required    | Sklearn-compatible clusterer or classifier                         |
| `study_params`        | `StudyParams`                  | Required    | Parameter variation specification                                  |
| `template_integrator` | `TemplateIntegrator` or `None` | `None`      | Template ICs for supervised classifiers                            |
| `save_to`             | `str` or `None`                | `"results"` | Output folder path, or `None` to disable saving                    |
| `verbose`             | `bool`                         | `False`     | Show detailed logs from each BSE run                               |

!!! note "Required Components"
    Unlike `BasinStabilityEstimator`, which auto-creates defaults, `BasinStabilityStudy` requires explicit components for `solver`, `feature_extractor`, and `estimator`. This ensures consistent configuration across all parameter combinations.

### Running the Study

Call `run()` to execute basin stability estimation for each parameter combination:

```python
from pybasin.basin_stability_study import BasinStabilityStudy

bs_study = BasinStabilityStudy(
    n=10_000,
    ode_system=ode,
    sampler=sampler,
    solver=solver,
    feature_extractor=feature_extractor,
    estimator=clusterer,
    study_params=study_params,
    save_to="results",
)

results = bs_study.run()
```

The method logs progress with parameter values and basin stability results for each run. GPU acceleration is used automatically when available. Memory is freed after each iteration to prevent accumulation during long studies.

### Accessing Results

After `run()` completes, results are available through properties:

| Property                   | Type                    | Description                                       |
| -------------------------- | ----------------------- | ------------------------------------------------- |
| `results`                  | `list[StudyResult]`     | Full results including labels and amplitudes      |
| `study_labels`             | `list[dict[str, Any]]`  | Parameter values for each run                     |
| `basin_stabilities`        | `list[dict[str, float]]`| Basin stability fractions per run                 |
| `studied_parameter_names`  | `list[str]`             | Names of the varied parameters                    |

Each `StudyResult` dict contains:

```python
{
    "study_label": {"T": 0.5, "alpha": 0.1},  # Parameter values
    "basin_stability": {"0": 0.42, "1": 0.58}, # BS fractions
    "errors": {"0": 0.01, "1": 0.01},          # Error estimates
    "n_samples": 10000,
    "labels": [...],                           # Cluster assignments
    "bifurcation_amplitudes": tensor(...),     # Steady-state amplitudes
}
```

---

## Visualization

Two plotters work with `BasinStabilityStudy` results.

### MatplotlibStudyPlotter

Creates static matplotlib figures for basin stability variation and bifurcation diagrams.

```python
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter

plotter = MatplotlibStudyPlotter(bs_study)

# Basin stability vs parameter value
plotter.plot_basin_stability_variation()

# Bifurcation diagram showing attractor amplitudes
plotter.plot_bifurcation_diagram(dof=[0, 1])

# Save all figures to bs_study.save_to
plotter.save()
plotter.show()  # or use plt.show() directly
```

For multi-parameter studies, specify which parameter to plot on the x-axis:

```python
plotter.plot_basin_stability_variation(parameters=["T"])
plotter.plot_bifurcation_diagram(dof=[0, 1], parameters=["T"])
```

Results are grouped by the other parameters, with each group rendered as a separate curve.

### InteractivePlotter

Launches a Dash web app for exploring results interactively:

```python
from pybasin.plotters.interactive_plotter import InteractivePlotter

plotter = InteractivePlotter(
    bs_study,
    state_labels={0: "theta", 1: "omega"},
)
plotter.run(port=8050)  # Opens browser at localhost:8050
```

See the [Plotters guide](plotters.md) for detailed plotter documentation.

---

## Saving Results

Call `save()` to write results to a JSON file in the `save_to` folder:

```python
bs_study.save()
```

The JSON includes studied parameters, basin stability values for each configuration, region of interest, and ODE system equations.

---

## Complete Example

A typical parameter study workflow:

```python
import numpy as np
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.study_params import SweepStudyParams

# Assume setup_system() returns configured components
props = setup_system()

study_params = SweepStudyParams(
    name='ode_system.params["sigma"]',
    values=list(np.arange(0.12, 0.18, 0.01)),
)

bs_study = BasinStabilityStudy(
    n=10_000,
    ode_system=props["ode_system"],
    sampler=props["sampler"],
    solver=props["solver"],
    feature_extractor=props["feature_extractor"],
    estimator=props["estimator"],
    study_params=study_params,
    template_integrator=props.get("template_integrator"),
    save_to="results_sigma",
)

bs_study.run()

# Visualize
plotter = MatplotlibStudyPlotter(bs_study)
plotter.plot_basin_stability_variation()
plotter.plot_bifurcation_diagram(dof=[0, 1, 2])
plotter.save()  # or plotter.show() for interactive display

# Save
bs_study.save()
```

For full API details, see the [BasinStabilityStudy API reference](../api/parameter-studies.md).
