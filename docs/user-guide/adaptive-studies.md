# Adaptive Parameter Studies

!!! note "Documentation in Progress"
This page is under construction.

## Use Case

Study how basin stability changes with a system parameter using `BasinStabilityStudy`.

## The `BasinStabilityStudy` Class

Runs BSE multiple times for different parameter values, returning parameter values, BS values, and full results per run.

## Example: Pendulum Damping Study

```python
import numpy as np
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.study_params import SweepStudyParams

study_params = SweepStudyParams(
    name='ode_system.params["gamma"]',
    values=np.linspace(0.1, 0.5, 10),
)

as_bse = BasinStabilityStudy(
    n=10_000,
    ode_system=pendulum_ode,
    sampler=sampler,
    solver=solver,
    feature_extractor=feature_extractor,
    estimator=predictor,
    study_params=study_params,
)
labels, bs_vals, results = as_bse.estimate_as_bs()
```

## Visualization with `ASPlotter`

Use the `ASPlotter` class to create parameter vs BS bifurcation diagrams.
