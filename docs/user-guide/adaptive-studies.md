# Adaptive Parameter Studies

!!! note "Documentation in Progress"
This page is under construction.

## Use Case

Study how basin stability changes with a system parameter using `ASBasinStabilityEstimator`.

## The `ASBasinStabilityEstimator` Class

Runs BSE multiple times for different parameter values, returning parameter values, BS values, and full results per run.

## Example: Pendulum Damping Study

```python
from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator

as_bse = ASBasinStabilityEstimator(
    n=10_000,
    ode_system=pendulum_ode,
    sampler=sampler,
    as_params={"gamma": np.linspace(0.1, 0.5, 10)},
)
params, bs_vals, results = as_bse.estimate_as_bs()
```

## Visualization with `ASPlotter`

Use the `ASPlotter` class to create parameter vs BS bifurcation diagrams.
