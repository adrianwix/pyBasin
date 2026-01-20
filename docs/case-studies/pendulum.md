# Pendulum

!!! note "Documentation in Progress"
This page is under construction.

## System Description

Driven damped pendulum:

$$\ddot{\theta} + \gamma \dot{\theta} + \sin(\theta) = A \cos(\omega t)$$

## Attractors

- **Fixed Point (FP)**: Pendulum settles to equilibrium
- **Limit Cycle (LC)**: Periodic oscillation

## Minimal Example

```python
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator

props = setup_pendulum_system()
bse = BasinStabilityEstimator(
    ode_system=props["ode_system"],
    sampler=props["sampler"],
)
basin_stability = bse.estimate_bs()
print(basin_stability)  # {'FP': 0.52, 'LC': 0.48}
```

## Expected Results

From integration tests:

```json
{ "FP": 0.518, "LC": 0.482 }
```
