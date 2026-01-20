# Basin Stability Estimator Overview

!!! note "Documentation in Progress"
This page is under construction.

## What is Basin Stability?

Basin stability is a nonlinear measure of stability that quantifies the probability that a system returns to a given attractor when perturbed with random initial conditions.

## The `BasinStabilityEstimator` Class

The `BasinStabilityEstimator` is the core class for computing basin stability values.

### Constructor Parameters

| Parameter           | Type                | Default                  | Description                 |
| ------------------- | ------------------- | ------------------------ | --------------------------- |
| `ode_system`        | `ODESystemProtocol` | Required                 | The dynamical system        |
| `sampler`           | `Sampler`           | Required                 | Initial condition generator |
| `n`                 | `int`               | `10_000`                 | Number of samples           |
| `solver`            | `SolverProtocol`    | Auto-detect              | ODE integrator              |
| `feature_extractor` | `FeatureExtractor`  | `TorchFeatureExtractor`  | Feature computation         |
| `predictor`         | `LabelPredictor`    | `HDBSCANClusterer`       | Classification method       |
| `feature_selector`  | `BaseEstimator`     | `DefaultFeatureSelector` | Feature filtering           |
| `detect_unbounded`  | `bool`              | `True`                   | Stop diverging trajectories |
| `save_to`           | `str`               | `None`                   | Output directory            |

## Default Flow

```
Sample ICs → Integrate ODEs → Detect Unbounded → Extract Features
→ Filter Features → Cluster/Classify → Compute BS Values
```

## Automatic Solver Selection

- If `ode_system` is `JaxODESystem` → uses `JaxSolver`
- If `ode_system` is `ODESystem` → uses `TorchDiffEqSolver`

## Unboundedness Detection

Only active when `detect_unbounded=True` AND solver is `JaxSolver` with `event_fn`.

See [Handling Unbounded Trajectories](../guides/unbounded-trajectories.md) for details.

## Output Attributes

- `bse.bs_vals`: Dict of basin stability values per class
- `bse.y0`: Initial conditions tensor
- `bse.solution`: Solution object with trajectories, features, labels
