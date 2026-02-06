# Lorenz System

## System Description

Lorenz "broken butterfly" attractor:

$$
\begin{aligned}
\dot{x} &= \sigma(y - x) \\
\dot{y} &= rx - y - xz \\
\dot{z} &= xy - bz
\end{aligned}
$$

## Attractors

- **chaos y_1**: Positive x wing (butterfly1)
- **chaos y_2**: Negative x wing (butterfly2)
- **unbounded**: Trajectories that escape to infinity

## Key Feature

Demonstrates unboundedness detection with `event_fn`.

## Reproduction Code

### Setup

{{ load_snippet("case_studies/lorenz/setup_lorenz_system.py::setup_lorenz_system") }}

### Main Estimation

{{ load_snippet("case_studies/lorenz/main_lorenz.py::main") }}

## Case 1: Baseline Results

### Comparison with MATLAB bSTAB

{{ comparison_table("lorenz_case1") }}

### Visualizations

#### Basin Stability

![Basin Stability](../assets/case_studies/lorenz_case1_basin_stability.png)

#### State Space

![State Space](../assets/case_studies/lorenz_case1_state_space.png)

#### Feature Space

![Feature Space](../assets/case_studies/lorenz_case1_feature_space.png)

## Case 2: Sigma Parameter Sweep

### Comparison with MATLAB bSTAB

{{ comparison_table("lorenz_case2") }}

### Visualizations

#### Basin Stability Variation

![Basin Stability Variation](../assets/case_studies/lorenz_case2_basin_stability_variation.png)

#### Bifurcation Diagram

![Bifurcation Diagram](../assets/case_studies/lorenz_case2_bifurcation_diagram.png)

## Case 3: Solver rtol Convergence Study

This hyperparameter study demonstrates the effect of ODE solver relative tolerance on basin stability estimation. Coarse tolerances (rtol=1e-3) produce inaccurate results, while finer tolerances converge to consistent values.

### Comparison with MATLAB bSTAB

{{ comparison_table("lorenz_case3") }}

### Visualizations

#### Basin Stability Variation

![Basin Stability Variation](../assets/case_studies/lorenz_case3_basin_stability_variation.png)

#### Bifurcation Diagram

![Bifurcation Diagram](../assets/case_studies/lorenz_case3_bifurcation_diagram.png)
