# Friction Oscillator

## System Description

Mass-spring-damper with friction:

$$m\ddot{x} + c\dot{x} + kx = F_{friction}(v_{belt} - \dot{x})$$

## Attractors

- **FP**: Fixed point (stick state)
- **LC**: Limit cycle (stick-slip oscillation)

## Reproduction Code

### Setup

{{ load_snippet("case_studies/friction/setup_friction_system.py::setup_friction_system") }}

### Main Estimation

{{ load_snippet("case_studies/friction/main_friction.py::main") }}

## Case 1: Baseline Results

### Comparison with MATLAB bSTAB

{{ comparison_table("friction_case1") }}

### Visualizations

#### Basin Stability

![Basin Stability](../assets/case_studies/friction_case1_basin_stability.png)

#### State Space

![State Space](../assets/case_studies/friction_case1_state_space.png)

#### Feature Space

![Feature Space](../assets/case_studies/friction_case1_feature_space.png)

## Case 2: v_d Parameter Sweep

### Comparison with MATLAB bSTAB

{{ comparison_table("friction_case2") }}

### Visualizations

#### Basin Stability Variation

![Basin Stability Variation](../assets/case_studies/friction_case2_basin_stability_variation.png)

#### Bifurcation Diagram

![Bifurcation Diagram](../assets/case_studies/friction_case2_bifurcation_diagram.png)
