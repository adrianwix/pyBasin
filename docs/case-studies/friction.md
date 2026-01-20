# Friction Oscillator

!!! note "Documentation in Progress"
This page is under construction.

## System Description

Mass-spring-damper with friction:

$$m\ddot{x} + c\dot{x} + kx = F_{friction}(v_{belt} - \dot{x})$$

## Attractors

- **stick_slip**: Stick-slip limit cycle
- **sliding**: Continuous sliding motion

## Expected Results

From integration tests:

```json
{ "stick_slip": 0.65, "sliding": 0.35 }
```
