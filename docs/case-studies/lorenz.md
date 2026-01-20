# Lorenz System

!!! note "Documentation in Progress"
This page is under construction.

## System Description

Lorenz "broken butterfly" attractor:

$$\dot{x} = \sigma(y - x)$$
$$\dot{y} = rx - y - xz$$
$$\dot{z} = xy - bz$$

## Attractors

- **butterfly_pos**: Positive x wing
- **butterfly_neg**: Negative x wing
- **unbounded**: Trajectories that escape to infinity

## Key Feature

Demonstrates unboundedness detection with `event_fn`.

## Expected Results

From integration tests:

```json
{ "butterfly_pos": 0.33, "butterfly_neg": 0.33, "unbounded": 0.34 }
```
