# Duffing Oscillator

!!! note "Documentation in Progress"
This page is under construction.

## System Description

Duffing oscillator with cubic nonlinearity:

$$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)$$

## Attractors

- **LC_pos**: Limit cycle with positive amplitude
- **LC_neg**: Limit cycle with negative amplitude

## Expected Results

From integration tests:

```json
{ "LC_pos": 0.513, "LC_neg": 0.487 }
```
