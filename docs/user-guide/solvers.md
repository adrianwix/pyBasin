# Solvers

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Solvers integrate the ODE system from initial conditions.

## Solver Protocol

All solvers implement:

- Method: `integrate(ode_system, initial_conditions) -> (t, y)`
- Property: `device`

## Available Solvers

| Class                 | Backend       | GPU Support | Event Functions     | Recommended For             |
| --------------------- | ------------- | ----------- | ------------------- | --------------------------- |
| `JaxSolver`           | JAX/Diffrax   | ✅ CUDA     | ✅ Yes              | **Default for performance** |
| `TorchDiffEqSolver`   | torchdiffeq   | ✅ CUDA     | ❌ Batch limitation | PyTorch ecosystems          |
| `TorchOdeSolver`      | torchode      | ✅ CUDA     | ❌ No               | Alternative PyTorch         |
| `ScipyParallelSolver` | scipy/sklearn | ❌ CPU only | ❌ No               | Debugging, reference        |

## JaxSolver (Recommended)

```python
from pybasin.solvers.jax_solver import JaxSolver

solver = JaxSolver(
    device="cuda",
    t_span=(0.0, 1000.0),
    n_steps=1000,
    method="Dopri5",  # or "Tsit5"
    rtol=1e-8,
    atol=1e-6,
    event_fn=my_stop_event,  # Optional: stop unbounded
)
```

## Event Functions for Unbounded Detection

```python
import jax.numpy as jnp

def stop_event(t, y, args):
    """Stop when |y| > 200."""
    return 200.0 - jnp.max(jnp.abs(y))
```

## TorchDiffEqSolver

```python
from pybasin.solvers import TorchDiffEqSolver

solver = TorchDiffEqSolver(
    device="cuda",
    t_span=(0.0, 1000.0),
    method="dopri5",
)
```

## Performance Comparison

See the [Solver Comparison](../benchmarks/solvers.md) benchmark for detailed performance data.
