"""JAX-based ODE solvers.

This package provides high-performance ODE solvers using JAX and Diffrax.

Solvers
-------
JaxSolver : Native JAX solver for JaxODESystem (fastest)
    Use this when you have an ODE system defined with pure JAX operations.

JaxForPytorchSolver : JAX solver for PyTorch ODESystem (compatibility)
    Use this when you need to use an existing PyTorch ODE system with JAX.
    Performance is limited due to PyTorch callbacks.

```python
from pybasin.jax_ode_system import JaxODESystem
from pybasin.solvers import JaxSolver
import jax.numpy as jnp


class MyODE(JaxODESystem):
    def ode(self, t, y):
        return -y

    def get_str(self):
        return "decay"


solver = JaxSolver(time_span=(0, 10), n_steps=100, device="cuda")
y0 = jnp.array([[1.0]])
```
    t, y = solver.integrate(MyODE({}), y0)
"""

from pybasin.solvers.jax_for_pytorch_solver import JaxForPytorchSolver
from pybasin.solvers.jax_solver import JaxSolver

# Backward compatibility: expose JaxForPytorchSolver as JaxPytorchSolver alias
# For users who were using `from pybasin.solver import JaxSolver` (for PyTorch ODEs),
# they should now use `from pybasin.solvers import JaxForPytorchSolver`

__all__ = ["JaxSolver", "JaxForPytorchSolver"]
