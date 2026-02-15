"""Pendulum ODE compiled to native C via mypyc.

Write the ODE as a standard typed Python function by compiling
the module with mypyc (bundled with mypy).

Unlike Cython, mypyc produces CPython C extensions -- there is no way
to export a raw C function pointer with a custom signature. The ODE
must still be called through a ctypes callback trampoline, so the FFI
overhead per call remains. The speedup comes only from the ODE math
itself being compiled to C rather than interpreted.

Build:
    cd experiments/zig/zig_mypyc/odes
    uv run mypyc pendulum_ode.py
"""

from math import sin


def compute(
    theta: float,
    theta_dot: float,
    alpha: float,
    torque: float,
    k: float,
) -> tuple[float, float]:
    """Pendulum ODE: d0/dt = 0', d0'/dt = -a*0' + T - K*sin(0).

    Pure math function -- no ctypes types, so mypyc can compile
    all operations (float arithmetic + sin) to native C.
    """
    d_theta: float = theta_dot
    d_theta_dot: float = -alpha * theta_dot + torque - k * sin(theta)
    return (d_theta, d_theta_dot)
