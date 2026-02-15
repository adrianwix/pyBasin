# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Pendulum ODE compiled to native C via Cython.

Write the ODE in Python-like syntax, compile once with Cython,
and pass the resulting C function pointer to the Zig solver.
The hot loop runs entirely in native code — no Python, no GIL.

Build:
    cythonize -i pendulum_ode.pyx

The exported `get_fn_ptr()` returns the raw C function address
that the Zig solver can call directly.
"""

from libc.math cimport sin


cdef void pendulum_ode(
    double t,
    const double* y,
    double* dydt,
    const double* params,
    size_t dim,
) noexcept nogil:
    """
    Pendulum ODE: dθ/dt = θ̇, dθ̇/dt = -α·θ̇ + T - K·sin(θ)

    This function has the exact same C signature as the Zig solver's
    COdeFn callback type. Cython compiles it to pure C — calling it
    from Zig has the same cost as calling any C function (~nanoseconds).

    `nogil` means Python's GIL is NOT held during execution,
    enabling safe multi-threaded calls from Zig's thread pool.
    """
    cdef double theta = y[0]
    cdef double theta_dot = y[1]

    cdef double alpha = params[0]
    cdef double torque = params[1]
    cdef double k = params[2]

    dydt[0] = theta_dot
    dydt[1] = -alpha * theta_dot + torque - k * sin(theta)


def get_fn_ptr() -> int:
    """Return the raw C function pointer as a Python integer.

    Python passes this to ctypes, which hands it to the Zig solver.
    From Zig's perspective, it's just a `*const fn(...)` — a normal
    C function pointer with zero overhead per call.
    """
    return <size_t>&pendulum_ode
