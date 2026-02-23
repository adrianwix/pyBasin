"""Benchmark the Zig Dopri5 solver on the pendulum system (Zig ODE vs SymPy C ODE).

Region of interest matches case_studies/pendulum/setup_pendulum_system.py:
  theta: [-pi + arcsin(T/K), pi + arcsin(T/K)]
  omega: [-10, 10]

Usage:
    uv run python -m zigode.benchmark_zig_solver
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from zigode.sympy_ode import SymPyODE
from zigode.zig_solver import ODEDefinition, ZigODE, ZigODESolver

N_ICS: int = 100_000
N_STEPS: int = 10_000
T_SPAN: tuple[float, float] = (0.0, 1000.0)
N_RUNS: int = 3
SEED: int = 42

PARAMS: dict[str, float] = {"alpha": 0.1, "T": 0.5, "K": 1.0}

PENDULUM_ZIG: ZigODE = ZigODE(name="pendulum", param_names=["alpha", "T", "K"])


def _make_sympy_pendulum() -> SymPyODE:
    """Create a SymPy pendulum ODE (identical RHS to ``pendulum.zig``)."""
    theta, dtheta = sp.symbols("theta dtheta")
    alpha, T, K = sp.symbols("alpha T K")
    return SymPyODE(
        name="pendulum_sympy",
        state=[theta, dtheta],
        params=[alpha, T, K],
        rhs=[dtheta, -alpha * dtheta + T - K * sp.sin(theta)],
    )


def _benchmark_ode(
    label: str,
    ode: ODEDefinition,
    solver: ZigODESolver,
    y0: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    t_eval: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> None:
    """Run and print benchmark for a single ODE."""
    print(f"=== {label} ===")

    _, y = solver.solve(ode, y0[:10], T_SPAN, t_eval, PARAMS)

    times: list[float] = []
    for i in range(N_RUNS):
        start: float = time.perf_counter()
        _, y = solver.solve(ode, y0, T_SPAN, t_eval, PARAMS)
        elapsed: float = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    best: float = min(times)
    mean: float = float(np.mean(times))

    print()
    print(f"  y shape: {y.shape}")
    print(f"  Memory:  {y.nbytes / 1e9:.2f} GB")
    print(f"  Best:    {best:.3f}s")
    print(f"  Mean:    {mean:.3f}s")
    print(f"  Throughput: {N_ICS / best:,.0f} ICs/s")
    print(f"  Per IC:  {best / N_ICS * 1e6:.1f} µs")
    print()


def main() -> None:
    pendulum_sympy: SymPyODE = _make_sympy_pendulum()
    solver = ZigODESolver()

    t_eval: np.ndarray[tuple[int], np.dtype[np.float64]] = np.linspace(
        T_SPAN[0], T_SPAN[1], N_STEPS
    )

    theta_eq: float = float(np.arcsin(PARAMS["T"] / PARAMS["K"]))
    theta_min: float = -np.pi + theta_eq
    theta_max: float = np.pi + theta_eq
    omega_min: float = -10.0
    omega_max: float = 10.0

    rng = np.random.default_rng(SEED)
    y0: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.column_stack(
        [
            rng.uniform(theta_min, theta_max, N_ICS),
            rng.uniform(omega_min, omega_max, N_ICS),
        ]
    )

    print(f"ICs: {N_ICS:,}  |  Steps: {N_STEPS:,}  |  t_span: {T_SPAN}")
    print(f"Region: theta=[{theta_min:.4f}, {theta_max:.4f}], omega=[{omega_min}, {omega_max}]")
    print(f"y0 shape: {y0.shape}")
    print()

    _benchmark_ode("Zig ODE (pendulum.zig)", PENDULUM_ZIG, solver, y0, t_eval)
    _benchmark_ode("SymPy ODE (pendulum_sympy.c)", pendulum_sympy, solver, y0, t_eval)


if __name__ == "__main__":
    main()
