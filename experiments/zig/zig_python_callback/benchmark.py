"""Benchmark: Zig Dopri5 solver with Python ODE callback via ctypes.

This is the "naive" approach — the ODE is defined in Python and called
back from Zig via the C FFI boundary on every RK stage evaluation.

Expected overhead: each ODE call crosses Python↔Zig boundary (~1-10 μs),
and with ~7 calls/step × ~thousands of steps per IC, this dominates runtime.
"""

import ctypes
import math
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

N_SAMPLES = 10000
DIM = 2
T0 = 0.0
T1 = 1000.0
T_STEADY = 900.0
N_SAVE_TOTAL = 10000
FP_THRESHOLD = 0.01

# ============================================================
# ODE callback (runs in Python, called from Zig)
# ============================================================

# C function type: void(double t, const double* y, double* dydt, const double* params, size_t dim)
ODE_FUNC_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
)


@ODE_FUNC_TYPE
def pendulum_ode(
    t: float,
    y: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
    dydt: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
    params: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
    dim: int,
) -> None:
    """Pendulum ODE: dθ/dt = θ̇, dθ̇/dt = -α·θ̇ + T - K·sin(θ)"""
    theta = y[0]
    theta_dot = y[1]

    alpha = params[0]
    torque = params[1]
    k = params[2]

    dydt[0] = theta_dot
    dydt[1] = -alpha * theta_dot + torque - k * math.sin(theta)


# ============================================================
# Load Zig shared library
# ============================================================

LIB_PATH = Path(__file__).parent / "zig-out" / "lib" / "libzigsolve.so"


def load_library() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        msg = (
            f"Shared library not found at {LIB_PATH}\n"
            f"Build it first: cd experiments/zig_cffi && zig build -Doptimize=ReleaseFast"
        )
        raise FileNotFoundError(msg)

    lib = ctypes.CDLL(str(LIB_PATH))

    # solve_ivp(ode_fn, params, y0, dim, t0, t1, rtol, atol, t_eval, n_eval, result, max_steps)
    lib.solve_ivp.argtypes = [
        ODE_FUNC_TYPE,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
    ]
    lib.solve_ivp.restype = ctypes.c_int32

    # solve_ivp_batch(ode_fn, params, ics, n_ics, dim, t0, t1, rtol, atol, t_eval, n_eval, result, max_steps)
    lib.solve_ivp_batch.argtypes = [
        ODE_FUNC_TYPE,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
    ]
    lib.solve_ivp_batch.restype = ctypes.c_int32

    return lib


# ============================================================
# Basin stability computation
# ============================================================


def generate_initial_conditions(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    seed: int = 42,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    rng = np.random.default_rng(seed)
    offset = np.arcsin(params[1] / params[2])
    theta_min = -np.pi + offset
    theta_max = np.pi + offset

    theta = rng.uniform(theta_min, theta_max, N_SAMPLES)
    theta_dot = rng.uniform(-10.0, 10.0, N_SAMPLES)

    return np.column_stack([theta, theta_dot])


def classify_trajectories(
    result: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    t_eval: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    """Classify each trajectory as FP (0) or LC (1)."""
    steady_mask = t_eval >= T_STEADY
    steady_sol = result[:, steady_mask, 1]

    max_vals = np.max(steady_sol, axis=1)
    mean_vals = np.mean(steady_sol, axis=1)
    delta = np.abs(max_vals - mean_vals)

    return (delta >= FP_THRESHOLD).astype(np.int32)


def main() -> None:
    print("=" * 60)
    print("Zig CFFI Benchmark: Python ODE callback")
    print("=" * 60)

    lib = load_library()

    params = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    t_eval = np.linspace(T0, T1, N_SAVE_TOTAL, dtype=np.float64)
    ics = generate_initial_conditions(params)

    print(f"\nParameters: alpha={params[0]}, T={params[1]}, K={params[2]}")
    print(f"N_SAMPLES={N_SAMPLES}, dim={DIM}")
    print(f"Saving {N_SAVE_TOTAL} points from t={T0:.0f} to t={T1:.0f}")
    print("Solver: Zig Dopri5 (rtol=1e-8, atol=1e-6)")
    print("ODE: Python callback (crosses FFI boundary each call)")

    # --- Single IC warmup ---
    print("\n" + "-" * 60)
    print("Warmup: solving 1 IC...")
    print("-" * 60)
    warmup_result = np.empty(N_SAVE_TOTAL * DIM, dtype=np.float64)
    ret = lib.solve_ivp(
        pendulum_ode,
        params,
        ics[0],
        DIM,
        T0,
        T1,
        1e-8,
        1e-6,
        t_eval,
        N_SAVE_TOTAL,
        warmup_result,
        1_000_000,
    )
    if ret != 0:
        print(f"Solver failed with code {ret}")
        return
    print("done")

    # --- Batch solve ---
    print("\n" + "-" * 60)
    n_batch = min(N_SAMPLES, 100)
    print(f"Solving {n_batch} ICs (batch, single-threaded)...")
    print("-" * 60)

    batch_result = np.empty(n_batch * N_SAVE_TOTAL * DIM, dtype=np.float64)

    start = time.perf_counter()
    ret = lib.solve_ivp_batch(
        pendulum_ode,
        params,
        np.ascontiguousarray(ics[:n_batch]),
        n_batch,
        DIM,
        T0,
        T1,
        1e-8,
        1e-6,
        t_eval,
        N_SAVE_TOTAL,
        batch_result,
        1_000_000,
    )
    elapsed = time.perf_counter() - start

    if ret != 0:
        print(f"Solver failed with code {ret}")
        return

    elapsed_ms = elapsed * 1000
    us_per_ic = (elapsed * 1_000_000) / n_batch

    print(f"\n{n_batch} ICs done in {elapsed_ms:.1f} ms ({us_per_ic:.1f} μs per IC)")

    # Extrapolate to full 10k
    est_full_s = (elapsed / n_batch) * N_SAMPLES
    print(f"Estimated time for {N_SAMPLES} ICs: {est_full_s:.1f} s")

    # --- Classify ---
    trajectories = batch_result.reshape(n_batch, N_SAVE_TOTAL, DIM)
    labels = classify_trajectories(trajectories, t_eval)

    fp_count = int(np.sum(labels == 0))
    lc_count = int(np.sum(labels == 1))
    fp_frac = fp_count / n_batch
    lc_frac = lc_count / n_batch

    print("\n" + "=" * 60)
    print(f"Basin Stability Results ({n_batch} ICs)")
    print("=" * 60)
    print(f"  Fixed Point (FP): {fp_count:5} / {n_batch}  =  {fp_frac:.4f}")
    print(f"  Limit Cycle (LC): {lc_count:5} / {n_batch}  =  {lc_frac:.4f}")
    print("=" * 60)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Performance comparison (estimated for 10,000 ICs)")
    print("=" * 60)
    print("  Zig native (24 threads):     ~494  ms")
    print(f"  Zig + Python ODE callback:   ~{est_full_s * 1000:.0f} ms")
    print("  Diffrax/JAX (JIT, CPU):    ~15238  ms")
    print(f"  Overhead from Python callback: ~{est_full_s / 0.494:.0f}x vs native Zig")
    print("=" * 60)


if __name__ == "__main__":
    main()
